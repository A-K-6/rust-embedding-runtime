#!/usr/bin/env python3
"""Bounded Linux ABBA comparison of two real F32 JSONL runtime binaries."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import time

from compare import distribution, parse_cpu_list, validate_affinity, validate_worker_cpus, worker_affinity

HERE = Path(__file__).resolve().parent
REFERENCE = HERE/'image-reference-0.3.0.json'
REFERENCE_SHA = 'd1dc177ea465ef242e44d254a5c004e3cbcffc35df481891304a825cef21ffbb'
GIB = 1024**3
THREAD_ENV = dict(RAYON_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', BLIS_NUM_THREADS='2',
    TOKENIZERS_PARALLELISM='false', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')


class ProbeError(Exception):
    """Only fixed error codes are serialized, never exception text or worker logs."""


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    return {'name':path.name, 'sha256':file_sha(path)}


def cgroup(pid):
    relative = next(line[3:] for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    # A cgroup namespace may expose its own root at /sys/fs/cgroup.
    root = Path('/sys/fs/cgroup')/relative.lstrip('/')
    if not (root/'cpu.max').is_file():
        root = Path('/sys/fs/cgroup')
    names = ('cpu.max','cpu.stat','memory.max','memory.current','memory.peak','memory.events',
             'cpuset.cpus.effective')
    return {name:(root/name).read_text().strip() for name in names if (root/name).exists()}


def controlled(resources):
    try:
        quota, period = map(int, resources['cpu.max'].split())
        return period > 0 and quota == 2*period and int(resources['memory.max']) == 3*GIB
    except (KeyError, ValueError):
        return False


def snapshot(pid):
    root = Path(f'/proc/{pid}')
    status = dict(line.split(':',1) for line in (root/'status').read_text().splitlines())
    stat = (root/'stat').read_text().rsplit(') ',1)[1].split()
    rollup = dict(line.split(':',1) for line in (root/'smaps_rollup').read_text().splitlines() if ':' in line)
    return {'rss_mib':int(status['VmRSS'].split()[0])/1024,
        'peak_rss_mib':int(status['VmHWM'].split()[0])/1024, 'threads':int(status['Threads']),
        'smaps_rollup_mib':{key:int(rollup[key].split()[0])/1024
                            for key in ('Rss','Pss','Anonymous','Private_Dirty')},
        'minor_faults':int(stat[7]), 'major_faults':int(stat[9]),
        'cpu_ticks':int(stat[11])+int(stat[12]), 'cgroup':cgroup(pid)}


def worker_environment(pid):
    values = {}
    for part in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
        key, separator, value = part.partition(b'=')
        name = os.fsdecode(key)
        if separator and name in THREAD_ENV:
            values[name] = os.fsdecode(value)
    if values != THREAD_ENV:
        raise ProbeError('worker_environment_mismatch')
    return values


def parity(actual, expected):
    if len(actual) != 768 or len(expected) != 768 or any(type(x) not in (int,float) or not math.isfinite(x) for x in actual):
        raise ProbeError('invalid_vector')
    actual_norm = sum(x*x for x in actual)
    if abs(actual_norm-1.) > 1e-4:
        raise ProbeError('invalid_vector_norm')
    denominator = math.sqrt(actual_norm*sum(x*x for x in expected))
    if not denominator:
        raise ProbeError('invalid_vector_norm')
    maximum = max(abs(a-b) for a,b in zip(actual,expected))
    cosine = sum(a*b for a,b in zip(actual,expected))/denominator
    return {'max_absolute_error':maximum, 'cosine':cosine,
            'passed':maximum <= 1e-6 and cosine >= .99999999}


def ready_metadata(value):
    if (value.get('threads') != 2 or value.get('device') != 'cpu' or value.get('dtype') != 'f32'
            or value.get('weight_quantization') != 'f32'):
        raise ProbeError('worker_not_two_thread_cpu_f32')
    if value.get('cpu_backend') not in ('aocl-dlp','accelerate','mkl','gemm'):
        raise ProbeError('unknown_cpu_backend')
    result = {key:value[key] for key in ('threads','device','dtype','weight_quantization','cpu_backend')}
    for key in ('load_ms','linear_weight_bytes','packed_weight_bytes','max_tokens'):
        if key in value:
            if type(value[key]) not in (int,float) or not math.isfinite(value[key]) or value[key] < 0:
                raise ProbeError('invalid_readiness_metric')
            result[key] = value[key]
    for key in ('startup_allocator_trimmed','fused_rope','fused_gelu','fused_rms_norm','gemm_avx512_runtime_dispatch'):
        if key in value:
            if type(value[key]) is not bool:
                raise ProbeError('invalid_readiness_flag')
            result[key] = value[key]
    return result


class Worker:
    def __init__(self, binary, model, cpus):
        env = os.environ.copy()
        for key in ('LD_PRELOAD','LD_AUDIT','OMP_DYNAMIC','OMP_PROC_BIND'):
            env.pop(key, None)
        env.update(THREAD_ENV)
        self.process = subprocess.Popen([shutil.which('taskset') or 'taskset','-c',','.join(map(str,cpus)),
            str(binary),'--quant','f32',str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.buffers = {'stdout':b'', 'stderr':b''}
        self.received = 0
        try:
            for name in self.buffers:
                self.selector.register(getattr(self.process,name), selectors.EVENT_READ, name)
        except BaseException:
            self.close()
            raise

    def receive(self, wanted, timeout):
        deadline = time.monotonic()+timeout
        while True:
            while b'\n' in self.buffers[wanted]:
                raw, self.buffers[wanted] = self.buffers[wanted].split(b'\n',1)
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError):
                    if wanted == 'stderr':
                        continue
                    raise ProbeError('invalid_worker_json') from None
                if not isinstance(value,dict):
                    raise ProbeError('invalid_worker_record')
                return value
            if wanted == 'stdout':
                self.buffers['stderr'] = b''  # Drain diagnostics, never publish logs.
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise ProbeError('worker_response_deadline')
            events = self.selector.select(remaining)
            if not events:
                raise ProbeError('worker_response_deadline')
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    self.selector.unregister(key.fileobj)
                    if key.data == wanted:
                        raise ProbeError('worker_eof')
                    continue
                self.received += len(chunk)
                if self.received > 4*1024*1024:
                    raise ProbeError('worker_output_budget')
                self.buffers[key.data] += chunk
                if len(self.buffers[key.data]) > 1024*1024:
                    raise ProbeError('worker_line_budget')

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for name in ('stdin','stdout','stderr'):
            getattr(self.process,name).close()
        self.selector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path, help='Previous embedding-jsonl executable')
    parser.add_argument('--candidate', required=True, type=Path, help='Candidate embedding-jsonl executable')
    parser.add_argument('--model', required=True, type=Path, help='Pinned local text-only model assets')
    parser.add_argument('--cpus', required=True, type=parse_cpu_list, help='Two physical cores sharing the last-level cache')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-uncontrolled', action='store_true', help='Explicitly label missing quota/memory control; never hide it')
    args = parser.parse_args()
    started = time.monotonic()
    trials = [{'trial':i,'runtime':name,'status':'not_started','rows':[]}
              for i,name in enumerate(('previous','candidate','candidate','previous'))]
    result = {'schema_version':1,'status':'starting','trials':trials,
        'budget':{'wall_seconds':300,'startup_seconds':60,'response_seconds':30,
                  'inferences_per_trial':24,'total_inferences':96},
        'protocol':{'threads':2,'order':['previous','candidate','candidate','previous'],
                    'synthetic_id_length':128,'calls_per_cycle':4,'cycles':6,
                    'timings_include_first_use':True,'quantile_method':'nearest rank',
                    'memory_method':'process RSS and smaps_rollup after readiness and each inference; VmHWM retained separately'},
        'allow_uncontrolled':args.allow_uncontrolled, 'attempted_inferences':0,
        'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'), 'environment':THREAD_ENV}
    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary = args.output.with_name(args.output.name+'.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        temporary.replace(args.output)
    def expire(*_):
        raise ProbeError('overall_wall_budget')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(300)
    exit_code = 1
    try:
        if platform.system() != 'Linux' or not shutil.which('taskset'):
            raise ProbeError('linux_taskset_required')
        if file_sha(REFERENCE) != REFERENCE_SHA:
            raise ProbeError('reference_hash_mismatch')
        reference = json.loads(REFERENCE.read_text())
        if len(reference['cases']) != 3:
            raise ProbeError('reference_case_count')
        result['reference'] = artifact(REFERENCE)
        result['model_revision'] = reference['model_revision']
        result['model_assets'] = {}
        for name, expected in reference['model_asset_sha256'].items():
            value = artifact(args.model/name)
            if value['sha256'] != expected:
                raise ProbeError('model_asset_hash_mismatch')
            result['model_assets'][name] = value
        binaries = {'previous':args.previous.expanduser().absolute(), 'candidate':args.candidate.expanduser().absolute()}
        result['binaries'] = {name:artifact(path) for name,path in binaries.items()}
        if result['binaries']['previous']['sha256'] == result['binaries']['candidate']['sha256']:
            raise ProbeError('identical_binaries_not_a_version_comparison')
        result['driver'] = artifact(Path(__file__))
        result['driver_dependencies'] = {'compare.py':artifact(HERE/'compare.py')}
        result['driver_cgroup'] = cgroup(os.getpid())
        result['resource_conditions_met'] = controlled(result['driver_cgroup'])
        result['control_label'] = 'controlled' if result['resource_conditions_met'] else 'uncontrolled'
        if not result['resource_conditions_met'] and not args.allow_uncontrolled:
            raise ProbeError('requires_two_cpu_three_gib_cgroup')
        result['cpu_selection'] = validate_worker_cpus(args.cpus)
        result['driver_affinity'] = sorted(os.sched_getaffinity(0))
        requests = []
        for repeat in range(6):
            requests.extend(({'text':case['text'],'dims':768}, case['embedding'],
                {'kind':'official','case_index':i,'repeat':repeat,'expected_tokens':case['tokens']})
                for i,case in enumerate(reference['cases']))
            requests.append(({'ids':[2]*128,'dims':768},None,
                             {'kind':'synthetic_ids','length':128,'repeat':repeat,'expected_tokens':128}))
        synthetic_reference = None
        for trial in trials:
            worker = None
            trial['status'] = 'starting'
            save()
            try:
                trial['binary_sha256'] = file_sha(binaries[trial['runtime']])
                if trial['binary_sha256'] != result['binaries'][trial['runtime']]['sha256']:
                    raise ProbeError('binary_changed_before_trial')
                launch = time.monotonic()
                worker = Worker(binaries[trial['runtime']],args.model.expanduser().absolute(),args.cpus)
                startup_deadline = launch+60
                while True:
                    ready = worker.receive('stderr',max(.001,startup_deadline-time.monotonic()))
                    if ready.get('ready') is True:
                        break
                    if time.monotonic() >= startup_deadline:
                        raise ProbeError('startup_deadline')
                trial['ready'] = ready_metadata(ready)
                trial['startup_wall_ms'] = (time.monotonic()-launch)*1000
                trial['environment'] = worker_environment(worker.process.pid)
                trial['after_ready'] = snapshot(worker.process.pid)
                trial['resource_conditions_met'] = controlled(trial['after_ready']['cgroup'])
                if not trial['resource_conditions_met']:
                    result.update(resource_conditions_met=False,control_label='uncontrolled')
                    if not args.allow_uncontrolled:
                        raise ProbeError('worker_cgroup_uncontrolled')
                trial['affinity'] = worker_affinity(worker.process.pid)
                validate_affinity(trial['affinity'],args.cpus)
                for request, expected, metadata in requests:
                    if result['attempted_inferences'] >= 96:
                        raise ProbeError('inference_budget')
                    row = dict(metadata,status='started')
                    trial['rows'].append(row)
                    result['attempted_inferences'] += 1
                    sent = time.monotonic()
                    worker.process.stdin.write(json.dumps(request).encode()+b'\n')
                    worker.process.stdin.flush()
                    value = worker.receive('stdout',30)
                    row['wall_ms'] = (time.monotonic()-sent)*1000
                    if 'error' in value:
                        raise ProbeError('worker_inference_error')
                    if value.get('tokens') != metadata['expected_tokens']:
                        raise ProbeError('token_count_mismatch')
                    elapsed = value.get('elapsed_ms')
                    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed < 0:
                        raise ProbeError('invalid_encoder_timing')
                    row.update(encoder_ms=elapsed,tokens=value['tokens'])
                    vector = value.get('embedding')
                    if not isinstance(vector,list):
                        raise ProbeError('missing_vector')
                    if expected is None:
                        if synthetic_reference is None:
                            parity(vector,vector)  # Validate before retaining the private in-memory baseline.
                            synthetic_reference = vector
                        expected = synthetic_reference
                    row['parity'] = parity(vector,expected)
                    row['resources'] = snapshot(worker.process.pid)
                    validate_affinity(worker_affinity(worker.process.pid),args.cpus)
                    if not controlled(row['resources']['cgroup']):
                        trial['resource_conditions_met'] = False
                        result.update(resource_conditions_met=False,control_label='uncontrolled')
                        if not args.allow_uncontrolled:
                            raise ProbeError('worker_cgroup_changed')
                    if not row['parity']['passed']:
                        raise ProbeError('numerical_parity_failed')
                    row['status'] = 'complete'
                    save()
                if file_sha(binaries[trial['runtime']]) != trial['binary_sha256']:
                    raise ProbeError('binary_changed_during_trial')
                trial['status'] = 'complete'
                trial['timings_ms'] = distribution([row['wall_ms'] for row in trial['rows']])
                trial['official_timings_ms'] = distribution([row['wall_ms'] for row in trial['rows'] if row['kind']=='official'])
                trial['after_first'] = trial['rows'][0]['resources']
                trial['after_first_cycle'] = trial['rows'][3]['resources']
                trial['after_all'] = trial['rows'][-1]['resources']
            except Exception as error:
                trial.update(status='failed',error_code=str(error) if isinstance(error,ProbeError) else type(error).__name__)
                if trial['rows'] and trial['rows'][-1]['status']=='started':
                    trial['rows'][-1].update(status='failed',error_code=trial['error_code'])
                if isinstance(error,ProbeError) and str(error)=='overall_wall_budget':
                    raise
            finally:
                if worker is not None:
                    worker.close()
                    trial['worker_exit_code'] = worker.process.returncode
                    trial['worker_reaped'] = worker.process.poll() is not None
                save()
        result['status'] = 'complete' if all(trial['status']=='complete' for trial in trials) else 'failed'
        exit_code = 0 if result['status']=='complete' else 1
    except Exception as error:
        result.update(status='failed',error_code=str(error) if isinstance(error,ProbeError) else type(error).__name__)
    finally:
        signal.alarm(0)
        for trial in trials:
            if trial['status']=='not_started':
                trial['not_started_reason'] = result.get('error_code','earlier_failure')
        save()
    print(json.dumps({'status':result['status'],'attempted_inferences':result['attempted_inferences'],
                      'error_code':result.get('error_code')}))
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
