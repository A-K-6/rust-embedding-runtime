#!/usr/bin/env python3
"""Compare native Rust CPU inference against official Transformers CPU inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKENDS = ('transformers_eager', 'transformers_sdpa', 'rust_f32')
WAIT_ENV_KEYS = ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_DYNAMIC', 'OMP_PROC_BIND')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_wait_environment(pid=None):
    if pid is None:
        return {key: os.environ.get(key) for key in WAIT_ENV_KEYS}
    wanted = {key.encode(): key for key in WAIT_ENV_KEYS}
    values = {}
    for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
        key, separator, value = item.partition(b'=')
        if separator and key in wanted:
            values[wanted[key]] = os.fsdecode(value)
    return {key: values.get(key) for key in WAIT_ENV_KEYS}


def distribution(values):
    values = sorted(values)
    return {'count': len(values), 'median': values[len(values)//2] if len(values)%2 else sum(values[len(values)//2-1:len(values)//2+1])/2,
            'p95': values[max(0, math.ceil(len(values)*0.95)-1)], 'mean': sum(values)/len(values), 'min': values[0], 'max': values[-1]}


def source_hashes(root=ROOT):
    """Hash source and project build configuration, including untracked files."""
    paths = set((root/'src').rglob('*.rs'))
    paths.update(path for path in (root/'crates').rglob('*') if path.is_file() and 'target' not in path.relative_to(root).parts and path.suffix in ('.rs', '.c', '.h', '.toml'))
    paths.update(root/name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'rust-toolchain', 'rust-toolchain.toml', 'scripts/build_aocl.py') if (root/name).is_file())
    paths.update(root/'.cargo'/name for name in ('config', 'config.toml') if (root/'.cargo'/name).is_file())
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(paths)}


def parse_cpu_list(value):
    """Parse Linux CPU lists, including the kernel's compact range notation."""
    cpus = []
    for part in value.split(','):
        bounds = part.split('-')
        if len(bounds) == 1 and bounds[0].isdigit():
            cpus.append(int(bounds[0]))
        elif len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
            first, last = map(int, bounds)
            if first > last:
                raise ValueError('CPU ranges must be ascending')
            cpus.extend(range(first, last+1))
        else:
            raise ValueError('CPU list must contain nonnegative CPU IDs or ascending ranges')
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError('CPU list must be nonempty and contain no duplicates')
    return sorted(cpus)


def validate_worker_cpus(cpus, sys_cpu=Path('/sys/devices/system/cpu'), allowed=None):
    """Require two physical cores in one last-level cache domain, without guessing IDs."""
    if len(cpus) != 2:
        raise ValueError('The frozen two-thread protocol requires exactly two logical CPU IDs')
    if allowed is not None and not set(cpus).issubset(allowed):
        raise ValueError('Requested CPUs are not online')
    topology = []
    for cpu_id in cpus:
        directory = sys_cpu/f'cpu{cpu_id}'
        item = {'cpu': cpu_id, 'core_id': int((directory/'topology/core_id').read_text()),
                'physical_package_id': int((directory/'topology/physical_package_id').read_text())}
        caches = []
        for index in (directory/'cache').glob('index*'):
            cache_type = (index/'type').read_text().strip()
            if cache_type in ('Unified', 'Data'):
                caches.append({'level': int((index/'level').read_text()), 'type': cache_type,
                               'shared_cpus': parse_cpu_list((index/'shared_cpu_list').read_text().strip())})
        if not caches:
            raise ValueError('CPU cache topology is unavailable; choose cores only after inspecting topology')
        item['last_level_cache'] = max(caches, key=lambda cache: cache['level'])
        topology.append(item)
    if len({(item['physical_package_id'], item['core_id']) for item in topology}) != 2:
        raise ValueError('Requested CPU IDs are SMT siblings, not two distinct physical cores')
    if topology[0]['last_level_cache'] != topology[1]['last_level_cache']:
        raise ValueError('Requested physical cores do not share a last-level cache domain')
    return {'requested_cpus': cpus, 'distinct_physical_cores': True,
            'same_last_level_cache_domain': True, 'topology': topology}


def worker_affinity(pid):
    """Record all worker thread masks after imports and thread-pool initialization."""
    if not hasattr(os, 'sched_getaffinity'):
        return None
    masks = {}
    for task in Path(f'/proc/{pid}/task').iterdir():
        try:
            mask = tuple(sorted(os.sched_getaffinity(int(task.name))))
        except ProcessLookupError:
            continue  # A transient framework thread exited while sampling.
        masks[mask] = masks.get(mask, 0)+1
    return {'process_cpus': sorted(os.sched_getaffinity(pid)),
            'thread_masks': [{'cpus': list(mask), 'threads': count} for mask, count in sorted(masks.items())]}


def validate_affinity(record, cpus):
    if cpus is not None and (record is None or record['process_cpus'] != cpus or
                            not record['thread_masks'] or
                            any(mask['cpus'] != cpus for mask in record['thread_masks'])):
        raise ValueError('Worker or framework threads changed the requested CPU affinity')


def validate_acceptance(result):
    """A completed run must preserve numerical outputs and all authored rankings."""
    if not result.get('parity_pass'):
        raise ValueError('Output parity gate failed')
    if set(result['backends']) != set(BACKENDS):
        raise ValueError('Missing backend results')
    if set(result['parity']) != set(BACKENDS):
        raise ValueError('Missing numerical comparisons')
    for name, entry in result['backends'].items():
        parity = result['parity'][name]
        if (parity['vectors'] != 33 or not math.isfinite(parity['max_abs_error']) or
                not math.isfinite(parity['min_cosine']) or parity['max_abs_error'] > 1e-4 or parity['min_cosine'] < 0.99999):
            raise ValueError(f'{name}: numerical comparison gate failed')
        if entry['retrieval'] != {'queries': 20, 'hit_at_1': 20, 'hit_at_5': 20}:
            raise ValueError(f'{name}: authored retrieval gate failed')
        if any(not row['top5'] or row['top5'][0] != row['expected'] for row in entry['rows']):
            raise ValueError(f'{name}: repeated-request retrieval gate failed')
        if result['ranking_agreement'][name] != 20:
            raise ValueError(f'{name}: ordered top-five agreement gate failed')
        if result['ranking_agreement_all_requests'][name] != len(entry['rows']):
            raise ValueError(f'{name}: repeated-request ranking agreement gate failed')


class Worker:
    def __init__(self, command, cpus=None):
        if cpus is not None:
            command = [shutil.which('taskset') or 'taskset', '--cpu-list', ','.join(map(str, cpus)), *command]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding='utf-8', bufsize=1,
                                        env={**os.environ, 'RAYON_NUM_THREADS':'2', 'TOKENIZERS_PARALLELISM':'false',
                                             'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                                             'VECLIB_MAXIMUM_THREADS':'2', 'BLIS_NUM_THREADS':'2',
                                             'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'})
        self.events = queue.Queue()
        self.output = queue.Queue()
        def read(stream, target):
            for line in stream:
                target.put(line)
            target.put(None)
        threading.Thread(target=read,args=(self.process.stderr,self.events),daemon=True).start()
        threading.Thread(target=read,args=(self.process.stdout,self.output),daemon=True).start()
        start=time.perf_counter()
        self.ready = None
        try:
            while True:
                line=self.events.get(timeout=180)
                if line is None:
                    raise RuntimeError('worker exited before readiness')
                try:
                    value=json.loads(line)
                    if value.get('ready'):
                        self.ready=value
                        break
                except json.JSONDecodeError:
                    # Public startup diagnostics; never retained in benchmark output.
                    print(line.rstrip(), file=sys.stderr)
            self.load_wall_ms=1000*(time.perf_counter()-start)
            self.affinity = worker_affinity(self.process.pid)
            validate_affinity(self.affinity, cpus)
        except BaseException:
            self.close()
            raise
    def encode(self, request):
        start=time.perf_counter()
        self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        line=self.output.get(timeout=180)
        if line is None:
            raise RuntimeError('worker exited during inference')
        value=json.loads(line)
        wall=1000*(time.perf_counter()-start)
        if 'error' in value:
            raise RuntimeError(value['error'])
        vector=value['embedding']
        if len(vector)!=768 or not all(math.isfinite(x) for x in vector) or abs(sum(x*x for x in vector)-1)>1e-4:
            raise ValueError('invalid embedding')
        return value, wall
    def memory(self):
        pid = self.process.pid
        values = {'rss_mib': int(subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True).strip())/1024}
        if platform.system() == 'Darwin':
            # macOS RSS excludes compressed pages; use the kernel footprint ledger.
            import ctypes
            class Usage(ctypes.Structure):
                _fields_ = [('uuid', ctypes.c_ubyte*16)] + [(name, ctypes.c_uint64) for name in [
                    'user_time','system_time','idle_wakeups','interrupt_wakeups','pageins',
                    'wired_size','resident_size','physical_footprint','start_time','exit_time']]
            usage = Usage()
            proc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            proc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            proc.proc_pid_rusage.restype = ctypes.c_int
            if proc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
                raise OSError(ctypes.get_errno(), 'Cannot measure process footprint')
            values['physical_footprint_mib'] = usage.physical_footprint / 1024**2
        else:
            status = Path(f'/proc/{pid}/status').read_text()
            for line in status.splitlines():
                if line.startswith('VmHWM:'):
                    values['peak_rss_mib'] = int(line.split()[1])/1024
        return values
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait()


def validate_ready(name, ready):
    if ready.get('threads') != 2:
        raise ValueError('Backend must report two CPU kernel threads')
    if name.startswith('transformers_'):
        if ready.get('parameter_dtype') != 'torch.float32' or ready.get('device') != 'cpu':
            raise ValueError('Reference must use CPU float32 parameters')
        if ready.get('attention') != name.removeprefix('transformers_'):
            raise ValueError('Reference attention backend mismatch')
        if ready.get('interop_threads') != 1:
            raise ValueError('Reference must use one CPU inter-op thread')
    elif ready.get('weight_quantization') != 'f32' or ready.get('device') != 'cpu' or ready.get('dtype') != 'f32':
        raise ValueError('Rust must use F32 linear weights')


def cpu():
    if platform.system()=='Darwin':
        return subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):return line.split(':',1)[1].strip()
    return platform.machine()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rust',required=True,type=Path)
    p.add_argument('--python',required=True,type=Path,help='Python with reference requirements installed')
    p.add_argument('--model',default=ROOT/'model',type=Path)
    p.add_argument('--order',default='transformers_eager,transformers_sdpa,rust_f32',help='Each backend exactly once; reverse for a counterbalanced second run')
    p.add_argument('--cpus',type=parse_cpu_list,help='Linux only: bind worker processes and their threads to two distinct physical cores in one last-level cache domain; leave the driver unpinned')
    p.add_argument('--build-provenance',default='unrecorded',help='Explicit declaration of the build command, features and flags used for this binary; runtime environment is not evidence of build flags')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to overwrite benchmark evidence')
    cpu_selection = None
    if args.cpus is not None:
        if platform.system() != 'Linux' or not hasattr(os, 'sched_getaffinity'):
            p.error('--cpus requires Linux affinity and sysfs topology')
        if shutil.which('taskset') is None:
            p.error('--cpus requires taskset')
        # Driver affinity may deliberately be disjoint from the workers. taskset
        # and post-start masks prove whether the child may use the requested CPUs.
        online = parse_cpu_list(Path('/sys/devices/system/cpu/online').read_text().strip())
        cpu_selection = validate_worker_cpus(args.cpus, allowed=online)
    fixture_path=HERE/'fixtures.json'
    if sha(fixture_path)!=json.loads((HERE/'freeze.json').read_text())['fixtures_sha256']:
        raise ValueError('Frozen fixture checksum mismatch')
    data=json.loads(fixture_path.read_text())
    manifest=json.loads((args.model/'manifest.json').read_text())
    if manifest['revision'] != data['model_revision']:raise ValueError('Model revision label mismatch')
    if data['protocol']['threads'] != 2 or data['protocol']['dimensions'] != 768 or len(data['queries']) != 20:raise ValueError('Unsupported protocol settings')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
    # Reuse the pinned-model verifier without importing the inference frameworks.
    expected={'model.safetensors':'1535bba554267ed2eb95d24c9eab0b7d19a9855d347adcbd44b354e74c7e13d9',
              'tokenizer.json':'4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4',
              'config.json':'b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3'}
    for name, checksum in expected.items():
        if sha(args.model/name)!=checksum:raise ValueError('Model checksum mismatch')
    result={'status':'running','fixture_sha256':sha(fixture_path),'model_revision':data['model_revision'],
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'rust_binary_sha256':sha(args.rust), 'cargo_lock_sha256':sha(ROOT/'Cargo.lock'), 'driver_python':platform.python_version(),
            'rust_source_sha256':sha(ROOT/'src/lib.rs'), 'driver_sha256':sha(Path(__file__)),
            'source_file_sha256':source_hashes(),
            'build_provenance':{'caller_declaration':args.build_provenance,
                                'verification':'Caller supplied; not inferred from runtime environment or independently verified by this driver.'},
            'reference_worker_sha256':sha(HERE/'reference_worker.py'),
            'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
            'hardware':{'cpu':cpu(),'architecture':platform.machine(),'os':platform.system(),'os_release':platform.release(),
                        'logical_cpus':os.cpu_count(),'affinity_cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None},
            'worker_cpu_selection':cpu_selection,
            'runtime_wait_environment':runtime_wait_environment(),
            'protocol':data['protocol'],
            'measurement':'Batch 1; uncached; serial. wall_ms includes tokenization, JSONL IPC and vector serialization; engine_ms excludes tokenization and IPC. No HTTP/network/database/ranking in either latency.',
            'cpu_limit':'Two intra-op/Rayon threads; PyTorch inter-op 1; BLAS/OpenMP environment capped at 2. Native host, no exclusive CPU reservation or hard quota.',
            'backends':{},'parity':{}}
    baseline_vectors={}
    reference_name='transformers_eager'
    commands=[('transformers_eager',[str(args.python),str(HERE/'reference_worker.py'),str(args.model),'--attention','eager']),
              ('transformers_sdpa',[str(args.python),str(HERE/'reference_worker.py'),str(args.model),'--attention','sdpa']),
              ('rust_f32',[str(args.rust),str(args.model)])]
    names=args.order.split(',')
    available=dict(commands)
    if len(names)!=3 or set(names)!=set(available):p.error('--order must name each backend exactly once')
    commands=[(name,available[name]) for name in names]
    result['backend_order']=names
    all_vectors={}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    try:
        for name,command in commands:
            print(f'Starting {name}',flush=True)
            worker=Worker(command, cpus=args.cpus)
            try:
                validate_ready(name, worker.ready)
                wait_environment = runtime_wait_environment(worker.process.pid if platform.system() == 'Linux' else None)
                if wait_environment != result['runtime_wait_environment']:
                    raise ValueError('Worker wait environment differs from the declared spawn environment')
                entry={'ready':worker.ready,'load_wall_ms':worker.load_wall_ms,'rows':[],'length_rows':[],
                       'ready_affinity':worker.affinity, 'text_token_ids_max_abs_error':0.0,
                       'runtime_wait_environment':wait_environment,
                       'wait_environment_verification':'linux_proc_environ' if platform.system() == 'Linux' else 'spawn_environment'}
                corpus=[]
                vectors={}
                for doc in data['documents']:
                    text=data['document_template'].format(**doc)
                    value,_=worker.encode({'text':text})
                    vectors['doc:'+doc['id']]=value['embedding']
                    corpus.append((doc['id'],value['embedding']))
                for q in data['queries'][:data['protocol']['warmup']]:
                    worker.encode({'text':data['query_prefix']+q['text']})
                for repeat in range(data['protocol']['repeats']):
                    for q in data['queries']:
                        text=data['query_prefix']+q['text']
                        value,wall=worker.encode({'text':text})
                        ids=tokenizer.encode(text,add_special_tokens=True).ids
                        if value['tokens']!=len(ids):raise ValueError('Tokenizer mismatch')
                        ranked=sorted(((fid,sum(a*b for a,b in zip(value['embedding'],v))) for fid,v in corpus),key=lambda r:r[1],reverse=True)
                        entry['rows'].append({'id':q['id'],'repeat':repeat,'tokens':value['tokens'],
                                              'engine_ms':value['elapsed_ms'],'wall_ms':wall,
                                              'top5':[fid for fid,_ in ranked[:5]],'expected':q['expected']})
                        if repeat==0:
                            vectors['query:'+q['id']]=value['embedding']
                            explicit,_=worker.encode({'ids':ids})
                            delta=max(abs(a-b) for a,b in zip(value['embedding'],explicit['embedding']))
                            entry['text_token_ids_max_abs_error']=max(entry['text_token_ids_max_abs_error'],delta)
                            if delta>1e-6:raise ValueError('Text and common token IDs disagree')
                    print(f'{name}: repeat {repeat+1} complete',flush=True)
                entry['short_query_memory'] = worker.memory()
                entry['short_query_affinity'] = worker_affinity(worker.process.pid)
                validate_affinity(entry['short_query_affinity'], args.cpus)
                body=tokenizer.encode(data['length_source']*200,add_special_tokens=False).ids
                for n in data['length_tokens']:
                    # Keep BOS/EOS fixed and all backends on exactly the same IDs.
                    ids=[2]+body[:n-2]+[1]
                    worker.encode({'ids':ids})
                    for repeat in range(data['protocol']['length_repeats']):
                        value,wall=worker.encode({'ids':ids})
                        entry['length_rows'].append({'tokens':n,'repeat':repeat,'engine_ms':value['elapsed_ms'],'wall_ms':wall})
                        if repeat==0:vectors['length:'+str(n)]=value['embedding']
                entry['post_long_memory']=worker.memory()
                entry['post_long_affinity'] = worker_affinity(worker.process.pid)
                validate_affinity(entry['post_long_affinity'], args.cpus)
                # Retain the original post-long field for existing result consumers.
                entry['memory']=entry['post_long_memory']
                entry['steady_rss_mib']=entry['memory']['rss_mib']
                entry['latency_ms']={metric:distribution([r[metric] for r in entry['rows']]) for metric in ['engine_ms','wall_ms']}
                first=[r for r in entry['rows'] if r['repeat']==0]
                entry['retrieval']={'queries':len(first),'hit_at_1':sum(r['top5'][0]==r['expected'] for r in first),
                                    'hit_at_5':sum(r['expected'] in r['top5'] for r in first)}
                result['backends'][name]=entry
                all_vectors[name]=vectors
                save()
            finally:worker.close()
        baseline_vectors=all_vectors[reference_name]
        for name,vectors in all_vectors.items():
            comparisons=[]
            for key,reference in baseline_vectors.items():
                vector=vectors[key]
                comparisons.append({'id':key,'max_abs_error':max(abs(a-b) for a,b in zip(vector,reference)),
                                    'cosine':sum(a*b for a,b in zip(vector,reference))/math.sqrt(sum(a*a for a in vector)*sum(b*b for b in reference))})
            result['parity'][name]={'reference':reference_name,'vectors':len(comparisons),
                                    'max_abs_error':max(r['max_abs_error'] for r in comparisons),
                                    'min_cosine':min(r['cosine'] for r in comparisons),'comparisons':comparisons}
        result['parity_pass'] = all(r['max_abs_error'] <= 1e-4 and r['min_cosine'] >= 0.99999 for r in result['parity'].values())
        reference_rows=[r for r in result['backends'][reference_name]['rows'] if r['repeat']==0]
        result['ranking_agreement']={name:sum(a['top5']==b['top5'] for a,b in zip(reference_rows,[r for r in entry['rows'] if r['repeat']==0])) for name,entry in result['backends'].items()}
        reference_all = result['backends'][reference_name]['rows']
        result['ranking_agreement_all_requests'] = {}
        for name, entry in result['backends'].items():
            if [(row['id'], row['repeat']) for row in reference_all] != [(row['id'], row['repeat']) for row in entry['rows']]:
                raise ValueError('Measured query rows do not match the reference schedule')
            result['ranking_agreement_all_requests'][name] = sum(a['top5'] == b['top5'] for a, b in zip(reference_all, entry['rows']))
        validate_acceptance(result)
        result['status']='complete'
    except BaseException as error:
        result['status']='failed';result['failure_type']=type(error).__name__
        raise
    finally:save()
    print(json.dumps({'status':result['status'],'hardware':result['hardware'],'parity':{n:{k:v for k,v in r.items() if k!='comparisons'} for n,r in result['parity'].items()}}),flush=True)

if __name__=='__main__':main()
