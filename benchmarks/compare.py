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
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = sorted(values)
    return {'count': len(values), 'median': values[len(values)//2] if len(values)%2 else sum(values[len(values)//2-1:len(values)//2+1])/2,
            'p95': values[max(0, math.ceil(len(values)*0.95)-1)], 'mean': sum(values)/len(values), 'min': values[0], 'max': values[-1]}


class Worker:
    def __init__(self, command):
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
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to overwrite benchmark evidence')
    fixture_path=HERE/'fixtures.json'
    if sha(fixture_path)!=json.loads((HERE/'freeze.json').read_text())['fixtures_sha256']:
        raise ValueError('Frozen fixture checksum mismatch')
    data=json.loads(fixture_path.read_text())
    manifest=json.loads((args.model/'manifest.json').read_text())
    if manifest['revision'] != data['model_revision']:raise ValueError('Model revision label mismatch')
    if data['protocol']['threads'] != 2 or data['protocol']['dimensions'] != 768:raise ValueError('Unsupported protocol settings')
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
            'reference_worker_sha256':sha(HERE/'reference_worker.py'),
            'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
            'hardware':{'cpu':cpu(),'architecture':platform.machine(),'os':platform.system(),'os_release':platform.release(),
                        'logical_cpus':os.cpu_count(),'affinity_cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None},
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
            worker=Worker(command)
            try:
                validate_ready(name, worker.ready)
                entry={'ready':worker.ready,'load_wall_ms':worker.load_wall_ms,'rows':[],'length_rows':[], 'text_token_ids_max_abs_error':0.0}
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
                body=tokenizer.encode(data['length_source']*200,add_special_tokens=False).ids
                for n in data['length_tokens']:
                    # Keep BOS/EOS fixed and all backends on exactly the same IDs.
                    ids=[2]+body[:n-2]+[1]
                    worker.encode({'ids':ids})
                    for repeat in range(data['protocol']['length_repeats']):
                        value,wall=worker.encode({'ids':ids})
                        entry['length_rows'].append({'tokens':n,'repeat':repeat,'engine_ms':value['elapsed_ms'],'wall_ms':wall})
                        if repeat==0:vectors['length:'+str(n)]=value['embedding']
                entry['memory']=worker.memory()
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
        if not result['parity_pass']:raise ValueError('Output parity gate failed')
        result['status']='complete'
    except BaseException as error:
        result['status']='failed';result['failure_type']=type(error).__name__
        raise
    finally:save()
    print(json.dumps({'status':result['status'],'hardware':result['hardware'],'parity':{n:{k:v for k,v in r.items() if k!='comparisons'} for n,r in result['parity'].items()}}),flush=True)

if __name__=='__main__':main()
