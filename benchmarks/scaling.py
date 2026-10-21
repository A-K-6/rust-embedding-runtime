#!/usr/bin/env python3
"""Exploratory equal-thread CPU sweep; separate from the six-order primary comparison."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from compare import (BACKENDS, HERE, ROOT, Worker, cpu, distribution, parse_cpu_list, sha,
                     source_hashes, validate_acceptance, validate_affinity, worker_affinity)

SETTINGS = (1, 2, 4, 8, 16)
MODEL_HASHES = {
    'model.safetensors': '1535bba554267ed2eb95d24c9eab0b7d19a9855d347adcbd44b354e74c7e13d9',
    'tokenizer.json': '4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4',
    'config.json': 'b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3',
}
THREAD_ENV = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
LABELS = {'transformers_eager': 'PyTorch eager', 'transformers_sdpa': 'PyTorch SDPA', 'rust_f32': 'Rust F32'}


def worker_command(name, args, threads):
    command = ([str(args.rust.resolve()), str(args.model.resolve())] if name == 'rust_f32' else
               [str(args.python.expanduser().absolute()), str(HERE/'reference_worker.py'), str(args.model.resolve()),
                '--attention', name.removeprefix('transformers_'), '--threads', str(threads)])
    return ['env', *(key+'='+str(threads) for key in THREAD_ENV), *command]


def topology(cpus, sys_cpu=Path('/sys/devices/system/cpu')):
    online = parse_cpu_list((sys_cpu/'online').read_text().strip())
    if len(cpus) < max(SETTINGS) or not set(cpus).issubset(online):
        raise ValueError('Provide at least 16 distinct online physical-core CPU IDs')
    result = []
    for cpu_id in cpus:
        directory = sys_cpu/f'cpu{cpu_id}'
        caches = []
        for index in (directory/'cache').glob('index*'):
            kind = (index/'type').read_text().strip()
            if kind in ('Unified', 'Data'):
                item = {'level': int((index/'level').read_text()), 'type': kind,
                        'shared_cpus': parse_cpu_list((index/'shared_cpu_list').read_text().strip())}
                if (index/'id').is_file():
                    item['id'] = int((index/'id').read_text())
                caches.append(item)
        if not caches:
            raise ValueError('CPU cache topology is unavailable')
        result.append({'cpu': cpu_id, 'core_id': int((directory/'topology/core_id').read_text()),
                       'physical_package_id': int((directory/'topology/physical_package_id').read_text()),
                       'last_level_cache': max(caches, key=lambda item: item['level'])})
    if len({(item['physical_package_id'], item['core_id']) for item in result}) != len(cpus):
        raise ValueError('CPU list contains SMT siblings rather than distinct physical cores')
    return result


def validate_ready(name, ready, threads):
    if ready.get('threads') != threads or ready.get('device') != 'cpu':
        raise ValueError('Worker did not report the requested CPU thread count')
    if name.startswith('transformers_'):
        if (ready.get('parameter_dtype') != 'torch.float32' or ready.get('interop_threads') != 1 or
                ready.get('attention') != name.removeprefix('transformers_')):
            raise ValueError('Reference must report CPU F32, matching attention and one inter-op thread')
    elif ready.get('dtype') != 'f32' or ready.get('weight_quantization') != 'f32':
        raise ValueError('Rust worker must report F32 compute and weights')


def parity_and_quality(setting, vectors):
    reference = vectors['transformers_eager']
    setting['parity'] = {}
    rows = setting['backends']['transformers_eager']['rows']
    setting['ranking_agreement'] = {}
    setting['ranking_agreement_all_requests'] = {}
    for name, current in vectors.items():
        if set(current) != set(reference) or len(current) != 33:
            raise ValueError('Expected 33 document, query and long-input vectors per backend')
        comparisons = []
        for key, expected in reference.items():
            actual = current[key]
            cosine = sum(a*b for a, b in zip(actual, expected))/math.sqrt(sum(a*a for a in actual)*sum(b*b for b in expected))
            comparisons.append({'id': key, 'max_abs_error': max(abs(a-b) for a, b in zip(actual, expected)), 'cosine': cosine})
        setting['parity'][name] = {'reference': 'transformers_eager', 'vectors': len(comparisons),
                                   'max_abs_error': max(item['max_abs_error'] for item in comparisons),
                                   'min_cosine': min(item['cosine'] for item in comparisons), 'comparisons': comparisons}
        current_rows = setting['backends'][name]['rows']
        if [(row['id'], row['repeat']) for row in rows] != [(row['id'], row['repeat']) for row in current_rows]:
            raise ValueError('Query schedules differ between backends')
        setting['ranking_agreement'][name] = sum(a['top5'] == b['top5'] for a, b in zip(rows, current_rows) if a['repeat'] == 0)
        setting['ranking_agreement_all_requests'][name] = sum(a['top5'] == b['top5'] for a, b in zip(rows, current_rows))
    setting['parity_pass'] = all(item['max_abs_error'] <= 1e-4 and item['min_cosine'] >= .99999 for item in setting['parity'].values())
    validate_acceptance(setting)
    setting['status'] = 'complete'


def report(result):
    lines = ['# Exploratory CPU thread scaling', '',
             '**Secondary screening: one rotated backend order per thread setting. This does not replace the six-order primary benchmark.**', '',
             f"Status: **{result['status']}**. CPU: {result['hardware']['cpu']}.", '',
             'Each completed setting uses 20 authored multilingual queries, 100 measured requests per backend, '
             'and identical explicit 128/512/1,024-token inputs with three measurements per length. '
             'Thread count and physical-core allocation change together. No fixed frequency or exclusive core reservation; '
             'cross-cache-domain settings are reported, not hidden.', '',
             '| Threads | Backend | Engine median / p95 (ms) | Wall median / p95 (ms) | Short / post-long / peak RSS (MiB) | 128 / 512 / 1,024 token wall medians (ms) |',
             '| ---: | --- | ---: | ---: | ---: | ---: |']
    for count, setting in result['settings'].items():
        for name, entry in setting['backends'].items():
            if 'latency_ms' not in entry:
                continue
            latency = [f'{entry["latency_ms"][metric]["median"]:.2f} / {entry["latency_ms"][metric]["p95"]:.2f}' for metric in ('engine_ms', 'wall_ms')]
            memory = f'{entry["short_query_memory"]["rss_mib"]:.1f} / {entry["post_long_memory"]["rss_mib"]:.1f} / {entry["post_long_memory"]["peak_rss_mib"]:.1f}'
            lengths = ' / '.join(f'{entry["length_latency_ms"][str(length)]["wall_ms"]["median"]:.2f}' for length in (128, 512, 1024))
            lines.append(f'| {count} | {LABELS[name]} | '+' | '.join(latency)+f' | {memory} | {lengths} |')
    for count, setting in result['settings'].items():
        lines += ['', f'{count} threads: quality status **{setting["status"]}**, worker CPUs {setting["worker_cpus"]}; '
                  f'LLC domains {setting["last_level_cache_domains"]}; order '+', '.join(LABELS[name] for name in setting['backend_order'])+'.', '']
    lines += ['Numerical agreement requires 33 vectors/backend, maximum absolute error at most 1e-4 and cosine at least .99999. '
              'Every completed setting requires 20/20 authored hit@1/5 plus ordered top-five agreement across all 100 requests. '
              'The small authored pilot and repeated measurements do not establish general relevance or performance across all CPUs. '
              'Memory is process RSS after short queries and long inputs, plus kernel peak RSS after long inputs.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust', required=True, type=Path)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--model', default=ROOT/'model', type=Path)
    parser.add_argument('--cpus', type=parse_cpu_list, default=parse_cpu_list('0-15'), help='At least 16 distinct physical-core CPU IDs; use the first N in ascending CPU-ID order for each setting')
    parser.add_argument('--build-provenance', required=True, help='Caller declaration of exact build command, features and flags; not inferred from runtime environment')
    parser.add_argument('--max-seconds', type=int, default=3600, help='Overall time budget; unfinished evidence is saved with partial status')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64') or shutil.which('taskset') is None:
        parser.error('This sweep requires native Linux AMD64 with taskset')
    if not 60 <= args.max_seconds <= 7200:
        parser.error('--max-seconds must be between 60 and 7200')
    if args.output_dir.exists():
        parser.error('Refusing to overwrite scaling evidence')
    cpus = topology(args.cpus)
    if sha(HERE/'fixtures.json') != json.loads((HERE/'freeze.json').read_text())['fixtures_sha256']:
        raise ValueError('Frozen fixture checksum mismatch')
    data = json.loads((HERE/'fixtures.json').read_text())
    if len(data['queries']) != 20 or data['protocol']['repeats'] != 5 or data['protocol']['length_repeats'] != 3 or data['protocol']['warmup'] != 5:
        raise ValueError('Unexpected frozen fixture protocol')
    if data['protocol']['dimensions'] != 768 or data['length_tokens'] != [128, 512, 1024]:
        raise ValueError('Unexpected frozen fixture dimensions or lengths')
    if json.loads((args.model/'manifest.json').read_text())['revision'] != data['model_revision']:
        raise ValueError('Model revision mismatch')
    for name, expected in MODEL_HASHES.items():
        if sha(args.model/name) != expected:
            raise ValueError('Pinned model asset checksum mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model/'tokenizer.json'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = {'source_file_sha256': source_hashes(), 'rust_binary_sha256': sha(args.rust),
                  'compare_driver_sha256': sha(HERE/'compare.py'), 'reference_worker_sha256': sha(HERE/'reference_worker.py'),
                  'scaling_driver_sha256': sha(Path(__file__)), 'fixture_sha256': sha(HERE/'fixtures.json')}
    result = {'status': 'running', 'scope': 'Exploratory single-order equal-thread physical-core scaling; secondary screening only.',
              **provenance, 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'model_revision': data['model_revision'], 'model_asset_sha256': MODEL_HASHES,
              'build_provenance': {'caller_declaration': args.build_provenance, 'independently_verified': False},
              'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
              'hardware': {'cpu': cpu(), 'architecture': platform.machine(), 'os': platform.system(), 'logical_cpus': os.cpu_count(),
                           'driver_affinity_cpus': sorted(os.sched_getaffinity(0)), 'cpu_topology': cpus},
              'frozen_protocol': data['protocol'], 'thread_settings': list(SETTINGS), 'settings': {},
              'schedule': [{'threads': count, 'order': list(BACKENDS[index % 3:]+BACKENDS[:index % 3]), 'worker_cpus': args.cpus[:count]}
                           for index, count in enumerate(SETTINGS)],
              'measurement': 'engine_ms excludes tokenization and IPC; wall_ms includes tokenization, JSONL IPC and vector serialization. No HTTP or ranking in latency.',
              'max_seconds': args.max_seconds}
    def save():
        result['elapsed_seconds'] = time.perf_counter()-started
        (args.output_dir/'scaling.json').write_text(json.dumps(result, indent=2)+'\n')
        (args.output_dir/'REPORT.md').write_text(report(result))
    def check():
        if time.perf_counter()-started > args.max_seconds:
            raise TimeoutError('Exploratory sweep time budget reached')
    def unchanged():
        if (source_hashes() != provenance['source_file_sha256'] or sha(args.rust) != provenance['rust_binary_sha256'] or
                any(sha(path) != provenance[key] for path, key in ((HERE/'compare.py', 'compare_driver_sha256'),
                    (HERE/'reference_worker.py', 'reference_worker_sha256'), (Path(__file__), 'scaling_driver_sha256'), (HERE/'fixtures.json', 'fixture_sha256')))):
            raise ValueError('Source, binary or benchmark driver changed during scaling')
    save()
    try:
        body = tokenizer.encode(data['length_source']*200, add_special_tokens=False).ids
        for scheduled in result['schedule']:
            check(); unchanged()
            count, selected = scheduled['threads'], scheduled['worker_cpus']
            domains = sorted({tuple(item['last_level_cache']['shared_cpus']) for item in cpus[:count]})
            setting = {'status': 'running', 'worker_cpus': selected, 'backend_order': scheduled['order'],
                       'last_level_cache_domains': [list(domain) for domain in domains],
                       'thread_environment': {key: str(count) for key in THREAD_ENV}, 'backends': {}}
            result['settings'][str(count)] = setting
            vectors = {}
            for name in scheduled['order']:
                check(); unchanged()
                command = worker_command(name, args, count)
                print(f'Starting exploratory {count} threads: {name}', flush=True)
                worker = Worker(command, cpus=selected)
                try:
                    validate_ready(name, worker.ready, count)
                    entry = {'ready': worker.ready, 'ready_affinity': worker.affinity, 'load_wall_ms': worker.load_wall_ms,
                             'rows': [], 'length_rows': [], 'text_token_ids_max_abs_error': 0.0}
                    setting['backends'][name] = entry
                    current, corpus = {}, []
                    def encode(request):
                        check()
                        value, wall = worker.encode(request)
                        if not math.isfinite(value['elapsed_ms']) or value['elapsed_ms'] <= 0 or not math.isfinite(wall) or wall <= 0:
                            raise ValueError('Inference timing must be finite and positive')
                        if 'ids' in request and value['tokens'] != len(request['ids']):
                            raise ValueError('Explicit-token count mismatch')
                        return value, wall
                    for doc in data['documents']:
                        value, _ = encode({'text': data['document_template'].format(**doc)})
                        current['doc:'+doc['id']] = value['embedding']; corpus.append((doc['id'], value['embedding']))
                    for query in data['queries'][:5]:
                        encode({'text': data['query_prefix']+query['text']})
                    for repeat in range(5):
                        for query in data['queries']:
                            text = data['query_prefix']+query['text']
                            value, wall = encode({'text': text})
                            ids = tokenizer.encode(text, add_special_tokens=True).ids
                            if value['tokens'] != len(ids):
                                raise ValueError('Tokenizer count mismatch')
                            ranked = sorted(corpus, key=lambda item: sum(a*b for a, b in zip(value['embedding'], item[1])), reverse=True)
                            entry['rows'].append({'id': query['id'], 'repeat': repeat, 'tokens': value['tokens'], 'engine_ms': value['elapsed_ms'],
                                                  'wall_ms': wall, 'top5': [item[0] for item in ranked[:5]], 'expected': query['expected']})
                            if repeat == 0:
                                current['query:'+query['id']] = value['embedding']
                                explicit, _ = encode({'ids': ids})
                                delta = max(abs(a-b) for a, b in zip(value['embedding'], explicit['embedding']))
                                entry['text_token_ids_max_abs_error'] = max(entry['text_token_ids_max_abs_error'], delta)
                                if delta > 1e-6:
                                    raise ValueError('Text and common token-ID embeddings disagree')
                    entry['short_query_memory'] = worker.memory()
                    entry['short_query_affinity'] = worker_affinity(worker.process.pid); validate_affinity(entry['short_query_affinity'], selected)
                    for length in data['length_tokens']:
                        ids = [2]+body[:length-2]+[1]; encode({'ids': ids})
                        for repeat in range(3):
                            value, wall = encode({'ids': ids})
                            entry['length_rows'].append({'tokens': length, 'repeat': repeat, 'engine_ms': value['elapsed_ms'], 'wall_ms': wall})
                            if repeat == 0:
                                current['length:'+str(length)] = value['embedding']
                    entry['post_long_memory'] = worker.memory()
                    entry['post_long_affinity'] = worker_affinity(worker.process.pid); validate_affinity(entry['post_long_affinity'], selected)
                    entry['latency_ms'] = {metric: distribution([row[metric] for row in entry['rows']]) for metric in ('engine_ms', 'wall_ms')}
                    entry['length_latency_ms'] = {str(length): {metric: distribution([row[metric] for row in entry['length_rows'] if row['tokens'] == length])
                                                for metric in ('engine_ms', 'wall_ms')} for length in data['length_tokens']}
                    first = entry['rows'][:20]
                    entry['retrieval'] = {'queries': 20, 'hit_at_1': sum(row['top5'][0] == row['expected'] for row in first),
                                          'hit_at_5': sum(row['expected'] in row['top5'] for row in first)}
                    vectors[name] = current
                    save()
                finally:
                    worker.close()
            parity_and_quality(setting, vectors); save()
        unchanged(); result['status'] = 'complete'
    except BaseException as error:
        result['status'] = 'partial' if isinstance(error, TimeoutError) else 'failed'
        result['failure_type'] = type(error).__name__
        for setting in result['settings'].values():
            if setting['status'] == 'running':
                setting['status'] = result['status']
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
