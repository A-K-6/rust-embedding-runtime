#!/usr/bin/env python3
"""Exploratory Rust-only GNU OpenMP wait-policy scheduling probe."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from compare import HERE, ROOT, Worker, cpu, distribution, parse_cpu_list, sha, source_hashes, validate_affinity, worker_affinity
from scaling import MODEL_HASHES, THREAD_ENV, topology, validate_ready

COUNTS = (2, 4, 8, 16)
WAIT_VARS = ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')


def process_cpu_seconds(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')


def command(args, count, policy):
    environment = {key: str(count) for key in THREAD_ENV}
    environment.update(dict.fromkeys(WAIT_VARS))
    if policy == 'passive':
        environment.update(OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
    return (['env', '-u', WAIT_VARS[0], '-u', WAIT_VARS[1]]+
            [key+'='+value for key, value in environment.items() if value is not None]+
            [str(args.rust.resolve()), str(args.model.resolve())]), environment


def gate(entry, vectors, reference):
    expected_vectors, expected_rows = reference
    if len(vectors) != 33 or set(vectors) != set(expected_vectors) or len(entry['rows']) != 100:
        raise ValueError('Expected 33 vectors and 100 query rows')
    comparisons = []
    for key, expected in expected_vectors.items():
        actual = vectors[key]
        comparisons.append({'id': key, 'max_abs_error': max(abs(a-b) for a, b in zip(actual, expected)),
                            'cosine': sum(a*b for a, b in zip(actual, expected))/math.sqrt(sum(a*a for a in actual)*sum(b*b for b in expected))})
    entry['parity'] = {'reference_run': 1, 'vectors': 33, 'max_abs_error': max(item['max_abs_error'] for item in comparisons),
                       'min_cosine': min(item['cosine'] for item in comparisons), 'comparisons': comparisons}
    first = entry['rows'][:20]
    entry['retrieval'] = {'queries': 20, 'hit_at_1': sum(row['top5'][0] == row['expected'] for row in first),
                          'hit_at_5': sum(row['expected'] in row['top5'] for row in first)}
    if [(row['id'], row['repeat']) for row in expected_rows] != [(row['id'], row['repeat']) for row in entry['rows']]:
        raise ValueError('Query schedules differ from first default Rust process')
    entry['ranking_agreement_all_requests'] = sum(a['top5'] == b['top5'] for a, b in zip(entry['rows'], expected_rows))
    if (not math.isfinite(entry['parity']['min_cosine']) or not math.isfinite(entry['parity']['max_abs_error']) or entry['parity']['max_abs_error'] > 1e-6 or entry['parity']['min_cosine'] < .99999999 or
            entry['retrieval'] != {'queries': 20, 'hit_at_1': 20, 'hit_at_5': 20} or
            any(row['top5'][0] != row['expected'] for row in entry['rows']) or entry['ranking_agreement_all_requests'] != 100):
        raise ValueError('Rust wait-policy numerical or retrieval gate failed')
    entry['status'] = 'complete'


def report(result):
    lines = ['# Exploratory Rust OpenMP wait-policy probe', '',
             '**Rust scheduling screening only; this is not a new PyTorch comparison or a headline performance claim.**', '',
             f"Status: **{result['status']}**. CPU: {result['hardware']['cpu']}.", '',
             'Default explicitly unsets OMP_WAIT_POLICY and GOMP_SPINCOUNT. Passive sets PASSIVE and 0. '
             'Each thread count runs both fresh-process orders, default → passive and passive → default. '
             'Idle OpenMP workers competing with Rayon is a hypothesis; no speedup is assumed.', '',
             '| Run | Threads | Order | Policy | Engine median / p95 (ms) | Wall median / p95 (ms) | Short / post-long / peak RSS (MiB) | 128 / 512 / 1,024 token wall medians (ms) |',
             '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for entry in result['runs']:
        if 'latency_ms' not in entry:
            continue
        latency = [' / '.join(f'{entry["latency_ms"][metric][percentile]:.2f}' for percentile in ('median', 'p95')) for metric in ('engine_ms', 'wall_ms')]
        memory = f'{entry["short_query_memory"]["rss_mib"]:.1f} / {entry["post_long_memory"]["rss_mib"]:.1f} / {entry["post_long_memory"]["peak_rss_mib"]:.1f}'
        lengths = ' / '.join(f'{entry["length_latency_ms"][str(length)]["wall_ms"]["median"]:.2f}' for length in (128, 512, 1024))
        lines.append(f'| {entry["run"]} | {entry["threads"]} | {entry["order"]} | {entry["policy"]} | '+' | '.join(latency)+f' | {memory} | {lengths} |')
    lines += ['', 'Every completed worker has 100 raw short-query timings, five warmups and ten document encodes excluded from query latency. '
              'Long inputs use one warmup and three timed samples per length. All 33-vector and 100-ranking agreement uses the first default Rust worker; '
              'the small 20-label retrieval pilot does not establish general relevance. '
              'Engine time excludes tokenization and IPC; wall time includes tokenization, JSONL IPC and serialization. Neither includes ranking or HTTP. '
              'Actual worker masks, physical-core/cache topology and whitelisted process environment are saved in wait-policy.json. '
              'Cross-cache allocation, shared-host interference and frequency changes remain uncontrolled.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust', required=True, type=Path)
    parser.add_argument('--model', default=ROOT/'model', type=Path)
    parser.add_argument('--cpus', type=parse_cpu_list, default=parse_cpu_list('0-15'))
    parser.add_argument('--build-provenance', required=True)
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('amd64', 'x86_64') or shutil.which('taskset') is None:
        parser.error('This probe requires native Linux AMD64, GNU env and taskset')
    if args.output_dir.exists() or not 60 <= args.max_seconds <= 7200:
        parser.error('Use a new output directory and max-seconds between 60 and 7200')
    cpus = topology(args.cpus)
    data = json.loads((HERE/'fixtures.json').read_text())
    if sha(HERE/'fixtures.json') != json.loads((HERE/'freeze.json').read_text())['fixtures_sha256']:
        raise ValueError('Frozen fixture checksum mismatch')
    if len(data['queries']) != 20 or len(data['documents']) != 10 or data['protocol'] != {'threads': 2, 'warmup': 5, 'repeats': 5, 'length_repeats': 3, 'dimensions': 768, 'batch_size': 1, 'cache': False}:
        raise ValueError('Unexpected frozen protocol')
    if data['length_tokens'] != [128, 512, 1024] or json.loads((args.model/'manifest.json').read_text())['revision'] != data['model_revision']:
        raise ValueError('Model revision or length protocol mismatch')
    if any(sha(args.model/name) != checksum for name, checksum in MODEL_HASHES.items()):
        raise ValueError('Pinned model checksum mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model/'tokenizer.json'))
    drivers = {name: sha(HERE/name) for name in ('compare.py', 'scaling.py', 'wait_policy.py', 'fixtures.json')}
    sources, binary = source_hashes(), sha(args.rust)
    schedule = [{'threads': count, 'order': order+1, 'policy': policy, 'worker_cpus': args.cpus[:count]}
                for count in COUNTS for order, policies in enumerate((('default', 'passive'), ('passive', 'default'))) for policy in policies]
    result = {'status': 'running', 'scope': 'Exploratory Rust-only GNU OpenMP scheduling probe', 'schedule': schedule, 'runs': [],
              'hardware': {'cpu': cpu(), 'architecture': platform.machine(), 'cpu_topology': cpus, 'driver_affinity': sorted(os.sched_getaffinity(0))},
              'source_file_sha256': sources, 'rust_binary_sha256': binary, 'driver_sha256': drivers,
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'build_provenance': {'caller_declaration': args.build_provenance, 'independently_verified': False},
              'model_revision': data['model_revision'], 'model_asset_sha256': MODEL_HASHES, 'frozen_protocol': data['protocol'], 'max_seconds': args.max_seconds}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, reference = time.perf_counter(), None
    def save():
        result['elapsed_seconds'] = time.perf_counter()-started
        (args.output_dir/'wait-policy.json').write_text(json.dumps(result, indent=2)+'\n')
        (args.output_dir/'REPORT.md').write_text(report(result))
    def check():
        if time.perf_counter()-started > args.max_seconds:
            raise TimeoutError('Wait-policy probe time budget reached')
    save()
    try:
        body = tokenizer.encode(data['length_source']*200, add_special_tokens=False).ids
        for index, scheduled in enumerate(schedule):
            check()
            if source_hashes() != sources or sha(args.rust) != binary or any(sha(HERE/name) != checksum for name, checksum in drivers.items()):
                raise ValueError('Source, binary or driver changed during probe')
            count, selected = scheduled['threads'], scheduled['worker_cpus']
            argv, environment = command(args, count, scheduled['policy'])
            entry = {'run': index+1, **scheduled, 'status': 'running', 'environment': environment, 'rows': [], 'length_rows': [],
                     'last_level_cache_domains': sorted({tuple(item['last_level_cache']['shared_cpus']) for item in cpus[:count]}), 'text_token_ids_max_abs_error': 0.0}
            result['runs'].append(entry)
            print(f'Starting wait-policy worker {index+1}/16: threads={count}, policy={scheduled["policy"]}', flush=True)
            worker = Worker(argv, cpus=selected)
            try:
                validate_ready('rust_f32', worker.ready, count)
                if worker.ready.get('cpu_backend') != 'aocl-dlp':
                    raise ValueError('OpenMP probe requires the AOCL-DLP Rust backend')
                entry.update(ready=worker.ready, ready_affinity=worker.affinity, load_wall_ms=worker.load_wall_ms)
                wanted = {key.encode() for key in environment}
                observed = dict(item.decode().split('=', 1) for item in Path(f'/proc/{worker.process.pid}/environ').read_bytes().split(b'\0') if b'=' in item and item.split(b'=', 1)[0] in wanted)
                entry['observed_environment'] = {key: observed.get(key) for key in environment}
                if entry['observed_environment'] != environment:
                    raise ValueError('Worker thread or wait-policy environment mismatch')
                def encode(request):
                    check(); value, wall = worker.encode(request)
                    if not math.isfinite(value['elapsed_ms']) or value['elapsed_ms'] <= 0 or not math.isfinite(wall) or wall <= 0 or ('ids' in request and value['tokens'] != len(request['ids'])):
                        raise ValueError('Invalid timing or explicit-token count')
                    return value, wall
                vectors, corpus = {}, []
                for doc in data['documents']:
                    value, _ = encode({'text': data['document_template'].format(**doc)})
                    vectors['doc:'+doc['id']] = value['embedding']; corpus.append((doc['id'], value['embedding']))
                for query in data['queries'][:5]:
                    encode({'text': data['query_prefix']+query['text']})
                cpu_start, wall_start = process_cpu_seconds(worker.process.pid), time.perf_counter()
                for repeat in range(5):
                    for query in data['queries']:
                        text = data['query_prefix']+query['text']; value, wall = encode({'text': text})
                        ids = tokenizer.encode(text, add_special_tokens=True).ids
                        if value['tokens'] != len(ids):
                            raise ValueError('Tokenizer count mismatch')
                        ranked = sorted(corpus, key=lambda item: sum(a*b for a, b in zip(value['embedding'], item[1])), reverse=True)
                        entry['rows'].append({'id': query['id'], 'repeat': repeat, 'tokens': value['tokens'], 'engine_ms': value['elapsed_ms'], 'wall_ms': wall,
                                              'top5': [item[0] for item in ranked[:5]], 'expected': query['expected']})
                        if repeat == 0:
                            vectors['query:'+query['id']] = value['embedding']; explicit, _ = encode({'ids': ids})
                            delta = max(abs(a-b) for a, b in zip(value['embedding'], explicit['embedding']))
                            entry['text_token_ids_max_abs_error'] = max(entry['text_token_ids_max_abs_error'], delta)
                            if delta > 1e-6:
                                raise ValueError('Text and common token-ID embeddings disagree')
                entry['query_block_worker_cpu_seconds'] = process_cpu_seconds(worker.process.pid)-cpu_start
                entry['query_block_wall_seconds'] = time.perf_counter()-wall_start
                entry['short_query_memory'] = worker.memory(); entry['short_query_affinity'] = worker_affinity(worker.process.pid)
                validate_affinity(entry['short_query_affinity'], selected)
                for length in data['length_tokens']:
                    ids = [2]+body[:length-2]+[1]; encode({'ids': ids})
                    for repeat in range(3):
                        value, wall = encode({'ids': ids}); entry['length_rows'].append({'tokens': length, 'repeat': repeat, 'engine_ms': value['elapsed_ms'], 'wall_ms': wall})
                        if repeat == 0:
                            vectors['length:'+str(length)] = value['embedding']
                entry['post_long_memory'] = worker.memory(); entry['post_long_affinity'] = worker_affinity(worker.process.pid)
                validate_affinity(entry['post_long_affinity'], selected)
                entry['latency_ms'] = {metric: distribution([row[metric] for row in entry['rows']]) for metric in ('engine_ms', 'wall_ms')}
                entry['length_latency_ms'] = {str(length): {metric: distribution([row[metric] for row in entry['length_rows'] if row['tokens'] == length])
                                            for metric in ('engine_ms', 'wall_ms')} for length in data['length_tokens']}
                if reference is None:
                    reference = vectors, entry['rows']
                gate(entry, vectors, reference); save()
            finally:
                worker.close()
        if source_hashes() != sources or sha(args.rust) != binary or any(sha(HERE/name) != checksum for name, checksum in drivers.items()):
            raise ValueError('Source, binary or driver changed during probe')
        result['status'] = 'complete'
    except BaseException as error:
        result['status'] = 'partial' if isinstance(error, TimeoutError) else 'failed'; result['failure_type'] = type(error).__name__
        for entry in result['runs']:
            if entry['status'] == 'running':
                entry['status'] = result['status']
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
