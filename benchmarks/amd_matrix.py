#!/usr/bin/env python3
"""Run all six CPU backend orders on one AMD64 host, then validate and summarize."""
import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess

from compare import BACKENDS, HERE, ROOT, distribution, parse_cpu_list, runtime_wait_environment, sha, source_hashes, validate_acceptance

LABELS = {'transformers_eager': 'PyTorch eager', 'transformers_sdpa': 'PyTorch SDPA', 'rust_f32': 'Rust F32'}
ORDERS = tuple(itertools.permutations(BACKENDS))
MEMORY_METRICS = ('short_rss_mib', 'post_long_rss_mib', 'peak_rss_mib')
LATENCY_METRICS = ('engine_ms', 'wall_ms')
PERCENTILES = ('median', 'p95')


def interpreter_path(path):
    # A virtualenv's python is often a symlink. Resolving it loses pyvenv.cfg
    # discovery and silently runs the base interpreter without its packages.
    return str(path.expanduser().absolute())


def memory_samples(entry):
    return {'short_rss_mib': entry['short_query_memory']['rss_mib'],
            'post_long_rss_mib': entry['post_long_memory']['rss_mib'],
            'peak_rss_mib': entry['post_long_memory']['peak_rss_mib']}


def timings(rows):
    for row in rows:
        for metric in LATENCY_METRICS:
            if not isinstance(row[metric], (float, int)) or not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError('Latency samples must be finite and positive')
    return {metric: distribution([row[metric] for row in rows]) for metric in LATENCY_METRICS}


def latency_win(rust, reference):
    return all(rust[metric][percentile] < reference[metric][percentile]
               for metric in LATENCY_METRICS for percentile in PERCENTILES)


def relative_latency(rust, reference):
    return {metric: {percentile: {'speedup': reference[metric][percentile]/rust[metric][percentile],
                                 'reduction_percent': 100*(1-rust[metric][percentile]/reference[metric][percentile])}
                     for percentile in PERCENTILES} for metric in LATENCY_METRICS}


def aggregate(runs, files):
    if len(runs) != len(ORDERS) or len(files) != len(ORDERS):
        raise ValueError('Exactly six complete runs are required')
    wait_environment = runs[0].get('runtime_wait_environment')
    if any(run.get('runtime_wait_environment') != wait_environment for run in runs):
        raise ValueError('Run provenance changed: runtime_wait_environment')
    shared = ('fixture_sha256', 'model_revision', 'source_revision', 'rust_binary_sha256',
              'cargo_lock_sha256', 'rust_source_sha256', 'source_file_sha256', 'driver_sha256',
              'reference_worker_sha256', 'build_provenance', 'rustc', 'driver_python',
              'hardware', 'protocol', 'worker_cpu_selection')
    protocol = runs[0]['protocol']
    if protocol['threads'] != 2 or protocol['repeats'] != 5 or protocol['length_repeats'] != 3:
        raise ValueError('Matrix requires the unchanged frozen two-thread protocol')
    for run, order in zip(runs, ORDERS):
        if run['status'] != 'complete':
            raise ValueError('A failed or incomplete run cannot be aggregated')
        validate_acceptance(run)
        if run['backend_order'] != list(order):
            raise ValueError('Run does not match the prespecified backend order')
        for key in shared:
            if run[key] != runs[0][key]:
                raise ValueError(f'Run provenance changed: {key}')
        for name in BACKENDS:
            entry = run['backends'][name]
            if wait_environment is not None and entry.get('runtime_wait_environment') != wait_environment:
                raise ValueError('Backend runtime wait environment differs from the run declaration')
            stable_ready = {key: value for key, value in entry['ready'].items() if key != 'load_ms'}
            first_ready = {key: value for key, value in runs[0]['backends'][name]['ready'].items() if key != 'load_ms'}
            if stable_ready != first_ready:
                raise ValueError(f'Backend readiness changed: {name}')
            if len(entry['rows']) != 100:
                raise ValueError('Each backend must contain 100 measured query requests per order')
            for length in (128, 512, 1024):
                if sum(row['tokens'] == length for row in entry['length_rows']) != 3:
                    raise ValueError('Each length requires three measured requests per order')
            if len(entry['length_rows']) != 9:
                raise ValueError('Unexpected length rows')
            for value in memory_samples(entry).values():
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Memory samples must be finite and positive')
    summary = {key: runs[0][key] for key in shared}
    summary['runtime_wait_environment'] = wait_environment
    summary['runtime_wait_environment_recorded'] = wait_environment is not None
    summary.update({'status': 'complete', 'schedule': [list(order) for order in ORDERS], 'raw_files': files,
                    'backends': {}, 'comparisons': {},
                    'scope': 'Native AMD64 CPU; same frozen public fixture; batch one; two kernel threads; six sequential orders. No core reservation, fixed frequency, HTTP or database in these timings.',
                    'memory_method': 'Process RSS after short queries and after long inputs; kernel peak RSS after long inputs. Aggregates use the largest of six samples, not sums or averages.',
                    'acceptance_method': 'Quality gates every run. Speed and memory goal requires lower engine AND wall median/p95 and all three memory metrics versus BOTH baselines, aggregated and in every order. Long-input comparisons are reported separately; no statistical significance claim.'})
    for name in BACKENDS:
        entries = [run['backends'][name] for run in runs]
        samples = [memory_samples(entry) for entry in entries]
        summary['backends'][name] = {
            'query': timings([row for entry in entries for row in entry['rows']]),
            'lengths': {str(length): timings([row for entry in entries for row in entry['length_rows'] if row['tokens'] == length])
                        for length in (128, 512, 1024)},
            'memory': {metric: max(sample[metric] for sample in samples) for metric in MEMORY_METRICS},
            'memory_samples': samples,
            'per_order_query': [timings(entry['rows']) for entry in entries],
            'ready': entries[0]['ready'],
            'max_abs_error': max(run['parity'][name]['max_abs_error'] for run in runs),
            'min_cosine': min(run['parity'][name]['min_cosine'] for run in runs),
            'retrieval_per_order': [entry['retrieval'] for entry in entries],
        }
    rust = summary['backends']['rust_f32']
    for name in BACKENDS[:2]:
        reference = summary['backends'][name]
        orders = []
        for index, (rust_query, ref_query) in enumerate(zip(rust['per_order_query'], reference['per_order_query'])):
            orders.append({'order': index+1, 'query': relative_latency(rust_query, ref_query),
                           'all_query_metrics_faster': latency_win(rust_query, ref_query),
                           'all_memory_metrics_lower': all(rust['memory_samples'][index][metric] < reference['memory_samples'][index][metric]
                                                           for metric in MEMORY_METRICS)})
        comparison = {'query': relative_latency(rust['query'], reference['query']),
                      'lengths': {length: relative_latency(rust['lengths'][length], reference['lengths'][length]) for length in rust['lengths']},
                      'memory_reduction_percent': {metric: 100*(1-rust['memory'][metric]/reference['memory'][metric]) for metric in MEMORY_METRICS},
                      'aggregate_query_win': latency_win(rust['query'], reference['query']),
                      'aggregate_memory_win': all(rust['memory'][metric] < reference['memory'][metric] for metric in MEMORY_METRICS),
                      'all_long_input_metrics_faster': all(latency_win(rust['lengths'][length], reference['lengths'][length]) for length in rust['lengths']),
                      'per_order': orders,
                      'query_win_in_all_orders': all(order['all_query_metrics_faster'] for order in orders),
                      'memory_win_in_all_orders': all(order['all_memory_metrics_lower'] for order in orders),
                      'minimum_order_speedup': min(order['query'][metric][percentile]['speedup'] for order in orders
                                                   for metric in LATENCY_METRICS for percentile in PERCENTILES)}
        comparison['speed_and_memory_goal_met'] = all(comparison[key] for key in (
            'aggregate_query_win', 'aggregate_memory_win', 'query_win_in_all_orders', 'memory_win_in_all_orders'))
        summary['comparisons'][name] = comparison
    summary['speed_and_memory_goal_met'] = all(value['speed_and_memory_goal_met'] for value in summary['comparisons'].values())
    summary['all_long_input_metrics_faster_than_both'] = all(value['all_long_input_metrics_faster'] for value in summary['comparisons'].values())
    return summary


def markdown(summary):
    lines = ['# AMD64 CPU matrix', '', summary['scope'], '',
             f"CPU: {summary['hardware']['cpu']}. Each backend has 600 measured short-query requests; each long-input length has 18.", '',
             'Runtime wait environment: '+(json.dumps(summary['runtime_wait_environment'], sort_keys=True) if summary['runtime_wait_environment_recorded'] else 'unrecorded in this legacy evidence')+'.', '',
             '| Backend | Engine median / p95 (ms) | Wall median / p95 (ms) | Short RSS / post-long RSS / peak RSS (MiB) |',
             '| --- | ---: | ---: | ---: |']
    for name in BACKENDS:
        entry = summary['backends'][name]
        values = [f"{entry['query'][metric]['median']:.2f} / {entry['query'][metric]['p95']:.2f}" for metric in LATENCY_METRICS]
        memory = ' / '.join(f'{entry["memory"][metric]:.1f}' for metric in MEMORY_METRICS)
        lines.append('| '+LABELS[name]+' | '+' | '.join(values)+' | '+memory+' |')
    lines += ['', '| Backend | 128 tokens engine / wall median (ms) | 512 tokens engine / wall median (ms) | 1,024 tokens engine / wall median (ms) |',
              '| --- | ---: | ---: | ---: |']
    for name in BACKENDS:
        values = [f"{summary['backends'][name]['lengths'][str(length)]['engine_ms']['median']:.2f} / {summary['backends'][name]['lengths'][str(length)]['wall_ms']['median']:.2f}"
                  for length in (128, 512, 1024)]
        lines.append('| '+LABELS[name]+' | '+' | '.join(values)+' |')
    lines += ['', '| Order | Sequence | Eager engine / wall median / p95 (ms) | SDPA engine / wall median / p95 (ms) | Rust engine / wall median / p95 (ms) |',
              '| --- | --- | ---: | ---: | ---: |']
    for index, order in enumerate(summary['schedule']):
        values = []
        for name in BACKENDS:
            query = summary['backends'][name]['per_order_query'][index]
            values.append('; '.join(f'{query[metric]["median"]:.2f} / {query[metric]["p95"]:.2f}' for metric in LATENCY_METRICS))
        lines.append(f'| {index+1} | '+', '.join(LABELS[name] for name in order)+' | '+' | '.join(values)+' |')
    lines += ['', f"Short-query speed and memory goal versus both baselines, including every order: **{summary['speed_and_memory_goal_met']}**.",
              f"Faster at every reported long-input metric versus both: **{summary['all_long_input_metrics_faster_than_both']}**.", '',
              summary['acceptance_method'], '', summary['memory_method'], '',
              'The 20 authored retrieval labels are a small correctness pilot. Repeating them does not create additional relevance labels. '
              'Results describe this CPU and software build; they do not establish performance on every consumer or server CPU.', '',
              'Raw runs: '+', '.join(f'[{name}]({name})' for name in summary['raw_files'])+'.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust', required=True, type=Path)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--model', default=ROOT/'model', type=Path)
    parser.add_argument('--cpus', type=parse_cpu_list)
    parser.add_argument('--build-provenance', required=True, help='Explicit build command, feature and flag declaration; not inferred from runtime environment')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        parser.error('This matrix requires a native Linux AMD64 host')
    if args.output_dir.exists():
        parser.error('Refusing to overwrite benchmark evidence; choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = source_hashes()
    binary_hash = sha(args.rust)
    driver_hash = sha(HERE/'compare.py')
    reference_hash = sha(HERE/'reference_worker.py')
    schedule = [{'file': f'run-{index+1:02d}.json', 'backend_order': list(order)} for index, order in enumerate(ORDERS)]
    manifest = {'status': 'running', 'schedule': schedule, 'matrix_driver_sha256': sha(Path(__file__)),
                'source_file_sha256': source, 'compare_driver_sha256': driver_hash, 'rust_binary_sha256': binary_hash,
                'reference_worker_sha256': reference_hash,
                'runtime_wait_environment': runtime_wait_environment(),
                'build_provenance_declaration': args.build_provenance,
                'worker_cpus': args.cpus, 'driver_affinity_modified': False,
                'note': 'Order fixed before measurement. Caller declaration records build flags; runtime environment does not prove them.'}
    manifest_path = args.output_dir/'matrix.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    save_manifest()
    runs = []
    def validate_unchanged():
        if runtime_wait_environment() != manifest['runtime_wait_environment']:
            raise ValueError('Runtime wait environment changed during the matrix')
        if (source_hashes() != source or sha(args.rust) != binary_hash or sha(HERE/'compare.py') != driver_hash or
                sha(HERE/'reference_worker.py') != reference_hash or sha(Path(__file__)) != manifest['matrix_driver_sha256']):
            raise ValueError('Source, binary or benchmark driver changed during the matrix')
    try:
        for item in schedule:
            validate_unchanged()
            command = [interpreter_path(args.python), str(HERE/'compare.py'), '--python', interpreter_path(args.python),
                       '--rust', str(args.rust.resolve()), '--model', str(args.model.resolve()),
                       '--order', ','.join(item['backend_order']), '--build-provenance', args.build_provenance,
                       '--output', str((args.output_dir/item['file']).resolve())]
            if args.cpus is not None:
                command.extend(['--cpus', ','.join(map(str, args.cpus))])
            subprocess.run(command, check=True, cwd=ROOT)
            run = json.loads((args.output_dir/item['file']).read_text())
            if run.get('runtime_wait_environment') != manifest['runtime_wait_environment']:
                raise ValueError('New run is missing the prespecified runtime wait environment')
            if run['source_file_sha256'] != source or run['rust_binary_sha256'] != binary_hash or run['driver_sha256'] != driver_hash:
                raise ValueError('Run provenance does not match the prespecified matrix')
            runs.append(run)
            manifest['completed_runs'] = len(runs)
            save_manifest()
        validate_unchanged()
        summary = aggregate(runs, [item['file'] for item in schedule])
        summary['matrix_driver_sha256'] = manifest['matrix_driver_sha256']
        (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        (args.output_dir/'REPORT.md').write_text(markdown(summary))
        manifest['status'] = 'complete'
        manifest['speed_and_memory_goal_met'] = summary['speed_and_memory_goal_met']
        print(json.dumps({'status': 'complete', 'speed_and_memory_goal_met': summary['speed_and_memory_goal_met']}), flush=True)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['failure_type'] = type(error).__name__
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    main()
