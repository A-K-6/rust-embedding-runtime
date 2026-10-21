#!/usr/bin/env python3
"""Render README charts from measured summaries; never run model inference.

Apple charts retain the historical v0.2 Accelerate results. AMD charts require
the completed six-order v0.3 AOCL matrix; missing AMD data is an error, rather
than a fallback to the historical AMD measurements. An optional scaling chart
combines the exploratory PyTorch scaling sweep and Rust wait-policy probe.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
BACKENDS = (('transformers_eager', 'PyTorch eager'),
            ('transformers_sdpa', 'PyTorch SDPA'), ('rust_f32', 'Rust F32'))
METRIC_COLORS = ('#2563eb', '#93c5fd')
BACKEND_COLORS = ('#64748b', '#2563eb', '#0f766e')
TEXT = '#475569'


def positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite, positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{field} must be a finite, positive number')
    return float(value)


def read_stats(entry, field, expected_count):
    if entry.get('count') != expected_count:
        raise ValueError(f'{field} requires {expected_count} measured timings')
    stats = {metric: positive(entry[metric], f'{field}.{metric}')
             for metric in ('median', 'p95')}
    if stats['p95'] < stats['median']:
        raise ValueError(f'{field}.p95 must not be below its median')
    return stats


def load_data(historical_path, amd_path):
    historical = json.loads(historical_path.read_text())['arm64-apple-m2']
    amd = json.loads(amd_path.read_text())
    if amd.get('status') != 'complete' or len(amd.get('schedule', [])) != 6:
        raise ValueError('AMD charts require a complete six-order summary')
    wait_environment = amd.get('runtime_wait_environment') or {}
    if (wait_environment.get('OMP_WAIT_POLICY') != 'PASSIVE'
            or wait_environment.get('GOMP_SPINCOUNT') != '0'):
        raise ValueError('AMD headline charts require the recorded PASSIVE / 0 environment')
    hosts = [
        {'title': 'Apple M2', 'runtime': 'v0.2 · Accelerate · native macOS',
         'samples': '200 query timings / backend · 2 orders',
         'memory_label': 'Physical footprint (MiB)', 'memory_count': 2, 'backends': {}},
        {'title': 'AMD Ryzen 9 9955HX',
         'runtime': 'v0.3 · AOCL F32 · native Linux\nOpenMP: PASSIVE / 0, all backends',
         'samples': '600 query timings / backend · 6 orders',
         'memory_label': 'Process RSS (MiB)', 'memory_count': 6, 'backends': {}},
    ]
    for name, _ in BACKENDS:
        old = historical['backends'][name]
        new = amd['backends'][name]
        if len(old['memory_samples']) != 2 or len(new['memory_samples']) != 6:
            raise ValueError('Expected two historical Apple and six AMD memory samples')
        hosts[0]['backends'][name] = {
            'wall_ms': read_stats(old['wall_ms'], f'Apple.{name}.wall_ms', 200),
            'memory_mib': positive(old['memory_mib'], f'Apple.{name}.memory_mib')}
        hosts[1]['backends'][name] = {
            'wall_ms': read_stats(new['query']['wall_ms'], f'AMD.{name}.wall_ms', 600),
            'memory_mib': positive(new['memory']['post_long_rss_mib'],
                                   f'AMD.{name}.post_long_rss_mib'),
            'lengths': {str(length): {
                metric: read_stats(new['lengths'][str(length)][metric],
                                   f'AMD.{name}.{length}.{metric}', 18)
                for metric in ('engine_ms', 'wall_ms')}
                for length in (128, 512, 1024)}}
    return hosts


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#cbd5e1',
                         'text.color': '#172033', 'axes.labelcolor': TEXT,
                         'xtick.color': TEXT, 'ytick.color': '#172033',
                         'savefig.facecolor': 'white'})


def base(title, subtitle, hosts):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.68, bottom=0.27, wspace=0.66)
    fig.text(0.035, 0.945, title, fontsize=21, weight='bold')
    fig.text(0.035, 0.887, subtitle, fontsize=11.5, color=TEXT)
    for ax, host in zip(axes, hosts):
        ax.set_title(host['title'] + '\n' + host['runtime'],
                     fontsize=12, weight='bold', pad=26)
        ax.text(0.5, 1.025, host['samples'], transform=ax.transAxes,
                ha='center', fontsize=9.5, color=TEXT)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e2e8f0', linewidth=0.8)
        ax.set_yticks(range(3), [label for _, label in BACKENDS])
        ax.tick_params(axis='y', length=0, pad=9)
        ax.invert_yaxis()
    return fig, axes


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=180)
    plt.close(fig)


def latency_chart(hosts, output):
    fig, axes = base('Uncached CPU latency',
                     'Same pinned text model · batch 1 · two kernel threads · lower is better within each panel',
                     hosts)
    for ax, host in zip(axes, hosts):
        for offset, metric, color in zip((-0.18, 0.18), ('median', 'p95'), METRIC_COLORS):
            values = [host['backends'][name]['wall_ms'][metric] for name, _ in BACKENDS]
            bars = ax.barh([i + offset for i in range(3)], values, height=0.31, color=color)
            ax.bar_label(bars, labels=[f'{value:.1f}' for value in values], padding=5, fontsize=11)
        peak = max(host['backends'][name]['wall_ms']['p95'] for name, _ in BACKENDS)
        ax.set_xlim(0, peak * 1.27)
        ax.set_xlabel('Request wall time (ms)', fontsize=11, labelpad=9)
    fig.legend(handles=[Patch(color=color, label=label)
                        for color, label in zip(METRIC_COLORS, ('Median', 'p95'))],
               loc='lower left', bbox_to_anchor=(0.03, 0.145), ncol=2, frameon=False, fontsize=11)
    fig.text(0.035, 0.10, '15–28-token authored queries; repeated timings do not add independent examples.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.061, 'Wall time includes tokenization and JSONL IPC; HTTP and network time are excluded.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.023, 'Different runtime versions and independent x-axis scales. Compare backends within each machine.',
             fontsize=10, color=TEXT)
    save(fig, output, 'latency')


def memory_chart(hosts, output):
    memory_hosts = [dict(host, samples=f"Largest of {host['memory_count']} post-long-input samples")
                    for host in hosts]
    fig, axes = base('Process memory after the long-input tests',
                     'Highest measured sample per backend · lower is better within each panel', memory_hosts)
    for ax, host in zip(axes, memory_hosts):
        values = [host['backends'][name]['memory_mib'] for name, _ in BACKENDS]
        bars = ax.barh(range(3), values, height=0.54, color=('#94a3b8', '#94a3b8', '#0f766e'))
        ax.bar_label(bars, labels=[f'{value:,.0f}' for value in values],
                     padding=6, fontsize=12, weight='bold')
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_xlabel(host['memory_label'], fontsize=11, labelpad=9)
        baseline = host['backends']['transformers_sdpa']['memory_mib']
        reduction = 100 * (1 - values[-1] / baseline)
        change = f'{abs(reduction):.1f}% {"less" if reduction >= 0 else "more"} memory than SDPA'
        ax.text(0.5, -0.28, change, transform=ax.transAxes, color='#0f766e' if reduction >= 0 else TEXT,
                fontsize=11, weight='bold', ha='center')
    fig.text(0.035, 0.095, 'macOS footprint accounts for compressed memory; Linux RSS measures resident pages.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.058, 'The metrics differ across machines. Compare backends within each panel; x-axis scales are independent.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.021, 'Rust retains the original BF16 vocabulary table; encoder math and AOCL packed linear weights are F32.',
             fontsize=10, color=TEXT)
    save(fig, output, 'memory')


def amd_long_chart(amd, output):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.71, bottom=0.25, wspace=0.28)
    fig.text(0.035, 0.945, 'AMD long-input CPU latency', fontsize=21, weight='bold')
    fig.text(0.035, 0.883, 'Ryzen 9 9955HX · v0.3 AOCL F32 · two kernel threads · 18 timings per length / backend',
             fontsize=11.5, color=TEXT)
    lengths = (128, 512, 1024)
    for ax, (metric, title) in zip(axes, (('engine_ms', 'Encoder + pooling'),
                                        ('wall_ms', 'Request wall time'))):
        for (name, label), color, marker, line in zip(
                BACKENDS, BACKEND_COLORS, ('o', 's', 'D'), ('--', ':', '-')):
            values = [amd['backends'][name]['lengths'][str(length)][metric]['median']
                      for length in lengths]
            ax.plot(lengths, values, marker=marker, linestyle=line,
                    markersize=6, linewidth=2.2, color=color, label=label)
        ax.set_title(title, fontsize=12, weight='bold', pad=14)
        ax.set_xticks(lengths, ['128', '512', '1,024'])
        ax.set_xlabel('Input tokens', fontsize=11, labelpad=9)
        ax.set_ylabel('Median latency (ms)', fontsize=11, labelpad=9)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e2e8f0', linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.03, 0.105),
               ncol=3, frameon=False, fontsize=11)
    fig.text(0.035, 0.064, 'Same model and token IDs · batch 1 · six orders · OpenMP PASSIVE / 0 for all backends.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.023, 'Medians summarize a small timing sample. Full distributions, quality gates and limits are in the benchmark reports.',
             fontsize=10, color=TEXT)
    save(fig, output, 'amd-long-latency')


def pooled_wall(rows, expected_count, field):
    """Pool raw requests, rather than average per-process summary statistics."""
    if len(rows) != expected_count:
        raise ValueError(f'{field} requires {expected_count} measured requests')
    values = sorted(positive(row['wall_ms'], f'{field}.wall_ms') for row in rows)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {'median': median, 'p95': values[math.ceil(len(values) * 0.95) - 1]}


def physical_cores(topology, selected, expected, field):
    if len(selected) != expected or len(set(selected)) != expected:
        raise ValueError(f'{field} requires {expected} distinct worker CPUs')
    by_cpu = {item['cpu']: (item['physical_package_id'], item['core_id']) for item in topology}
    if len({by_cpu[cpu] for cpu in selected}) != expected:
        raise ValueError(f'{field} must use one CPU per physical core')


def load_scaling(scaling_path, wait_path):
    scaling = json.loads(scaling_path.read_text())
    wait = json.loads(wait_path.read_text())
    if scaling.get('status') != 'complete' or wait.get('status') != 'complete':
        raise ValueError('The optional scaling chart requires both complete exploratory datasets')
    for field in ('model_revision', 'model_asset_sha256'):
        if scaling[field] != wait[field]:
            raise ValueError(f'Exploratory probe provenance differs: {field}')
    if scaling['hardware']['cpu'] != wait['hardware']['cpu']:
        raise ValueError('Exploratory probes must use the same CPU')
    for data in (scaling, wait):
        if data['frozen_protocol']['batch_size'] != 1 or data['frozen_protocol']['cache']:
            raise ValueError('Exploratory charts require uncached batch-one inference')

    series = []
    for name, label, color in (
            ('transformers_eager', 'PyTorch eager', '#64748b'),
            ('transformers_sdpa', 'PyTorch SDPA', '#2563eb')):
        values = {}
        for count in (1, 2, 4, 8, 16):
            setting = scaling['settings'][str(count)]
            if setting['status'] != 'complete':
                raise ValueError(f'Incomplete PyTorch scaling setting: {count}')
            physical_cores(scaling['hardware']['cpu_topology'], setting['worker_cpus'],
                           count, f'PyTorch {count}')
            values[count] = pooled_wall(setting['backends'][name]['rows'], 100,
                                        f'{name}.{count}')
        series.append({'label': label, 'color': color, 'line': '-', 'marker': 'o', 'values': values})

    for policy, label, color, line, marker in (
            ('passive', 'Rust PASSIVE / 0', '#0f766e', '-', 'D'),
            ('default', 'Rust OpenMP default', '#ea580c', '--', 's')):
        values = {}
        for count in (2, 4, 8, 16):
            entries = [entry for entry in wait['runs']
                       if entry['threads'] == count and entry['policy'] == policy]
            if len(entries) != 2 or {entry['order'] for entry in entries} != {1, 2}:
                raise ValueError(f'Rust {policy}/{count} requires both probe orders')
            rows = []
            for entry in entries:
                if entry['status'] != 'complete' or len(entry['rows']) != 100:
                    raise ValueError(f'Incomplete Rust wait-policy worker: {policy}/{count}')
                physical_cores(wait['hardware']['cpu_topology'], entry['worker_cpus'],
                               count, f'Rust {policy}/{count}')
                if policy == 'passive':
                    expected = {'OMP_WAIT_POLICY': 'PASSIVE', 'GOMP_SPINCOUNT': '0'}
                else:
                    expected = {'OMP_WAIT_POLICY': None, 'GOMP_SPINCOUNT': None}
                if any(entry['observed_environment'].get(key) != value
                       for key, value in expected.items()):
                    raise ValueError(f'Unexpected observed Rust wait policy: {policy}/{count}')
                rows.extend(entry['rows'])
            values[count] = pooled_wall(rows, 200, f'Rust.{policy}.{count}')
        series.append({'label': label, 'color': color, 'line': line,
                       'marker': marker, 'values': values})
    return series


def amd_scaling_chart(series, output):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.32, wspace=0.25)
    fig.text(0.035, 0.945, 'AMD CPU scaling · exploratory', fontsize=21, weight='bold')
    fig.text(0.035, 0.888, 'Ryzen 9 9955HX · same pinned model · batch 1 · uncached requests',
             fontsize=11.5, color=TEXT)
    all_values = [stats[metric] for entry in series for stats in entry['values'].values()
                  for metric in ('median', 'p95')]
    log_scale = max(all_values) / min(all_values) >= 10
    for ax, (metric, title) in zip(axes, (('median', 'Median request wall time'),
                                        ('p95', 'p95 request wall time'))):
        for entry in series:
            counts = sorted(entry['values'])
            values = [entry['values'][count][metric] for count in counts]
            ax.plot(counts, values, color=entry['color'], linestyle=entry['line'],
                    marker=entry['marker'], linewidth=2.2, markersize=6, label=entry['label'])
        ax.set_title(title, fontsize=12, weight='bold', pad=14)
        ax.set_xscale('log', base=2)
        ax.set_xticks((1, 2, 4, 8, 16), ['1', '2', '4', '8', '16'])
        ax.set_xlabel('Physical cores / kernel threads', fontsize=11, labelpad=9)
        ax.set_ylabel('Wall time (ms)' + (' · log scale' if log_scale else ''),
                      fontsize=11, labelpad=9)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(min(all_values) / 1.2, max(all_values) * 1.2)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        else:
            ax.set_ylim(0, max(all_values) * 1.15)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e2e8f0', linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.03, 0.178),
               ncol=4, frameon=False, fontsize=10.5)
    fig.text(0.035, 0.145, 'PyTorch: earlier scaling sweep, 100 timings per setting, one order.', fontsize=10, color=TEXT)
    fig.text(0.035, 0.111, 'Rust: later wait-policy probe, 200 pooled raw timings per policy / setting, two orders (A–B–B–A).',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.077, 'PASSIVE / 0 means OMP_WAIT_POLICY=PASSIVE, GOMP_SPINCOUNT=0; default explicitly unsets both.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.043, 'Unequal schedules and order counts: exploratory screening, separate from the six-order headline comparison.',
             fontsize=10, color=TEXT)
    fig.text(0.035, 0.009, 'Sixteen-core workers span two cache domains and overlap driver cores. Other CPUs are untested.',
             fontsize=10, color=TEXT)
    save(fig, output, 'amd-scaling')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', type=Path, default=HERE / 'results/summary.json')
    parser.add_argument('--amd', type=Path,
                        default=HERE / 'results/amd-aocl-passive-balanced-2026-10-22/summary.json')
    parser.add_argument('--scaling', type=Path,
                        default=HERE / 'results/amd-aocl-scaling-2026-10-22/scaling.json')
    parser.add_argument('--wait-policy', type=Path,
                        default=HERE / 'results/amd-aocl-wait-policy-2026-10-22/wait-policy.json')
    parser.add_argument('--only-scaling', action='store_true',
                        help='Render only the optional exploratory scaling chart')
    parser.add_argument('--output', type=Path, default=HERE / 'charts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    if not args.only_scaling:
        hosts = load_data(args.historical, args.amd)
        latency_chart(hosts, args.output)
        memory_chart(hosts, args.output)
        amd_long_chart(hosts[1], args.output)
    if args.only_scaling or (args.scaling.is_file() and args.wait_policy.is_file()):
        amd_scaling_chart(load_scaling(args.scaling, args.wait_policy), args.output)


if __name__ == '__main__':
    main()
