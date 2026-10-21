#!/usr/bin/env python3
"""Render README charts from the validated, checked-in summary (no model run)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE/'charts'
OUT.mkdir(exist_ok=True)
DATA = json.loads((HERE/'results/summary.json').read_text())
HOSTS = [('arm64-apple-m2', 'Apple M2', 'Native macOS · Accelerate'),
         ('amd64-ryzen-9955hx', 'AMD Ryzen 9 9955HX', 'Native Linux · portable GEMM / MKL')]
BACKENDS = [('transformers_eager', 'PyTorch eager'),
            ('transformers_sdpa', 'PyTorch SDPA'), ('rust_f32', 'Rust F32')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.edgecolor': '#cbd5e1',
                     'text.color': '#172033', 'axes.labelcolor': '#475569',
                     'xtick.color': '#475569', 'ytick.color': '#172033',
                     'savefig.facecolor': 'white'})


def base(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.70, bottom=0.21, wspace=0.64)
    fig.text(0.04, 0.94, title, fontsize=21, weight='bold')
    fig.text(0.04, 0.875, subtitle, fontsize=11.5, color='#475569')
    for ax, (host, title, subtitle) in zip(axes, HOSTS):
        ax.set_title(title+'\n'+subtitle, fontsize=12, weight='bold', pad=15)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e2e8f0', linewidth=0.8)
        ax.set_yticks(range(3), [label for _, label in BACKENDS])
        ax.tick_params(axis='y', length=0, pad=9)
        ax.invert_yaxis()
    return fig, axes


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=180)
    plt.close(fig)


fig, axes = base('Uncached CPU latency',
                 'Same pinned text model · batch 1 · two kernel threads · 200 short-query timings per backend / machine')
colors = ['#2563eb', '#93c5fd']
for ax, (host, _, _) in zip(axes, HOSTS):
    for offset, metric, color in zip((-0.18, 0.18), ('median', 'p95'), colors):
        values = [DATA[host]['backends'][name]['wall_ms'][metric] for name, _ in BACKENDS]
        bars = ax.barh([i+offset for i in range(3)], values, height=0.31, color=color)
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=5, fontsize=11)
    peak = max(DATA[host]['backends'][name]['wall_ms']['p95'] for name, _ in BACKENDS)
    ax.set_xlim(0, peak*1.25)
    ax.set_xlabel('Request wall time (ms) · lower is better', fontsize=10.5, labelpad=9)
fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, ('Median', 'p95'))],
           loc='lower left', bbox_to_anchor=(0.035, 0.065), ncol=2, frameon=False, fontsize=11)
fig.text(0.04, 0.025, '15–28-token authored queries. AMD winner changes with run order: treat as latency parity. Independent x-axis scales.',
         fontsize=10, color='#475569')
save(fig, 'latency')

fig, axes = base('Process memory after the long-input tests',
                 'Larger of two post-test samples · compare backends within each machine · lower is better')
for ax, (host, _, _) in zip(axes, HOSTS):
    values = [DATA[host]['backends'][name]['memory_mib'] for name, _ in BACKENDS]
    bars = ax.barh(range(3), values, height=0.54, color=['#94a3b8', '#94a3b8', '#0f766e'])
    ax.bar_label(bars, labels=[f'{v:,.0f}' for v in values], padding=6, fontsize=12, weight='bold')
    ax.set_xlim(0, max(values)*1.30)
    metric = 'Physical footprint (MiB)' if host.startswith('arm') else 'RSS (MiB)'
    ax.set_xlabel(metric, fontsize=11, labelpad=9)
    baseline = DATA[host]['backends']['transformers_sdpa']['memory_mib']
    reduction = 100*(1-values[-1]/baseline)
    ax.text(0.5, -0.23, f'{reduction:.1f}% lower memory than SDPA', transform=ax.transAxes,
            color='#0f766e', fontsize=11, weight='bold', ha='center')
fig.text(0.04, 0.055, 'macOS footprint accounts for compressed memory; Linux RSS is a different metric. No cross-machine memory claim.',
         fontsize=10, color='#475569')
fig.text(0.04, 0.02, 'Rust retains the original BF16 vocabulary table; encoder math is F32. Full method, raw runs and limits: benchmarks/REPORT.md',
         fontsize=10, color='#475569')
save(fig, 'memory')
