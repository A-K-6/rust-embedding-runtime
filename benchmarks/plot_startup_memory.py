#!/usr/bin/env python3
"""Render the scalar startup-memory probe; requires matplotlib."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if data['status'] != 'complete' or len(data['trials']) != 4:
        raise ValueError('A complete four-trial report is required')
    trials = data['trials']
    labels = [f"{i+1}: {'Before' if t['runtime']=='previous' else 'Cleanup'}" for i,t in enumerate(trials)]
    colors = ['#667085' if t['runtime']=='previous' else '#3274b8' for t in trials]
    fig, ax = plt.subplots(figsize=(9.5,4.5), layout='constrained')
    ax.bar(labels, [t['after_all']['rss_mib'] for t in trials], color=colors, label='RSS after 24 inputs')
    ax.scatter(labels, [t['after_ready']['rss_mib'] for t in trials], marker='v', color='#111827', s=45, label='RSS after readiness', zorder=3)
    ax.scatter(labels, [t['after_all']['peak_rss_mib'] for t in trials], marker='D', facecolors='white', edgecolors='#111827', s=40, label='Process peak including load', zorder=3)
    for i,t in enumerate(trials):
        ax.text(i,t['after_all']['rss_mib']-75,f"{t['after_all']['rss_mib']:.1f}",ha='center',fontsize=10,color='white')
    ax.set(ylabel='Process memory (MiB)',ylim=(0,1400),title='Ryzen 9 9955HX: one startup cleanup, four fresh processes')
    ax.legend(loc='upper center', ncols=3, fontsize=9, frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=.15)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
