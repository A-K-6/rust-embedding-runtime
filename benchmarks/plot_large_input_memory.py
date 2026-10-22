#!/usr/bin/env python3
"""Plot all calls from a complete large_input_memory.py result as PNG or SVG."""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def metric(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Expected a finite, nonnegative measurement')
    return value


def validated_trials(data):
    trials = data.get('trials', [])
    if (data.get('status') != 'complete' or data.get('resource_conditions_met') is not True
            or data.get('attempted_inferences') != 44 or len(trials) != 4
            or [trial['runtime'] for trial in trials] != ['previous', 'candidate', 'candidate', 'previous']):
        raise ValueError('A complete controlled four-trial ABBA result is required')
    for trial in trials:
        if trial.get('status') != 'complete' or len(trial.get('rows', [])) != 11:
            raise ValueError('All eleven calls in every trial must be retained')
        for record in [trial['after_ready'], trial['after_all']]:
            metric(record['rss_mib'])
            metric(record['peak_rss_mib'])
        for row in trial['rows']:
            if row.get('status') != 'complete' or row.get('parity', {}).get('passed') is not True:
                raise ValueError('Every call must have completed and passed numerical checks')
            metric(row['wall_ms'])
            metric(row['resources']['rss_mib'])
            metric(row['resources']['peak_rss_mib'])
        for length in (512, 1024):
            longs = [row for row in trial['rows'] if row['kind'] == 'synthetic_ids' and row['length'] == length]
            recoveries = [row for row in trial['rows'] if row['phase'] == 'recovery' and row['after_length'] == length]
            if len(longs) != 2 or len(recoveries) != 2:
                raise ValueError('Both rounds of each long input and recovery are required')
    return trials


def draw(data, output):
    trials = validated_trials(data)
    labels = [f"Trial {index+1}\n{'Startup only' if trial['runtime'] == 'previous' else 'Large-input cleanup'}"
              for index, trial in enumerate(trials)]
    colors = ['#667085' if trial['runtime'] == 'previous' else '#2563a6' for trial in trials]
    fig = plt.figure(figsize=(12, 9.6), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1.05, .95])
    latency_axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    memory_axes = [fig.add_subplot(grid[1, column]) for column in range(2)]
    peak_axis = fig.add_subplot(grid[2, :])
    for column, length in enumerate((512, 1024)):
        latency, memory = latency_axes[column], memory_axes[column]
        for index, (trial, color) in enumerate(zip(trials, colors)):
            longs = [row for row in trial['rows'] if row['kind'] == 'synthetic_ids' and row['length'] == length]
            recoveries = [row for row in trial['rows'] if row['phase'] == 'recovery' and row['after_length'] == length]
            for repeat, (long, recovery) in enumerate(zip(longs, recoveries)):
                offset = -.09 if repeat == 0 else .09
                latency.scatter(index+offset, long['wall_ms'], s=55, marker='o',
                    facecolors=color if repeat == 0 else 'white', edgecolors=color,
                    label=('First at this length' if repeat == 0 else 'Second at this length') if index == 0 else None,
                    zorder=3)
                memory.scatter(index+offset-.025, long['resources']['rss_mib'], s=50,
                    marker='o', color=color, label='After long input' if index == repeat == 0 else None,
                    zorder=3)
                memory.scatter(index+offset+.025, recovery['resources']['rss_mib'], s=55,
                    marker='v', facecolors='white', edgecolors=color,
                    label='After next short input' if index == repeat == 0 else None, zorder=3)
        latency.set(title=f'{length} tokens: every long-input call', ylabel='Wall time (ms)', ylim=(0, None))
        memory.set(title=f'{length} tokens: retained memory', ylabel='Process RSS (MiB)', ylim=(0, None))
        for axis in (latency, memory):
            axis.set_xticks(range(4), labels, fontsize=8)
            axis.legend(frameon=False, fontsize=8, loc='upper left')
    for index, (trial, color) in enumerate(zip(trials, colors)):
        records = ((-.14, trial['after_ready']['rss_mib'], 'o', 'RSS after readiness'),
                   (0, trial['after_ready']['peak_rss_mib'], '^', 'Peak through readiness'),
                   (.14, trial['after_all']['peak_rss_mib'], 'D', 'Peak through all 11 calls'))
        for offset, value, marker, label in records:
            peak_axis.scatter(index+offset, value, s=55, marker=marker,
                facecolors=color if marker == 'o' else 'white', edgecolors=color,
                label=label if index == 0 else None, zorder=3)
    peak_axis.set(title='Readiness RSS and process peak: separate from retained memory',
                  ylabel='Process memory (MiB)', ylim=(0, None))
    peak_axis.set_xticks(range(4), labels, fontsize=9)
    peak_axis.legend(frameon=False, fontsize=8, ncols=3, loc='upper left')
    for axis in [*latency_axes, *memory_axes, peak_axis]:
        axis.set_xlim(-.5, 3.5)
        axis.set_ylim(0, axis.get_ylim()[1]*1.30)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.16)
    fig.suptitle('Large-input cleanup: four fresh two-thread processes', fontsize=15)
    fig.supxlabel('44 timed calls; first use retained. CPU quota: 2 cores; memory limit: 3 GiB.\n'
                  'Peak RSS includes startup. Lengths are plotted separately; no pooled tail-latency claim.', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() not in ('.png', '.svg'):
        parser.error('--output must have a .png or .svg extension')
    draw(json.loads(args.result.read_text()), args.output)


if __name__ == '__main__':
    main()
