#!/usr/bin/env python3
"""Validate and summarize the checked-in counterbalanced benchmark runs."""
import hashlib
import json
import subprocess
from pathlib import Path
from compare import distribution

HERE = Path(__file__).resolve().parent
BACKENDS = ('transformers_eager', 'transformers_sdpa', 'rust_f32')
LABELS = {'transformers_eager': 'PyTorch eager', 'transformers_sdpa': 'PyTorch SDPA', 'rust_f32': 'Rust F32'}


def summarize():
    hosts = {}
    for host in ('arm64-apple-m2', 'amd64-ryzen-9955hx'):
        paths = [HERE/'results'/f'{host}-{order}.json' for order in ('forward', 'reverse')]
        runs = [json.loads(p.read_text()) for p in paths]
        assert all(r['status'] == 'complete' and r['parity_pass'] for r in runs)
        for key in ('fixture_sha256', 'model_revision', 'source_revision', 'cargo_lock_sha256',
                    'rust_source_sha256', 'driver_sha256', 'reference_worker_sha256', 'rust_binary_sha256', 'protocol'):
            assert runs[0][key] == runs[1][key], key
        assert runs[0]['backend_order'] == list(reversed(runs[1]['backend_order']))
        for name, path in (('rust_source_sha256', HERE.parent/'src/lib.rs'),
                           ('fixture_sha256', HERE/'fixtures.json'),
                           ('driver_sha256', HERE/'compare.py'),
                           ('reference_worker_sha256', HERE/'reference_worker.py')):
            assert runs[0][name] == hashlib.sha256(path.read_bytes()).hexdigest(), name
        # A packaging patch may change the package version after measurement.
        # Verify the measured lockfile against its recorded commit, and require
        # the model/activation source to remain identical in the current tree.
        revision = runs[0]['source_revision']
        locked = subprocess.check_output(['git', 'show', f'{revision}:Cargo.lock'], cwd=HERE)
        assert runs[0]['cargo_lock_sha256'] == hashlib.sha256(locked).hexdigest()
        for name in ('src/lib.rs', 'src/activation.rs'):
            measured = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=HERE)
            assert measured == (HERE.parent/name).read_bytes(), name
        memory_key = 'physical_footprint_mib' if host.startswith('arm') else 'rss_mib'
        entry = {'hardware': runs[0]['hardware'], 'source_revision': runs[0]['source_revision'],
                 'memory_metric': memory_key, 'files': [p.name for p in paths], 'backends': {}}
        for name in BACKENDS:
            backends = [r['backends'][name] for r in runs]
            rows = [row for b in backends for row in b['rows']]
            assert len(rows) == 200
            assert all(b['retrieval'] == {'queries': 20, 'hit_at_1': 20, 'hit_at_5': 20} for b in backends)
            assert all(r['ranking_agreement'][name] == 20 for r in runs)
            entry['backends'][name] = {
                'wall_ms': distribution([row['wall_ms'] for row in rows]),
                'engine_ms': distribution([row['engine_ms'] for row in rows]),
                'memory_mib': max(b['memory'][memory_key] for b in backends),
                'memory_samples': [b['memory'] for b in backends],
                'per_order_wall_ms': [b['latency_ms']['wall_ms'] for b in backends],
                'length_wall_ms': {str(n): distribution([row['wall_ms'] for b in backends for row in b['length_rows'] if row['tokens'] == n])
                                   for n in (128, 512, 1024)},
                'max_abs_error': max(r['parity'][name]['max_abs_error'] for r in runs),
                'min_cosine': min(r['parity'][name]['min_cosine'] for r in runs),
                'ready': backends[0]['ready'],
            }
        rust = entry['backends']['rust_f32']
        references = [entry['backends'][name] for name in BACKENDS[:2]]
        entry['rust_lower_median_p95_and_memory_than_both_references'] = (
            all(rust['wall_ms'][metric] < min(b['wall_ms'][metric] for b in references) for metric in ('median', 'p95'))
            and rust['memory_mib'] < min(b['memory_mib'] for b in references))
        entry['rust_query_latency_win_in_both_orders'] = all(
            rust['per_order_wall_ms'][i][metric] < min(b['per_order_wall_ms'][i][metric] for b in references)
            for i in range(2) for metric in ('median', 'p95'))
        hosts[host] = entry
    (HERE/'results'/'summary.json').write_text(json.dumps(hosts, indent=2)+'\n')
    lines = []
    for host, entry in hosts.items():
        lines.extend([f'## {entry["hardware"]["cpu"]}', '',
                      '| Backend | Query median / p95 (ms) | Memory (MiB) | 128 / 512 / 1,024 tokens, median (ms) |',
                      '| --- | ---: | ---: | ---: |'])
        for name in BACKENDS:
            b = entry['backends'][name]
            lengths = ' / '.join(f'{b["length_wall_ms"][str(n)]["median"]:.1f}' for n in (128, 512, 1024))
            lines.append(f'| {LABELS[name]} | {b["wall_ms"]["median"]:.1f} / {b["wall_ms"]["p95"]:.1f} | {b["memory_mib"]:.0f} | {lengths} |')
        lines.extend(['', '| Order | PyTorch eager median / p95 | PyTorch SDPA median / p95 | Rust median / p95 |',
                      '| --- | ---: | ---: | ---: |'])
        for index, order in enumerate(('Forward', 'Reverse')):
            values = [entry['backends'][name]['per_order_wall_ms'][index] for name in BACKENDS]
            lines.append('| '+order+' | '+' | '.join(f'{v["median"]:.1f} / {v["p95"]:.1f}' for v in values)+' |')
        lines.extend(['', 'Raw runs: '+', '.join(f'[{p}](results/{p})' for p in entry['files'])+'.', ''])
    return '\n'.join(lines), hosts


if __name__ == '__main__':
    table, hosts = summarize()
    print(table)
    print(json.dumps({h: e['rust_lower_median_p95_and_memory_than_both_references'] for h, e in hosts.items()}))
