"""Model-free acceptance, topology and provenance checks for the AMD harness."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from amd_matrix import BACKENDS, ORDERS, aggregate, markdown, interpreter_path
from compare import parse_cpu_list, source_hashes, validate_acceptance, validate_affinity, validate_worker_cpus


def valid_runs():
    """Small synthetic timings exercise acceptance; they are not benchmark evidence."""
    common = {'fixture_sha256': 'fixture', 'model_revision': 'model', 'source_revision': 'source',
              'rust_binary_sha256': 'binary', 'cargo_lock_sha256': 'lock', 'rust_source_sha256': 'lib',
              'source_file_sha256': {'src/nested/kernel.rs': 'kernel'}, 'driver_sha256': 'driver',
              'reference_worker_sha256': 'reference', 'build_provenance': {'caller_declaration': 'explicit test declaration'},
              'rustc': 'test compiler', 'driver_python': 'test Python', 'hardware': {'cpu': 'Synthetic test CPU'},
              'protocol': {'threads': 2, 'repeats': 5, 'length_repeats': 3}, 'worker_cpu_selection': None,
              'status': 'complete', 'parity_pass': True, 'parity': {}, 'backends': {},
              'ranking_agreement': {}, 'ranking_agreement_all_requests': {}}
    for name in BACKENDS:
        latency = 8 if name == 'rust_f32' else 10
        memory = 800 if name == 'rust_f32' else 1200
        rows = [{'id': f'q{query}', 'repeat': repeat, 'engine_ms': latency, 'wall_ms': latency+1,
                 'top5': [f'doc{query}', 'other'], 'expected': f'doc{query}'}
                for repeat in range(5) for query in range(20)]
        common['backends'][name] = {
            'rows': rows,
            'length_rows': [{'tokens': length, 'repeat': repeat, 'engine_ms': latency*2, 'wall_ms': latency*2+1}
                            for length in (128, 512, 1024) for repeat in range(3)],
            'short_query_memory': {'rss_mib': memory},
            'post_long_memory': {'rss_mib': memory+100, 'peak_rss_mib': memory+200},
            'retrieval': {'queries': 20, 'hit_at_1': 20, 'hit_at_5': 20}, 'ready': {'threads': 2}}
        common['parity'][name] = {'vectors': 33, 'max_abs_error': 1e-7, 'min_cosine': 1.0}
        common['ranking_agreement'][name] = 20
        common['ranking_agreement_all_requests'][name] = 100
    runs = []
    for order in ORDERS:
        run = copy.deepcopy(common)
        run['backend_order'] = list(order)
        runs.append(run)
    return runs


class AcceptanceTests(unittest.TestCase):
    def test_complete_matrix_and_safe_report(self):
        summary = aggregate(valid_runs(), [f'run-{i}.json' for i in range(6)])
        self.assertTrue(summary['speed_and_memory_goal_met'])
        self.assertEqual(summary['backends']['rust_f32']['query']['wall_ms']['count'], 600)
        self.assertEqual(summary['backends']['rust_f32']['lengths']['1024']['engine_ms']['count'], 18)
        self.assertIn('every order', markdown(summary))
        json.dumps(summary, allow_nan=False)

    def test_negative_quality_and_numerical_gates(self):
        for mutate in (
            lambda run: run['backends']['rust_f32']['retrieval'].update(hit_at_1=19),
            lambda run: run['ranking_agreement'].update(rust_f32=19),
            lambda run: run['ranking_agreement_all_requests'].update(rust_f32=99),
            lambda run: run['parity']['rust_f32'].update(max_abs_error=float('nan')),
            lambda run: run['parity']['rust_f32'].update(vectors=32),
            lambda run: run['backends']['rust_f32']['rows'][-1].update(top5=['wrong']),
        ):
            run = valid_runs()[0]
            mutate(run)
            with self.assertRaises(ValueError):
                validate_acceptance(run)

    def test_one_slow_order_prevents_pooled_win_claim(self):
        runs = valid_runs()
        for row in runs[-1]['backends']['rust_f32']['rows']:
            row.update(engine_ms=10.01, wall_ms=11.01)
        summary = aggregate(runs, [f'{i}.json' for i in range(6)])
        self.assertFalse(summary['speed_and_memory_goal_met'])
        for comparison in summary['comparisons'].values():
            self.assertFalse(comparison['query_win_in_all_orders'])
            self.assertLess(comparison['minimum_order_speedup'], 1)

    def test_engine_and_memory_must_also_win(self):
        for metric in ('engine_ms', 'memory'):
            runs = valid_runs()
            if metric == 'engine_ms':
                for run in runs:
                    for row in run['backends']['rust_f32']['rows']:
                        row['engine_ms'] = 10.1
            else:
                runs[-1]['backends']['rust_f32']['short_query_memory']['rss_mib'] = 1201
            summary = aggregate(runs, [f'{i}.json' for i in range(6)])
            self.assertFalse(summary['speed_and_memory_goal_met'])

    def test_changed_provenance_order_or_missing_run_rejected(self):
        for mutate in (
            lambda runs: runs[-1]['source_file_sha256'].update({'src/new.rs': 'different'}),
            lambda runs: runs[-1].update(status='failed'),
            lambda runs: runs[-1].update(backend_order=list(ORDERS[0])),
            lambda runs: runs[-1]['backends']['rust_f32']['rows'][0].update(wall_ms=float('inf')),
        ):
            runs = valid_runs()
            mutate(runs)
            with self.assertRaises(ValueError):
                aggregate(runs, [f'{i}.json' for i in range(6)])
        with self.assertRaises(ValueError):
            aggregate(valid_runs()[:5], [f'{i}.json' for i in range(5)])


class AffinityAndProvenanceTests(unittest.TestCase):
    def test_cpu_parser_rejects_duplicates_invalid_and_reverse_ranges(self):
        self.assertEqual(parse_cpu_list('2,4-5'), [2, 4, 5])
        for value in ('', '-1', '2,2', '2-4,3', '4-2', 'all', '1-2-3'):
            with self.assertRaises(ValueError):
                parse_cpu_list(value)

    @staticmethod
    def core(root, cpu, core, package=0, cache='2,4', level=3):
        directory = root/f'cpu{cpu}'
        (directory/'topology').mkdir(parents=True)
        (directory/'topology/core_id').write_text(str(core))
        (directory/'topology/physical_package_id').write_text(str(package))
        cache_dir = directory/'cache/index3'
        cache_dir.mkdir(parents=True)
        (cache_dir/'level').write_text(str(level))
        (cache_dir/'type').write_text('Unified')
        (cache_dir/'shared_cpu_list').write_text(cache)

    def test_physical_and_cache_topology_instead_of_assumed_cpu_numbering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.core(root, 2, core=7)
            self.core(root, 4, core=8)
            self.assertTrue(validate_worker_cpus([2, 4], root, allowed={2, 4})['same_last_level_cache_domain'])
            (root/'cpu4/topology/core_id').write_text('7')
            with self.assertRaisesRegex(ValueError, 'SMT'):
                validate_worker_cpus([2, 4], root)
            (root/'cpu4/topology/core_id').write_text('8')
            (root/'cpu4/cache/index3/shared_cpu_list').write_text('4,6')
            with self.assertRaisesRegex(ValueError, 'cache domain'):
                validate_worker_cpus([2, 4], root)
            with self.assertRaisesRegex(ValueError, 'online'):
                validate_worker_cpus([2, 4], root, allowed={2})

    def test_all_worker_threads_must_preserve_requested_mask(self):
        good = {'process_cpus': [2, 4], 'thread_masks': [{'cpus': [2, 4], 'threads': 3}]}
        validate_affinity(good, [2, 4])
        good['thread_masks'].append({'cpus': [2, 4, 6], 'threads': 1})
        with self.assertRaises(ValueError):
            validate_affinity(good, [2, 4])

    def test_recursive_sources_and_build_configuration_are_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'src/nested').mkdir(parents=True)
            (root/'.cargo').mkdir()
            for name in ('src/lib.rs', 'src/nested/kernel.rs', 'Cargo.toml', 'Cargo.lock', '.cargo/config.toml'):
                (root/name).write_text(name)
            hashes = source_hashes(root)
            self.assertEqual(len(hashes), 5)
            (root/'src/nested/kernel.rs').write_text('changed')
            self.assertNotEqual(hashes, source_hashes(root))


class InterpreterTests(unittest.TestCase):
    def test_subprocess_preserves_virtual_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary)/'environment'
            subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(environment)], check=True)
            python = environment/'bin/python'
            prefix = subprocess.check_output([interpreter_path(python), '-c', 'import sys; print(sys.prefix)'], text=True).strip()
            self.assertEqual(Path(prefix).resolve(), environment.resolve())


if __name__ == '__main__':
    unittest.main()
