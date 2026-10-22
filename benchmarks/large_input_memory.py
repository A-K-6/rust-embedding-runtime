#!/usr/bin/env python3
"""Bounded Linux ABBA probe of startup-only versus large-input heap cleanup.

Requires local profile-enabled F32 JSONL executables, pinned assets, cgroup v2,
two distinct physical worker cores sharing a cache, a two-core CPU quota and a
3-GiB memory limit. No downloads, retries or answer caching. Every first-use call
is timed. Synthetic agreement is separate from official-reference correctness.
The driver shares the CPU quota. Process peak RSS includes startup and is never
described as a steady-memory saving. Profiles overlap and must not be summed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sys
import time

from compare import distribution, parse_cpu_list, validate_affinity, validate_worker_cpus, worker_affinity
from startup_memory import (HERE, REFERENCE, REFERENCE_SHA, THREAD_ENV, ProbeError, Worker,
    cgroup, controlled, file_sha, parity, ready_metadata, snapshot, worker_environment)

ORDER = ('previous', 'candidate', 'candidate', 'previous')
PROFILE_KEYS = frozenset(('activation', 'allocator_cleanup', 'attention', 'attention_output',
    'embedding_lookup', 'encoder_total', 'ffn_total', 'linear', 'ple_block', 'ple_projection',
    'pool_and_projection', 'positions_and_masks', 'qkv_and_norm', 'rms_norm',
    'rope_and_repeat', 'tokenization'))


def requests(cases):
    """Eleven fixed calls per fresh process; each long call has a short recovery."""
    for index, case in enumerate(cases):
        yield ({'text': case['text'], 'dims': 768}, case['embedding'],
               {'kind': 'official', 'case_index': index, 'phase': 'initial',
                'expected_tokens': case['tokens']})
    for repeat in range(2):
        for index, length in enumerate((512, 1024)):
            yield ({'ids': [2]*length, 'dims': 768}, None,
                   {'kind': 'synthetic_ids', 'length': length, 'repeat': repeat,
                    'phase': 'long', 'expected_tokens': length})
            case_index = (2*repeat+index) % len(cases)
            case = cases[case_index]
            yield ({'text': case['text'], 'dims': 768}, case['embedding'],
                   {'kind': 'official', 'case_index': case_index, 'repeat': repeat,
                    'phase': 'recovery', 'after_length': length,
                    'expected_tokens': case['tokens']})


def profile(value):
    times = value.get('profile_ms')
    if (not isinstance(times, dict) or not {'encoder_total', 'ffn_total'}.issubset(times)
            or set(times)-PROFILE_KEYS or any(type(v) not in (int, float)
                or not math.isfinite(v) or v < 0 for v in times.values())):
        raise ProbeError('valid_profile_feature_required')
    return times


def summarize(rows):
    """Never pool latency across input lengths, official cases or recovery phase."""
    groups = {}
    for row in rows:
        if row['kind'] == 'synthetic_ids':
            key = f"synthetic_{row['length']}"
        else:
            key = f"official_{row['case_index']}_{row['tokens']}_tokens_{row['phase']}"
            if row['phase'] == 'recovery':
                key += f"_after_{row['after_length']}"
        groups.setdefault(key, []).append(row)
    return {key: {'tokens': entries[0]['tokens'],
        'wall_ms': distribution([row['wall_ms'] for row in entries]),
        'encoder_ms': distribution([row['encoder_ms'] for row in entries]),
        'post_call_rss_mib': [row['resources']['rss_mib'] for row in entries],
        'process_peak_rss_mib': [row['resources']['peak_rss_mib'] for row in entries]}
        for key, entries in groups.items()}


def revision(value):
    if not re.fullmatch(r'[0-9a-f]{40}', value):
        raise argparse.ArgumentTypeError('Expected a full lowercase Git revision')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path, help='Startup-only cleanup JSONL executable')
    parser.add_argument('--candidate', required=True, type=Path, help='Large-input cleanup JSONL executable')
    parser.add_argument('--model', required=True, type=Path, help='Pinned local text-only model assets')
    parser.add_argument('--cpus', required=True, type=parse_cpu_list)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--previous-source-revision', type=revision)
    parser.add_argument('--candidate-source-revision', type=revision)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name+'.tmp').exists():
        parser.error('Refusing to overwrite existing evidence')
    started = time.monotonic()
    trials = [{'trial': i, 'runtime': name, 'status': 'not_started', 'rows': []}
              for i, name in enumerate(ORDER)]
    result = {'schema_version': 1, 'status': 'starting', 'trials': trials,
        'budget': {'wall_seconds': 300, 'work_seconds': 294, 'cleanup_seconds': 6,
            'startup_seconds': 60, 'response_seconds': 30,
            'inferences_per_trial': 11, 'total_inferences': 44},
        'protocol': {'threads': 2, 'order': list(ORDER), 'synthetic_lengths': [512, 1024],
            'official_initial_calls': 3, 'long_and_recovery_rounds': 2,
            'timings_include_first_use': True, 'quantile_method': 'nearest rank',
            'latency_groups': 'exact token length, case and initial/recovery phase',
            'peak_memory_includes_startup': True, 'driver_shares_cpu_quota': True,
            'synthetic_reference': 'first previous result for each length; agreement only'},
        'environment': THREAD_ENV, 'attempted_inferences': 0,
        'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'),
        'driver_python': platform.python_version()}
    binaries = {'previous': args.previous.expanduser().absolute(),
                'candidate': args.candidate.expanduser().absolute()}

    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name+'.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)

    def expire(*_):
        raise ProbeError('overall_wall_budget')

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(294)  # Reserve bounded worker termination time within 300 seconds.
    exit_code = 1
    try:
        if platform.system() != 'Linux' or not shutil.which('taskset'):
            raise ProbeError('linux_taskset_required')
        if file_sha(REFERENCE) != REFERENCE_SHA:
            raise ProbeError('reference_hash_mismatch')
        reference = json.loads(REFERENCE.read_text())
        if len(reference['cases']) != 3:
            raise ProbeError('reference_case_count')
        result['reference_sha256'] = REFERENCE_SHA
        result['model_revision'] = reference['model_revision']
        result['model_asset_sha256'] = {}
        for name, expected in reference['model_asset_sha256'].items():
            actual = file_sha(args.model/name)
            if actual != expected:
                raise ProbeError('model_asset_hash_mismatch')
            result['model_asset_sha256'][name] = actual
        result['binary_sha256'] = {name: file_sha(path) for name, path in binaries.items()}
        if result['binary_sha256']['previous'] == result['binary_sha256']['candidate']:
            raise ProbeError('identical_binaries_not_a_version_comparison')
        result['declared_source_revisions'] = {'previous': args.previous_source_revision,
                                               'candidate': args.candidate_source_revision}
        result['helper_sha256'] = {name: file_sha(HERE/name)
            for name in ('large_input_memory.py', 'startup_memory.py', 'compare.py')}
        result['driver_cgroup'] = cgroup(os.getpid())
        if not controlled(result['driver_cgroup']):
            raise ProbeError('requires_two_cpu_three_gib_cgroup')
        result['resource_conditions_met'] = True
        result['cpu_selection'] = validate_worker_cpus(args.cpus)
        result['driver_affinity'] = sorted(os.sched_getaffinity(0))
        result['cpu_model'] = next(line.split(':', 1)[1].strip()
            for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name'))
        sequence = list(requests(reference['cases']))
        if len(sequence) != 11 or len(sequence)*len(trials) > 60:
            raise ProbeError('protocol_request_budget')
        synthetic_reference = {}
        previous_backend = None
        for trial in trials:
            worker = None
            trial['status'] = 'starting'
            save()
            try:
                if file_sha(binaries[trial['runtime']]) != result['binary_sha256'][trial['runtime']]:
                    raise ProbeError('binary_changed_before_trial')
                launch = time.monotonic()
                worker = Worker(binaries[trial['runtime']], args.model.expanduser().absolute(), args.cpus)
                startup_deadline = launch+60
                while True:
                    ready = worker.receive('stderr', max(.001, startup_deadline-time.monotonic()))
                    if ready.get('ready') is True:
                        break
                    if time.monotonic() >= startup_deadline:
                        raise ProbeError('startup_deadline')
                trial['ready'] = ready_metadata(ready)
                if trial['ready'].get('startup_allocator_trimmed') is not True:
                    raise ProbeError('startup_cleanup_required_for_both_binaries')
                backend = trial['ready']['cpu_backend']
                previous_backend = previous_backend or backend
                if backend != previous_backend:
                    raise ProbeError('backend_mismatch')
                trial['startup_wall_ms'] = (time.monotonic()-launch)*1000
                trial['environment'] = worker_environment(worker.process.pid)
                trial['after_ready'] = snapshot(worker.process.pid)
                if not controlled(trial['after_ready']['cgroup']):
                    raise ProbeError('worker_cgroup_uncontrolled')
                trial['affinity'] = worker_affinity(worker.process.pid)
                validate_affinity(trial['affinity'], args.cpus)
                for request, expected, metadata in sequence:
                    if result['attempted_inferences'] >= 44:
                        raise ProbeError('inference_budget')
                    row = dict(metadata, status='started')
                    trial['rows'].append(row)
                    result['attempted_inferences'] += 1
                    sent = time.monotonic()
                    worker.process.stdin.write(json.dumps(request).encode()+b'\n')
                    worker.process.stdin.flush()
                    value = worker.receive('stdout', 30)
                    row['wall_ms'] = (time.monotonic()-sent)*1000
                    if 'error' in value:
                        raise ProbeError('worker_inference_error')
                    if value.get('tokens') != metadata['expected_tokens']:
                        raise ProbeError('token_count_mismatch')
                    elapsed = value.get('elapsed_ms')
                    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                        raise ProbeError('invalid_encoder_timing')
                    row.update(encoder_ms=elapsed, tokens=value['tokens'], profile_ms=profile(value))
                    cleanup_expected = trial['runtime'] == 'candidate' and metadata['kind'] == 'synthetic_ids'
                    if ('allocator_cleanup' in row['profile_ms']) != cleanup_expected:
                        raise ProbeError('cleanup_binary_semantics_mismatch')
                    vector = value.get('embedding')
                    if not isinstance(vector, list):
                        raise ProbeError('missing_vector')
                    if expected is None:
                        length = metadata['length']
                        if length not in synthetic_reference:
                            if trial['runtime'] != 'previous':
                                raise ProbeError('previous_synthetic_reference_missing')
                            parity(vector, vector)
                            synthetic_reference[length] = vector
                        expected = synthetic_reference[length]
                    row['parity'] = parity(vector, expected)
                    row['resources'] = snapshot(worker.process.pid)
                    validate_affinity(worker_affinity(worker.process.pid), args.cpus)
                    if not controlled(row['resources']['cgroup']):
                        raise ProbeError('worker_cgroup_changed')
                    if not row['parity']['passed']:
                        raise ProbeError('numerical_parity_failed')
                    row['status'] = 'complete'
                    save()
                if file_sha(binaries[trial['runtime']]) != result['binary_sha256'][trial['runtime']]:
                    raise ProbeError('binary_changed_during_trial')
                trial.update(status='complete', timings_by_input=summarize(trial['rows']),
                             after_all=trial['rows'][-1]['resources'])
            except Exception as error:
                code = str(error) if isinstance(error, ProbeError) else type(error).__name__
                trial.update(status='failed', error_code=code)
                if trial['rows'] and trial['rows'][-1]['status'] == 'started':
                    trial['rows'][-1].update(status='failed', error_code=code)
                raise
            finally:
                if worker is not None:
                    worker.close()
                    trial['worker_exit_code'] = worker.process.returncode
                    trial['worker_reaped'] = worker.process.poll() is not None
                save()
        result['status'] = 'complete'
        exit_code = 0
    except Exception as error:
        result.update(status='failed', error_code=str(error) if isinstance(error, ProbeError) else type(error).__name__)
    finally:
        signal.alarm(0)
        for trial in trials:
            if trial['status'] == 'not_started':
                trial['not_started_reason'] = result.get('error_code', 'earlier_failure')
        save()
    print(json.dumps({'status': result['status'], 'attempted_inferences': result['attempted_inferences'],
                      'error_code': result.get('error_code')}))
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
