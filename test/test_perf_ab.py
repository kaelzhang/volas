import pytest

from scripts import perf_pairs


def sample(seconds):
    return {'benchmarks': [{'name': 'test_api[construct-volas]',
                           'params': {'indicator': 'construct', 'candidate': 'volas'},
                           'stats': {'median': seconds, 'mean': seconds}}]}


def test_paired_ratios_cancel_drift_and_reject_one_outlier():
    # Ratio of pooled medians is 1.6; the paired median is exactly 1.
    base = [sample(x) for x in (10, 10, 10, 40, 40, 40)]
    head = [sample(x) for x in (10, 10, 1000, 40, 40, 40)]
    b, h = perf_pairs.summarize(base, head)
    assert h['benchmarks'][0]['stats']['median'] / b['benchmarks'][0]['stats']['median'] == 1
    assert h['paired']['items'][0]['median_ratio'] == 1
    assert h['paired']['items'][0]['relative_mad'] == 0


def test_paired_ratios_detect_consistent_slowdown():
    b, h = perf_pairs.summarize([sample(x) for x in (10, 20, 40, 80, 160, 320)],
                              [sample(x * 2) for x in (10, 20, 40, 80, 160, 320)])
    assert h['benchmarks'][0]['stats']['median'] / b['benchmarks'][0]['stats']['median'] == 2


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_invalid_measurements_raise(bad):
    with pytest.raises(ValueError, match='positive finite'):
        perf_pairs.summarize([sample(10)] * 6, [sample(bad)] * 6)


def test_missing_round_item_raises():
    with pytest.raises(ValueError, match='same benchmark items'):
        perf_pairs.summarize([sample(10)] * 6, [sample(10)] * 5 + [{'benchmarks': []}])


def test_unbalanced_rounds_raise():
    with pytest.raises(ValueError, match='even.*at least 6'):
        perf_pairs.summarize([sample(10)] * 5, [sample(10)] * 5)


def test_gate_thresholds_apply_to_paired_ratios(tmp_path, capsys):
    import json
    from scripts import perf_gate

    for factor, verdict in ((1, 0), (1.11, 1), (2, 1)):
        b, h = perf_pairs.summarize([sample(10e-6)] * 6, [sample(10e-6 * factor)] * 6)
        for name, data in (('base', b), ('head', h)):
            (tmp_path / f'{name}.json').write_text(json.dumps(data))
        assert perf_gate.main([str(tmp_path / 'head.json'), '--base',
                               str(tmp_path / 'base.json')]) == verdict
    assert 'slowed by > 60%' in capsys.readouterr().out


def test_gate_keeps_five_microsecond_floor(tmp_path, capsys):
    import json
    from scripts import perf_gate

    b, h = perf_pairs.summarize([sample(4e-6)] * 6, [sample(40e-6)] * 6)
    for name, data in (('base', b), ('head', h)):
        (tmp_path / f'{name}.json').write_text(json.dumps(data))
    assert perf_gate.main([str(tmp_path / 'head.json'), '--base', str(tmp_path / 'base.json')]) == 0
    assert 'no comparable items above the time floor' in capsys.readouterr().out


def test_runner_uses_balanced_order_identical_layout_and_paired_files(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        path = command[-1].split('=', 1)[1]
        data = sample(10 if command[3] == 'base-site' else 20)
        from pathlib import Path
        Path(path).write_text(json.dumps(data))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(perf_pairs.subprocess, 'run', run)
    perf_pairs.run_pairs(sys.executable, tmp_path, {'base': 'base-site', 'head': 'head-site'},
                         tmp_path, 6, ['--benchmark-only'])
    assert [command[3] for command, _ in calls] == ['base-site', 'head-site', 'head-site', 'base-site'] * 3
    assert all(command[:3] == [sys.executable, '-c', perf_pairs.WORKER] for command, _ in calls)
    assert all(kwargs['cwd'] == tmp_path for _, kwargs in calls)
    assert all(kwargs['env']['VOLAS_BENCHMARK_PAIRS'] == '6' for _, kwargs in calls)
    h = json.loads((tmp_path / 'head.json').read_text())
    assert h['paired']['items'][0]['median_ratio'] == 2


def test_runner_propagates_failure_with_log(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    def run(command, **kwargs):
        kwargs['stdout'].write('benchmark exploded\n')
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(perf_pairs.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='benchmark failed with exit 7'):
        perf_pairs.run_pairs(sys.executable, tmp_path, {'base': 'b', 'head': 'h'}, tmp_path, 6, [])
    assert 'benchmark exploded' in capsys.readouterr().err
    assert not (tmp_path / 'head.json').exists()


def test_worker_rejects_import_outside_isolated_site(tmp_path):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, '-c', perf_pairs.WORKER, str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'isolated import failed' in result.stderr


def test_single_item_limit_still_fails_when_geomean_passes(tmp_path, capsys):
    import copy
    import json
    from scripts import perf_gate

    base, head = [], []
    for _ in range(6):
        b, h = {'benchmarks': []}, {'benchmarks': []}
        for index in range(20):
            entry = copy.deepcopy(sample(10e-6)['benchmarks'][0])
            entry['name'] = f'test_api[op{index}-volas]'
            entry['params']['indicator'] = f'op{index}'
            b['benchmarks'].append(entry)
            entry = copy.deepcopy(entry)
            if index == 0:
                entry['stats'] = {'median': 20e-6, 'mean': 20e-6}
            h['benchmarks'].append(entry)
        base.append(b)
        head.append(h)
    b, h = perf_pairs.summarize(base, head)
    for name, data in (('base', b), ('head', h)):
        (tmp_path / f'{name}.json').write_text(json.dumps(data))
    assert perf_gate.main([str(tmp_path / 'head.json'), '--base', str(tmp_path / 'base.json')]) == 1
    output = capsys.readouterr().out
    assert 'geomean HEAD/base = 1.035' in output
    assert 'slowed by > 60%' in output
    assert 'broad slowdown' not in output


def test_main_forwards_benchmark_options(tmp_path, monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(sys, 'argv', ['perf_pairs.py', '--base-site', 'b', '--head-site', 'h',
                                    '--root', str(tmp_path), '--output', str(tmp_path),
                                    '--rounds', '8', '-k', 'test_api and volas', '--benchmark-only'])
    monkeypatch.setattr(perf_pairs, 'run_pairs', lambda *args: calls.append(args))
    perf_pairs.main()
    assert calls == [(sys.executable, tmp_path, {'base': 'b', 'head': 'h'}, tmp_path, 8,
                      ['-k', 'test_api and volas', '--benchmark-only'])]


def test_pairing_preserves_win_count_minimum_metric():
    base, head = [sample(10)] * 6, [sample(20)] * 6
    for side, minimum in ((base, 8), (head, 12)):
        for run in side:
            run['benchmarks'][0]['stats']['min'] = minimum
    b, h = perf_pairs.summarize(base, head)
    assert b['benchmarks'][0]['stats']['min'] == 8
    assert h['benchmarks'][0]['stats']['min'] == 12


def test_unequal_pair_counts_raise():
    with pytest.raises(ValueError, match='equal paired rounds'):
        perf_pairs.summarize([sample(10)] * 6, [sample(10)] * 8)


def test_script_entrypoint_writes_comparison(tmp_path, monkeypatch):
    import json
    import runpy
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.setattr(sys, 'argv', ['perf_pairs.py', '--base-site', 'b', '--head-site', 'h',
                                    '--root', str(tmp_path), '--output', str(tmp_path)])

    def run(command, **kwargs):
        Path(command[-1].split('=', 1)[1]).write_text(json.dumps(sample(10)))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', run)
    runpy.run_path(str(Path(perf_pairs.__file__)), run_name='__main__')
    assert json.loads((tmp_path / 'head.json').read_text())['paired']['rounds'] == 6


def test_item_pairing_uses_names_across_reordered_rounds():
    import copy

    first = sample(10)['benchmarks'][0]
    second = copy.deepcopy(first)
    second['name'] = 'test_api[mask-volas]'
    second['params']['indicator'] = 'mask'
    second['stats'] = {'mean': 40, 'median': 40}
    b = {'benchmarks': [first, second]}
    h = copy.deepcopy(b)
    h['benchmarks'][0]['stats'] = {'mean': 20, 'median': 20}
    h['benchmarks'][1]['stats'] = {'mean': 10, 'median': 10}
    h['benchmarks'].reverse()
    base, head = perf_pairs.summarize([b] * 6, [h] * 6)
    assert [x['stats']['median'] for x in base['benchmarks']] == [10, 40]
    assert [x['stats']['median'] for x in head['benchmarks']] == [20, 10]


@pytest.mark.parametrize('side, status', [('base', 2), ('head', 1)])
def test_main_distinguishes_base_plumbing_from_head_failures(tmp_path, monkeypatch, side, status):
    import sys

    monkeypatch.setattr(sys, 'argv', ['perf_pairs.py', '--base-site', 'b', '--head-site', 'h',
                                    '--root', str(tmp_path), '--output', str(tmp_path)])

    def fail(*args):
        raise perf_pairs.BenchmarkFailure(side, 7)

    monkeypatch.setattr(perf_pairs, 'run_pairs', fail)
    with pytest.raises(SystemExit) as error:
        perf_pairs.main()
    assert error.value.code == status


def test_paired_budget_preserves_total_samples_and_default(tmp_path):
    import os
    import subprocess
    import sys

    for pairs, expected in ((None, 500), ('6', 84), ('10', 50)):
        environment = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / 'numba'))
        environment.pop('VOLAS_BENCHMARK_PAIRS', None)
        if pairs:
            environment['VOLAS_BENCHMARK_PAIRS'] = pairs
        result = subprocess.run([sys.executable, '-c',
                                 'from test.test_benchmark import APPEND_ROUNDS; print(APPEND_ROUNDS)'],
                                env=environment, capture_output=True, text=True, check=True)
        assert int(result.stdout.strip()) == expected
