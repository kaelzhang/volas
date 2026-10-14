import copy
import json

import pytest

from scripts import perf_gate, perf_pairs


def row(section, seconds=10e-6, item='ema:20'):
    return {'name': f'test_{section}[{item}-volas]',
            'params': {'candidate': 'volas', 'indicator': item},
            'stats': {'median': seconds, 'mean': seconds}}


def verdict(tmp_path, base, head):
    for side, data in [('base', base), ('head', head)]:
        (tmp_path / f'{side}.json').write_text(json.dumps(data))
    return perf_gate.main([str(tmp_path / 'head.json'), '--base', str(tmp_path / 'base.json')])


@pytest.mark.parametrize('section', ['coverage', 'coverage_extended', 'coverage_after_append',
                                    'first_cache', 'append', 'windowed_stream', 'calc', 'api', 'api_interop', 'tf_live'])
def test_every_volas_section_detects_slowdown(tmp_path, section):
    base = {'benchmarks': [row(section)]}
    head = {'benchmarks': [row(section, 30e-6)]}
    assert verdict(tmp_path, base, head) == 1


def test_same_directive_retains_each_coverage_section():
    data = {'benchmarks': [row(s) for s in ['coverage', 'coverage_extended', 'coverage_after_append']]}
    assert len(perf_gate._volas_medians(data)) == 3


def test_subfloor_items_count_as_section_sum(tmp_path):
    base = {'benchmarks': [row('append', 1e-6, str(i)) for i in range(3)]}
    head = copy.deepcopy(base)
    for entry in head['benchmarks']:
        entry['stats'] = {'median': 3e-6, 'mean': 3e-6}
    assert verdict(tmp_path, base, head) == 1


def test_duplicate_names_in_different_groups_survive_pairing():
    a, b = row('coverage'), row('coverage')
    a['group'], b['group'] = 'coverage', 'extended'
    data = {'benchmarks': [a, b]}
    changed = copy.deepcopy(data)
    changed['benchmarks'][0]['stats'] = {'median': 20e-6, 'mean': 20e-6}
    changed['benchmarks'][1]['stats'] = {'median': 40e-6, 'mean': 40e-6}
    changed['benchmarks'].reverse()
    base, head = perf_pairs.summarize([data] * 6, [changed] * 6)
    assert [r['stats']['median'] for r in base['benchmarks']] == [10e-6, 10e-6]
    assert [r['stats']['median'] for r in head['benchmarks']] == [20e-6, 40e-6]


def test_cost_bounds_keep_strict_limits_and_reject_noise_outlier(capsys):
    from scripts.perf_costs import cost_verdict

    assert cost_verdict([{'check': .5}] * 5 + [{'check': 20}])
    assert not cost_verdict([{'check': 1.0}] * 6)
    assert 'normalized ratio 1.000 >= 1' in capsys.readouterr().out


@pytest.mark.parametrize('samples', [[], [{}, {}], [{'a': .5}, {'b': .5}], [{'a': float('nan')}]])
def test_cost_bounds_cannot_silently_disappear(samples):
    from scripts.perf_costs import cost_verdict

    with pytest.raises((ValueError, IndexError)):
        cost_verdict(samples)


def test_cost_guard_executes_correctness_without_enforcing_instrumented_times(tmp_path):
    from types import SimpleNamespace
    from test.conftest import cost_guard
    from scripts.perf_costs import cost_verdict

    config = SimpleNamespace(getoption=lambda option: None)
    request = SimpleNamespace(config=config, node=SimpleNamespace(nodeid='cost-test'))
    guard = cost_guard.__wrapped__(request)
    guard(200, 1)  # instrumentation may weight paths differently
    assert not hasattr(config, '_volas_cost_checks')
    config.getoption = lambda option: str(tmp_path / 'cost.json')
    guard(200, 1)
    assert not cost_verdict([config._volas_cost_checks] * 6)


def test_subfloor_sum_is_paired_before_median(tmp_path, capsys):
    base, head = [], []
    # Each row's median is .1x, but each pair's section total is exactly 1x.
    for i in range(6):
        b = {'benchmarks': [row('append', 1e-6, str(j)) for j in range(3)]}
        h = {'benchmarks': [row('append', (2.8 if i % 3 == j else .1) * 1e-6, str(j))
                            for j in range(3)]}
        base.append(b)
        head.append(h)
    b, h = perf_pairs.summarize(base, head)
    assert verdict(tmp_path, b, h) == 0
    assert 'geomean HEAD/base = 1.000' in capsys.readouterr().out


@pytest.mark.parametrize('reverse, order', [('0', ['a', 'b']), ('1', ['b', 'a'])])
def test_cost_pair_balances_order_without_swapping_meaning(monkeypatch, reverse, order):
    from types import SimpleNamespace
    from test.conftest import cost_guard

    monkeypatch.setenv('VOLAS_COST_REVERSE', reverse)
    guard = cost_guard.__wrapped__(SimpleNamespace())
    calls = []
    def first():
        calls.append('a')
        return 2
    def second():
        calls.append('b')
        return 7
    assert guard.pair(first, second) == (2, 7)
    assert calls == order


def test_indicator_probe_does_not_run_unrelated_cost_checks(tmp_path, monkeypatch):
    import subprocess
    import sys

    commands = []
    def run(command, **kwargs):
        commands.append(command)
        path = command[-1].split('=', 1)[1]
        from pathlib import Path
        Path(path).write_text(json.dumps({'benchmarks': [row('coverage')]}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(perf_pairs.subprocess, 'run', run)
    perf_pairs.run_pairs(sys.executable, tmp_path, {'base': 'b', 'head': 'h'}, tmp_path, 6,
                         ['--benchmark-only', '--volas-benchmark-indicator=ema:20'])
    assert len(commands) == 12
    assert all('--volas-cost-json=' not in command[-1] for command in commands)


@pytest.mark.parametrize('pairs, expected', [(None, [3, 5, 20]), ('6', [1, 1, 4]), ('10', [1, 1, 2])])
def test_timing_budget_keeps_default_and_total_samples(monkeypatch, pairs, expected):
    from scripts.perf_costs import timing_rounds

    monkeypatch.delenv('VOLAS_BENCHMARK_PAIRS', raising=False)
    if pairs:
        monkeypatch.setenv('VOLAS_BENCHMARK_PAIRS', pairs)
    actual = [timing_rounds(count) for count in [3, 5, 20]]
    assert actual == expected
    assert all(value * int(pairs or '1') >= count for value, count in zip(actual, [3, 5, 20]))


def _shell_arguments(tmp_path, script):
    import os
    import subprocess
    import sys

    fake = tmp_path / 'python'
    fake.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
args = sys.argv[1:]
if args[:2] == ['-m', 'maturin']:
    output = pathlib.Path(args[args.index('--out') + 1])
    output.mkdir(parents=True)
    (output / 'fake.whl').touch()
elif args and args[0].endswith('perf_pairs.py'):
    pathlib.Path(os.environ['PAIRS_CAPTURE']).write_text(json.dumps(args))
''')
    fake.chmod(0o755)
    site = tmp_path / 'base'
    site.mkdir()
    capture = tmp_path / 'arguments.json'
    environment = dict(os.environ, VOLAS_PYTHON=str(fake), BASE_SITE=str(site),
                       PAIRS_CAPTURE=str(capture), PERF_AB_JSON_DIR='', PERF_AB_KSELECT='')
    result = subprocess.run(['bash', str(script), 'HEAD'], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(capture.read_text())


def test_full_shell_protocol_selects_every_section_and_release_costs(tmp_path):
    from pathlib import Path

    arguments = _shell_arguments(tmp_path, Path(perf_pairs.__file__).with_name('perf_ab.sh'))
    selector = arguments[arguments.index('-k') + 1]
    assert selector == 'volas or test_tf_live or test_first_cache or test_range'
    assert '--volas-cost-checks' in arguments
    assert '--benchmark-group-by=func' in arguments
