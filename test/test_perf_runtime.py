"""Result lifetime and allocator controls are part of the measurement contract."""

from types import SimpleNamespace

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from scripts import perf_pairs, perf_runtime


def lifetime():
    state = {'timed': False, 'drops': [], 'calls': 0}

    def timer():
        state['timed'] = not state['timed']
        return int(not state['timed'])

    class Result:
        def __del__(self):
            state['drops'].append(state['timed'])

    def operation():
        state['calls'] += 1
        return Result()

    return state, timer, operation


@pytest.mark.parametrize('setup', [False, True])
def test_cpu_pedantic_releases_each_result_after_both_clocks(monkeypatch, setup):
    state, timer, operation = lifetime()
    cap = perf_runtime.Capture()

    def prepare():
        assert not state['timed']
        return (), {}

    cap.pedantic(operation, setup=prepare if setup else None, iterations=1)
    monkeypatch.setattr(perf_runtime.time, 'perf_counter_ns', timer)
    monkeypatch.setattr(perf_runtime.time, 'process_time_ns', lambda: 0)
    perf_runtime.measure(cap, 3)
    assert state['calls'] == 3
    assert state['drops'] == [False] * 3


@pytest.mark.parametrize('candidate', ['volas', 'pandas', 'polars'])
@pytest.mark.parametrize('operation_name', ['tails', 'middle'])
def test_long_selection_retains_result_until_timer_stops(monkeypatch, candidate, operation_name):
    from test import test_benchmark as bench

    state, timer, operation = lifetime()

    class Driver:
        # Exercise the installed pytest-benchmark runner, including its actual
        # automatic-loop versus single-call result lifetimes.
        _make_runner = BenchmarkFixture._make_runner
        pedantic = BenchmarkFixture._raw_pedantic
        _disable_gc = False
        disabled = False
        cprofile = None
        _timer = staticmethod(timer)

        def _make_stats(self, iterations):
            assert iterations == 1
            return SimpleNamespace(update=lambda duration: None)

        def __call__(self, fn):
            self._make_runner(fn, (), {})(range(1))

    monkeypatch.setattr(bench, 'API_REG', {candidate: {operation_name: operation}})
    monkeypatch.setattr(bench, 'APPEND_ROUNDS', 3)
    monkeypatch.setattr(bench, 'pl', object())
    bench.test_api(Driver(), operation_name, candidate)
    assert state['drops'] and not any(state['drops'])
    assert state['calls'] == 13  # ten warmups, then the fixed three-sample budget


@pytest.mark.parametrize('platform', ['darwin', 'linux', 'win32'])
def test_runtime_controls_are_platform_scoped_and_do_not_mutate_parent(monkeypatch, platform):
    monkeypatch.setattr(perf_runtime.sys, 'platform', platform)
    monkeypatch.setenv('MallocMaxMagazines', '16')
    monkeypatch.setenv('MallocMaxMediumMagazines', '8')
    monkeypatch.setenv('NUMBA_CACHE_DIR', 'private-cache')
    env = perf_runtime.environment()
    expected = ('1', '1') if platform == 'darwin' else ('16', '8')
    assert (env['MallocMaxMagazines'], env['MallocMaxMediumMagazines']) == expected
    assert env['NUMBA_CACHE_DIR'] == 'private-cache'
    assert perf_runtime.os.environ['MallocMaxMagazines'] == '16'
    assert perf_runtime.os.environ['MallocMaxMediumMagazines'] == '8'


def test_wall_workers_control_allocator_before_start(tmp_path, monkeypatch):
    import json
    import subprocess
    from test.test_perf_ab import sample

    monkeypatch.setattr(perf_pairs.sys, 'platform', 'darwin')
    monkeypatch.setenv('MallocMaxMagazines', '16')
    monkeypatch.setenv('MallocMaxMediumMagazines', '8')
    environments = []

    def run(command, **kwargs):
        from pathlib import Path
        environments.append(dict(kwargs['env']))
        Path(command[-1].split('=', 1)[1]).write_text(json.dumps(sample(10)))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(perf_pairs.processes, 'run_child', run)
    perf_pairs.run_pairs('python', tmp_path, {'base': 'b', 'head': 'h'}, tmp_path, 6, ['-k', 'volas'])
    assert len(environments) == 12
    assert all(e['MallocMaxMagazines'] == e['MallocMaxMediumMagazines'] == '1' for e in environments)
