"""Result lifetime and allocator controls are part of the measurement contract."""

from pathlib import Path
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
    windows = {'wall': False, 'cpu': False}
    drops = []
    calls = []

    def clock(name):
        def tick():
            windows[name] = not windows[name]
            return int(not windows[name])
        return tick

    class Result:
        def __del__(self):
            drops.append(dict(windows))

    def operation():
        assert all(windows.values())
        calls.append(1)
        return Result()

    cap = perf_runtime.Capture()

    def prepare():
        assert not any(windows.values())
        return (), {}

    cap.pedantic(operation, setup=prepare if setup else None, iterations=1)
    monkeypatch.setattr(perf_runtime.time, 'perf_counter_ns', clock('wall'))
    monkeypatch.setattr(perf_runtime.time, 'process_time_ns', clock('cpu'))
    assert perf_runtime.measure(cap, 3) == {'wall': 1, 'cpu': 1}
    assert len(calls) == 3
    assert drops == [{'wall': False, 'cpu': False}] * 3


class PedanticDriver:
    """Use the installed runner as the lifetime oracle, without a pytest fixture."""

    _make_runner = BenchmarkFixture._make_runner
    pedantic = BenchmarkFixture._raw_pedantic
    _disable_gc = False
    disabled = False
    cprofile = None

    def __init__(self, timer):
        self._timer = timer

    def _make_stats(self, iterations):
        assert iterations == 1
        return SimpleNamespace(update=lambda duration: None)


@pytest.mark.parametrize('setup', [False, True])
def test_cpu_live_result_trace_matches_real_pedantic(monkeypatch, setup):
    def trace(cpu):
        events = []
        live = set()
        timed = False
        serial = 0

        def timer():
            nonlocal timed
            timed = not timed
            events.append(('start' if timed else 'end', tuple(sorted(live))))
            return int(not timed)

        class Result:
            def __init__(self, number):
                self.number = number
                live.add(number)

            def __del__(self):
                events.append(('drop', self.number, timed))
                live.remove(self.number)

        def operation(value, *, flag):
            nonlocal serial
            assert (value, flag) == (7, True)
            events.append(('call', tuple(sorted(live))))
            serial += 1
            return Result(serial)

        def prepare():
            assert not timed
            events.append(('setup', tuple(sorted(live))))
            return (7,), {'flag': True}

        options = {'setup': prepare} if setup else {'args': (7,), 'kwargs': {'flag': True}}
        if cpu:
            cap = perf_runtime.Capture()
            cap.pedantic(operation, iterations=1, **options)
            monkeypatch.setattr(perf_runtime.time, 'perf_counter_ns', timer)
            monkeypatch.setattr(perf_runtime.time, 'process_time_ns', lambda: 0)
            perf_runtime.measure(cap, 3)
        else:
            PedanticDriver(timer).pedantic(operation, iterations=1, rounds=3, **options)
        assert not live
        return events

    wall = trace(False)
    assert [event[1] for event in wall if event[0] == 'call'] == [(), (1,), (2,)]
    assert trace(True) == wall


@pytest.mark.parametrize('rounds', [0, 1, 10])
@pytest.mark.parametrize('setup', [False, True])
def test_capture_preserves_declared_warmup_and_prepares_calls(monkeypatch, rounds, setup):
    events = []
    cap = perf_runtime.Capture()

    def operation(value, *, flag):
        events.append(('call', value, flag))

    def prepare():
        events.append(('setup',))
        return (7,), {'flag': True}

    options = {'setup': prepare} if setup else {'args': (7,), 'kwargs': {'flag': True}}
    cap.pedantic(operation, warmup_rounds=rounds, iterations=1, **options)
    assert cap.warmup_rounds == rounds

    def clock():
        pytest.fail('warmup must be outside the measurement clocks')

    monkeypatch.setattr(perf_runtime.time, 'perf_counter_ns', clock)
    monkeypatch.setattr(perf_runtime.time, 'process_time_ns', clock)
    perf_runtime.warmup(cap)
    expected = ([('setup',)] if setup else []) + [('call', 7, True)]
    assert events == expected * rounds
    events.clear()
    monkeypatch.setattr(perf_runtime.time, 'perf_counter_ns', lambda: 0)
    monkeypatch.setattr(perf_runtime.time, 'process_time_ns', lambda: 0)
    perf_runtime.measure(cap, 2)
    assert events == expected * 2


def test_development_distinguishes_wall_and_cpu_protocols():
    text = ' '.join((Path(__file__).resolve().parents[1] / 'DEVELOPMENT.md').read_text().split())
    assert 'Wall protocol:' in text and '500 rounds' in text
    assert 'CPU protocol:' in text and '200 pairs' in text and 'two runs' in text
    assert 'time.process_time_ns' in text and 'warmup_rounds' in text


def test_development_limits_magazine_claim_to_affected_classes():
    text = ' '.join((Path(__file__).resolve().parents[1] / 'DEVELOPMENT.md').read_text().split())
    assert 'affected allocation classes' in text
    assert 'CPU migration' in text and 'local reuse state' in text
    assert 'controlled, warmed creation cost of a selection' in text


@pytest.mark.parametrize('warmup_rounds', [None, 0, 1, 10])
def test_cpu_worker_uses_runtime_warmup_once_per_captured_call(monkeypatch, warmup_rounds):
    import io
    import json
    import runpy
    import sys
    import test

    events = []

    def operation():
        events.append('call')

    def target(benchmark):
        if warmup_rounds is None:
            benchmark(operation)
        else:
            benchmark.pedantic(operation, warmup_rounds=warmup_rounds, iterations=1)

    frame = lambda data: None
    bench = SimpleNamespace(VolasDataFrame=frame, ARR={}, COLUMNS=[], PERIODS=[], test_probe=target)
    monkeypatch.setattr(test, 'test_benchmark', bench, raising=False)
    monkeypatch.setitem(sys.modules, 'test.test_benchmark', bench)
    monkeypatch.setitem(sys.modules, 'volas_rs', SimpleNamespace(__file__='/site/__init__.py'))
    monkeypatch.setitem(sys.modules, 'volas_rs.volas_rs',
                        SimpleNamespace(__file__='/site/native.so', DataFrame=frame))
    monkeypatch.setitem(sys.modules, 'perf_runtime', perf_runtime)
    monkeypatch.setattr(perf_runtime, 'measure', lambda cap, reps: events.append('measure'))
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(sys, 'argv', ['perf_cpu.py', 'worker', '/site'])
    request = json.dumps({'row': {'name': 'test_probe'}, 'reps': 2}) + '\n'
    monkeypatch.setattr(sys, 'stdin', io.StringIO(request * 2))
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(root / 'scripts/perf_cpu.py'), run_name='__main__')
    assert exit_info.value.code is None
    assert events == ['call'] * (5 if warmup_rounds is None else warmup_rounds) + ['measure'] * 2


def test_capture_warmup_matches_real_pedantic_lifetimes():
    def trace(cpu):
        events = []
        live = 0

        class Result:
            def __init__(self):
                nonlocal live
                live += 1

            def __del__(self):
                nonlocal live
                live -= 1
                events.append(('drop', live))

        def operation():
            events.append(('call', live))
            return Result()

        if cpu:
            cap = perf_runtime.Capture()
            cap.pedantic(operation, warmup_rounds=10, iterations=1)
            perf_runtime.warmup(cap)
            perf_runtime.measure(cap, 3)
        else:
            PedanticDriver(lambda: 0).pedantic(operation, warmup_rounds=10, iterations=1, rounds=3)
        assert live == 0
        return events

    assert trace(True) == trace(False)


def test_prepare_uses_declared_arguments_when_setup_returns_none():
    cap = perf_runtime.Capture()
    cap.pedantic(lambda value: value, args=(7,), setup=lambda: None)
    assert perf_runtime.prepare(cap) == ((7,), {})


def test_prepare_rejects_competing_argument_sources():
    cap = perf_runtime.Capture()
    cap.pedantic(lambda value: value, args=(7,), setup=lambda: ((8,), {}))
    with pytest.raises(TypeError, match="Can't use `args` or `kwargs`"):
        perf_runtime.prepare(cap)


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
