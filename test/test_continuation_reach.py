"""Every command's continuation reach (`spec::reads_past_lookback`, the one per-command
fact behind `lookback::reach`, and a recursion anywhere in a command): a tail slice of
any length keeps every cached column and continues it exactly — from its carried state
and the rows before it the slice keeps (its lead), including each recursive node in
a composite."""

import numpy as np
import pytest

import volas

from .harness.inventory import executable_lines

from test.exact import assert_exact, same_exact

rng = np.random.default_rng(7)
N = 100
_c = 100 + np.cumsum(rng.normal(0, 1, N + 3))
COLS = {'open': _c + rng.normal(0, .3, N + 3), 'high': _c + 1 + rng.random(N + 3),
        'low': _c - 1 - rng.random(N + 3), 'close': _c,
        'volume': rng.integers(1, 1000, N + 3).astype(float)}

LINES = executable_lines()


def _same(got, want):
    return same_exact(got, want)


def frame(a, b):
    return volas.DataFrame({k: v[a:b] for k, v in COLS.items()}, index=list(range(a, b)))


@pytest.mark.parametrize('line', LINES, ids=lambda ln: ln.form + ln.series_suffix)
def test_a_tail_slice_of_any_length_continues_exactly(line):
    d = line.form + line.series_suffix
    lb = volas.directive_lookback(d)
    if lb > 60:
        pytest.skip('lookback beyond the fixture')
    base = frame(0, N)
    base[d]
    want = frame(0, N + 2).exec(d, dtype='object')[-2:]  # a bool line may be missing
    for keep in sorted({max(lb - 1, 1), max(lb, 1), lb + 1, lb + 2, N}):
        tail = base.iloc[N - keep:]
        assert tail.is_computed(volas.directive_stringify(d))
        tail.append(frame(N, N + 2))
        got = tail[d].to_numpy(dtype='object')[-2:]
        assert _same(got, want), (d, keep, got, want)


@pytest.mark.parametrize('d', ['kcw:5', 'stoch_momentum:5', 'close+ema:3@open'])
@pytest.mark.parametrize('start', [0, 1, 37, N - 1])
def test_a_recursion_continues_from_the_history_a_slice_keeps(d, start):
    # Every recursion carries continuation state; the slice retains its history boundary.
    base = frame(0, N)
    base[d]
    name = volas.directive_stringify(d)
    sliced = base.iloc[start:]
    assert sliced.is_computed(name)
    sliced.append(frame(N, N + 2))
    assert _same(sliced[d].to_numpy(dtype='object'), frame(0, N + 2).exec(d, dtype='object')[start:])


@pytest.mark.parametrize('d', ['crsi:3,2,5', 'stoch_momentum:5', 'close+crsi:3,2,5'])
def test_recursive_composites_keep_their_states_across_a_slice(d):
    # Each constituent recursion carries its state across a dropped head.
    base = frame(0, N)
    base[d]
    name = volas.directive_stringify(d)
    whole = base.iloc[0:]
    assert whole.is_computed(name)
    whole.append(frame(N, N + 2))
    assert _same(whole[d].to_numpy(dtype='object')[-2:], frame(0, N + 2).exec(d, dtype='object')[-2:])
    tail = base.iloc[1:]
    assert tail.is_computed(name)
    tail.append(frame(N, N + 2))
    assert _same(tail[d].to_numpy(dtype="object")[-2:], frame(0, N + 2).exec(d, dtype="object")[-2:])


@pytest.mark.parametrize('d', ['close+ema:3@open', 'ema:5@(ema:3)', 'close-ma:5+obv'])
def test_a_composite_carries_its_recursions_states_across_a_slice(d):
    # each recursion in it carries its own state (a state chain): a slice past the
    # column's first row keeps it and continues exactly
    base = frame(0, N)
    base[d]
    tail = base.iloc[5:]
    assert tail.is_computed(volas.directive_stringify(d))
    tail.append(frame(N, N + 2))
    assert _same(tail[d].to_numpy(dtype='object')[-2:], frame(0, N + 2).exec(d, dtype='object')[-2:])


def test_a_window_keeps_recursive_composites():
    seed = {k: v[:40] for k, v in COLS.items()}
    crsi = volas.DataFrame(seed, window=5, max_lookback=['crsi:3,2,5'])
    want = volas.DataFrame(seed).exec('crsi:3,2,5')[-5:]
    assert_exact(np.asarray(crsi['crsi:3,2,5']).view(np.uint64), want.view(np.uint64))
    w = volas.DataFrame(seed, window=5, max_lookback=['keltner:5'])
    for i in range(40, N):
        w.append(volas.DataFrame({k: v[i:i + 1] for k, v in COLS.items()}))
    assert w._physical_height < N - 40
    assert w.is_computed('keltner:5')
    want = volas.DataFrame({k: v[:N] for k, v in COLS.items()}).exec('keltner:5')[-5:]
    assert same_exact(w['keltner:5'].to_numpy(), want)


@pytest.mark.parametrize('d', ['ema:3', 'ema:5@(ema:3)', 'close>ema:5', 'close-ma:5+obv'])
def test_a_new_column_on_a_slice_keeps_its_recursions(d):
    # adding an unrelated column writes no input: the carried states stay and the
    # recursion continues exactly
    base = frame(0, N)
    base[d]
    s = base.iloc[5:]
    s['u0'] = np.zeros(len(s))
    s.append(volas.DataFrame({**{k: v[N:N + 1] for k, v in COLS.items()}, 'u0': [0.]}, index=[N]))
    assert _same(s[d].to_numpy(dtype='object')[-1:], frame(0, N + 1).exec(d, dtype='object')[-1:])


@pytest.mark.parametrize('d', ['ema:3', 'ema:5@(ema:3)', 'close>ema:5', 'close-ma:5+obv'])
@pytest.mark.parametrize('read', ['item', 'fulfill', 'get_column'])
def test_an_input_write_on_a_slice_replays_its_recursions(d, read):
    # The write invalidates terminal states; the retained checkpoint and lead replay
    # the original history for both recursive and window columns.
    base = frame(0, N)
    base[d]
    base['ma:3']
    s = base.iloc[5:]
    s.iat[3, 3] = 1.0
    reader = {'item': lambda: s[d], 'fulfill': s.fulfill,
              'get_column': lambda: s.get_column(volas.directive_stringify(d))}[read]
    reader()
    want = {k: v[:N].copy() for k, v in COLS.items()}
    want['close'][8] = 1.0
    oracle = volas.DataFrame(want)
    assert _same(s[d].to_numpy(dtype='object'), oracle.exec(d, dtype='object')[5:])
    got = s.get_column('ma:3').to_numpy()
    assert same_exact(got, oracle.exec('ma:3')[5:])
