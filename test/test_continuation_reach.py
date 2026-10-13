"""Every command's continuation reach (`spec::reads_past_lookback`, the one per-command
fact behind `lookback::reach`, and a recursion anywhere in a command): a tail slice of
any length either keeps a cached column it can continue exactly or no longer caches it
— an append never produces a number the full history does not."""

import numpy as np
import pytest

import volas

from .harness.inventory import executable_lines

rng = np.random.default_rng(7)
N = 100
_c = 100 + np.cumsum(rng.normal(0, 1, N + 3))
COLS = {'open': _c + rng.normal(0, .3, N + 3), 'high': _c + 1 + rng.random(N + 3),
        'low': _c - 1 - rng.random(N + 3), 'close': _c,
        'volume': rng.integers(1, 1000, N + 3).astype(float)}

LINES = executable_lines()


def _same(got, want):
    """Equal cells, a missing cell (volas.NA / NaN) equal to a missing one."""
    def cell(v):
        return None if v is volas.NA or (isinstance(v, float) and v != v) else v
    return [cell(v) for v in got] == [cell(v) for v in want]


def frame(a, b):
    return volas.DataFrame({k: v[a:b] for k, v in COLS.items()}, index=list(range(a, b)))


@pytest.mark.parametrize('line', LINES, ids=lambda ln: ln.form + ln.series_suffix)
def test_a_tail_slice_continues_exactly_or_not_at_all(line):
    d = line.form + line.series_suffix
    lb = volas.directive_lookback(d)
    if lb > 60:
        pytest.skip('lookback beyond the fixture')
    base = frame(0, N)
    base[d]
    want = frame(0, N + 2).exec(d, dtype='object')[-2:]  # a bool line may be missing
    for keep in sorted({max(lb - 1, 1), max(lb, 1), lb + 1, lb + 2, N}):
        tail = base.iloc[N - keep:]
        cached = tail.is_computed(volas.directive_stringify(d))
        tail.append(frame(N, N + 2))
        if cached:
            got = tail[d].to_numpy(dtype='object')[-2:]
            assert _same(got, want), (d, keep, got, want)


@pytest.mark.parametrize('d', ['crsi:3,2,5', 'stoch_momentum:5', 'close+crsi:3,2,5'])
def test_a_recursion_without_a_state_continues_only_from_its_history_start(d):
    # no carried state: a slice from the column's first row recomputes it all (kept,
    # exact); a later start would restart the recursion, so the column is plain data
    base = frame(0, N)
    base[d]
    name = volas.directive_stringify(d)
    whole = base.iloc[0:]
    assert whole.is_computed(name)
    whole.append(frame(N, N + 2))
    assert _same(whole[d].to_numpy(dtype='object')[-2:], frame(0, N + 2).exec(d, dtype='object')[-2:])
    assert not base.iloc[1:].is_computed(name)


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


def test_a_window_refuses_a_recursion_without_a_carried_state():
    # a bounded window drops its older rows: a recursion it keeps exactly carries a state
    # it replays from (a checkpoint); one without (crsi recomputes) cannot be declared —
    # recomputing it over the retained rows would restart its history there
    seed = {k: v[:40] for k, v in COLS.items()}
    with pytest.raises(ValueError, match="cannot keep 'crsi:3,2,5' exact on a windowed frame"):
        volas.DataFrame(seed, window=5, max_lookback=['crsi:3,2,5'])
    w = volas.DataFrame(seed, window=5, max_lookback=['keltner:5'])
    for i in range(40, N):
        w.append(volas.DataFrame({k: v[i:i + 1] for k, v in COLS.items()}))
    assert w._physical_height < N - 40
    assert w.is_computed('keltner:5')
    want = volas.DataFrame({k: v[:N] for k, v in COLS.items()}).exec('keltner:5')[-5:]
    assert np.array_equal(w['keltner:5'].to_numpy(), want)


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
def test_an_input_write_on_a_slice_loses_its_recursions(d, read):
    # the write discards the carried states and the slice lacks the history's start:
    # every refresh of the recursion raises, and a window column beside it stays exact
    base = frame(0, N)
    base[d]
    base['ma:3']
    s = base.iloc[5:]
    s.iat[3, 3] = 1.0
    reader = {'item': lambda: s[d], 'fulfill': s.fulfill,
              'get_column': lambda: s.get_column(volas.directive_stringify(d))}[read]
    with pytest.raises(ValueError, match='cannot recompute'):
        reader()
    want = {k: v[5:N].copy() for k, v in COLS.items()}
    want['close'][3] = 1.0
    got = s.get_column('ma:3').to_numpy()
    assert np.array_equal(got, volas.DataFrame(want).exec('ma:3'), equal_nan=True)
