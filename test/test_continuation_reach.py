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


@pytest.mark.parametrize('d', ['crsi:3,2,5', 'stoch_momentum:5', 'close+ema:3@open'])
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


def test_a_window_keeps_a_stateless_recursion_across_its_compaction():
    # a bounded window drops its older rows by design: its cached columns recompute over
    # the rows it retains (the max_lookback margin), so a recursion stays cached
    w = volas.DataFrame({k: v[:40] for k, v in COLS.items()}, window=5, max_lookback=['stoch_momentum:5'])
    w['stoch_momentum:5']
    for i in range(40, N):
        w.append(volas.DataFrame({k: v[i:i + 1] for k, v in COLS.items()}))
    assert w._physical_height < N - 40
    assert w.is_computed('stoch_momentum:5')
    assert not np.isnan(w['stoch_momentum:5'].to_numpy()[-1])
