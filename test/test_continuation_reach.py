"""Every command's continuation reach (`spec::reads_past_lookback`, the one per-command
fact behind `lookback::resume_reach` / `continuation_reach`): a tail slice of any length
either keeps a cached column it can continue exactly or no longer caches it — an
append never produces a number the full history does not."""

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

# Composites without a carried state re-warm on a slice's own rows (the open H19
# defect, pinned in test_harness_defects); they are not a continuation-reach question.
REWARM = {'kcw', 'stoch_momentum'}

LINES = [ln for ln in executable_lines() if ln.head.split('.')[0] not in REWARM]


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
    want = frame(0, N + 2).exec(d)[-2:]
    for keep in sorted({max(lb - 1, 1), max(lb, 1), lb + 1, lb + 2}):
        tail = base.iloc[N - keep:]
        cached = tail.is_computed(volas.directive_stringify(d))
        tail.append(frame(N, N + 2))
        if cached:
            got = tail[d].to_numpy()[-2:]
            assert np.array_equal(got, want, equal_nan=True), (d, keep, got, want)
