"""A contiguous slice carries the cached indicator's *body* (a correct full-history
snapshot of the visible rows). It does NOT enable exact continuation of a stateful
indicator across slice->append: the slice has dropped its head, so a recursive value
(which depends on the whole prefix) cannot be continued without carrying its state.
That used to appear to work only because the incremental refresh was wrong on both the
sliced and non-sliced sides by the same amount (BUG 1, now fixed)."""

import numpy as np
import pytest
import volas
from volas import DataFrame


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + rng.random(n)
    low = close - rng.random(n)
    return {'high': high, 'low': low, 'close': close}


# State-carry now continues the EMA-recursion family (ema / smma / macd.signal) exactly
# across a head-dropping slice, so those flip from xfail to a real pass. The Wilder/KDJ
# recursions (rsi / kdj) are not yet state-carry-converted, so they stay xfail(strict):
# a slice dropped their head and they cannot be continued past it without their state.
_NO_SLICE_CARRY = frozenset({'kdj.j', 'rsi:14'})
_SLICE_CARRY_PARAMS = [
    pytest.param(
        d,
        marks=pytest.mark.xfail(
            reason='Wilder/KDJ recursion not yet state-carry-converted: a slice dropped '
            'its head, so it cannot be continued past it without carrying its state.',
            strict=True,
        ),
    )
    if d in _NO_SLICE_CARRY
    else d
    for d in ['ema:12', 'kdj.j', 'macd.signal', 'rsi:14', 'smma:7']
]


@pytest.mark.parametrize('directive', _SLICE_CARRY_PARAMS)
def test_slice_then_append_matches_nonsliced(directive):
    data = _ohlc(80, seed=1)
    bar = {'high': [data['high'][-1] + 0.5],
           'low': [data['low'][-1] - 0.5],
           'close': [data['close'][-1] + 0.3]}

    # reference: cache on the FULL frame, append a bar, fulfill.
    full = DataFrame({k: v.copy() for k, v in data.items()})
    _ = full[directive]
    full.append(DataFrame(bar))
    full.fulfill()
    ref_tail = full[directive].to_numpy()[-1]

    # sliced: cache on the full frame, slice off the first 40 rows (>= lookback),
    # then append the SAME bar and fulfill.
    sliced = DataFrame({k: v.copy() for k, v in data.items()})
    _ = sliced[directive]
    sliced = sliced.iloc[40:]
    sliced.append(DataFrame(bar))
    sliced.fulfill()
    got_tail = sliced[directive].to_numpy()[-1]

    # exact continuation: the appended row matches the non-sliced computation.
    assert got_tail == pytest.approx(ref_tail, rel=1e-12, nan_ok=True)


def test_slice_carries_correct_body_values():
    data = _ohlc(80, seed=2)
    df = DataFrame({k: v.copy() for k, v in data.items()})
    full_j = df['kdj.j'].to_numpy()
    sl = df.iloc[40:]
    # the carried cached values equal the full-history values for those rows
    np.testing.assert_allclose(sl['kdj.j'].to_numpy(), full_j[40:], rtol=1e-12)


def test_short_slice_drops_continuation_not_silently_wrong():
    data = _ohlc(80, seed=3)
    df = DataFrame({k: v.copy() for k, v in data.items()})
    _ = df['kdj.j']                       # lookback 27
    short = df.iloc[60:]                   # 20 rows < 27 -> not continuable
    body = short['kdj.j'].to_numpy().copy()
    short.append(DataFrame({'high': [1.0], 'low': [1.0], 'close': [1.0]}))
    short.fulfill()
    # the carried body values are unchanged (we did not corrupt them with a
    # wrong-seed re-warmup); the new row is honest (NaN), not silent garbage.
    np.testing.assert_allclose(short['kdj.j'].to_numpy()[:20], body, rtol=1e-12)
    assert np.isnan(short['kdj.j'].to_numpy()[-1])
