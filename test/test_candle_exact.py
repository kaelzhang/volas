"""Candle averages follow the exact arithmetic on near-tie bars.

TA-Lib slides each candle average as a plain running total, whose rounding drifts over a
long history: on prices quantised to a 0.1 tick (decimals that are not exact binary
fractions) a doji's `body <= 0.1 · avg(high − low)` test flips on near-tie bars after a
few thousand rows. volas slides the same average through the exact, re-anchored window
sum every moving average uses, so its decision is the exact one. This pins it against
rational arithmetic on the float inputs themselves.
"""

from fractions import Fraction as F

import numpy as np
import pytest

from volas import DataFrame

from test.exact import assert_exact

N = 20_000


def _quantised(seed=1):
    r = np.random.default_rng(seed)
    c = np.round(100 + np.cumsum(r.standard_normal(N)), 1)
    o = c + np.round(r.normal(0, 0.6, N), 1)
    h = np.maximum(o, c) + np.round(r.exponential(0.5, N), 1)
    lo = np.minimum(o, c) - np.round(r.exponential(0.5, N), 1)
    return o, h, lo, c


def test_doji_decides_by_the_exact_average_over_a_long_quantised_history():
    o, h, lo, c = _quantised()
    got = np.asarray(DataFrame(dict(open=o, high=h, low=lo, close=c)).exec("cdl.doji"), float)
    # exact window sums of the float ranges, by prefix sums of rationals
    prefix = [F(0)]
    for hi, low in zip(h, lo):
        prefix.append(prefix[-1] + F(hi) - F(low))
    exact_ties = 0
    for i in range(10, N):
        body = abs(F(c[i]) - F(o[i]))
        threshold = F(1, 10) * (prefix[i] - prefix[i - 10]) / 10
        if body == threshold:
            exact_ties += 1  # the float rounding of the average decides an exact tie
            continue
        assert got[i] == (100.0 if body <= threshold else 0.0), i
    assert exact_ties < 50


def test_talib_running_totals_drift_on_the_same_history():
    talib = pytest.importorskip("talib")
    o, h, lo, c = _quantised()
    got = np.asarray(DataFrame(dict(open=o, high=h, low=lo, close=c)).exec("cdl.doji"), float)
    want = np.asarray(talib.CDLDOJI(o, h, lo, c), float)
    differ = np.flatnonzero(~np.isnan(got) & (got != want))
    # the documented divergence: a handful of near-tie bars late in the history
    assert 0 < len(differ) < 10 and differ.min() > 10_000


@pytest.mark.parametrize("pattern", ["doji", "hammer", "hangingman", "longline", "tristar", "3whitesoldiers",
                                     "stalledpattern", "hikkakemod", "breakaway", "mathold"])
def test_a_live_candle_refresh_is_the_full_computation_bit_for_bit(pattern):
    """The live refresh recomputes the last window from the rows' absolute positions, so
    the anchored sums — and every decision — are the full computation's."""
    o, h, lo, c = (x[:2_000] for x in _quantised(2))
    data = dict(open=o, high=h, low=lo, close=c)
    directive = f"cdl.{pattern}"
    seed = 1_700
    df = DataFrame({k: v[:seed] for k, v in data.items()})
    df[directive]
    for i in range(seed, len(c)):
        df.append(DataFrame({k: v[i:i + 1] for k, v in data.items()}))
    full = np.asarray(DataFrame(data).exec(directive), float)
    assert_exact(np.asarray(df[directive], float), full)


@pytest.mark.parametrize('pattern', ['harami', 'haramicross'])
@pytest.mark.parametrize('bearish', [False, True])
@pytest.mark.parametrize('placement,strength', [('outside', 0.0), ('inside', 100.0), ('touch', 80.0)])
def test_harami_signal_has_positive_zero_when_not_contained(pattern, bearish, placement, strength):
    o, c = np.full(12, 10.0), np.full(12, 11.0)
    o[10], c[10] = (14.0, 10.0) if bearish else (10.0, 14.0)
    endpoints = {'outside': (15.0, 15.1), 'inside': (12.0, 12.1), 'touch': (14.0, 13.9)}
    o[11], c[11] = endpoints[placement]
    if bearish:
        o[11], c[11] = 24.0 - o[11], 24.0 - c[11]
    data = dict(open=o, high=np.maximum(o, c) + 1.0, low=np.minimum(o, c) - 1.0, close=c)
    directive = 'cdl.' + pattern
    # TA-Lib emits integer signal codes: no signal has positive zero in float64.
    want = np.array([strength if bearish or strength == 0 else -strength])
    full = DataFrame(data)
    assert_exact(full.exec(directive)[-1:], want)
    for kwargs in ({}, {'window': 3, 'max_lookback': [directive]}):
        live = DataFrame({k: v[:11] for k, v in data.items()}, **kwargs)
        live[directive]
        live.append(DataFrame({k: v[11:] for k, v in data.items()}))
        assert_exact(live[directive].to_numpy()[-1:], want)
