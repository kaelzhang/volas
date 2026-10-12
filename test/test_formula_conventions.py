"""Indicator formula conventions against their oracles — the regression suite for the
zero tests, warm-up boundaries, argument orders and evaluation orders of the
TA-Lib-compatible commands (TA-Lib is the oracle, contract #30; each deliberate
divergence is documented in INDICATORS.md / docs/migration-talib.md and pinned here).

Every case runs the full computation and, where a command resumes, the live paths
(a cached column continued by `append`) — both must give the oracle's value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame

talib = pytest.importorskip('talib')

TENCENT_FULL = str((Path(__file__).parent / 'data' / 'tencent_full.csv').resolve())


def _ohlcv(n=300, seed=11, scale=1.0):
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    o = c + rng.normal(0.0, 0.3, n)
    h = np.maximum(o, c) + rng.random(n)
    l = np.minimum(o, c) - rng.random(n)
    v = rng.random(n) * 1e6 + 1.0
    return {'open': o * scale, 'high': h * scale, 'low': l * scale, 'close': c * scale, 'volume': v}


def _frame(data):
    return DataFrame({k: np.asarray(v, dtype=float) for k, v in data.items()})


def _exec(data, directive):
    return np.asarray(_frame(data).exec(directive), dtype=float)


def _appended(data, directive, cut):
    """The column cached on the first `cut` rows, then continued by one-row appends."""
    df = _frame({k: v[:cut] for k, v in data.items()})
    df[directive]
    for i in range(cut, len(data['close'])):
        df.append({k: float(v[i]) for k, v in data.items()})
    return np.asarray(df[directive].to_numpy(), dtype=float)


def _same(got, want, exact=False):
    got, want = np.asarray(got, float), np.asarray(want, float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg='NaN masks differ')
    m = ~np.isnan(want)
    if exact:
        np.testing.assert_array_equal(got[m], want[m])
    else:
        np.testing.assert_allclose(got[m], want[m], rtol=1e-9, atol=1e-12)


# --- RSI / CMO / StochRSI: an exact zero test on avgGain + avgLoss --------------------

def test_rsi_of_a_flat_or_falling_series_is_zero_as_in_talib():
    flat = np.full(40, 5.0)
    falling = (100.0 - np.arange(40.0)) * 1e-11
    for close in (flat, falling):
        _same(_exec({'close': close}, 'rsi:14'), talib.RSI(close, 14), exact=True)
    assert _exec({'close': flat}, 'rsi:14')[20] == 0.0


def test_rsi_keeps_its_scale_free_value_on_tiny_prices():
    # TA-Lib's absolute 1e-14 zero test sends a real ×1e-15 random walk to 0; the exact
    # test keeps the value the same walk has at any other scale (documented divergence).
    c = _ohlcv()['close']
    _same(_exec({'close': c * 1e-10}, 'rsi:14'), talib.RSI(c * 1e-10, 14))
    _same(_exec({'close': c * 1e-15}, 'rsi:14'), _exec({'close': c}, 'rsi:14'))
    _same(_exec({'close': c * 1e-15}, 'cmo:14'), _exec({'close': c}, 'cmo:14'))
    assert np.nanmax(np.abs(talib.RSI(c * 1e-15, 14))) == 0.0  # the oracle's own zeroing


@pytest.mark.parametrize('directive', ['rsi:14', 'cmo:14', 'stochrsi.k', 'stochrsi.d', 'crsi'])
def test_a_flat_tail_continues_on_append_as_a_full_recompute(directive):
    c = np.r_[_ohlcv(150)['close'], np.full(60, 101.25)]
    full = _exec({'close': c}, directive)
    _same(_appended({'close': c}, directive, 140), full, exact=True)


def test_stochrsi_of_a_flat_series_matches_talib():
    flat = np.r_[np.full(40, 5.0), 5.0 + np.arange(20.0)]
    k, d = talib.STOCHRSI(flat, 14, 5, 3, 0)
    m = ~np.isnan(d)  # TA-Lib delays %K to %D's start; volas emits each line at its own
    _same(_exec({'close': flat}, 'stochrsi.k')[m], k[m])
    _same(_exec({'close': flat}, 'stochrsi.d')[m], d[m])


# --- change: an exact zero test on the base -------------------------------------------

def test_change_on_tiny_prices_is_not_dropped():
    _same(_exec({'close': [1e-11, 2e-11, 3e-11]}, 'change:2'), [np.nan, 1.0, 0.5])
    _same(_exec({'close': [0.0, 2.0, 3.0]}, 'change:2'), [np.nan, np.nan, 0.5])  # zero base: NaN


# --- RSV: an exact zero test on the range, as TA-Lib STOCHF -----------------------------

def test_rsv_of_a_tiny_range_matches_stochf():
    d = _ohlcv(60, scale=1e-11)
    fastk, _ = talib.STOCHF(d['high'], d['low'], d['close'], 1, 1, 0)
    _same(_exec(d, 'rsv:1'), fastk)


# --- NATR: a zero close gives 0 at its own row ------------------------------------------

def test_natr_zero_close_is_zero_at_the_current_row():
    h, l, c = np.array([1, 2, 1, 3, 4.0]), np.array([1, 2, 0, 3, 2.0]), np.array([1, 2, 0, 3, 3.0])
    got = _exec({'high': h, 'low': l, 'close': c}, 'natr:1')
    assert got[2] == 0.0 and np.isfinite(got).sum() == 4
    assert np.isnan(got[0])
    d = {'high': h, 'low': l, 'close': c}
    _same(_appended(d, 'natr:1', 2), got, exact=True)


# --- ±DM at period 1: the raw one-bar movement from bar 1 -------------------------------

@pytest.mark.parametrize('name,fn', [('plus_dm', talib.PLUS_DM), ('minus_dm', talib.MINUS_DM)])
def test_dm_period_one_starts_at_bar_one(name, fn):
    d = _ohlcv(80)
    want = fn(d['high'], d['low'], 1)
    _same(_exec(d, f'{name}:1'), want, exact=True)
    assert np.isnan(_exec(d, f'{name}:1')[0])
    assert volas.directive_lookback(f'{name}:1') == 1
    _same(_appended(d, f'{name}:1', 3), want, exact=True)


# --- OBV: a flat bar adds nothing — its volume is never multiplied by 0 ---------------

def test_obv_flat_bar_with_infinite_volume_leaves_the_total_unchanged():
    close, volume = np.array([1, 1, 2, 3.0]), np.array([10, np.inf, 3, 4.0])
    _same(_exec({'close': close, 'volume': volume}, 'obv'), talib.OBV(close, volume), exact=True)
    _same(_exec({'close': close, 'volume': volume}, 'obv'), [10, 10, 13, 17])
    _same(_appended({'close': close, 'volume': volume}, 'obv', 1), [10, 10, 13, 17], exact=True)


# --- AD / ADOSC / CMF / III / WVAD: one range-flow rule, zero flow when H == L ----------

def _limit_bar_data():
    d = _ohlcv(120, seed=5)
    for i in (40, 41, 90):  # limit bars: no range, close above / below open
        d['high'][i] = d['low'][i] = d['close'][i]
    d['open'][41] = d['close'][41] - 1.0
    return d


def _flow(location, h, l, v):
    rng = h - l
    return np.where(rng > 0, location / np.where(rng > 0, rng, 1.0) * v, 0.0)


def test_a_limit_bar_carries_no_money_flow():
    d = _limit_bar_data()
    o, h, l, c, v = (d[k] for k in ('open', 'high', 'low', 'close', 'volume'))
    mfv = _flow((c - l) - (h - c), h, l, v)
    cmf = np.array([mfv[i - 19:i + 1].sum() / v[i - 19:i + 1].sum() if i >= 19 else np.nan for i in range(len(c))])
    _same(_exec(d, 'cmf:20'), cmf)
    w = _flow(c - o, h, l, v)
    wvad = np.array([w[i - 23:i + 1].sum() if i >= 23 else np.nan for i in range(len(c))])
    _same(_exec(d, 'wvad:24'), wvad)
    _same(_exec(d, 'iii'), mfv)
    _same(_exec(d, 'ad'), talib.AD(h, l, c, v))
    _same(_exec(d, 'adosc:3,10'), talib.ADOSC(h, l, c, v, 3, 10))
    for directive in ('cmf:20', 'wvad:24'):
        _same(_appended(d, directive, 30), _exec(d, directive), exact=True)


def test_a_missing_range_input_is_missing_not_zero_flow():
    d = _limit_bar_data()
    d['high'][60] = np.nan
    got = _exec(d, 'iii')
    assert np.isnan(got[60]) and np.isfinite(np.delete(got, 60)).all()
    assert np.isnan(_exec(d, 'cmf:5')[60:65]).all() and np.isfinite(_exec(d, 'cmf:5')[65])
