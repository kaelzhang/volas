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
from test import ht_reference as R

from test.exact import assert_exact

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
    got, want = np.asarray(got), np.asarray(want)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg='NaN masks differ')
    m = ~np.isnan(want)
    if exact:
        assert_exact(got, want)
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

def test_obv_skips_a_flat_bar_with_infinite_volume():
    close, volume = np.array([1, 1, 2, 3.0]), np.array([10, np.inf, 3, 4.0])
    keep = np.isfinite(close) & np.isfinite(volume)
    want = np.full(len(close), np.nan)
    want[keep] = talib.OBV(close[keep], volume[keep])
    _same(_exec({'close': close, 'volume': volume}, 'obv'), want, exact=True)
    _same(want, [10.0, np.nan, 13.0, 17.0], exact=True)
    _same(_appended({'close': close, 'volume': volume}, 'obv', 1), want, exact=True)


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


# --- the generic MA: period 1 is the identity for every type ----------------------------

@pytest.mark.parametrize('matype', range(9))
def test_generic_ma_of_period_one_is_the_input(matype):
    d = _ohlcv(60)
    _same(_exec(d, f'ma:1,{matype}'), talib.MA(d['close'], 1, matype), exact=True)
    assert volas.directive_lookback(f'ma:1,{matype}') == 0
    _same(_appended(d, f'ma:1,{matype}', 5), d['close'], exact=True)


def test_dispatchers_smooth_with_the_period_one_identity():
    d = _ohlcv(120)
    h, l, c = d['high'], d['low'], d['close']
    k, dd = talib.STOCH(h, l, c, 5, 1, 7, 3, 0)
    _same(_exec(d, 'stoch.k:5,1,7,3,0')[~np.isnan(k)], k[~np.isnan(k)])
    _same(_exec(d, 'stoch.d:5,1,7,3,0')[~np.isnan(dd)], dd[~np.isnan(dd)])
    # a 1-period KAMA signal is the MACD line itself (TA-Lib: TA_MA(…, 1, KAMA) = input)
    _same(_exec(d, 'macdext.signal:12,1,26,1,1,6'), _exec(d, 'macdext:12,1,26,1'), exact=True)
    assert volas.directive_lookback('stoch.k:5,1,7,3,0') == 4
    _same(_appended(d, 'stochrsi.d:14,5,1,7', 40), _exec(d, 'stochrsi.d:14,5,1,7'), exact=True)


# --- price oscillators: TA-Lib's order (the longer period is the slow average) ----------

def test_macd_with_the_slow_period_first_is_talibs_and_does_not_crash():
    # TA-Lib swaps the periods; its MACD line itself seeds the EMAs off-by-alignment (a
    # documented quirk), so the line is checked against the clean EMA difference.
    d = _ohlcv(120)
    c = d['close']
    _same(_exec(d, 'macd:14,7'), talib.EMA(c, 7) - talib.EMA(c, 14))
    for swapped, ordered in (('macd:14,7', 'macd:7,14'), ('macd.signal:14,7,5', 'macd.signal:7,14,5'),
                             ('macd.histogram:14,7,5', 'macd.histogram:7,14,5')):
        _same(_exec(d, swapped), _exec(d, ordered), exact=True)
        _same(_appended(d, swapped, 30), _exec(d, ordered), exact=True)


@pytest.mark.parametrize('directive,fn', [
    ('apo:10,4,1', lambda c: talib.APO(c, 10, 4, 1)),
    ('ppo:10,4,0', lambda c: talib.PPO(c, 10, 4, 0)),
])
def test_oscillators_with_the_slow_period_first_match_talib(directive, fn):
    d = _ohlcv(120)
    want = fn(d['close'])
    m = ~np.isnan(want)
    _same(_exec(d, directive)[m], want[m])
    _same(_appended(d, directive, 30), _exec(d, directive), exact=True)


def test_macdext_swaps_its_averages_with_their_types():
    d = _ohlcv(120)
    for swapped, ordered in (('macdext:14,1,7,0', 'macdext:7,0,14,1'),
                             ('macdext.signal:14,1,7,0,5,2', 'macdext.signal:7,0,14,1,5,2')):
        _same(_exec(d, swapped), _exec(d, ordered), exact=True)
        _same(_appended(d, swapped, 30), _exec(d, ordered), exact=True)


# --- ULTOSC: one window definition, so the future cannot change the past ----------------

def _ultosc_ref(h, l, c, p):
    bp = np.full(len(c), np.nan)
    tr = np.full(len(c), np.nan)
    bp[1:] = c[1:] - np.minimum(l[1:], c[:-1])
    tr[1:] = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    out = np.full(len(c), np.nan)
    for i in range(max(p), len(c)):
        out[i] = 100 * sum(w * bp[i - q + 1:i + 1].sum() / tr[i - q + 1:i + 1].sum()
                           for w, q in zip((4, 2, 1), p)) / 7
    return out


@pytest.mark.parametrize('periods', [(7, 14, 28), (4, 3, 2), (28, 14, 7), (7, 28, 14)])
def test_ultosc_weights_its_periods_by_position_and_is_prefix_invariant(periods):
    d = _ohlcv(160)
    directive = 'ultosc:%d,%d,%d' % periods
    got = _exec(d, directive)
    _same(got, _ultosc_ref(d['high'], d['low'], d['close'], periods))
    for bad in (np.nan, np.inf, -np.inf):
        tail = {k: np.r_[v, v[-3:]] for k, v in d.items()}
        tail['close'][-2] = bad
        tail['high'][-1] = bad
        _same(_exec(tail, directive)[:160], got, exact=True)


# --- Ichimoku: the leading span waits for the longer of its two midpoints ---------------

def test_senkou_a_cached_before_its_first_value_continues_on_append():
    d = _ohlcv(80)
    assert volas.directive_lookback('ichimoku.senkou_a:9,1,52') == 9
    full = _exec(d, 'ichimoku.senkou_a:9,1,52')
    assert np.isnan(full[:9]).all() and np.isfinite(full[9])
    _same(_appended(d, 'ichimoku.senkou_a:9,1,52', 4), full, exact=True)


# --- the Hilbert family: TA-Lib's C recurrence, unfused, from the first published row ---

# The oracle is TA-Lib's C source transcribed in plain (unfused) arithmetic
# (`test/ht_reference.py`): bit-exact on every platform, flat and step input included.
# The comparison with the installed TA-Lib build lives in test_talib_parity.
HT = [
    ('ht_dcperiod', R.ht_dcperiod),
    ('ht_phasor', lambda c: R.ht_phasor(c)[0]),
    ('ht_phasor.quadrature', lambda c: R.ht_phasor(c)[1]),
    ('ht_dcphase', R.ht_dcphase),
    ('ht_sine', lambda c: R.ht_sine(c)[0]),
    ('ht_sine.leadsine', lambda c: R.ht_sine(c)[1]),
    ('ht_trendline', R.ht_trendline),
    ('ht_trendmode', R.ht_trendmode),
    ('mama', lambda c: R.mama(c)[0]),
    ('mama.fama', lambda c: R.mama(c)[1]),
]


@pytest.mark.parametrize('directive,fn', HT, ids=[d for d, _ in HT])
def test_hilbert_matches_the_c_recurrence_bit_for_bit(directive, fn):
    close = volas.read_csv(TENCENT_FULL)['close'].to_numpy()
    _same(_exec({'close': close}, directive), fn([float(x) for x in close]), exact=True)


@pytest.mark.parametrize('directive,fn', HT, ids=[d for d, _ in HT])
def test_hilbert_on_degenerate_input_matches_the_c_recurrence(directive, fn):
    # On flat / step prices the DC-phase DFT works on rounding noise: only the C source's
    # exact evaluation order reproduces the phase it settles on.
    for close in (np.full(200, 10.0), np.r_[np.full(100, 10.0), np.full(100, 12.0)], np.full(200, 1234.567)):
        _same(_exec({'close': close}, directive), fn([float(x) for x in close]), exact=True)


@pytest.mark.parametrize('directive', [d for d, _ in HT])
def test_hilbert_append_is_bit_identical_to_the_full_recompute(directive):
    close = volas.read_csv(TENCENT_FULL)['close'].to_numpy()[:260]
    _same(_appended({'close': close}, directive, 200), _exec({'close': close}, directive), exact=True)
