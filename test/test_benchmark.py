"""Multi-library performance comparison for OHLCV technical-indicator computation.

Three report sections:

  1. ``append`` — a new bar arrives → produce the updated indicator. ``volas`` and
     ``stock_pandas`` refresh their cached column incrementally (``O(lookback)``);
     pandas / polars / talib have no indicator cache, so a new bar means
     recomputing the series (``O(n)`` — the honest cost for them). Every candidate
     is measured through ``pedantic`` with the **same** round count, so the
     ``rounds`` column is comparable (see ``APPEND_ROUNDS``).
  2. ``calc`` — compute the indicator over the whole series (batch), all libraries.
  3. ``coverage`` — every indicator **both volas and TA-Lib implement** (the set the
     parity suite aligns), timed **volas vs TA-Lib only**. An indicator only one of
     them has is omitted.

Candidates: ``pandas`` (idiomatic), ``pandas_ta`` (the ``.ta`` accessor),
``stock_pandas`` (StockDataFrame), ``polars`` (rolling / ewm), ``talib`` (the C
library), ``volas`` (the directive kernel). A candidate that cannot express a
given indicator is dropped from the registry (probed once at build time) and
simply shows no bar for it.

Run::

    make benchmark                 # console table + benchmark-report.html (installs .[dev,benchmark])
    make benchmark INDICATOR=roc:10 # one coverage row, stdout only (not archived)

pandas / stock_pandas / talib live in the ``dev`` extra (the parity tests use them
as oracles); polars and pandas-ta live in the ``benchmark`` extra.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_pandas import StockDataFrame
from volas import DataFrame as VolasDataFrame

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None
try:
    import talib
except ImportError:  # pragma: no cover
    talib = None
try:
    import pandas_ta  # noqa: F401  (registers the `.ta` DataFrame accessor)
    _HAVE_PANDAS_TA = True
except ImportError:  # pragma: no cover
    _HAVE_PANDAS_TA = False

DATA = Path(__file__).parent / 'data' / 'tencent_full.csv'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']

_CSV = pd.read_csv(DATA)
ARR = {c: _CSV[c].to_numpy(dtype=float) for c in COLUMNS}
N = len(_CSV)

INDICATORS = [
    'ma:20', 'ema:12', 'macd', 'macd.signal', 'boll.upper',
    'willr:14', 'rsi:14', 'atr:14', 'llv:10', 'hhv:10',
]
# The append section omits the rolling extrema (llv / hhv): their incremental refresh
# is a trivial running min/max and not an informative cross-library append comparison
# (they remain in the batch `calc` section and in coverage's MIN/MAX-derived family).
APPEND_INDICATORS = [i for i in INDICATORS if i not in ('llv:10', 'hhv:10')]
CANDIDATES = ['pandas', 'pandas_ta', 'stock_pandas', 'polars', 'talib', 'volas']

# A varying per-row period (2..30) for MAVP (TA-Lib's variable-period MA), supplied to
# volas as a `periods` column and to TA-Lib as the periods array.
PERIODS = (np.arange(N, dtype=float) % 29.0) + 2.0

# Extended coverage is opt-in by directive: these families have already shown
# length-sensitive behavior where the 1999-row Tencent fixture can hide fixed-cost
# vs streaming tradeoffs. The default full-coverage row stays one row per
# TA-Lib-backed indicator; these generated lengths become extra report columns.
EXTENDED_COVERAGE_LENGTHS = {
    directive: (100, 250, 20_000)
    for directive in [
        'hhv:10',
        'aroon.up:14',
        'aroon.down:14',
        'aroonosc:14',
        'stoch.d',
        'stochf.d',
        'stochrsi.d',
        'mfi:14',
        'roc:10',
        'mama',
        'ht_dcperiod',
        # Long-data candle candidates that still need TA-Lib comparisons outside
        # the historical Tencent fixture.
        'cdl.3linestrike',
        'cdl.breakaway',
        'cdl.hikkake',
        'cdl.tristar',
    ]
}
EXTENDED_LENGTHS = sorted({n for lengths in EXTENDED_COVERAGE_LENGTHS.values() for n in lengths})

HAVE = {
    'pandas': True,
    'pandas_ta': _HAVE_PANDAS_TA,
    'stock_pandas': True,
    'polars': pl is not None,
    'talib': talib is not None,
    'volas': True,
}


def _generated_ohlcv(n, seed=20260606):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = close + rng.normal(0.0, 0.2, n)
    high = np.maximum(open_, close) + rng.random(n) * 1.5
    low = np.minimum(open_, close) - rng.random(n) * 1.5
    volume = np.abs(rng.normal(1.0e6, 2.0e5, n)) + 1.0
    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'periods': (np.arange(n, dtype=float) % 29.0) + 2.0,
    }

# Every append candidate is measured through `pedantic` with this fixed round count
# (volas / stock_pandas need fresh per-round state, so they cannot use the
# auto-calibrated `benchmark()` the recompute candidates would otherwise get —
# that asymmetry is exactly why the rounds used to be 50 vs ~30k). A uniform count
# keeps the timings comparable and stable.
APPEND_ROUNDS = 500

# Bumped whenever the measurement protocol changes meaning (not when indicators
# are added): leader comparisons across different methodologies are flagged in
# the leader report instead of read as code-driven movement.
#   v2: the append sections measure the STEADY-STATE per-bar cost (the setup
#       performs one warm-up append, so the one-time Vec capacity-doubling
#       memcpy of a fresh frame is no longer charged to every measured round).
METHODOLOGY = "append-steady-state-v2"


# --- pandas (idiomatic, per indicator) -------------------------------------

def _pd_ma(csv):
    return csv['close'].rolling(20).mean()


def _pd_ema(csv):
    return csv['close'].ewm(span=12, adjust=True, min_periods=12).mean()


def _pd_macd(csv):
    c = csv['close']
    return (c.ewm(span=12, adjust=True, min_periods=12).mean()
            - c.ewm(span=26, adjust=True, min_periods=26).mean())


def _pd_macd_signal(csv):
    return _pd_macd(csv).ewm(span=9, adjust=True, min_periods=9).mean()


def _pd_boll_upper(csv):
    c = csv['close']
    return c.rolling(20).mean() + 2.0 * c.rolling(20).std(ddof=0)


def _pd_willr(csv):
    h, lo, c = csv['high'], csv['low'], csv['close']
    hh, ll = h.rolling(14).max(), lo.rolling(14).min()
    return -100.0 * (hh - c) / (hh - ll)


def _pd_rsi(csv):
    c = csv['close']
    delta = c.diff()
    ag = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=True, min_periods=14).mean()
    al = (-delta).clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=True, min_periods=14).mean()
    return 100.0 - 100.0 / (1.0 + ag / al)


def _pd_atr(csv):
    h, lo, c = csv['high'], csv['low'], csv['close']
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def _pd_llv(csv):
    return csv['low'].rolling(10).min()


def _pd_hhv(csv):
    return csv['high'].rolling(10).max()


PANDAS_CALC = {
    'ma:20': _pd_ma, 'ema:12': _pd_ema, 'macd': _pd_macd,
    'macd.signal': _pd_macd_signal, 'boll.upper': _pd_boll_upper, 'willr:14': _pd_willr,
    'rsi:14': _pd_rsi, 'atr:14': _pd_atr, 'llv:10': _pd_llv, 'hhv:10': _pd_hhv,
}


# --- pandas-ta (the `.ta` DataFrame accessor) ------------------------------
# Each call returns a Series (single-output) or a DataFrame (macd / bbands);
# for the multi-output ones the band is selected by column-name prefix so the
# mapping survives pandas-ta's version-dependent column suffixes
# (e.g. `BBU_20_2.0_2.0`). pandas-ta has no rolling-min-of-low / rolling-max-of-
# high study, so llv / hhv are intentionally absent (skipped per candidate).

def _pta_band(result, prefix):
    for col in result.columns:
        if col.startswith(prefix):
            return result[col]
    raise KeyError(prefix)  # pragma: no cover - guards a pandas-ta rename


def _pta_calc(indicator):
    table = {
        'ma:20': lambda d: d.ta.sma(length=20),
        'ema:12': lambda d: d.ta.ema(length=12),
        'macd': lambda d: _pta_band(d.ta.macd(fast=12, slow=26, signal=9), 'MACD_'),
        'macd.signal': lambda d: _pta_band(d.ta.macd(fast=12, slow=26, signal=9), 'MACDs'),
        'boll.upper': lambda d: _pta_band(d.ta.bbands(length=20, std=2), 'BBU'),
        'willr:14': lambda d: d.ta.willr(length=14),
        'rsi:14': lambda d: d.ta.rsi(length=14),
        'atr:14': lambda d: d.ta.atr(length=14),
    }
    return table.get(indicator)


# --- polars (rolling / ewm expressions) ------------------------------------

def _pl_expr(indicator):
    if pl is None:
        return None
    c = pl.col('close')
    fast = c.ewm_mean(span=12, adjust=True)
    slow = c.ewm_mean(span=26, adjust=True)
    macd = fast - slow
    delta = c.diff()
    ag = pl.when(delta > 0).then(delta).otherwise(0.0).ewm_mean(alpha=1.0 / 14, adjust=True)
    al = pl.when(delta < 0).then(-delta).otherwise(0.0).ewm_mean(alpha=1.0 / 14, adjust=True)
    h, lo, pc = pl.col('high'), pl.col('low'), c.shift(1)
    tr = pl.max_horizontal(h - lo, (h - pc).abs(), (lo - pc).abs())
    hh, ll = h.rolling_max(14), lo.rolling_min(14)
    return {
        'ma:20': c.rolling_mean(20),
        'ema:12': c.ewm_mean(span=12, adjust=True),
        'macd': macd,
        'macd.signal': macd.ewm_mean(span=9, adjust=True),
        'boll.upper': c.rolling_mean(20) + 2.0 * c.rolling_std(20, ddof=0),
        'willr:14': -100.0 * (hh - c) / (hh - ll),
        'rsi:14': 100.0 - 100.0 / (1.0 + ag / al),
        'atr:14': tr.rolling_mean(14),
        'llv:10': pl.col('low').rolling_min(10),
        'hhv:10': pl.col('high').rolling_max(10),
    }[indicator]


def _pl_calc(indicator):
    expr = _pl_expr(indicator)
    return lambda pf: pf.select(expr)


# --- TA-Lib (the C library): the core 10 -----------------------------------

def _talib_calc(indicator):
    if talib is None:
        return None

    def run(a):
        c, h, lo = a['close'], a['high'], a['low']
        return {
            'ma:20': lambda: talib.SMA(c, 20),
            'ema:12': lambda: talib.EMA(c, 12),
            'macd': lambda: talib.MACD(c, 12, 26, 9)[0],
            'macd.signal': lambda: talib.MACD(c, 12, 26, 9)[1],
            'boll.upper': lambda: talib.BBANDS(c, 20, 2.0, 2.0)[0],
            'willr:14': lambda: talib.WILLR(h, lo, c, 14),
            'rsi:14': lambda: talib.RSI(c, 14),
            'atr:14': lambda: talib.ATR(h, lo, c, 14),
            'llv:10': lambda: talib.MIN(lo, 10),
            'hhv:10': lambda: talib.MAX(h, 10),
        }[indicator]()

    return run


# --- the calc registry: CALC[candidate][indicator] = fn --------------------
#
# A candidate only keeps an indicator it can ACTUALLY express: the registry
# probes every (candidate, indicator) fn once against a small synthetic frame at
# build time and drops the ones that raise. So a library missing a study
# (stock_pandas has no `willr`; pandas-ta has no rolling extrema) simply shows no
# bar for it — the same per-library gap polars/talib already get from a `None`
# entry — instead of failing the benchmark at run time.

def _probe_state(candidate):
    n = 60
    a = {c: v for c, v in _generated_ohlcv(n).items() if c in COLUMNS}
    if candidate in ('pandas', 'pandas_ta'):
        return pd.DataFrame(a)
    if candidate == 'stock_pandas':
        return StockDataFrame(pd.DataFrame(a))
    if candidate == 'polars':
        return pl.DataFrame(a) if pl else None
    if candidate == 'talib':
        return a
    return VolasDataFrame({**a, 'periods': PERIODS[:n]})  # volas


def _calc_registry():
    reg = {
        'pandas': dict(PANDAS_CALC),
        'pandas_ta': {k: _pta_calc(k) for k in INDICATORS} if _HAVE_PANDAS_TA else {},
        'stock_pandas': {k: (lambda spd, d=k: spd.exec(d, create_column=False)) for k in INDICATORS},
        'polars': {k: _pl_calc(k) for k in INDICATORS} if pl else {},
        'talib': {k: _talib_calc(k) for k in INDICATORS} if talib else {},
        'volas': {k: (lambda v, d=k: v.exec(d)) for k in INDICATORS},
    }
    out = {}
    for cand, m in reg.items():
        probe = _probe_state(cand) if HAVE.get(cand) else None
        kept = {}
        for k, fn in m.items():
            if fn is None:
                continue
            if probe is None:
                kept[k] = fn          # library absent — HAVE gate skips it at run time
                continue
            try:
                fn(probe)
            except Exception:
                continue              # candidate cannot express this indicator
            kept[k] = fn
        out[cand] = kept
    return out


CALC = _calc_registry()


# --- coverage: every indicator BOTH volas and TA-Lib implement -------------
#
# (directive, TA-Lib call) pairs drawn from the TA-Lib parity suite — the
# authoritative volas∩TA-Lib set. Includes the core indicators also charted in the
# append / calc sections, so Full Coverage is the complete set. Timed volas-vs-TA-Lib
# only. A multi-output TA-Lib function indexes the output the directive selects, and
# every band of a multi-band indicator is listed as its own row.

# Pattern-recognition function names, fetched once; the candle directives map to them.
_PATTERN_NAMES = talib.get_function_groups()['Pattern Recognition'] if talib is not None else []


def _coverage_pairs(arr):
    if talib is None:
        return []
    c, h, lo, o, v = (arr['close'], arr['high'], arr['low'], arr['open'], arr['volume'])
    periods = arr['periods']
    pairs = [
        # core indicators — also shown across libraries in the append / calc charts,
        # listed here too so Full Coverage is the complete volas-vs-TA-Lib set. Each
        # multi-band indicator is timed band-by-band (request: every band separately).
        ('ma:20', lambda: talib.MA(c, 20, 0)),
        ('ema:12', lambda: talib.EMA(c, 12)),
        ('rsi:14', lambda: talib.RSI(c, 14)),
        ('atr:14', lambda: talib.ATR(h, lo, c, 14)),
        ('llv:10', lambda: talib.MIN(lo, 10)),
        ('hhv:10', lambda: talib.MAX(h, 10)),
        # MACD bands. volas's line is the clean EMA(fast)-EMA(slow); TA-Lib's own MACD
        # (its internally-inconsistent quirk) is the natural speed reference.
        ('macd', lambda: talib.MACD(c, 12, 26, 9)[0]),
        ('macd.signal', lambda: talib.MACD(c, 12, 26, 9)[1]),
        ('macd.histogram', lambda: talib.MACD(c, 12, 26, 9)[2]),
        # Bollinger bands (upper / middle / lower). bbw (band-width) has no native
        # TA-Lib function, so it is neither a coverage row nor a cross-library
        # calc/append chart indicator (willr took its slot there); its correctness
        # is still checked in the parity suite.
        ('boll.upper', lambda: talib.BBANDS(c, 20, 2.0, 2.0, 0)[0]),
        ('boll.middle', lambda: talib.BBANDS(c, 20, 2.0, 2.0, 0)[1]),
        ('boll.lower', lambda: talib.BBANDS(c, 20, 2.0, 2.0, 0)[2]),
        # price transforms
        ('avgprice', lambda: talib.AVGPRICE(o, h, lo, c)),
        ('medprice', lambda: talib.MEDPRICE(h, lo)),
        ('typprice', lambda: talib.TYPPRICE(h, lo, c)),
        ('wclprice', lambda: talib.WCLPRICE(h, lo, c)),
        # overlap moving averages
        ('wma:30', lambda: talib.WMA(c, 30)),
        ('dema:30', lambda: talib.DEMA(c, 30)),
        ('tema:30', lambda: talib.TEMA(c, 30)),
        ('trima:30', lambda: talib.TRIMA(c, 30)),
        ('t3:5', lambda: talib.T3(c, 5)),
        ('kama:30', lambda: talib.KAMA(c, 30)),
        ('sar', lambda: talib.SAR(h, lo)),
        # linear regression family
        ('linearreg:14', lambda: talib.LINEARREG(c, 14)),
        ('linearreg_slope:14', lambda: talib.LINEARREG_SLOPE(c, 14)),
        ('linearreg_intercept:14', lambda: talib.LINEARREG_INTERCEPT(c, 14)),
        ('linearreg_angle:14', lambda: talib.LINEARREG_ANGLE(c, 14)),
        ('tsf:14', lambda: talib.TSF(c, 14)),
        # volume
        ('obv', lambda: talib.OBV(c, v)),
        ('ad', lambda: talib.AD(h, lo, c, v)),
        ('adosc', lambda: talib.ADOSC(h, lo, c, v)),
        # variance / stddev
        ('var:5', lambda: talib.VAR(c, 5)),
        ('stddev:5', lambda: talib.STDDEV(c, 5)),
        # math operators
        ('sum:30', lambda: talib.SUM(c, 30)),
        ('maxindex:30', lambda: talib.MAXINDEX(c, 30)),
        ('minindex:30', lambda: talib.MININDEX(c, 30)),
        ('minmax.min:30', lambda: talib.MINMAX(c, 30)[0]),
        ('minmax.max:30', lambda: talib.MINMAX(c, 30)[1]),
        # aroon
        ('aroon.up:14', lambda: talib.AROON(h, lo, 14)[1]),
        ('aroon.down:14', lambda: talib.AROON(h, lo, 14)[0]),
        ('aroonosc:14', lambda: talib.AROONOSC(h, lo, 14)),
        # price oscillators
        ('apo', lambda: talib.APO(c, 12, 26, 0)),
        ('ppo', lambda: talib.PPO(c, 12, 26, 0)),
        # stochastics
        ('stoch.k', lambda: talib.STOCH(h, lo, c)[0]),
        ('stoch.d', lambda: talib.STOCH(h, lo, c)[1]),
        ('stochf.k', lambda: talib.STOCHF(h, lo, c)[0]),
        ('stochf.d', lambda: talib.STOCHF(h, lo, c)[1]),
        ('stochrsi.k', lambda: talib.STOCHRSI(c)[0]),
        ('stochrsi.d', lambda: talib.STOCHRSI(c)[1]),
        # correlation / beta
        ('correl:30@high,low', lambda: talib.CORREL(h, lo, 30)),
        ('beta:5@high,low', lambda: talib.BETA(h, lo, 5)),
        # directional movement
        ('plus_dm:14', lambda: talib.PLUS_DM(h, lo, 14)),
        ('minus_dm:14', lambda: talib.MINUS_DM(h, lo, 14)),
        ('plus_di:14', lambda: talib.PLUS_DI(h, lo, c, 14)),
        ('minus_di:14', lambda: talib.MINUS_DI(h, lo, c, 14)),
        ('dx:14', lambda: talib.DX(h, lo, c, 14)),
        ('adx:14', lambda: talib.ADX(h, lo, c, 14)),
        ('adxr:14', lambda: talib.ADXR(h, lo, c, 14)),
        # acceleration bands
        ('accbands.upper:20', lambda: talib.ACCBANDS(h, lo, c, 20)[0]),
        ('accbands:20', lambda: talib.ACCBANDS(h, lo, c, 20)[1]),
        ('accbands.lower:20', lambda: talib.ACCBANDS(h, lo, c, 20)[2]),
        # other oscillators
        ('cci:14', lambda: talib.CCI(h, lo, c, 14)),
        ('mfi:14', lambda: talib.MFI(h, lo, c, v, 14)),
        ('trix:30', lambda: talib.TRIX(c, 30)),
        ('ultosc', lambda: talib.ULTOSC(h, lo, c)),
        ('bop', lambda: talib.BOP(o, h, lo, c)),
        ('cmo:14', lambda: talib.CMO(c, 14)),
        ('natr:14', lambda: talib.NATR(h, lo, c, 14)),
        # range based
        ('midpoint:14', lambda: talib.MIDPOINT(c, 14)),
        ('midprice:14', lambda: talib.MIDPRICE(h, lo, 14)),
        ('willr:14', lambda: talib.WILLR(h, lo, c, 14)),
        # momentum
        ('mom:10', lambda: talib.MOM(c, 10)),
        ('roc:10', lambda: talib.ROC(c, 10)),
        ('rocp:10', lambda: talib.ROCP(c, 10)),
        ('rocr:10', lambda: talib.ROCR(c, 10)),
        ('rocr100:10', lambda: talib.ROCR100(c, 10)),
        # volatility
        ('tr', lambda: talib.TRANGE(h, lo, c)),
        # extended MACD variants (every band timed separately)
        ('macdext', lambda: talib.MACDEXT(c)[0]),
        ('macdext.signal', lambda: talib.MACDEXT(c)[1]),
        ('macdext.histogram', lambda: talib.MACDEXT(c)[2]),
        ('macdfix', lambda: talib.MACDFIX(c)[0]),
        ('macdfix.signal', lambda: talib.MACDFIX(c)[1]),
        ('macdfix.histogram', lambda: talib.MACDFIX(c)[2]),
        # adaptive / extended overlap studies
        ('mama', lambda: talib.MAMA(c)[0]),
        ('mama.fama', lambda: talib.MAMA(c)[1]),
        ('mavp:2,30@close,periods', lambda: talib.MAVP(c, periods, 2, 30, 0)),
        ('sarext', lambda: talib.SAREXT(h, lo)),
        # extended math-operator index pair
        ('minmaxindex.min:30', lambda: talib.MINMAXINDEX(c, 30)[0]),
        ('minmaxindex.max:30', lambda: talib.MINMAXINDEX(c, 30)[1]),
        # Hilbert-transform cycle family
        ('ht_trendline', lambda: talib.HT_TRENDLINE(c)),
        ('ht_dcperiod', lambda: talib.HT_DCPERIOD(c)),
        ('ht_dcphase', lambda: talib.HT_DCPHASE(c)),
        ('ht_phasor.inphase', lambda: talib.HT_PHASOR(c)[0]),
        ('ht_phasor.quadrature', lambda: talib.HT_PHASOR(c)[1]),
        ('ht_sine.sine', lambda: talib.HT_SINE(c)[0]),
        ('ht_sine.leadsine', lambda: talib.HT_SINE(c)[1]),
        ('ht_trendmode', lambda: talib.HT_TRENDMODE(c)),
    ]
    # Candlestick patterns: every TA-Lib CDL* maps to a volas `cdl.<name>` directive
    # (name = the CDL-stripped function name, lower-cased: CDL2CROWS -> cdl.2crows).
    # Auto-generated from TA-Lib's group so the full 61-pattern set stays in sync;
    # penetration patterns use TA-Lib's default. Both compute -100/0/100 over OHLC.
    for fn_name in _PATTERN_NAMES:
        talib_fn = getattr(talib, fn_name)
        pairs.append((
            f'cdl.{fn_name[3:].lower()}',
            lambda fn=talib_fn: fn(o, h, lo, c),
        ))
    return pairs


COVERAGE = dict(_coverage_pairs({**ARR, 'periods': PERIODS}))
COVERAGE_IDS = list(COVERAGE)


def talib_expected(directive, data):
    """The TA-Lib reference array for `directive`, computed on `data` (an OHLCV+`periods`
    dict). The single source of the directive->TA-Lib mapping, shared with the
    mutation-parity TA-Lib oracle (test_mutation_talib): `data` must carry every column
    the directive reads (open/high/low/close/volume, plus `periods` for MAVP)."""
    return np.asarray(dict(_coverage_pairs(data))[directive](), dtype=float)


# --- per-candidate state ----------------------------------------------------

@pytest.fixture(scope='module')
def states():
    st = {
        'pandas': _CSV,
        'stock_pandas': StockDataFrame(_CSV.copy()),
        # `periods` rides alongside OHLCV so coverage can exercise MAVP; every other
        # directive ignores the extra column.
        'volas': VolasDataFrame({**{c: ARR[c] for c in COLUMNS}, 'periods': PERIODS}),
    }
    if _HAVE_PANDAS_TA:
        st['pandas_ta'] = _CSV.copy()   # the `.ta` accessor reads OHLCV by column name
    if pl is not None:
        st['polars'] = pl.DataFrame({c: ARR[c] for c in COLUMNS})
    if talib is not None:
        st['talib'] = ARR
    return st


# --- section 1: append one new bar -> updated indicator --------------------

def _volas_append(indicator):
    """volas: refresh the cached directive's tail incrementally (O(lookback)).

    The setup appends one warm-up bar so the measured append is the STEADY-STATE
    per-bar cost a live session pays. A freshly constructed frame's column
    buffers have no spare capacity, so the very first append triggers the
    one-time amortized Vec doubling (a full-column memcpy, ~3us at n=2000);
    rebuilding the frame every round would charge that one-time event to every
    measured round, which is not what "a new bar arrives" costs in a live system.
    """
    warm_bar = VolasDataFrame({c: ARR[c][-2:-1] for c in COLUMNS})
    bar = VolasDataFrame({c: ARR[c][-1:] for c in COLUMNS})

    def setup():
        d = VolasDataFrame({c: ARR[c][:-2] for c in COLUMNS})
        _ = d[indicator]                       # cache over the first n-2 bars
        d.append(warm_bar)                     # one-time capacity growth +
        d.fulfill()                            # resume-state warm-up
        return (d,), {}

    def run(d):
        d.append(bar)
        d.fulfill()
    return run, setup


def _stock_pandas_append(indicator):
    """stock_pandas: append a bar and read the directive (its live path)."""
    bar = pd.DataFrame({c: ARR[c][-1:] for c in COLUMNS})

    def setup():
        sdf = StockDataFrame(pd.DataFrame({c: ARR[c][:-1] for c in COLUMNS}))
        _ = sdf[indicator]                     # cache over the first n-1 bars
        return (sdf,), {}

    def run(sdf):
        nxt = sdf.append(bar)
        _ = nxt[indicator]
    return run, setup


@pytest.mark.parametrize('candidate', CANDIDATES)
@pytest.mark.parametrize('indicator', APPEND_INDICATORS)
def test_append(benchmark, states, indicator, candidate):
    if not HAVE[candidate]:
        pytest.skip(f'{candidate} not installed')
    if CALC[candidate].get(indicator) is None:
        pytest.skip(f'{candidate} cannot express {indicator}')

    if candidate == 'volas':
        run, setup = _volas_append(indicator)
    elif candidate == 'stock_pandas':
        run, setup = _stock_pandas_append(indicator)
    else:
        # No incremental indicator cache: a new bar means recomputing the series
        # (the honest O(n) cost). Wrapped in pedantic with the same rounds as the
        # incremental candidates so the `rounds` column is comparable.
        fn, state = CALC[candidate][indicator], states[candidate]

        def run(_fn=fn, _state=state):
            _fn(_state)

        def setup():
            return (), {}
    benchmark.pedantic(run, setup=setup, rounds=APPEND_ROUNDS, iterations=1, warmup_rounds=10)


# --- section 2: batch indicator computation --------------------------------

@pytest.mark.parametrize('candidate', CANDIDATES)
@pytest.mark.parametrize('indicator', INDICATORS)
def test_calc(benchmark, states, indicator, candidate):
    if not HAVE[candidate]:
        pytest.skip(f'{candidate} not installed')
    fn = CALC[candidate].get(indicator)
    if fn is None:
        pytest.skip(f'{candidate} cannot express {indicator}')
    benchmark(fn, states[candidate])


# --- section 3: full coverage, volas vs TA-Lib only ------------------------

@pytest.mark.parametrize('candidate', ['talib', 'volas'])
@pytest.mark.parametrize('indicator', COVERAGE_IDS)
def test_coverage(benchmark, states, indicator, candidate):
    if talib is None:
        pytest.skip('talib not installed')
    if candidate == 'talib':
        benchmark(COVERAGE[indicator])
    else:
        v = states['volas']
        try:
            v.exec(indicator)  # confirm volas implements it before timing
        except Exception:
            pytest.skip(f'volas cannot express {indicator}')
        benchmark(lambda d=indicator: v.exec(d))


@pytest.fixture(scope='module')
def length_states():
    return {
        n: {
            'data': (data := _generated_ohlcv(n)),
            'volas': VolasDataFrame(data),
        }
        for n in EXTENDED_LENGTHS
    }


_EXTENDED_COVERAGE_PARAMS = [
    pytest.param(directive, directive, n, id=f'{directive}@n={n}')
    for directive, lengths in EXTENDED_COVERAGE_LENGTHS.items()
    for n in lengths
]


@pytest.mark.parametrize('candidate', ['talib', 'volas'])
@pytest.mark.parametrize('indicator,directive,length', _EXTENDED_COVERAGE_PARAMS)
def test_coverage_extended(benchmark, length_states, indicator, directive, length, candidate):
    """Representative volas-vs-TA-Lib coverage across data lengths.

    The full coverage table keeps the historical Tencent fixture as the primary row;
    this opt-in matrix catches length-sensitive regressions and becomes additional
    columns in the HTML report.
    """
    if talib is None:
        pytest.skip('talib not installed')
    state = length_states[length]
    if candidate == 'talib':
        benchmark(dict(_coverage_pairs(state['data']))[directive])
    else:
        v = state['volas']
        try:
            v.exec(directive)
        except Exception:
            pytest.skip(f'volas cannot express {directive}')
        benchmark(lambda d=directive: v.exec(d))


def _volas_coverage_after_append(indicator):
    """Cache one indicator over n-2 rows, warm one append (see `_volas_append`:
    steady-state, not the one-time capacity-doubling event), then measure the
    append+refresh of the final bar."""
    history = {c: ARR[c][:-2] for c in COLUMNS}
    history['periods'] = PERIODS[:-2]
    warm = {c: ARR[c][-2:-1] for c in COLUMNS}
    warm['periods'] = PERIODS[-2:-1]
    warm_df = VolasDataFrame(warm)
    bar = {c: ARR[c][-1:] for c in COLUMNS}
    bar['periods'] = PERIODS[-1:]
    bar_df = VolasDataFrame(bar)

    def setup():
        d = VolasDataFrame(history)
        _ = d[indicator]
        d.append(warm_df)
        d.fulfill()
        return (d,), {}

    def run(d):
        d.append(bar_df)
        d.fulfill()

    return run, setup


def _talib_coverage_after_append(indicator):
    def run():
        COVERAGE[indicator]()

    def setup():
        return (), {}

    return run, setup


@pytest.mark.parametrize('candidate', ['talib', 'volas'])
@pytest.mark.parametrize('indicator', COVERAGE_IDS)
def test_coverage_after_append(benchmark, indicator, candidate):
    """volas cached append+fulfill vs TA-Lib full recompute after one appended bar."""
    if talib is None:
        pytest.skip('talib not installed')
    if candidate == 'talib':
        run, setup = _talib_coverage_after_append(indicator)
    else:
        try:
            VolasDataFrame({**{c: ARR[c][:-1] for c in COLUMNS}, 'periods': PERIODS[:-1]})[indicator]
        except Exception:
            pytest.skip(f'volas cannot express {indicator}')
        run, setup = _volas_coverage_after_append(indicator)
    benchmark.pedantic(run, setup=setup, rounds=APPEND_ROUNDS, iterations=1, warmup_rounds=10)


# --- section 4: core DataFrame API (the data-handling flows, not indicators) -
#
# The plumbing a live system runs around every indicator call — frame
# construction, column access, row slicing, boolean masking, column assignment,
# copy — timed volas vs pandas / polars. These are the core APIs whose overhead
# the user wants tracked alongside the kernels.

_API_THRESH = float(_CSV['close'].median())


def _api_registry():
    """candidate -> {op: thunk}. `construct` rebuilds each call; the rest run over
    a prebuilt per-candidate frame; `setitem` overwrites a scratch column (so it
    stays idempotent under repeated timing)."""

    def cols():
        return {c: ARR[c] for c in COLUMNS}

    reg = {
        'pandas': (lambda pdf: {
            'construct': lambda: pd.DataFrame(cols()),
            'getcol': lambda: pdf['close'],
            'slice': lambda: pdf[100:1900],
            'mask': lambda: pdf[pdf['close'] > _API_THRESH],
            'setitem': lambda: pdf.__setitem__('scratch', ARR['close']),
            'copy': lambda: pdf.copy(),
        })(pd.DataFrame(cols())),
        'volas': (lambda vdf: {
            'construct': lambda: VolasDataFrame(cols()),
            'getcol': lambda: vdf['close'],
            'slice': lambda: vdf[100:1900],
            'mask': lambda: vdf[vdf['close'] > _API_THRESH],
            'setitem': lambda: vdf.__setitem__('scratch', ARR['close']),
            'copy': lambda: vdf.copy(),
        })(VolasDataFrame(cols())),
    }
    if pl is not None:
        ldf = pl.DataFrame(cols())
        reg['polars'] = {
            'construct': lambda: pl.DataFrame(cols()),
            'getcol': lambda: ldf['close'],
            'slice': lambda: ldf[100:1900],
            'mask': lambda: ldf.filter(pl.col('close') > _API_THRESH),
            # polars frames are immutable; the idiomatic "add a column" is with_columns.
            'setitem': lambda: ldf.with_columns(pl.Series('scratch', ARR['close'])),
            'copy': lambda: ldf.clone(),
        }
    return reg


API_REG = _api_registry()
API_OPS = ['construct', 'getcol', 'slice', 'mask', 'setitem', 'copy']
API_CANDIDATES = ['pandas', 'polars', 'volas']


@pytest.mark.parametrize('candidate', API_CANDIDATES)
# The op is exposed under the `indicator` param name so it shares the report grouping
# and the `--benchmark-group-by=param:indicator` console grouping with the other sections.
@pytest.mark.parametrize('indicator', API_OPS)
def test_api(benchmark, indicator, candidate):
    if candidate == 'polars' and pl is None:
        pytest.skip('polars not installed')
    fn = API_REG.get(candidate, {}).get(indicator)
    if fn is None:
        pytest.skip(f'{candidate} has no {indicator}')
    benchmark(fn)
