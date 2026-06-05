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

Candidates: ``pandas`` (idiomatic), ``stock_pandas`` (StockDataFrame), ``polars``
(rolling / ewm), ``talib`` (the C library), ``volas`` (the directive kernel).

Run::

    make benchmark                 # console table (installs .[dev,benchmark])
    make benchmark WEB_REPORT=1     # also (re)generate benchmark-report.html

pandas / stock_pandas / talib live in the ``dev`` extra (the parity tests use them
as oracles); polars lives in the ``benchmark`` extra.
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

DATA = Path(__file__).parent / 'data' / 'tencent_full.csv'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']

_CSV = pd.read_csv(DATA)
ARR = {c: _CSV[c].to_numpy(dtype=float) for c in COLUMNS}
N = len(_CSV)

INDICATORS = [
    'ma:20', 'ema:12', 'macd', 'macd.signal', 'boll.upper',
    'bbw', 'rsi:14', 'atr:14', 'llv:10', 'hhv:10',
]
# The append section omits the rolling extrema (llv / hhv): their incremental refresh
# is a trivial running min/max and not an informative cross-library append comparison
# (they remain in the batch `calc` section and in coverage's MIN/MAX-derived family).
APPEND_INDICATORS = [i for i in INDICATORS if i not in ('llv:10', 'hhv:10')]
CANDIDATES = ['pandas', 'stock_pandas', 'polars', 'talib', 'volas']

# A varying per-row period (2..30) for MAVP (TA-Lib's variable-period MA), supplied to
# volas as a `periods` column and to TA-Lib as the periods array.
PERIODS = (np.arange(N, dtype=float) % 29.0) + 2.0

HAVE = {
    'pandas': True,
    'stock_pandas': True,
    'polars': pl is not None,
    'talib': talib is not None,
    'volas': True,
}

# Every append candidate is measured through `pedantic` with this fixed round count
# (volas / stock_pandas need fresh per-round state, so they cannot use the
# auto-calibrated `benchmark()` the recompute candidates would otherwise get —
# that asymmetry is exactly why the rounds used to be 50 vs ~30k). A uniform count
# keeps the timings comparable and stable.
APPEND_ROUNDS = 500


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


def _pd_bbw(csv):
    c = csv['close']
    return 4.0 * c.rolling(20).std(ddof=0) / c.rolling(20).mean()


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
    'macd.signal': _pd_macd_signal, 'boll.upper': _pd_boll_upper, 'bbw': _pd_bbw,
    'rsi:14': _pd_rsi, 'atr:14': _pd_atr, 'llv:10': _pd_llv, 'hhv:10': _pd_hhv,
}


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
    return {
        'ma:20': c.rolling_mean(20),
        'ema:12': c.ewm_mean(span=12, adjust=True),
        'macd': macd,
        'macd.signal': macd.ewm_mean(span=9, adjust=True),
        'boll.upper': c.rolling_mean(20) + 2.0 * c.rolling_std(20, ddof=0),
        'bbw': 4.0 * c.rolling_std(20, ddof=0) / c.rolling_mean(20),
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
            'bbw': lambda: (lambda u, m, low: (u - low) / m)(*talib.BBANDS(c, 20, 2.0, 2.0)),
            'rsi:14': lambda: talib.RSI(c, 14),
            'atr:14': lambda: talib.ATR(h, lo, c, 14),
            'llv:10': lambda: talib.MIN(lo, 10),
            'hhv:10': lambda: talib.MAX(h, 10),
        }[indicator]()

    return run


# --- the calc registry: CALC[candidate][indicator] = fn | None -------------

def _calc_registry():
    reg = {
        'pandas': dict(PANDAS_CALC),
        'stock_pandas': {k: (lambda spd, d=k: spd.exec(d, create_column=False)) for k in INDICATORS},
        'polars': {k: _pl_calc(k) for k in INDICATORS} if pl else {},
        'talib': {k: _talib_calc(k) for k in INDICATORS} if talib else {},
        'volas': {k: (lambda v, d=k: v.exec(d)) for k in INDICATORS},
    }
    return {cand: {k: fn for k, fn in m.items() if fn is not None} for cand, m in reg.items()}


CALC = _calc_registry()


# --- coverage: every indicator BOTH volas and TA-Lib implement -------------
#
# (directive, TA-Lib call) pairs drawn from the TA-Lib parity suite — the
# authoritative volas∩TA-Lib set, beyond the core 10 above. Timed volas-vs-TA-Lib
# only. A multi-output TA-Lib function indexes the output the directive selects.

def _coverage_pairs():
    if talib is None:
        return []
    c, h, lo, o, v = (ARR['close'], ARR['high'], ARR['low'], ARR['open'], ARR['volume'])
    pairs = [
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
        # extended MACD variants
        ('macdext', lambda: talib.MACDEXT(c)[0]),
        ('macdext.signal', lambda: talib.MACDEXT(c)[1]),
        ('macdfix', lambda: talib.MACDFIX(c)[0]),
        ('macdfix.signal', lambda: talib.MACDFIX(c)[1]),
        # adaptive / extended overlap studies
        ('mama', lambda: talib.MAMA(c)[0]),
        ('mama.fama', lambda: talib.MAMA(c)[1]),
        ('mavp@close,periods', lambda: talib.MAVP(c, PERIODS, 2, 30, 0)),
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
    for fn_name in talib.get_function_groups()['Pattern Recognition']:
        talib_fn = getattr(talib, fn_name)
        pairs.append((
            f'cdl.{fn_name[3:].lower()}',
            lambda fn=talib_fn: fn(o, h, lo, c),
        ))
    return pairs


COVERAGE = dict(_coverage_pairs())
COVERAGE_IDS = list(COVERAGE)


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
    if pl is not None:
        st['polars'] = pl.DataFrame({c: ARR[c] for c in COLUMNS})
    if talib is not None:
        st['talib'] = ARR
    return st


# --- section 1: append one new bar -> updated indicator --------------------

def _volas_append(indicator):
    """volas: refresh the cached directive's tail incrementally (O(lookback))."""
    bar = VolasDataFrame({c: ARR[c][-1:] for c in COLUMNS})

    def setup():
        d = VolasDataFrame({c: ARR[c][:-1] for c in COLUMNS})
        _ = d[indicator]                       # cache over the first n-1 bars
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
