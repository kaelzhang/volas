"""Multi-library performance comparison for OHLCV technical-indicator
computation, plus the incremental "append one new bar" path.

Candidates (a library that is not importable, or an indicator a candidate cannot
express, is simply skipped — the web report omits it):

  * ``pandas``       — idiomatic per-indicator pandas.
  * ``stock_pandas`` — ``StockDataFrame`` directive engine (Rust backend).
  * ``polars``       — polars rolling / ewm expressions.
  * ``talib``        — TA-Lib (the C library) via its Python wheel.
  * ``duckdb``       — SQL window functions (only the *non-recursive* indicators;
                        ema / macd / rsi / atr cannot be expressed as a plain SQL
                        window, so duckdb is skipped for those).
  * ``volas``        — ``volas.DataFrame`` directive kernel.

Two benchmark groups per indicator:

  * ``calc``   — compute the indicator over the whole series (batch).
  * ``append`` — a new bar arrives → produce the updated indicator. ``volas`` and
                 ``stock_pandas`` refresh their cached column incrementally
                 (``O(lookback)``); the libraries with no indicator cache
                 (pandas / polars / talib / duckdb) must recompute the series
                 (``O(n)``) — that *is* the honest cost of a new bar for them.

Run::

    make benchmark                 # console table (installs .[dev,benchmark])
    make benchmark WEB_REPORT=1     # also (re)generate benchmark-report.html

pandas / stock_pandas live in the ``dev`` extra (the parity tests use them);
polars / talib / duckdb live in the ``benchmark`` extra (only this file uses them).
"""

from pathlib import Path

import pandas as pd
import pytest

from stock_pandas import StockDataFrame
from volas import DataFrame as VolasDataFrame

# Optional comparison libraries — absent ones are skipped, not errors.
try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None
try:
    import talib
except ImportError:  # pragma: no cover
    talib = None
try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

DATA = Path(__file__).parent / 'data' / 'tencent_full.csv'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']

_CSV = pd.read_csv(DATA)
ARR = {c: _CSV[c].to_numpy(dtype=float) for c in COLUMNS}
N = len(_CSV)

INDICATORS = [
    'ma:20', 'ema:12', 'macd', 'macd.signal', 'boll.upper',
    'bbw', 'rsi:14', 'atr:14', 'llv:10', 'hhv:10',
]
CANDIDATES = ['pandas', 'stock_pandas', 'polars', 'talib', 'duckdb', 'volas']

HAVE = {
    'pandas': True,
    'stock_pandas': True,
    'polars': pl is not None,
    'talib': talib is not None,
    'duckdb': duckdb is not None,
    'volas': True,
}


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


# --- TA-Lib (the C library) -------------------------------------------------

def _talib_calc(indicator):
    if talib is None:
        return None

    def run(a):
        c, h, lo = a['close'], a['high'], a['low']
        if indicator == 'ma:20':
            return talib.SMA(c, 20)
        if indicator == 'ema:12':
            return talib.EMA(c, 12)
        if indicator == 'macd':
            return talib.MACD(c, 12, 26, 9)[0]
        if indicator == 'macd.signal':
            return talib.MACD(c, 12, 26, 9)[1]
        if indicator == 'boll.upper':
            return talib.BBANDS(c, 20, 2.0, 2.0)[0]
        if indicator == 'bbw':
            u, m, low = talib.BBANDS(c, 20, 2.0, 2.0)
            return (u - low) / m
        if indicator == 'rsi:14':
            return talib.RSI(c, 14)
        if indicator == 'atr:14':
            return talib.ATR(h, lo, c, 14)
        if indicator == 'llv:10':
            return talib.MIN(lo, 10)
        if indicator == 'hhv:10':
            return talib.MAX(h, 10)
        raise KeyError(indicator)

    return run


# --- DuckDB (SQL window functions; non-recursive indicators only) ----------

# An ema / macd / rsi / atr is a recurrence that a plain SQL window cannot
# express, so duckdb is offered only for the windowed (non-recursive) indicators.
_DUCK_SQL = {
    'ma:20': 'avg(close) OVER (ORDER BY rn ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)',
    'boll.upper': ('avg(close) OVER w + 2.0 * stddev_pop(close) OVER w'),
    'bbw': '4.0 * stddev_pop(close) OVER w / avg(close) OVER w',
    'llv:10': 'min(low) OVER (ORDER BY rn ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)',
    'hhv:10': 'max(high) OVER (ORDER BY rn ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)',
}


def _duck_sql(indicator):
    expr = _DUCK_SQL[indicator]
    window = 'WINDOW w AS (ORDER BY rn ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)'
    tail = f' {window}' if ' OVER w ' in f' {expr} ' else ''
    return f'SELECT {expr} AS v FROM bars{tail} ORDER BY rn'


def _duck_name(indicator):
    """A safe PREPARE-statement name for an indicator key."""
    return 'd_' + ''.join(ch if ch.isalnum() else '_' for ch in indicator)


def _duck_calc(indicator):
    # Run a PREPAREd statement (prepared once in the `states` fixture) so the
    # per-call cost excludes SQL parsing — DuckDB's fairest idiom for a repeated
    # query. Even so its fixed per-query overhead (plan + window operator +
    # materialize back to NumPy) dwarfs a few microseconds of real work on a
    # ~16 KB single column: DuckDB is a large-analytical-query engine being
    # measured well outside its design point, not used "wrong" (verified: the gap
    # vs talib shrinks from ~120x at 2 K rows to ~14x at 1 M as the fixed overhead
    # amortizes; prepared + no redundant sort only trims ~30%).
    if duckdb is None or indicator not in _DUCK_SQL:
        return None
    name = _duck_name(indicator)
    return lambda con, n=name: con.execute(f'EXECUTE {n}').fetchnumpy()


# --- the calc registry: CALC[candidate][indicator] = fn | None -------------

def _calc_registry():
    reg = {
        'pandas': dict(PANDAS_CALC),
        'stock_pandas': {k: (lambda spd, d=k: spd.exec(d, create_column=False)) for k in INDICATORS},
        'polars': {k: _pl_calc(k) for k in INDICATORS} if pl else {},
        'talib': {k: _talib_calc(k) for k in INDICATORS} if talib else {},
        'duckdb': {k: _duck_calc(k) for k in INDICATORS} if duckdb else {},
        'volas': {k: (lambda v, d=k: v.exec(d)) for k in INDICATORS},
    }
    # prune the indicators a candidate cannot express
    return {cand: {k: fn for k, fn in m.items() if fn is not None} for cand, m in reg.items()}


CALC = _calc_registry()


# --- per-candidate state ----------------------------------------------------

@pytest.fixture(scope='module')
def states():
    st = {
        'pandas': _CSV,
        'stock_pandas': StockDataFrame(_CSV.copy()),
        'volas': VolasDataFrame({c: ARR[c] for c in COLUMNS}),
    }
    if pl is not None:
        st['polars'] = pl.DataFrame({c: ARR[c] for c in COLUMNS})
    if talib is not None:
        st['talib'] = ARR
    if duckdb is not None:
        con = duckdb.connect()
        con.register('csv_df', pd.DataFrame({c: ARR[c] for c in COLUMNS}))
        con.execute('CREATE TABLE bars AS SELECT row_number() OVER () AS rn, * FROM csv_df')
        con.unregister('csv_df')
        # Prepare each supported query once so the timed call is a plain EXECUTE.
        for ind in _DUCK_SQL:
            con.execute(f'PREPARE {_duck_name(ind)} AS {_duck_sql(ind)}')
        st['duckdb'] = con
    return st


# --- group 1: batch indicator computation ----------------------------------

@pytest.mark.parametrize('candidate', CANDIDATES)
@pytest.mark.parametrize('indicator', INDICATORS)
def test_calc(benchmark, states, indicator, candidate):
    if not HAVE[candidate]:
        pytest.skip(f'{candidate} not installed')
    fn = CALC[candidate].get(indicator)
    if fn is None:
        pytest.skip(f'{candidate} cannot express {indicator}')
    benchmark(fn, states[candidate])


# --- group 2: append one new bar -> updated indicator ----------------------

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
@pytest.mark.parametrize('indicator', INDICATORS)
def test_append(benchmark, states, indicator, candidate):
    if not HAVE[candidate]:
        pytest.skip(f'{candidate} not installed')
    if CALC[candidate].get(indicator) is None:
        pytest.skip(f'{candidate} cannot express {indicator}')

    if candidate == 'volas':
        run, setup = _volas_append(indicator)
        benchmark.pedantic(run, setup=setup, rounds=50, iterations=1)
    elif candidate == 'stock_pandas':
        run, setup = _stock_pandas_append(indicator)
        benchmark.pedantic(run, setup=setup, rounds=50, iterations=1)
    else:
        # pandas / polars / talib / duckdb have no incremental indicator cache,
        # so a new bar means recomputing the series — the honest O(n) cost.
        fn = CALC[candidate][indicator]
        state = states[candidate]
        benchmark(fn, state)
