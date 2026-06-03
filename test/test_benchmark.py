"""Performance comparison: pandas-direct vs stock-pandas vs volas.

For each indicator we time three implementations on the same Tencent data:

1. ``pandas``       - a direct, idiomatic pandas implementation.
2. ``stock_pandas`` - ``StockDataFrame.exec(directive)`` (Rust backend).
3. ``volas``        - ``volas.DataFrame.exec(directive)`` (Rust kernel).

Run with::

    make benchmark
    # or
    pytest test/test_benchmark.py --benchmark-only --benchmark-group-by=param:spec
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_pandas import StockDataFrame
from volas import DataFrame as VolasDataFrame

DATA = Path(__file__).parent / 'data' / 'tencent_full.csv'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']


@pytest.fixture(scope='module')
def frames():
    csv = pd.read_csv(DATA)
    spd = StockDataFrame(csv.copy())
    volas_df = VolasDataFrame({c: csv[c].to_numpy(dtype=float) for c in COLUMNS})
    return csv, spd, volas_df


# --- pandas-direct implementations -----------------------------------------

def p_ma(csv):
    return csv['close'].rolling(20).mean()


def p_ema(csv):
    return csv['close'].ewm(span=12, adjust=True, min_periods=12).mean()


def p_macd(csv):
    c = csv['close']
    fast = c.ewm(span=12, adjust=True, min_periods=12).mean()
    slow = c.ewm(span=26, adjust=True, min_periods=26).mean()
    return fast - slow


def p_macd_signal(csv):
    macd = p_macd(csv)
    return macd.ewm(span=9, adjust=True, min_periods=9).mean()


def p_boll_upper(csv):
    c = csv['close']
    return c.rolling(20).mean() + 2.0 * c.rolling(20).std(ddof=0)


def p_bbw(csv):
    c = csv['close']
    ma = c.rolling(20).mean()
    std = c.rolling(20).std(ddof=0)
    return 4.0 * std / ma


def p_rsi(csv):
    c = csv['close']
    delta = c.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    ag = gains.ewm(alpha=1.0 / 14, adjust=True, min_periods=14).mean()
    al = losses.ewm(alpha=1.0 / 14, adjust=True, min_periods=14).mean()
    return 100.0 - 100.0 / (1.0 + ag / al)


def p_atr(csv):
    h, l, c = csv['high'], csv['low'], csv['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr.rolling(14).mean()


def p_llv(csv):
    return csv['low'].rolling(10).min()


def p_hhv(csv):
    return csv['high'].rolling(10).max()


# (name, pandas_fn, directive)
SPECS = [
    ('ma:20', p_ma, 'ma:20'),
    ('ema:12', p_ema, 'ema:12'),
    ('macd', p_macd, 'macd'),
    ('macd.signal', p_macd_signal, 'macd.signal'),
    ('boll.upper', p_boll_upper, 'boll.upper'),
    ('bbw', p_bbw, 'bbw'),
    ('rsi:14', p_rsi, 'rsi:14'),
    ('atr:14', p_atr, 'atr:14'),
    ('llv:10', p_llv, 'llv:10'),
    ('hhv:10', p_hhv, 'hhv:10'),
]

IDS = [s[0] for s in SPECS]


@pytest.mark.parametrize('impl', ['pandas', 'stock_pandas', 'volas'])
@pytest.mark.parametrize('spec', SPECS, ids=IDS)
def test_indicator(benchmark, frames, spec, impl):
    csv, spd, volas_df = frames
    _name, pandas_fn, directive = spec

    if impl == 'pandas':
        benchmark(pandas_fn, csv)
    elif impl == 'stock_pandas':
        benchmark(lambda: spd.exec(directive, create_column=False))
    else:
        benchmark(lambda: volas_df.exec(directive))
