"""Shared test fixtures and helpers (volas port of stock-pandas's test/common.py)."""

from pathlib import Path

import numpy as np
import pandas as pd

from volas import DataFrame

_data_dir = Path(__file__).parent / 'data'

COLUMNS = ['open', 'high', 'low', 'close', 'volume']
simple_list = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def read_tencent_csv(filename: str = 'tencent.csv') -> pd.DataFrame:
    """Read the raw Tencent kline CSV as a plain pandas DataFrame."""
    return pd.read_csv((_data_dir / filename).resolve())


def get_tencent(filename: str = 'tencent.csv') -> DataFrame:
    """Build a volas DataFrame of the numeric OHLCV columns from Tencent data."""
    csv = read_tencent_csv(filename)
    return DataFrame({c: csv[c].to_numpy(dtype=float) for c in COLUMNS})


def create_stock() -> DataFrame:
    """A small in-memory volas DataFrame, mirroring stock-pandas's create_stock()."""
    return DataFrame({
        'open': simple_list,
        'close': [x + 1 for x in simple_list],
        'high': [x + 10 for x in simple_list],
        'low': [x - 1 for x in simple_list],
        'volume': [x * 100 for x in simple_list],
    })


def get_last(series):
    """Last value of a Series-like / numpy array."""
    arr = series.to_numpy() if hasattr(series, 'to_numpy') else np.asarray(series)
    return arr[len(arr) - 1]


def to_fixed(n: float, precision: int = 4) -> str:
    return format(n, f'.{precision}f')
