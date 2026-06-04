"""Shared test fixtures and helpers (volas port of stock-pandas's test/common.py)."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from volas import DataFrame

_data_dir = Path(__file__).parent / 'data'

COLUMNS = ['open', 'high', 'low', 'close', 'volume']
TIME_KEY = 'time_key'
FORMAT = '%Y-%m-%d %H:%M:%S'
simple_list = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
names = 'abcdef'


def read_tencent_csv(filename: str = 'tencent.csv') -> pd.DataFrame:
    """Read the raw Tencent kline CSV as a plain pandas DataFrame."""
    return pd.read_csv((_data_dir / filename).resolve())


def get_tencent(date_col: bool = True, stock: bool = True, filename: str = 'tencent.csv'):
    """A volas DataFrame of the Tencent data (or the raw pandas frame when stock=False)."""
    csv = read_tencent_csv(filename)
    if not stock:
        return csv
    data = {c: csv[c].to_numpy(dtype=float) for c in COLUMNS}
    if date_col:
        data[TIME_KEY] = csv[TIME_KEY].to_numpy()
        return DataFrame(data, date_col=TIME_KEY)
    return DataFrame(data)


def get_1m_tencent(hour_offset: int = 0) -> pd.DataFrame:
    """Tencent data re-stamped at a 1-minute interval (raw pandas frame)."""
    csv = read_tencent_csv().copy()
    time_array = []
    date = datetime(2020, 1, 1, hour_offset)
    step = timedelta(minutes=1)
    for _ in range(len(csv)):
        time_array.append(date.strftime(FORMAT))
        date += step
    csv[TIME_KEY] = np.array(time_array)
    return csv[[TIME_KEY, *COLUMNS]]


def create_stock() -> DataFrame:
    """A small in-memory volas DataFrame, mirroring stock-pandas's create_stock()."""
    return DataFrame({
        'open': list(simple_list),
        'close': [x + 1 for x in simple_list],
        'high': [x + 10 for x in simple_list],
        'low': [x - 1 for x in simple_list],
        'volume': [x * 100 for x in simple_list],
    })


def get_stock_update() -> DataFrame:
    return DataFrame(dict(open=[8.0], close=[9.0], high=[18.0], low=[7.0], volume=[800.0]))


def get_last(series):
    arr = series.to_numpy() if hasattr(series, 'to_numpy') else np.asarray(series)
    return arr[len(arr) - 1]


def to_fixed(n: float, precision: int = 4) -> str:
    return format(n, f'.{precision}f')
