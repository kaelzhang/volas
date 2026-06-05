"""Shared test fixtures and helpers (volas port of stock-pandas's test/common.py).

Fully pandas-free: the Tencent kline data is loaded natively with
``volas.read_csv``. (The stock-pandas parity oracle, which does need a raw pandas
frame, builds its own in ``test_volas.py``.)
"""

from pathlib import Path

import numpy as np

from volas import DataFrame, read_csv

_data_dir = Path(__file__).parent / 'data'

COLUMNS = ['open', 'high', 'low', 'close', 'volume']
TIME_KEY = 'time_key'
simple_list = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
names = 'abcdef'


def _tencent_path(filename: str = 'tencent.csv') -> str:
    return str((_data_dir / filename).resolve())


def get_tencent(date_col: bool = True, filename: str = 'tencent.csv') -> DataFrame:
    """A native volas DataFrame of the Tencent kline data (OHLCV columns).

    With ``date_col`` (default) the ``time_key`` column becomes a DatetimeIndex;
    otherwise the frame keeps a default RangeIndex.
    """
    path = _tencent_path(filename)
    df = read_csv(path, parse_dates=[TIME_KEY], index_col=TIME_KEY) if date_col else read_csv(path)
    return df[COLUMNS]


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
