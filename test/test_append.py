"""volas append tests.

Ported / adapted from stock-pandas's ``test_append.py`` and
``test_commands_after_append.py``. volas ``append`` returns a NEW frame (pandas
semantics, not in place); a datetime index label is preserved across append.

Note: appending a single ``Row`` currently materialises its cells as ``f64``
(the Row is f64-only — see the DDD review), so a frame with an integer column
(e.g. ``volume``) cannot take a Row directly. The Row-append cases below use the
float OHLC columns; whole-DataFrame append preserves every dtype.
"""

from pathlib import Path

import pytest

import volas
from volas import DataFrame

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


@pytest.fixture
def stock():
    return volas.read_csv(TENCENT, parse_dates=['time_key'], index_col='time_key')


@pytest.fixture
def fstock(stock):
    # float-only OHLC view (so a Row, which is f64, can be appended back)
    return stock[['open', 'high', 'low', 'close']]


def test_append_dataframe_returns_new(stock):
    head = stock.iloc[:10]
    new = stock.iloc[10:15]
    out = head.append(new)
    assert isinstance(out, DataFrame)
    assert len(out) == 15
    assert len(head) == 10  # original is untouched (append is not in place)
    assert out.iloc[-1].name == new.iloc[-1].name  # datetime label preserved


def test_append_preserves_datetime_index_and_dtypes(stock):
    head = stock.iloc[:10]
    out = head.append(stock.iloc[10:12])
    assert str(out.index.dtype) == 'datetime64[ns]'
    assert out['volume'].dtype == 'int64'  # int column survives whole-frame append


def test_append_row(fstock):
    head = fstock.iloc[:10]
    row = fstock.iloc[10]
    out = head.append(row)
    assert len(out) == 11
    assert out.iloc[-1].name == row.name  # datetime label carried by the Row
    close = fstock['close'].to_numpy()
    assert out.iat[-1, out.columns.index('close')] == close[10]


def test_append_invalid_type_raises(stock):
    with pytest.raises(TypeError):
        stock.append(1)
    with pytest.raises(TypeError):
        stock.append('not-a-frame')


def test_commands_stable_after_append(stock):
    # ported from test_commands_after_append: appending future bars must not
    # change an already-computed past indicator value.
    cur = stock.iloc[0:-4]
    parts = [stock.iloc[-4:-3], stock.iloc[-3:-2], stock.iloc[-2:-1], stock.iloc[-1:]]
    index = -1
    j = cur['kdj.j'].iloc[index]
    for s in parts:
        index -= 1
        cur = cur.append(s)
        assert cur['kdj.j'].iloc[index] == j
