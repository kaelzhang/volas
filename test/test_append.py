"""volas append tests.

Ported / adapted from stock-pandas's ``test_append.py`` and
``test_commands_after_append.py``. volas ``append`` returns a NEW frame (pandas
semantics, not in place); a datetime index label is preserved across append. A
``Row`` is a faithful 1-row frame, so appending one preserves every column's
dtype (an integer ``volume`` stays ``int64``).
"""

from pathlib import Path

import pytest

import volas
from volas import DataFrame

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


@pytest.fixture
def stock():
    return volas.read_csv(TENCENT, parse_dates=['time_key'], index_col='time_key')


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


def test_append_row(stock):
    head = stock.iloc[:10]
    row = stock.iloc[10]                                   # a faithful Row
    out = head.append(row)
    assert len(out) == 11
    assert out.iloc[-1].name == row.name                  # datetime label carried by the Row
    assert out['volume'].dtype == 'int64'                 # int column preserved (Row is not f64-lossy)
    close = stock['close'].to_numpy()
    assert out['close'].to_numpy()[-1] == close[10]
    assert row['close'] == close[10]                      # row[col] scalar access


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
