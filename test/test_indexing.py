"""volas pandas-compatible indexing tests.

Ported from stock-pandas's ``test_indexing.py`` and pandas's indexing semantics.
This module exercises ``.iloc`` / ``.loc`` / ``.at`` / ``.iat`` / ``.index`` /
row ``.name`` / column-list selection over a DatetimeIndex and a RangeIndex;
the string-index (``set_index`` on a string column, ``.loc['b':]``) counterpart
lives in ``test_string_index.py``.
"""

from pathlib import Path

import pytest

import volas
from volas import DataFrame, Series

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


@pytest.fixture
def stock():
    return volas.read_csv(TENCENT, parse_dates=['time_key'], index_col='time_key')


# --- .iloc (positional) -----------------------------------------------------

def test_iloc_slice_returns_dataframe(stock):
    sub = stock.iloc[1:]
    assert isinstance(sub, DataFrame)
    assert len(sub) == len(stock) - 1


def test_iloc_int_row_label_round_trips(stock):
    row = stock.iloc[2]
    # the row's label resolves back to the same row through .loc
    assert stock.loc[row.name].name == row.name


def test_negative_iloc(stock):
    assert stock.iloc[-1].name == stock.iloc[len(stock) - 1].name


def test_iloc_out_of_range_raises(stock):
    with pytest.raises(IndexError):
        stock.iloc[len(stock)]


# --- .loc (label) -----------------------------------------------------------

def test_loc_slice_from_label(stock):
    # mirrors stock-pandas test_loc: take a row, slice .loc[row.name:]
    row = stock.iloc[1]
    sub = stock.loc[row.name:]
    assert isinstance(sub, DataFrame)
    assert sub.iloc[0].name == row.name
    assert len(sub) == len(stock) - 1


def test_loc_label_slice_is_inclusive(stock):
    lo = stock.iloc[1].name
    hi = stock.iloc[3].name
    sub = stock.loc[lo:hi]
    assert len(sub) == 3  # pandas .loc includes both endpoints


def test_loc_bad_label_raises(stock):
    with pytest.raises((KeyError, ValueError)):
        stock.loc['1900-01-01 00:00:00']


# --- .at / .iat (scalars) ---------------------------------------------------

def test_iat_scalar(stock):
    close = stock['close'].to_numpy()
    j = stock.columns.index('close')
    assert stock.iat[2, j] == close[2]
    assert stock.iat[-1, j] == close[-1]


def test_at_scalar(stock):
    close = stock['close'].to_numpy()
    label = stock.iloc[2].name
    assert stock.at[label, 'close'] == close[2]


# --- .index -----------------------------------------------------------------

def test_index_is_datetime(stock):
    assert str(stock.index.dtype) == 'datetime64[ns]'
    assert len(stock.index) == len(stock)


# --- column-list indexing (mirrors stock-pandas) ----------------------------

def test_indexing_with_column_list(stock):
    sub = stock[['close', 'open']]
    assert isinstance(sub, DataFrame)
    assert sub.columns == ['close', 'open']


# --- Series .iloc / .loc / .name --------------------------------------------

def test_series_iloc(stock):
    s = stock['close']
    close = s.to_numpy()
    assert s.iloc[-1] == close[-1]
    sub = s.iloc[1:3]
    assert isinstance(sub, Series)
    assert len(sub) == 2


def test_series_loc(stock):
    s = stock['close']
    close = s.to_numpy()
    label = stock.iloc[4].name
    assert s.loc[label] == close[4]


def test_series_name(stock):
    assert stock['close'].name == 'close'


# --- range (int) index ------------------------------------------------------

def test_range_index_iloc_and_iat():
    df = DataFrame({'a': [10.0, 20.0, 30.0], 'b': [1.0, 2.0, 3.0]})
    assert df.iloc[0].name == 0
    assert df.iloc[-1].name == 2
    assert df.iat[2, 0] == 30.0
    assert df.iat[-1, 1] == 3.0
