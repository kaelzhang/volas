"""volas append tests.

Ported / adapted from stock-pandas's ``test_append.py`` and
``test_commands_after_append.py``. volas ``append`` mutates the frame in place
(amortized O(1), like ``list.append``) and returns it — the live single-bar hot
path; a datetime index label is preserved across append. A ``Row`` is a faithful
1-row frame, so appending one preserves every column's dtype (an integer
``volume`` stays ``int64``).
"""

import time
from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


@pytest.fixture
def stock():
    return volas.read_csv(TENCENT, parse_dates=['time_key'], index_col='time_key')


def test_append_dataframe_in_place(stock):
    head = stock.iloc[:10]
    new = stock.iloc[10:15]
    out = head.append(new)
    assert isinstance(out, DataFrame)
    assert out is head      # append is in place and returns the same frame
    assert len(out) == 15
    assert len(head) == 15  # the original frame is mutated
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


# --- append a scalar bar dict (timestamp under the index-name key) ------------

def _bar_dict(stock, i):
    """The i-th row as a scalar bar dict: every column + the index-name timestamp key."""
    row = stock.iloc[i]
    bar = {c: row[c] for c in stock.columns}
    bar['time_key'] = row.name              # the convention key == the index's name
    return bar


def test_append_dict_bar(stock):
    head = stock.iloc[:10]
    bar = _bar_dict(stock, 10)
    out = head.append(bar)
    assert out is head and len(out) == 11
    assert out.iloc[-1].name == stock.iloc[10].name          # timestamp placed from the key
    assert out['volume'].dtype == 'int64'                    # int column stays int (no f64 demotion)
    assert out['close'].to_numpy()[-1] == stock['close'].to_numpy()[10]


def test_append_dict_missing_index_key_raises(stock):
    bar = _bar_dict(stock, 10)
    del bar['time_key']                                      # drop the timestamp key
    with pytest.raises(ValueError):
        stock.iloc[:10].append(bar)


def test_append_dict_missing_column_raises(stock):
    bar = _bar_dict(stock, 10)
    del bar['close']                                         # a data column is missing
    with pytest.raises(ValueError):
        stock.iloc[:10].append(bar)


def test_append_dict_unknown_key_raises(stock):
    bar = _bar_dict(stock, 10)
    bar['not_a_column'] = 1.0
    with pytest.raises(ValueError):
        stock.iloc[:10].append(bar)


def test_append_dict_skips_and_refreshes_cached_directive(stock):
    # a bar dict carries only raw columns; a cached directive column is auto-skipped
    # (not "missing") and refreshed by fulfill.
    head = stock.iloc[:10]
    _ = head['ma:3']                                 # cache a directive column
    head.append(_bar_dict(stock, 10))                # the bar provides only raw columns
    head.fulfill()
    assert len(head) == 11 and 'ma:3' in head.columns
    close = head['close'].to_numpy()
    assert abs(head['ma:3'].to_numpy()[-1] - close[-3:].mean()) < 1e-9


def test_append_dict_none_is_dtype_preserving_na(stock):
    # a missing scalar becomes a dtype-preserving NA — an int column keeps int64
    head = stock.iloc[:10]
    bar = _bar_dict(stock, 10)
    bar['volume'] = None
    head.append(bar)
    assert head['volume'].dtype == 'int64'           # not promoted to float
    assert head['volume'].to_list()[-1] is volas.NA
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


def test_append_is_in_place_and_returns_self(stock):
    # append mutates the frame in place (like list.append) and returns the same
    # object, so the live loop `df.append(bar)` grows one frame with no copy.
    head = stock.iloc[:10]
    out = head.append(stock.iloc[10:15])
    assert out is head            # returns the same object
    assert len(head) == 15        # the original frame is mutated in place
    assert len(out) == 15


def test_append_per_bar_is_amortized_constant_time():
    # a single-bar append must not scale with frame size — the whole point of the
    # in-place change (the immutable O(n)-copy append made a live loop O(n^2)).
    cols = ['open', 'high', 'low', 'close', 'volume']
    one = volas.DataFrame({c: [1.0] for c in cols})

    def per_bar_us(n):
        df = volas.DataFrame({c: np.arange(n, dtype=float) for c in cols})
        for _ in range(100):                       # warm: absorb capacity growth
            df.append(one)
        t = time.perf_counter()
        for _ in range(1000):
            df.append(one)
        return (time.perf_counter() - t) / 1000 * 1e6

    small = per_bar_us(2_000)
    large = per_bar_us(200_000)
    assert large < small * 5, (small, large)        # O(1)-ish, not ~100x
