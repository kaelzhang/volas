"""volas drop / slice + directive-stability tests.

Adapted from stock-pandas's ``test_manipulate.py``. The stock-pandas suite reads
private column-info internals and a string ``directive_stringify`` column name;
the portable essence — an indicator value is stable when rows are dropped and the
same bar is re-added, and stable across a positional slice — is kept. Float OHLC
columns are used so a single ``Row`` (f64-only) can be appended back.
"""

from pathlib import Path

import volas

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


def ohlc():
    # RangeIndex, float OHLC columns (so drop-by-position and Row append work).
    return volas.read_csv(TENCENT)[['open', 'high', 'low', 'close']]


def test_boll_stable_after_drop_and_readd():
    stock = ohlc()
    last_index = len(stock) - 1
    last_boll = stock['boll'].iloc[-1]

    origin_last = stock.iloc[last_index]            # a Row carrying its label
    dropped = stock.drop([last_index])
    assert len(dropped) == last_index

    restored = dropped.append(origin_last)
    assert restored['boll'].iloc[-1] == last_boll   # same bar back -> same value

    # a *different* last bar changes boll
    changed = dropped.append(
        volas.DataFrame({'open': [30.0], 'high': [30.0], 'low': [30.0], 'close': [30.0]})
    )
    assert changed['boll'].iloc[-1] != last_boll


def test_boll_stable_after_iloc_slice():
    stock = volas.read_csv(TENCENT)
    length = len(stock)
    last_boll = stock['boll'].iloc[-1]

    sliced = stock.iloc[10:]
    assert len(sliced) == length - 10
    # boll[-1] depends only on the last 20 rows, present in both frames
    assert sliced['boll'].iloc[-1] == last_boll


def test_list_of_non_strings_raises():
    # mirrors stock-pandas test_invalid_indexing: df[[1]] is not a column list
    stock = ohlc()
    try:
        stock[[1]]
    except Exception:
        return
    raise AssertionError('indexing by a list of ints should raise')
