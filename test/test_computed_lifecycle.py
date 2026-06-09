"""Computed-column dtype-aware lifecycle (audit EX-16 / EX-11)."""

import numpy as np
import pytest
import volas
from volas import DataFrame


def test_bool_directive_survives_append():        # EX-16 (was a crash)
    df = DataFrame({'open': [1., 2., 3.], 'close': [2., 1., 4.]})
    df['close > open']
    df = df.append(DataFrame({'open': [5.], 'close': [6.]}))
    assert df['close > open'].to_numpy().tolist() == [True, False, True, True]
    assert df['close > open'].dtype == 'bool'     # stays a usable mask


def test_append_missing_int_column_keeps_int_na():     # EX-11 (NA model)
    df = DataFrame({'f': [1., 2.], 'i': np.array([10, 20], dtype=np.int64)})
    df = df.append(DataFrame({'f': [3.]}))         # 'i' missing from the appended frame
    assert df['i'].dtype == 'int64'                # stays int64 — no upcast to float
    assert df['i'].to_list() == [10, 20, volas.NA]
    df.fulfill()
    col = df['i'].to_numpy()                        # NumPy export: int NA -> NaN
    assert col[0] == 10.0 and col[1] == 20.0 and np.isnan(col[2])


def test_append_missing_plain_bool_or_str_pads_na():   # EX-11 (NA model)
    df = DataFrame({'f': [1., 2.], 'flag': [True, False]})
    df = df.append(DataFrame({'f': [3.]}))         # plain bool missing -> bool+NA (was an error)
    assert df['flag'].dtype == 'bool'
    assert df['flag'].to_list() == [True, False, volas.NA]
    df2 = DataFrame({'f': [1., 2.], 's': ['a', 'b']})
    df2 = df2.append(DataFrame({'f': [3.]}))       # str missing -> str+NA (was an error)
    assert df2['s'].dtype == 'str'
    assert df2['s'].to_list() == ['a', 'b', volas.NA]


def test_stale_bulk_read_raises_until_fulfill():   # EX-1
    df = DataFrame({'open': [float(i) for i in range(30)],
                    'close': [float(i) + 1 for i in range(30)]})
    df['ma:5']
    df = df.append(DataFrame({'open': [99.0], 'close': [100.0]}))
    # single-column df[directive] auto-refreshes (always fresh) ...
    _ = df['ma:5']
    # ... but a bulk to_numpy on another still-stale frame raises:
    df2 = DataFrame({'open': [float(i) for i in range(30)],
                     'close': [float(i) + 1 for i in range(30)]})
    df2['ma:5']
    df2 = df2.append(DataFrame({'open': [99.0], 'close': [100.0]}))
    with pytest.raises(ValueError):
        df2.to_numpy()
    with pytest.raises(ValueError):
        df2.iloc[-1]
    df2.fulfill()
    assert df2.to_numpy().shape[0] == 31           # fresh after fulfill
