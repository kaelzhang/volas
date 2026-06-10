"""V2 (API contract C4): a directive whose operand resolves to a non-numeric
(str / datetime) column must raise, not silently funnel to all-NaN. The
directive engine is a core selling point — a silent all-NaN feature column is
the most dangerous live-trading failure."""

import numpy as np
import pytest
from volas import DataFrame


def test_directive_over_str_column_raises():
    df = DataFrame({'close': [1.0, 2.0, 3.0, 4.0], 'sym': ['a', 'a', 'b', 'b']})
    for d in ['ma:2@sym', 'sum:2@sym', 'ema:2@sym']:
        with pytest.raises(Exception):
            _ = df[d]


def test_directive_over_datetime_column_raises():
    df = DataFrame({'close': [1.0, 2.0, 3.0],
                    't': np.array(['2021-01-01', '2021-01-02', '2021-01-03'], dtype='datetime64[ns]')})
    with pytest.raises(Exception):
        _ = df['ma:2@t']


def test_directive_over_numeric_column_still_works():
    # a numeric (incl. int / bool) operand is unaffected
    df = DataFrame({'close': [1.0, 2.0, 3.0, 4.0], 'vol': [10, 20, 30, 40]})
    out = df['ma:2@vol']
    assert out.to_list()[1:] == [15.0, 25.0, 35.0]


def test_directive_arithmetic_over_str_raises():
    # a binary directive (close + sym) over a str column must also raise, not
    # silently funnel to NaN (same C4 guard at the operator level)
    df = DataFrame({'close': [1.0, 2.0, 3.0], 'sym': ['a', 'b', 'c']})
    for d in ['close+sym', 'close*sym', '-sym', 'close>sym']:
        with pytest.raises(Exception):
            _ = df[d]
