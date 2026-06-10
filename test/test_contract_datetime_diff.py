"""Contract D4/D5 + C4 — datetime.diff() must not return float64 nanoseconds
(datetime is a logical type, not f64; timedelta64 is the open O3 decision). It
raises like datetime subtraction, as do str.diff() and bool.diff() (bool
subtraction is unsupported). Numeric diff is unchanged."""

import numpy as np
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


def test_datetime_diff_raises():
    with pytest.raises(Exception):
        _s(['2024-01-01', '2024-01-02', '2024-01-03'], 'datetime64[ns]').diff()


def test_str_diff_raises():
    with pytest.raises(Exception):
        _s(['a', 'b', 'c']).diff()


def test_bool_diff_raises():
    # bool - bool subtraction is unsupported (use ^), so bool.diff raises too
    b = _s([1.0, 0.0, 1.0]) > 0.5
    with pytest.raises(Exception):
        b.diff()


def test_numeric_diff_unchanged():
    assert _s([1.0, 3.0, 6.0]).diff().to_list()[1:] == [2.0, 3.0]
    di = _s([1, 3, 6]).diff()
    assert di.dtype == 'int64' and di.to_list()[1:] == [2, 3]
