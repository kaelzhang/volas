"""Contract — Series.fillna accepts a same-dtype scalar fill, like where/mask
(P3-01). A str column fills NA with a string, a datetime column with a parsed
timestamp; an incompatible fill raises. Numeric fills keep their promotion rules
(an integral fill into an int column stays int; a fractional fill promotes to
float). This closes the fillna leg of the typed-scalar-fill family."""

import numpy as np
import pytest
import volas
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


def test_fillna_str_with_string():
    s = _s(['a', None, 'c'])
    out = s.fillna('z')
    assert out.dtype == 'object' or out.dtype == 'str'
    assert list(out.to_numpy()) == ['a', 'z', 'c']


def test_fillna_datetime_with_timestamp():
    s = _s(['2021-01-01', None, '2021-01-03']).astype('datetime64[ns]')
    out = s.fillna('2021-01-02')
    assert out.dtype == 'datetime64[ns]'
    assert list(out.to_numpy()) == [
        np.datetime64('2021-01-01'), np.datetime64('2021-01-02'), np.datetime64('2021-01-03')
    ]


def test_fillna_str_with_number_raises():
    with pytest.raises(Exception):
        _s(['a', None, 'c']).fillna(5)


# --- numeric fillna unchanged (promotion preserved) --------------------------

def test_fillna_int_with_integral_stays_int():
    s = _s([1, None, 3])
    out = s.fillna(9)
    assert out.dtype == 'int64'
    assert out[0] == 1 and out[1] == 9 and out[2] == 3


def test_fillna_int_with_fractional_promotes_float():
    out = _s([1, None, 3]).fillna(2.5)
    assert out.dtype == 'float64'
    assert list(out.to_numpy()) == [1.0, 2.5, 3.0]


def test_fillna_float_scattered():
    out = _s([0.0, 1.0, float('nan'), 3.0]).fillna(5.0)
    assert list(out.to_numpy()) == [0.0, 1.0, 5.0, 3.0]
