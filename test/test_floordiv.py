"""Series floor division (`//`) — pandas parity.

`int // int` stays integer and floors toward -inf; a present zero divisor promotes
the whole result to float (`inf` / `-inf` / `nan`), exactly as pandas; any float
operand is float; NA propagates; `bool // bool` errors (like true division).
"""

import numpy as np
import pandas as pd
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


def test_floordiv_int_floor_parity():
    a, b = _s([5, 7, -5, 8]), _s([2, 2, 2, 3])
    r = a // b
    p = pd.Series([5, 7, -5, 8]) // pd.Series([2, 2, 2, 3])
    assert r.dtype == 'int64' and r.to_list() == p.tolist() == [2, 3, -3, 2]


def test_floordiv_zero_divisor_promotes_to_float():
    r = _s([1, -1, 0]) // _s([0, 0, 0])
    assert r.dtype == 'float64'
    v = r.to_list()
    assert v[0] == float('inf') and v[1] == float('-inf') and np.isnan(v[2])


def test_floordiv_partial_zero_divisor_promotes_whole_column():
    # one present zero divisor -> the entire result is float (pandas behaviour)
    r = _s([1, 4]) // _s([0, 2])
    assert r.dtype == 'float64'
    v = r.to_list()
    assert v[0] == float('inf') and v[1] == 2.0


def test_floordiv_float():
    a, b = _s([5.0, 7.0, -5.0]), _s([2.0, 2.0, 2.0])
    assert (a // b).dtype == 'float64' and (a // b).to_list() == [2.0, 3.0, -3.0]


def test_floordiv_mixed_int_float_is_float():
    assert (_s([5, 7]) // _s([2.0, 2.0])).dtype == 'float64'
    assert (_s([5, 7]) // _s([2.0, 2.0])).to_list() == [2.0, 3.0]


def test_floordiv_scalar_and_reflected():
    assert (_s([5, 7, -5]) // 2).to_list() == [2, 3, -3]
    assert (10 // _s([3, 4, -5])).to_list() == [3, 2, -2]   # __rfloordiv__


def test_floordiv_i32_keeps_i32():
    r = _s([7, 8], np.int32) // _s([2, 3], np.int32)
    assert r.dtype == 'int32' and r.to_list() == [3, 2]


def test_floordiv_na_propagates_keeping_int():
    r = _s([10, None, 30]) // _s([3, 3, None])
    assert r.dtype == 'int64' and r.isna().to_list() == [False, True, True]
    assert r.to_list()[0] == 3


def test_floordiv_bool_bool_errors():
    with pytest.raises(Exception):
        _ = _s([True, False]) // _s([True, True])
