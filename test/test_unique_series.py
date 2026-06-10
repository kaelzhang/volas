"""V1 (API contract C1/C2): Series.unique() returns a Series that preserves the
dtype and volas.NA — NOT a numpy array that collapses nullable int/bool to
float64+NaN. Distinct values, one NA slot, in order of first appearance."""

import numpy as np
import volas
from volas import DataFrame

NA = volas.NA


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


def test_unique_returns_series():
    u = _s([3, 1, 3, 1, 2]).unique()
    assert type(u).__name__ == 'Series'
    assert u.to_list() == [3, 1, 2] and u.dtype == 'int64'


def test_unique_nullable_int_preserves_dtype_and_na():
    u = _s([3, None, 1, 3, None]).unique()
    assert u.dtype == 'int64'                      # NOT float64
    assert u.to_list() == [3, NA, 1]               # one NA slot, appearance order
    assert u.isna().to_list() == [False, True, False]


def test_unique_nullable_bool_preserves_dtype():
    u = DataFrame({'a': [True, None, False, True]})['a'].unique()
    assert u.dtype == 'bool'
    assert u.to_list() == [True, NA, False]


def test_unique_str_preserves_dtype():
    u = _s(['b', 'a', 'b', None, 'c']).unique()
    assert u.dtype == 'str'
    assert u.to_list() == ['b', 'a', NA, 'c']


def test_unique_float_keeps_nan_as_na():
    u = _s([2.0, float('nan'), 2.0, 1.0]).unique()
    assert u.dtype == 'float64' and u.to_list()[0] == 2.0
    assert u.isna().to_list() == [False, True, False]
