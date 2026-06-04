"""to_numpy common-dtype + dtype= + float32 export (audit EX-2/10/15)."""

import numpy as np
import pytest
from volas import DataFrame


def test_all_numeric_is_float64_matrix():
    df = DataFrame({'a': [1.0, 2.0], 'b': np.array([10, 20], dtype=np.int64)})
    arr = df.to_numpy()
    assert arr.dtype == np.float64 and arr.shape == (2, 2)


def test_mixed_frame_is_object_array_lossless():
    df = DataFrame({'a': [1.0, 2.0], 's': ['x', 'y']})
    arr = df.to_numpy()
    assert arr.dtype == object
    assert arr[1, 1] == 'y'          # string preserved, not NaN


def test_dtype_float32_export():
    df = DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    arr = df.to_numpy(dtype='float32')
    assert arr.dtype == np.float32


def test_dtype_float_over_string_raises():
    df = DataFrame({'a': [1.0], 's': ['x']})
    with pytest.raises(Exception):
        df.to_numpy(dtype='float64')


def test_series_to_numpy_dtype_and_array_protocol():
    s = DataFrame({'a': [1.0, 2.0, 3.0]})['a']
    assert s.to_numpy(dtype='float32').dtype == np.float32
    assert np.asarray(s, dtype=np.int64).dtype == np.int64   # __array__ honors dtype
