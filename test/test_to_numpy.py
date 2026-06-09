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


def test_dataframe_to_numpy_int_bool_na_is_nan():
    # 2-D row-major export must honor the validity bitmap: a missing int/bool cell
    # becomes NaN, not 0.0. Regression — to_row_major_f64 used get_f64, which drops
    # validity, so NA poisoned NumPy / torch pipelines with a real-looking 0.
    df = DataFrame({'i': [1, None, 3], 'b': [True, None, False]})
    arr = df.to_numpy()
    assert arr.dtype == np.float64
    assert arr[0].tolist() == [1.0, 1.0]
    assert np.isnan(arr[1]).all()                       # NA row -> [nan, nan], not [0, 0]
    assert arr[2, 0] == 3.0 and arr[2, 1] == 0.0        # b[2] is False -> a real 0.0, not NA
    # parity with the (already-correct) 1-D Series export
    assert np.isnan(df['i'].to_numpy()[1]) and np.isnan(df['b'].to_numpy()[1])


def test_dataframe_to_numpy_dtype_arg_keeps_na_as_nan():
    df = DataFrame({'i': [1, None, 3]})
    arr = df.to_numpy(dtype='float32')
    assert arr.dtype == np.float32 and np.isnan(arr[1, 0])


def test_row_to_numpy_int_bool_na_is_nan():
    df = DataFrame({'i': [1, None, 3], 'b': [True, None, False]})
    row = df.iloc[1].to_numpy()
    assert np.isnan(row).all()                          # [nan, nan], not [0, 0]
    # a present row is unaffected
    assert df.iloc[0].to_numpy().tolist() == [[1.0, 1.0]]
