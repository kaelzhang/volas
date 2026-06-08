"""volas frame operations: rename / astype.

Adapted from stock-pandas's ``test_basic.py`` (``test_astype`` and the
``rename`` usage in ``test_get_column``). Both return a NEW frame (the original
is untouched).
"""

import numpy as np

from volas import DataFrame


def test_rename_columns():
    df = DataFrame({'open': [2.0, 3.0], 'close': [3.0, 4.0]})
    out = df.rename(columns={'open': 'Open', 'close': 'Close'})
    assert out.columns == ['Open', 'Close']
    assert df.columns == ['open', 'close']  # original untouched
    np.testing.assert_array_equal(out['Open'].to_numpy(), [2.0, 3.0])


def test_rename_partial():
    df = DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})
    out = df.rename(columns={'b': 'B'})
    assert out.columns == ['a', 'B', 'c']


def test_astype_int_to_float():
    df = DataFrame({'a': np.array([1, 2, 3], dtype=np.int64), 'b': [1.5, 2.5, 3.5]})
    assert df['a'].dtype == 'int64'
    out = df.astype({'a': 'float'})
    assert out['a'].dtype == 'float64'
    assert df['a'].dtype == 'int64'  # original untouched
    np.testing.assert_array_equal(out['a'].to_numpy(), [1.0, 2.0, 3.0])


def test_astype_float_to_int():
    df = DataFrame({'a': [1.0, 2.0, 3.0]})
    out = df.astype({'a': 'int64'})
    assert out['a'].dtype == 'int64'
    np.testing.assert_array_equal(out['a'].to_numpy(), [1, 2, 3])


def test_astype_to_bool():
    df = DataFrame({'a': [0.0, 1.0, 2.0]})
    out = df.astype({'a': 'bool'})
    assert out['a'].dtype == 'bool'
    np.testing.assert_array_equal(out['a'].to_numpy(), [False, True, True])


def test_round_per_column_bankers():
    # banker's rounding per float column; non-float columns unchanged
    df = DataFrame({'a': [0.5, 1.5, 2.5], 's': ['x', 'y', 'z']})
    out = df.round(0)
    np.testing.assert_array_equal(out['a'].to_numpy(), [0, 2, 2])
    assert list(out['s'].to_numpy()) == ['x', 'y', 'z']


def test_astype_unknown_dtype_raises():
    df = DataFrame({'a': [1.0]})
    import pytest
    with pytest.raises(Exception):
        df.astype({'a': 'complex128'})
