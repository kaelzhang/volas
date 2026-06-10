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


def test_describe_numeric_columns_only():
    df = DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 's': ['w', 'x', 'y', 'z']})
    d = df.describe()
    assert d.columns == ['a']  # the string column is excluded
    assert list(np.asarray(d.index)) == ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    np.testing.assert_allclose(np.asarray(d['a'].to_numpy(), float),
                               [4.0, 2.5, 1.2909944487358056, 1.0, 1.75, 2.5, 3.25, 4.0])


def test_corr_cov_matrix():
    df = DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [1.0, 2.0, 3.0, 5.0]})
    c = df.corr()
    assert c.columns == ['a', 'b'] and list(np.asarray(c.index)) == ['a', 'b']
    m = np.asarray(c.to_numpy(), float)
    assert m[0][0] == 1.0 and m[1][1] == 1.0           # unit diagonal
    assert abs(m[0][1] - m[1][0]) < 1e-12              # symmetric
    assert abs(m[0][1] - 0.9827076298239906) < 1e-9
    assert abs(np.asarray(df.cov().to_numpy(), float)[0][1] - 2.1666666666666665) < 1e-9


def test_astype_unknown_dtype_raises():
    df = DataFrame({'a': [1.0]})
    import pytest
    with pytest.raises(Exception):
        df.astype({'a': 'complex128'})


def test_round_includes_narrow_numeric_dtypes():
    # P1-04: round must round float32 / int32 too (was: only f64/i64 touched)
    df = DataFrame({
        'f32': np.array([1.25, 2.75], dtype=np.float32),
        'i32': np.array([15, 25], dtype=np.int32),
        'f64': [1.25, 2.75],
        'i64': [15, 25],
    })
    r1 = df.round(1)
    assert r1['f32'].dtype == 'float32' and r1['i32'].dtype == 'int32'
    np.testing.assert_allclose(np.asarray(r1['f32'].to_numpy(), float), [1.2, 2.8], rtol=1e-5)
    np.testing.assert_allclose(np.asarray(r1['f64'].to_numpy(), float), [1.2, 2.8])
    rneg = df.round(-1)
    np.testing.assert_array_equal(np.asarray(rneg['i32'].to_numpy()), [20, 20])  # tens, like i64
    np.testing.assert_array_equal(np.asarray(rneg['i64'].to_numpy()), [20, 20])


def test_describe_corr_cov_include_narrow_numeric_dtypes():
    # P1-04: describe / corr / cov must treat float32 / int32 as numeric (str excluded)
    wide = DataFrame({
        'f32': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'i32': np.array([1, 2, 3], dtype=np.int32),
        'f64': [1.0, 2.0, 3.0],
        'i64': [1, 2, 3],
        's': ['a', 'b', 'c'],
    })
    assert wide.describe().columns == ['f32', 'i32', 'f64', 'i64']
    assert wide.corr().columns == ['f32', 'i32', 'f64', 'i64']
    assert wide.cov().columns == ['f32', 'i32', 'f64', 'i64']


def test_sem_skew_kurt_include_narrow_numeric_dtypes():
    # P1-04: the column reductions (sem/skew/kurt) must include float32 / int32
    df = DataFrame({
        'f32': np.array([1.0, 2.0, 4.0, 7.0], dtype=np.float32),
        'i32': np.array([1, 2, 4, 7], dtype=np.int32),
        'f64': [1.0, 2.0, 4.0, 7.0],
    })
    for op in ['sem', 'skew', 'kurt']:
        idx = list(np.asarray(getattr(df, op)().index))
        assert {'f32', 'i32', 'f64'}.issubset(set(idx)), f"{op}: {idx}"
