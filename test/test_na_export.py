"""A missing truth never reaches NumPy as a value that reads True or False.

NumPy has no missing bool: an object array would hold a ``nan`` (truthy) or ``None``
(falsy), so ``np.where`` / boolean indexing would act on an unknown signal as if it
were decided. A bool column holding NA therefore exports only when the caller says
what a missing cell is — ``na_value=``, ``dtype='float64'`` (``NaN``) or
``dtype='object'`` (``volas.NA``, whose truth value raises). Every NumPy route goes
through that one rule: ``Series.to_numpy``, ``np.asarray`` / ``__array__`` and
``DataFrame.exec``.
"""

import numpy as np
import pytest

import volas
from volas import NA

CLOSE = [1.0, 2.0, 3.0, 2.0, 5.0]
SIGNAL = 'close > ma:3'  # warm-up rows 0, 1 are missing truths


def _frame():
    return volas.DataFrame({'close': CLOSE})


def test_a_bool_series_with_na_refuses_a_bare_numpy_export():
    s = _frame()[SIGNAL]
    for export in (s.to_numpy, lambda: np.asarray(s), lambda: s.__array__()):
        with pytest.raises(ValueError, match='bool column with missing values'):
            export()


def test_exec_of_a_boolean_directive_with_missing_truths_refuses_a_bare_export():
    df = _frame()
    with pytest.raises(ValueError, match='bool column with missing values'):
        df.exec(SIGNAL)
    # the numpy pitfalls this prevents: a missing truth read as True / False
    with pytest.raises(ValueError):
        np.where(df.exec(SIGNAL), 1, 0)
    with pytest.raises(ValueError):
        np.asarray(CLOSE)[df.exec(SIGNAL)]


def test_saying_what_a_missing_cell_is_exports_it():
    df, s = _frame(), _frame()[SIGNAL]
    for got in (s.to_numpy(na_value=False), df.exec(SIGNAL, na_value=False)):
        assert got.dtype == np.bool_ and got.tolist() == [False, False, True, False, True]
    for got in (s.to_numpy(dtype='float64'), df.exec(SIGNAL, dtype='float64'),
                np.asarray(s, dtype=np.float64)):
        assert got.dtype == np.float64
        assert np.isnan(got[:2]).all() and got[2:].tolist() == [1.0, 0.0, 1.0]
    for got in (s.to_numpy(dtype='object'), df.exec(SIGNAL, dtype='object')):
        assert got[0] is NA and got[2] == True  # noqa: E712 - a numpy / Python bool
        with pytest.raises(TypeError):
            bool(got[0])  # an unknown truth has no truth value
    # a float fill into a float target: NaN stays NaN (never coerced through a bool)
    got = s.to_numpy(dtype='float64', na_value=np.nan)
    assert np.isnan(got[:2]).all() and got[2:].tolist() == [1.0, 0.0, 1.0]
    got = s.to_numpy(na_value=-1)  # a non-bool fill keeps every cell as it is
    assert got.dtype == object and got.tolist() == [-1, -1, True, False, True]
    assert s.to_numpy(na_value=np.bool_(True)).tolist() == [True, True, True, False, True]
    with pytest.raises(ValueError, match="dtype 'bool'"):
        s.to_numpy(dtype='bool')
    with pytest.raises(ValueError, match='bool column with missing values'):
        s.to_numpy(dtype='datetime64[ns]')


def test_float_exports_keep_working():
    df = _frame()
    # the Compton idiom: an indicator exported as float64 (warm-up NaN)
    ma = np.asarray(df.exec('ma:3'), dtype=np.float64)
    assert np.isnan(ma[:2]).all()
    # a dense boolean exports as bool, a float column / int column with NA as float
    assert df.exec('close > 0').dtype == np.bool_
    ints = volas.DataFrame({'n': ([1, 0, 3], [False, True, False], 'int64')})['n']
    assert np.isnan(ints.to_numpy()[1])


def test_a_frame_export_never_carries_a_truthy_nan_for_a_bool():
    df = _frame()
    df[SIGNAL]
    # the default numeric matrix is a float export (NaN in a float matrix) …
    m = df.to_numpy()
    assert m.dtype == np.float64 and np.isnan(m[0, 1])
    # … and the object export carries volas.NA
    assert df.to_numpy(dtype='object')[0, 1] is NA
    # pandas gets a masked boolean (pd.NA), not an object column of nan
    pd = pytest.importorskip('pandas')
    col = df.to_pandas()['close>ma:3']  # the canonical cached name
    assert str(col.dtype) == 'boolean' and col.isna().tolist()[:2] == [True, True]
    assert pd.isna(col.iloc[0])
