"""`Timestamp <op> ndarray` is element-wise, exactly as pandas' `Timestamp`.

A datetime64 array (e.g. ``df.index``) compares each element with the instant; an
object array compares each element with the Timestamp; any other array is
all-unequal and cannot be ordered. The scalar rule (a string / an integer is not an
instant) never applies to an array operand. The oracle is pandas 3.0.3: the same
expression with ``pd.Timestamp`` in place of ``volas.Timestamp``.
"""

import operator

import numpy as np
import pandas as pd
import pytest

import volas

OPS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
DT64 = np.array(['2021-06-15T09:30', '2021-06-15T12:00', '2021-06-16', 'NaT'], 'datetime64[ns]')


def _outcome(f):
    try:
        out = f()
    except TypeError:
        return TypeError
    return np.asarray(out).tolist()


def _pair(when, tz=None):
    return volas.Timestamp(when, tz=tz), pd.Timestamp(when, tz=tz)


ARRAYS = {
    'datetime64': DT64,
    'datetime64-2d': DT64[:4].reshape(2, 2),
    'int64': np.array([1, 2, 3]),
    'float64': np.array([1.5, np.nan]),
    'str': np.array(['2021-06-15 12:00', 'x']),
}


@pytest.mark.parametrize('tz', [None, 'UTC'])
@pytest.mark.parametrize('kind', list(ARRAYS))
@pytest.mark.parametrize('op', OPS, ids=lambda f: f.__name__)
def test_timestamp_against_arrays_matches_pandas(op, kind, tz):
    ours, oracle = _pair('2021-06-15 12:00', tz)
    arr = ARRAYS[kind]
    assert _outcome(lambda: op(ours, arr)) == _outcome(lambda: op(oracle, arr))
    assert _outcome(lambda: op(arr, ours)) == _outcome(lambda: op(arr, oracle))


@pytest.mark.parametrize('op', OPS, ids=lambda f: f.__name__)
def test_object_array_of_timestamps_is_element_wise(op):
    ours, oracle = _pair('2021-06-16')
    cells = np.array([volas.Timestamp('2021-06-15'), volas.Timestamp('2021-06-16'),
                      volas.Timestamp('2021-06-17')], dtype=object)
    pd_cells = np.array([pd.Timestamp(t.value) for t in cells], dtype=object)
    assert _outcome(lambda: op(cells, ours)) == _outcome(lambda: op(pd_cells, oracle))
    assert _outcome(lambda: op(ours, cells)) == _outcome(lambda: op(oracle, pd_cells))


def test_object_cells_exported_from_a_column_compare_element_wise():
    df = volas.DataFrame({'t': DT64[:3]})
    cells = df['t'].to_numpy(dtype='object')
    assert (cells == volas.Timestamp('2021-06-16', tz='UTC')).tolist() == [False, False, True]


def test_boolean_row_selection_by_index_comparison():
    df = volas.DataFrame({'c': [1.0, 2.0, 3.0]}, index=DT64[:3])
    ts = volas.Timestamp('2021-06-15 12:00')
    assert df[df.index >= ts]['c'].to_list() == [2.0, 3.0]
    assert df[ts <= df.index]['c'].to_list() == [2.0, 3.0]
    assert (df['c'].index > ts).tolist() == [False, False, True]
    pdf = pd.DataFrame({'c': [1.0, 2.0, 3.0]}, index=DT64[:3])
    assert pdf[pdf.index >= pd.Timestamp('2021-06-15 12:00')]['c'].tolist() == [2.0, 3.0]
