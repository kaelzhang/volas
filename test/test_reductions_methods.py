"""pandas-parity for the count / nunique / unique / sort_values / head / tail
methods (the NA-aware reductions the review flagged as missing).

count / nunique exclude missing; unique keeps a single missing slot in
appearance order; sort_values is stable with missing values last in BOTH
directions (pandas `na_position='last'`); head / tail slice data and index.
"""

import numpy as np
import pandas as pd
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


# --- count -------------------------------------------------------------------

def test_series_count_parity():
    for data in ([1.0, float('nan'), 3.0, float('nan'), 5.0], [5.0, 5.0, 5.0], [float('nan')]):
        assert _s(data).count() == int(pd.Series(data).count())
    # int / str / bool with NA (volas keeps dtype + validity)
    assert DataFrame({'a': [1, None, 3]})['a'].count() == 2
    assert DataFrame({'a': ['x', None, 'y']})['a'].count() == 2
    assert DataFrame({'a': [True, None, False, True]})['a'].count() == 3


def test_dataframe_count():
    df = DataFrame({'a': [1.0, float('nan'), 3.0], 'b': [1, 2, 3], 's': ['x', None, 'z']})
    c = df.count()
    assert list(np.asarray(c.index)) == ['a', 'b', 's']
    assert list(np.asarray(c.to_numpy())) == [2, 3, 2]
    assert c.dtype == 'int64'


# --- nunique -----------------------------------------------------------------

def test_series_nunique_parity():
    for data in ([1.0, 1.0, float('nan'), 2.0, float('nan')], [3.0, 3.0, 3.0]):
        assert _s(data).nunique() == int(pd.Series(data).nunique())
    assert DataFrame({'a': ['a', 'b', 'a', None]})['a'].nunique() == 2
    assert DataFrame({'a': [1, 1, 2, None, 3]})['a'].nunique() == 3


# --- unique ------------------------------------------------------------------

def test_series_unique_int_order():
    # appearance order, no NA -> a plain int64 array
    assert list(_s([3, 1, 3, 1, 2]).unique()) == [3, 1, 2]


def test_series_unique_keeps_one_na_slot():
    u = _s([2.0, float('nan'), 2.0, 1.0, float('nan')]).unique()
    assert u[0] == 2.0 and np.isnan(u[1]) and u[2] == 1.0 and len(u) == 3


def test_series_unique_string():
    assert list(_s(['b', 'a', 'b', 'c']).unique()) == ['b', 'a', 'c']


# --- sort_values -------------------------------------------------------------

def test_series_sort_values_na_last_both_directions():
    s = _s([3.0, 1.0, float('nan'), 2.0])
    asc = s.sort_values()
    assert asc.isna().to_list() == [False, False, False, True]   # NA sinks last
    assert asc.to_list()[:3] == [1.0, 2.0, 3.0]
    assert list(np.asarray(asc.index)) == [1, 3, 0, 2]           # index follows
    desc = s.sort_values(ascending=False)
    assert desc.isna().to_list() == [False, False, False, True]  # NA STILL last
    assert desc.to_list()[:3] == [3.0, 2.0, 1.0]


def test_series_sort_values_is_stable():
    # equal keys keep their original relative order (stable sort)
    s = DataFrame({'a': [1, 1, 1], 'k': [10, 20, 30]})['k']
    # sort the 'a' column (all equal) -> index order preserved
    a = DataFrame({'a': [2, 2, 1]})['a'].sort_values()
    assert list(np.asarray(a.index)) == [2, 0, 1]  # the 1 first, then the two 2s in order


def test_series_sort_values_string():
    s = _s(['banana', 'apple', 'cherry'])
    assert s.sort_values().to_list() == ['apple', 'banana', 'cherry']


# --- head / tail -------------------------------------------------------------

def test_series_head_tail():
    s = _s([0, 1, 2, 3, 4, 5, 6])
    assert list(np.asarray(s.head(3).to_numpy())) == [0, 1, 2]
    assert list(np.asarray(s.tail(2).to_numpy())) == [5, 6]
    assert list(np.asarray(s.head().to_numpy())) == [0, 1, 2, 3, 4]   # default n=5
    assert list(np.asarray(s.tail().to_numpy())) == [2, 3, 4, 5, 6]
    assert list(np.asarray(s.tail(2).index)) == [5, 6]               # index sliced too
    assert list(np.asarray(s.head(99).to_numpy())) == [0, 1, 2, 3, 4, 5, 6]  # n > len


def test_tail_preserves_range_index_offset():
    # a RangeIndex tail keeps the absolute labels (offset), not a reset 0.. — both
    # for DataFrame.tail and Series.tail (Index::slice materializes a non-zero start)
    df = DataFrame({'a': [0, 1, 2, 3, 4, 5, 6]})
    assert list(np.asarray(df.tail(2).index)) == [5, 6]
    assert list(np.asarray(df['a'].tail(2).index)) == [5, 6]
    assert list(np.asarray(df.head(3).index)) == [0, 1, 2]  # head (start 0) unchanged


# --- every storage dtype exercises its group / compare arm --------------------

def test_nunique_unique_all_dtypes():
    assert _s([2.0, 1.0, 2.0], np.float32).nunique() == 2
    assert list(_s([2.0, 1.0, 2.0], np.float32).unique()) == [2.0, 1.0]
    assert _s([2, 1, 2], np.int32).nunique() == 2
    assert list(_s([2, 1, 2], np.int32).unique()) == [2, 1]
    assert DataFrame({'a': [True, False, True, True]})['a'].nunique() == 2
    assert list(DataFrame({'a': [True, False, True]})['a'].unique()) == [True, False]
    dt = _s(['2021-01-02', '2021-01-01', '2021-01-02'], 'datetime64[ns]')
    assert dt.nunique() == 2
    u = dt.unique()
    assert u[0] == np.datetime64('2021-01-02') and u[1] == np.datetime64('2021-01-01') and len(u) == 2


def test_sort_values_all_dtypes():
    assert _s([3.0, 1.0, 2.0], np.float32).sort_values().to_list() == [1.0, 2.0, 3.0]
    assert _s([3, 1, 2], np.int32).sort_values().to_list() == [1, 2, 3]
    assert DataFrame({'a': [True, False, True]})['a'].sort_values().to_list() == [False, True, True]
    dt = _s(['2021-01-03', '2021-01-01', '2021-01-02'], 'datetime64[ns]')
    assert dt.sort_values().to_list() == [
        np.datetime64('2021-01-01'), np.datetime64('2021-01-02'), np.datetime64('2021-01-03')]


def test_sort_values_multiple_na_compare_each_other():
    # two+ NA values must both sink last (exercises the NA-vs-NA compare branch)
    s = _s([3.0, float('nan'), 1.0, float('nan'), 2.0])
    asc = s.sort_values()
    assert asc.to_list()[:3] == [1.0, 2.0, 3.0]
    assert asc.isna().to_list() == [False, False, False, True, True]


def test_nunique_unique_canonicalize_zero():
    # +0.0 and -0.0 are one value (pandas), and 0.0 participates as a real group
    s = _s([0.0, -0.0, 1.0, 0.0])
    assert s.nunique() == 2                         # {0.0, 1.0}
    u = s.unique()
    assert u[0] == 0.0 and u[1] == 1.0 and len(u) == 2
