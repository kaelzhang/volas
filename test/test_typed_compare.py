"""Typed element-wise comparison (==, !=, <, <=, >, >=) for str / datetime / bool
columns, plus scalar right-hand sides.

Comparison must NOT funnel through `to_f64_vec()` (which maps str -> NaN and loses
datetime precision). It compares native values and produces a non-nullable bool
mask (decision (2)): a missing slot follows IEEE — equality/ordering are False
there, `!=` is True (consistent with the existing numeric comparison). Scalars
(a str, a number) broadcast.
"""

import numpy as np
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


# --- string columns ----------------------------------------------------------

def test_str_eq_ne_elementwise():
    s = _s(['a', None, 'b'])
    t = _s(['a', None, 'x'])
    assert (s == t).to_list() == [True, False, False]   # NA slot -> False
    assert (s != t).to_list() == [False, True, True]    # NA slot -> True (IEEE)
    assert (s == t).dtype == 'bool'


def test_str_eq_scalar():
    s = _s(['a', None, 'b'])
    assert (s == 'a').to_list() == [True, False, False]
    assert (s != 'a').to_list() == [False, True, True]


def test_str_ordering():
    a = _s(['b', 'a', 'c'])
    b = _s(['c', 'a', 'a'])
    assert (a < b).to_list() == [True, False, False]
    assert (a <= b).to_list() == [True, True, False]
    assert (a > b).to_list() == [False, False, True]
    assert (a >= b).to_list() == [False, True, True]
    assert (a < 'b').to_list() == [False, True, False]  # scalar bound


# --- datetime columns (compare by value, no f64 precision loss) ---------------

def test_datetime_eq_elementwise():
    s = _s(['2021-01-01', 'NaT', '2021-01-03'], 'datetime64[ns]')
    t = _s(['2021-01-01', 'NaT', '2021-01-09'], 'datetime64[ns]')
    assert (s == t).to_list() == [True, False, False]
    assert (s != t).to_list() == [False, True, True]
    assert (s < t).to_list() == [False, False, True]


def test_datetime_precision_beyond_f64():
    # two timestamps 1 ns apart, far past 2**53 ns — an f64 funnel would collapse
    # them to equal; a typed i64 compare keeps them distinct
    big = np.datetime64('2262-04-11T23:47:16.854775806')
    big1 = big + np.timedelta64(1, 'ns')
    s = _s([big, big], 'datetime64[ns]')
    t = _s([big, big1], 'datetime64[ns]')
    assert (s == t).to_list() == [True, False]


# --- bool columns (validity-aware) -------------------------------------------

def test_bool_eq_elementwise():
    s = _s([True, None, False])
    t = _s([True, None, True])
    assert (s == t).to_list() == [True, False, False]
    assert (s != t).to_list() == [False, True, True]


# --- DataFrame comparison (per column, str by value) -------------------------

def test_dataframe_eq_str_column():
    a = DataFrame({'s': ['a', 'b', 'c'], 'n': [1, 2, 3]})
    b = DataFrame({'s': ['a', 'x', 'c'], 'n': [1, 2, 9]})
    eq = a == b
    assert eq['s'].to_list() == [True, False, True]
    assert eq['n'].to_list() == [True, True, False]


def test_compare_incompatible_kinds_errors():
    # comparing a str column to an int column is a hard error (no silent all-False
    # mask) — same-frame so the shared index passes alignment first
    df = DataFrame({'s': ['a', 'b'], 'i': [1, 2]})
    with pytest.raises(Exception):
        _ = df['s'] == df['i']
