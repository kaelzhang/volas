"""Contract D4/D5 + the order-based-ops principle — idxmax/idxmin/min/max/rank
work on any ORDERED dtype (numeric by value, str lexically, datetime by raw i64,
bool), NOT through the f64 funnel. Datetime must keep sub-256ns ordering (f64
collapses it past 2^53); str uses lexical order; numeric is unchanged."""

import numpy as np
import volas
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


# --- idxmax / idxmin ----------------------------------------------------------

def test_idxmax_datetime_sub_256ns():
    base = np.datetime64('2024-01-01T00:00:00.000000000')
    s = _s(base + np.array([0, 100, 200, 300], dtype='timedelta64[ns]'))
    # the 300ns row is the max; f64 would collapse the 100ns gaps and miss it
    assert s.idxmax() == 3 and s.idxmin() == 0


def test_idxmax_idxmin_str_lexical():
    DataFrame({'s': ['banana', 'apple', 'cherry'], 'i': [0, 1, 2]})['s']
    sx = DataFrame({'s': ['banana', 'apple', 'cherry'], 'i': [0, 1, 2]}).set_index('i')['s']
    assert sx.idxmax() == 2   # 'cherry' (lexical max) at label 2
    assert sx.idxmin() == 1   # 'apple' (lexical min) at label 1


def test_idxmax_numeric_unchanged():
    assert _s([3.0, 1.0, 4.0, 1.0]).idxmax() == 2
    assert _s([3, 1, 4, 1]).idxmin() == 1


def test_idxmax_skips_na():
    s = _s([1.0, float('nan'), 5.0, float('nan')])
    assert s.idxmax() == 2 and s.idxmin() == 0


# --- min / max (typed VALUE, not f64) -----------------------------------------

def test_minmax_datetime_keeps_instant():
    base = np.datetime64('2024-01-01T00:00:00.000000000')
    s = _s(base + np.array([0, 100, 200, 300], dtype='timedelta64[ns]'))
    # the extreme VALUE is a Timestamp (O1->B), exact to the nanosecond (no f64
    # collapse) — and it compares equal to the matching np.datetime64 instant.
    assert s.max() == base + np.timedelta64(300, 'ns')
    assert s.min() == base
    assert isinstance(s.max(), volas.Timestamp)


def test_minmax_str_lexical():
    s = _s(['banana', 'apple', 'cherry'])
    assert s.max() == 'cherry'   # lexical max, returned as a Python str
    assert s.min() == 'apple'
    assert isinstance(s.max(), str)


def test_minmax_numeric_unchanged():
    s = _s([3.0, 1.0, 4.0, 1.0])
    assert s.max() == 4.0 and s.min() == 1.0
    assert isinstance(s.max(), np.float64)        # numeric reductions stay numpy
    si = _s([3, 1, 4, 1])
    assert isinstance(si.max(), np.int64)


def test_minmax_bool():
    s = _s([False, True, False])
    assert s.max() == True and s.min() == False  # noqa: E712


# --- rank (order-based, result always float64) --------------------------------

def test_rank_str_lexical():
    s = _s(['banana', 'apple', 'cherry'])
    # apple<banana<cherry -> ranks 2,1,3; result dtype is float64
    assert list(s.rank().to_numpy()) == [2.0, 1.0, 3.0]


def test_rank_datetime():
    base = np.datetime64('2024-01-01T00:00:00.000000000')
    s = _s(base + np.array([0, 300, 100, 200], dtype='timedelta64[ns]'))
    assert list(s.rank().to_numpy()) == [1.0, 4.0, 2.0, 3.0]


def test_rank_numeric_unchanged():
    s = _s([3.0, 1.0, 4.0, 1.0])
    assert list(s.rank().to_numpy()) == [3.0, 1.5, 4.0, 1.5]


# --- cummax / cummin (order-based, dtype-preserving) --------------------------

def test_cummax_cummin_str_lexical():
    s = _s(['b', 'a', 'c'])
    assert list(s.cummax().to_numpy()) == ['b', 'b', 'c']
    assert list(s.cummin().to_numpy()) == ['b', 'a', 'a']


def test_cummax_cummin_datetime():
    base = np.datetime64('2024-01-01', 'ns')
    days = lambda d: base + np.timedelta64(d, 'D')  # noqa: E731
    s = _s(np.array([base, days(2), days(1)], dtype='datetime64[ns]'))
    assert list(s.cummax().to_numpy()) == [base, days(2), days(2)]
    assert list(s.cummin().to_numpy()) == [base, base, base]


def test_cummax_str_keeps_interior_na():
    # a missing cell stays missing; the running max ignores it (pandas semantics)
    s = _s(['b', None, 'a', 'd'])
    out = list(s.cummax().to_numpy())
    assert out[0] == 'b' and out[1] is None and out[2] == 'b' and out[3] == 'd'


def test_cummax_cummin_numeric_unchanged():
    s = _s([3.0, 1.0, 4.0, 1.0])
    assert list(s.cummax().to_numpy()) == [3.0, 3.0, 4.0, 4.0]
    assert list(s.cummin().to_numpy()) == [3.0, 1.0, 1.0, 1.0]
