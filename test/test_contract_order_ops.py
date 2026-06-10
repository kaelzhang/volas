"""Contract D4/D5 + the order-based-ops principle — idxmax/idxmin/min/max/rank
work on any ORDERED dtype (numeric by value, str lexically, datetime by raw i64,
bool), NOT through the f64 funnel. Datetime must keep sub-256ns ordering (f64
collapses it past 2^53); str uses lexical order; numeric is unchanged."""

import numpy as np
import pytest
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
    s = DataFrame({'s': ['banana', 'apple', 'cherry'], 'i': [0, 1, 2]})['s']
    sx = DataFrame({'s': ['banana', 'apple', 'cherry'], 'i': [0, 1, 2]}).set_index('i')['s']
    assert sx.idxmax() == 2   # 'cherry' (lexical max) at label 2
    assert sx.idxmin() == 1   # 'apple' (lexical min) at label 1


def test_idxmax_numeric_unchanged():
    assert _s([3.0, 1.0, 4.0, 1.0]).idxmax() == 2
    assert _s([3, 1, 4, 1]).idxmin() == 1


def test_idxmax_skips_na():
    s = _s([1.0, float('nan'), 5.0, float('nan')])
    assert s.idxmax() == 2 and s.idxmin() == 0
