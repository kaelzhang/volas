"""V3 / V4 (API contract C4): numeric-ARITHMETIC operations on a str (or datetime)
column must raise, not silently return 0.0 / NaN. str storage exists for labels,
not arithmetic — a silent numeric result is a lossy implicit conversion.

The ORDER-based ops (min/max/rank/idxmax/idxmin/sort_values) are NOT numeric:
they need only a comparison, so they work on any ordered dtype (str lexically,
datetime by instant) and are exercised in test_contract_order_ops.py — they are
deliberately absent from the raise list below."""

import numpy as np
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


# --- V3: numeric-arithmetic reductions on str raise --------------------------

@pytest.mark.parametrize('op', ['sum', 'prod', 'mean', 'var', 'std',
                                'median', 'sem', 'skew', 'kurt'])
def test_str_reduction_raises(op):
    with pytest.raises(Exception):
        getattr(_s(['a', 'b', 'c']), op)()


def test_str_order_based_reductions_work():
    # min/max/rank are order-based, so they operate on str lexically rather than
    # raising (the numeric-arithmetic reductions above still raise).
    s = _s(['banana', 'apple', 'cherry'])
    assert s.min() == 'apple' and s.max() == 'cherry'
    assert list(s.rank().to_numpy()) == [2.0, 1.0, 3.0]


def test_str_quantile_corr_raise():
    with pytest.raises(Exception):
        _s(['a', 'b']).quantile(0.5)
    with pytest.raises(Exception):
        _s(['a', 'b']).corr(_s(['x', 'y']))


def test_numeric_reductions_still_work():
    assert _s([1.0, 2.0, 3.0]).sum() == 6.0
    assert _s([1, 2, 3]).mean() == 2.0


# --- V4: arithmetic on str ---------------------------------------------------

def test_str_arithmetic_raises():
    s, t = _s(['a', 'b', 'c']), _s(['x', 'y', 'z'])
    for fn in [lambda: s + t, lambda: s - t, lambda: s * t, lambda: s / t, lambda: s // t, lambda: -s]:
        with pytest.raises(Exception):
            fn()


def test_corr_cov_either_operand_str_raises():
    # both require_numeric guards (self AND other) must fire
    n = _s([1.0, 2.0, 3.0])
    s = _s(['a', 'b', 'c'])
    for fn in [lambda: s.cov(n), lambda: n.corr(s), lambda: n.cov(s)]:
        with pytest.raises(Exception):
            fn()
