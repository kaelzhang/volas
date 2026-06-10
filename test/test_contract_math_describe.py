"""Contract C4 — the math-transform family and Series.describe() must reject
str/datetime (they funnel through to_f64_vec / map_f64, producing silent NaN for
str and meaningless sin-of-epoch / numeric-stats-of-epoch for datetime)."""

import numpy as np
import pytest
from volas import DataFrame

MATH = ['acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'exp', 'floor', 'ln',
        'log10', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


@pytest.mark.parametrize('op', MATH)
def test_math_transform_str_raises(op):
    with pytest.raises(Exception):
        getattr(_s(['a', 'b']), op)()


@pytest.mark.parametrize('op', MATH)
def test_math_transform_datetime_raises(op):
    dt = _s(['2024-01-01', '2024-01-02'], 'datetime64[ns]')
    with pytest.raises(Exception):
        getattr(dt, op)()


def test_math_transform_numeric_still_works():
    assert _s([1.0, 4.0]).sqrt().to_list() == [1.0, 2.0]
    assert _s([0.0]).cos().to_list() == [1.0]


def test_describe_str_raises():
    with pytest.raises(Exception):
        _s(['a', 'b', 'c']).describe()


def test_describe_datetime_raises():
    with pytest.raises(Exception):
        _s(['2024-01-01', '2024-01-02'], 'datetime64[ns]').describe()


def test_describe_numeric_still_works():
    d = _s([1.0, 2.0, 3.0, 4.0]).describe()
    assert d.to_list()[0] == 4.0  # count
