"""Contract C4 — any()/all() are bool/numeric truthiness reductions, so a str or
datetime column must raise rather than funnel through to_f64_vec (which silently
makes str all-NaN -> any()==False, and datetime epoch-ns -> any()==True: a
nonsense, dtype-dependent answer)."""

import numpy as np
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


@pytest.mark.parametrize('op', ['any', 'all'])
def test_any_all_on_str_raises(op):
    with pytest.raises(Exception):
        getattr(_s(['a', 'b', '']), op)()


@pytest.mark.parametrize('op', ['any', 'all'])
def test_any_all_on_datetime_raises(op):
    s = _s(['2021-01-01', '2021-01-02']).astype('datetime64[ns]')
    with pytest.raises(Exception):
        getattr(s, op)()


def test_any_all_numeric_and_bool_unchanged():
    assert _s([0.0, 0.0, 1.0]).any() == True   # noqa: E712
    assert _s([0.0, 0.0, 0.0]).any() == False  # noqa: E712
    assert _s([1.0, 2.0, 0.0]).all() == False  # noqa: E712
    assert _s([True, False, True]).any() == True   # noqa: E712
    assert _s([True, True, True]).all() == True    # noqa: E712
