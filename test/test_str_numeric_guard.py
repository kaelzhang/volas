"""V3 / V4 (API contract C4): numeric operations on a str (or datetime) column
must raise, not silently return 0.0 / NaN. str storage exists for labels, not
arithmetic — a silent numeric result is a lossy implicit conversion."""

import numpy as np
import pytest
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


# --- V3: numeric reductions on str -------------------------------------------

@pytest.mark.parametrize('op', ['sum', 'prod', 'mean', 'min', 'max', 'var', 'std',
                                'median', 'sem', 'skew', 'kurt', 'rank'])
def test_str_reduction_raises(op):
    with pytest.raises(Exception):
        getattr(_s(['a', 'b', 'c']), op)()


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
