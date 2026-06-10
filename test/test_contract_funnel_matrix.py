"""The f64-funnel contract matrix (P3-02).

The recurring contract gaps all had ONE root cause: the f64 funnel
(`to_f64_vec` / `map_f64` / the row-major export) leaking into an API surface
that should not ride the f64 channel for a str / datetime column. Fixing those
reactively (one finding at a time) kept leaving un-audited siblings, so this
module enumerates EVERY public funnel consumer and pins its classification, on
BOTH a str and a datetime column:

  * ARITHMETIC      — a numeric-only op; must RAISE on str/datetime (no silent
                      0.0 / NaN / epoch funnel). C4.
  * ORDER_BASED     — needs only a comparison, so it WORKS on any ordered dtype
                      (str lexically, datetime by instant), typed, no funnel.
  * EXPLICIT_BOUNDARY — an explicit export/cast; WORKS, producing a typed
                      (object / datetime64) result, never a silent funnel.

A new funnel consumer that is not listed here, or one that changes class, should
break this matrix — that is the point. It is the regression guard that the
audit's conclusions stay true.
"""

import numpy as np
import pytest
from volas import DataFrame


def _str():
    return DataFrame({'a': ['b', 'a', 'c']})['a']


def _datetime():
    return DataFrame({'a': ['2021-01-01', '2021-01-03', '2021-01-02']})['a'].astype('datetime64[ns]')


# --- ARITHMETIC: must raise on str AND datetime ------------------------------
# Every numeric reduction / moment / transform that funnels through to_f64_vec.
ARITHMETIC = {
    'sum': lambda s: s.sum(),
    'prod': lambda s: s.prod(),
    'mean': lambda s: s.mean(),
    'var': lambda s: s.var(),
    'std': lambda s: s.std(),
    'median': lambda s: s.median(),
    'sem': lambda s: s.sem(),
    'skew': lambda s: s.skew(),
    'kurt': lambda s: s.kurt(),
    'quantile': lambda s: s.quantile(0.5),
    'describe': lambda s: s.describe(),
    'any': lambda s: s.any(),
    'all': lambda s: s.all(),
    'corr': lambda s: s.corr(s),
    'cov': lambda s: s.cov(s),
    'diff': lambda s: s.diff(1),
    'cumsum': lambda s: s.cumsum(),
    'cumprod': lambda s: s.cumprod(),
    'clip': lambda s: s.clip(0, 1),
    # math transforms (the map_f64 family)
    'acos': lambda s: s.acos(), 'asin': lambda s: s.asin(), 'atan': lambda s: s.atan(),
    'ceil': lambda s: s.ceil(), 'cos': lambda s: s.cos(), 'cosh': lambda s: s.cosh(),
    'exp': lambda s: s.exp(), 'floor': lambda s: s.floor(), 'ln': lambda s: s.ln(),
    'log10': lambda s: s.log10(), 'sin': lambda s: s.sin(), 'sinh': lambda s: s.sinh(),
    'sqrt': lambda s: s.sqrt(), 'tan': lambda s: s.tan(), 'tanh': lambda s: s.tanh(),
}

# --- ORDER_BASED: must work on str AND datetime (typed, no funnel) -----------
ORDER_BASED = {
    'min': lambda s: s.min(),
    'max': lambda s: s.max(),
    'idxmax': lambda s: s.idxmax(),
    'idxmin': lambda s: s.idxmin(),
    'rank': lambda s: s.rank(),
    'sort_values': lambda s: s.sort_values(),
    'cummax': lambda s: s.cummax(),
    'cummin': lambda s: s.cummin(),
    'unique': lambda s: s.unique(),
    'shift': lambda s: s.shift(1),
}

# --- EXPLICIT_BOUNDARY: explicit export/cast works (typed result) ------------
EXPLICIT_BOUNDARY = {
    'to_numpy': lambda s: s.to_numpy(),
    'to_list': lambda s: s.to_list(),
}


@pytest.mark.parametrize('op', list(ARITHMETIC), ids=list(ARITHMETIC))
@pytest.mark.parametrize('col', ['str', 'datetime'])
def test_arithmetic_raises(op, col):
    s = _str() if col == 'str' else _datetime()
    # TypeError specifically (the require_numeric guard), so a missing method's
    # AttributeError cannot masquerade as a passing "it raises" assertion.
    with pytest.raises(TypeError):
        ARITHMETIC[op](s)


@pytest.mark.parametrize('op', list(ORDER_BASED), ids=list(ORDER_BASED))
@pytest.mark.parametrize('col', ['str', 'datetime'])
def test_order_based_works(op, col):
    s = _str() if col == 'str' else _datetime()
    ORDER_BASED[op](s)  # must not raise


@pytest.mark.parametrize('op', list(EXPLICIT_BOUNDARY), ids=list(EXPLICIT_BOUNDARY))
@pytest.mark.parametrize('col', ['str', 'datetime'])
def test_explicit_boundary_works_typed(op, col):
    s = _str() if col == 'str' else _datetime()
    out = EXPLICIT_BOUNDARY[op](s)
    if op == 'to_numpy':
        # the export is typed, never the f64 channel
        assert out.dtype == object if col == 'str' else out.dtype == np.dtype('datetime64[ns]')


def test_order_based_results_are_typed():
    # spot-check that ORDER_BASED really produced typed values, not f64
    assert _str().max() == 'c' and _str().min() == 'a'
    assert _datetime().max() == np.datetime64('2021-01-03')
    assert list(_str().cummax().to_numpy()) == ['b', 'b', 'c']
    assert list(_str().sort_values().to_numpy()) == ['a', 'b', 'c']
