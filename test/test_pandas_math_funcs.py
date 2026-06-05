"""Parity of volas's Series transcendental methods with numpy (the ufunc oracle).

volas exposes the numpy elementwise math functions as Series methods
(``s.sin()``, ``s.sqrt()``, ...), the same values pandas yields via
``np.sin(series)``. Each case asserts volas == numpy elementwise, including the
NaN that numpy emits outside a function's domain (``sqrt(-1)``, ``log(0)``,
``acos(2)``) — volas produces the same NaN, it just does not warn.
"""

import numpy as np
import pytest

import volas


def _series(values):
    return volas.DataFrame({"x": values})["x"]


DOMAIN = [0.1, 0.5, 1.0, 2.0, 4.0, 10.0]
SIGNED = [-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0]
UNIT = [-1.0, -0.5, 0.0, 0.5, 1.0]  # asin/acos domain

# (method, numpy fn, input values)
MATH_CASES = [
    ("sin", np.sin, SIGNED),
    ("cos", np.cos, SIGNED),
    ("tan", np.tan, SIGNED),
    ("asin", np.arcsin, UNIT),
    ("acos", np.arccos, UNIT),
    ("atan", np.arctan, SIGNED),
    ("sinh", np.sinh, SIGNED),
    ("cosh", np.cosh, SIGNED),
    ("tanh", np.tanh, SIGNED),
    ("exp", np.exp, SIGNED),
    ("ln", np.log, SIGNED),       # negative/zero -> nan/-inf, like numpy
    ("log10", np.log10, SIGNED),
    ("sqrt", np.sqrt, SIGNED),    # negative -> nan, like numpy
    ("floor", np.floor, [-2.5, -0.1, 0.0, 0.1, 1.9, 4.0]),
    ("ceil", np.ceil, [-2.5, -0.1, 0.0, 0.1, 1.9, 4.0]),
]


@pytest.mark.parametrize(
    "method,np_fn,values",
    [c for c in MATH_CASES],
    ids=[c[0] for c in MATH_CASES],
)
def test_math_method_matches_numpy(method, np_fn, values):
    got = np.asarray(getattr(_series(values), method)().to_numpy(), dtype=float)
    with np.errstate(all="ignore"):
        exp = np_fn(np.asarray(values, dtype=float))
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12, equal_nan=True), (
        f"{method}: {got.tolist()} != {exp.tolist()}"
    )


def test_math_propagates_nan():
    got = np.asarray(_series([1.0, float("nan"), 4.0]).sqrt().to_numpy(), dtype=float)
    assert got[0] == 1.0 and np.isnan(got[1]) and got[2] == 2.0
