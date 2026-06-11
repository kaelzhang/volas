"""Systematic audit — T1 (unary): shape-preserving element-wise transforms.

  abs round clip · cumsum cumprod cummax cummin · shift diff · ffill bfill ·
  15 math fns (acos…tanh)

Invariants: a unary op preserves the container (Series->Series) and length;
NA either propagates (math, abs, cum*) or is filled (ffill/bfill); values match
pandas exactly on the integer-valued basis.

Cell IDs:  T1.<fn>/D=<d>/N=<n>
"""

from __future__ import annotations

import math

import pytest

import volas
from . import audit_dims as A

_NUM = ("f64", "f32", "i64", "i32")
_MATH = ("acos", "asin", "atan", "ceil", "cos", "cosh", "exp", "floor",
         "ln", "log10", "sin", "sinh", "sqrt", "tan", "tanh")

# cumulative values over the basis: N0=[1,2,3], N1=[1,NA,3] (NA stays in place,
# the scan resumes past it — # pandas).
_CUM = {
    "cumsum": {"N0": [1, 3, 6], "N1": [1, None, 4]},
    "cumprod": {"N0": [1, 2, 6], "N1": [1, None, 3]},
    "cummax": {"N0": [1, 2, 3], "N1": [1, None, 3]},
    "cummin": {"N0": [1, 1, 1], "N1": [1, None, 1]},
}


def _vals_match(out, expected):
    mask = out.isna().to_list()
    assert mask == [e is None for e in expected]
    for got, exp, m in zip(out.to_list(), expected, mask):
        if not m:
            assert float(got) == float(exp)


# --- cumulative scans -------------------------------------------------------
@pytest.mark.parametrize("n", ("N0", "N1"))
@pytest.mark.parametrize("d", _NUM)
@pytest.mark.parametrize("fn", list(_CUM))
def test_cumulative(fn, d, n):
    out = getattr(A.series(d, n), fn)()
    assert isinstance(out, volas.Series) and len(out) == 3
    _vals_match(out, _CUM[fn][n])


# --- ffill / bfill ----------------------------------------------------------
@pytest.mark.parametrize("d", _NUM)
def test_ffill_bfill(d):
    s = A.series(d, "N1")            # [v0, NA, v2]
    _vals_match(s.ffill(), [1, 1, 3])
    _vals_match(s.bfill(), [1, 3, 3])
    # a leading hole has nothing to pull forward -> stays NA after ffill.
    lead = volas.DataFrame({"x": [None, 2.0, 3.0]})["x"]
    assert lead.ffill().isna().to_list() == [True, False, False]


# --- shift / diff (introduce NA at the boundary) ---------------------------
@pytest.mark.parametrize("d", _NUM)
def test_shift_diff(d):
    s = A.series(d, "N0")           # [1, 2, 3]
    _vals_match(s.shift(1), [None, 1, 2])
    _vals_match(s.shift(-1), [2, 3, None])
    _vals_match(s.diff(1), [None, 1, 1])


# --- abs / round / clip -----------------------------------------------------
def test_abs_round_clip():
    _vals_match(volas.DataFrame({"x": [-1.0, 2.0, -3.0]})["x"].abs(), [1, 2, 3])
    _vals_match(volas.DataFrame({"x": [1.27, 2.34]})["x"].round(1), [1.3, 2.3])
    _vals_match(A.series("f64", "N0").clip(1.5, 2.5), [1.5, 2.0, 2.5])
    # NA is preserved through abs (# C4 propagate).
    assert A.series("f64", "N1").abs().isna().to_list() == [False, True, False]


# --- the 15 math fns: NA propagation + shape + a value anchor --------------
@pytest.mark.parametrize("fn", _MATH)
def test_math_na_propagation_and_shape(fn):
    # 0.5 is inside every fn's domain (acos/asin need [-1,1], ln/sqrt need >0),
    # so the only missing output is the propagated input NA — not a domain NaN.
    s = volas.DataFrame({"x": [0.5, None, 0.5]})["x"]
    out = getattr(s, fn)()
    assert isinstance(out, volas.Series) and len(out) == 3
    assert out.isna().to_list() == [False, True, False], f"T1.{fn}: NA must propagate"


def test_math_value_anchors():
    _vals_match(volas.DataFrame({"x": [1.0, 4.0, 9.0]})["x"].sqrt(), [1.0, 2.0, 3.0])
    _vals_match(volas.DataFrame({"x": [0.0, 1.0]})["x"].exp(), [1.0, math.e])
    _vals_match(volas.DataFrame({"x": [1.0, 10.0]})["x"].log10(), [0.0, 1.0])
    _vals_match(volas.DataFrame({"x": [1.0, math.e]})["x"].ln(), [0.0, 1.0])
