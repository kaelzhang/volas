"""Systematic audit — T3 (binop): Series arithmetic / comparison / corr·cov.

The operand space is large (LHS-D × RHS-D × N × op), so per SPEC §8 the skeleton
is the *unambiguous core* + the high-risk NA cells, not the full cartesian:

  * value — computed directly from the integer-valued N-basis (1/2/3), so it is
    exact without a float oracle (P6).
  * container/length — Series OP Series -> Series, length preserved (`# C1`).
  * NA — *arithmetic* propagates (any NA operand -> NA, `# C2`), but *comparison*
    follows IEEE NaN semantics (`# O6`): NA==x is False, NA!=x is True, every
    ordering is False — a concrete bool, NOT a propagated NA. The two differ by
    design (volas chose IEEE over pandas-nullable propagation).
  * dtype *family* (int/float/bool) is asserted; exact integer-width promotion
    (i32⊕i32 -> i32 vs i64) and bool-arithmetic dtype are un-ruled design
    questions -> backlog meta-test, not hand-guessed.
  * guards — str ⊕ numeric arithmetic raises (a guard, never a panic — P7).

Cell IDs:  BINOP:<lhsD>op<rhsD>/N=<n> · CMP:<d>/<op> · CORR:<pair>
"""

from __future__ import annotations

import operator

import numpy as np
import pytest

import volas
from . import audit_dims as A

_ARITH_D = ("f64", "f32", "i64", "i32")  # numeric, excl. bool (bool-arith deferred)
ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul,
         "//": operator.floordiv, "/": operator.truediv}
COMPARE = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
           "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def _family(dtype: str) -> str:
    if dtype in ("float64", "float32"):
        return "float"
    if dtype in ("int64", "int32"):
        return "int"
    return dtype  # bool / str / datetime64[ns]


def _arith_family(lhs: str, rhs: str, op: str) -> str:
    if op == "/":
        return "float"                       # true division is always float
    if "f64" in (lhs, rhs) or "f32" in (lhs, rhs):
        return "float"
    return "int"                             # int ⊕ int


# --- arithmetic: value + container + dtype family, dense (N0) ---------------
@pytest.mark.parametrize("op", list(ARITH))
@pytest.mark.parametrize("rhs", _ARITH_D)
@pytest.mark.parametrize("lhs", _ARITH_D)
def test_arith_value(lhs, rhs, op):
    a, b = A.series(lhs, "N0"), A.series(rhs, "N0")
    out = ARITH[op](a, b)
    assert isinstance(out, volas.Series)                 # # C1
    assert len(out) == 3
    assert not any(out.isna().to_list())
    av = [float(x) for x in A._PRESENT[lhs]]
    bv = [float(x) for x in A._PRESENT[rhs]]
    want = [ARITH[op](x, y) for x, y in zip(av, bv)]
    assert [float(x) for x in out.to_list()] == want, f"BINOP:{lhs}{op}{rhs} value"
    assert _family(out.dtype) == _arith_family(lhs, rhs, op), f"BINOP:{lhs}{op}{rhs} dtype family"


# --- NA propagation: any NA operand -> NA result (C2) ----------------------
@pytest.mark.parametrize("op", list(ARITH))
@pytest.mark.parametrize("d", _ARITH_D)
def test_arith_na_propagates(d, op):
    a = A.series(d, "N1")            # [v0, NA, v2]
    b = A.series(d, "N0")            # [v0, v1, v2]
    out = ARITH[op](a, b)
    assert out.isna().to_list() == [False, True, False], "NA must propagate (# C2)"


# --- comparison: bool result, correct truth, NA -> IEEE (not propagated) ----
@pytest.mark.parametrize("op", list(COMPARE))
@pytest.mark.parametrize("d", ("f64", "i64"))
def test_comparison_value(d, op):
    a, b = A.series(d, "N0"), A.series(d, "N0")
    out = COMPARE[op](a, b)
    assert out.dtype == "bool"
    av = [float(x) for x in A._PRESENT[d]]
    want = [COMPARE[op](x, y) for x, y in zip(av, av)]
    assert out.to_list() == want


# Comparison against an NA operand follows IEEE NaN semantics (# O6, contract
# closed by HEAD 06d41bc), NOT pandas-nullable propagation: NA==x is False,
# NA!=x is True, every ordering is False — a concrete bool, never a propagated
# NA. (volas deliberately chose IEEE over pandas here; *arithmetic* still
# propagates NA — see test_arith_na_propagates — the two differ by design.)
@pytest.mark.parametrize("op,na_pos", [
    ("==", False), ("!=", True), ("<", False), ("<=", False), (">", False), (">=", False),
])
def test_comparison_na_is_ieee(op, na_pos):
    a = A.series("i64", "N1")       # [1, NA, 3]
    b = A.series("i64", "N0")       # [1, 2, 3]
    out = COMPARE[op](a, b)
    assert out.dtype == "bool"
    assert out.isna().to_list() == [False, False, False], f"CMP:{op} IEEE result is never NA"
    assert out.to_list()[1] == na_pos, f"CMP:{op} NA-position must be IEEE {na_pos}"


# --- scalar RHS: the input-representation (I-rep) axis ----------------------
@pytest.mark.parametrize("d", _ARITH_D)
def test_scalar_rhs(d):
    s = A.series(d, "N0")
    out = s + 10
    assert [float(x) for x in out.to_list()] == [v + 10 for v in (1.0, 2.0, 3.0)]
    # a numpy scalar (np.int64) takes a different pyo3 extraction path but the
    # same logical value -> identical result (I-rep equivalence).
    assert (s + np.int64(10)).to_list() == out.to_list()


# F7 (findings-ledger): an NA scalar operand is rejected with TypeError instead
# of poisoning the column to all-NA — inconsistent with column-NA arithmetic,
# which propagates correctly. xfail(strict).
def test_scalar_na_operand():
    s = A.series("i64", "N0")
    assert (s + volas.NA).isna().to_list() == [True, True, True]  # # C2 / # pandas


# --- guards: cross-family arithmetic raises (never panics, P7) --------------
def test_arith_guard_str_numeric():
    with pytest.raises((TypeError, ValueError)):
        A.series("str", "N0") + A.series("i64", "N0")


# --- corr / cov: two-Series reductions to a float scalar -------------------
def test_corr_cov():
    s = A.series("i64", "N0")                       # [1, 2, 3]
    rev = volas.DataFrame({"y": [3, 2, 1]})["y"]    # perfectly anti-correlated
    assert s.corr(s) == pytest.approx(1.0)
    assert s.corr(rev) == pytest.approx(-1.0)
    assert s.cov(s) == pytest.approx(1.0)           # sample var of 1,2,3
    assert s.cov(rev) == pytest.approx(-1.0)


def test_binop_promotion_tripwire():
    """Un-ruled dtype-promotion questions (SPEC §5, findings-ledger) — owner
    decision pending. NOT an oracle: this pins *today's* result so a silent
    promotion change trips the wire and forces a conscious ruling/diff.

    Open questions pinned: (1) integer-width — i32⊕i32 stays i32 (narrowest, like
    the fillna F1 ruling) vs pandas-nullable's widen-to-Int64; (2) f32 mixing;
    (3) bool arithmetic dtype (volas keeps bool where pandas gives int).
    """
    assert (A.series("i32", "N0") + A.series("i32", "N0")).dtype == "int32"   # narrowest
    assert (A.series("i32", "N0") + A.series("i64", "N0")).dtype == "int64"   # widen to max
    assert (A.series("f32", "N0") + A.series("f32", "N0")).dtype == "float32"
    assert (A.series("f32", "N0") + A.series("f64", "N0")).dtype == "float64"
    assert (A.series("f32", "N0") + A.series("i32", "N0")).dtype == "float64"  # int forces wide float
    b = volas.DataFrame({"x": [True, False, True]})["x"]
    assert (b + b).dtype == "bool"                                             # vs pandas int
