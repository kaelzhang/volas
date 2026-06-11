"""Systematic audit — numeric input-representation + value-boundary census (P8 §6.8).

Generalises the per-parameter value-category census from datetime to the numeric
surface: construction element-types, scalar operands, and the V-axis boundaries
(2^53 / ±inf / i64 limits). Mostly *confirms* volas is sound here (unlike
datetime) — which is itself the deliverable: solid behaviour pinned against
regression, with the few real divergences flagged.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import volas
from . import audit_irep as I

_col = lambda xs: volas.DataFrame({"x": xs})["x"]

# --- construction: element-type I-rep (dtype inference vs pandas) ------------
# np.int32 -> int64 is a deliberate volas divergence (it infers its default int
# from any int-like element; pandas preserves int32). Pinned, not a bug.
_CONSTRUCT_DTYPE = {
    "py-int": "int64", "py-float": "float64", "np-int64": "int64",
    "np-int32": "int64", "np-float64": "float64", "np-bool": "bool",
    "mixed-int-float": "float64", "np-array-i64": "int64", "np-array-f64": "float64",
}


@pytest.mark.parametrize("label,xs", I.numeric_list_inputs(),
                         ids=[l for l, _ in I.numeric_list_inputs()])
def test_numeric_construct_irep(label, xs):
    assert _col(xs).dtype == _CONSTRUCT_DTYPE[label], f"construct {label}"


# --- scalar-operand I-rep (Series + every numeric scalar form) --------------
@pytest.mark.parametrize("label,val", I.numeric_scalars(),
                         ids=[l for l, _ in I.numeric_scalars()])
def test_numeric_scalar_operand_irep(label, val):
    got = (_col([1, 2, 3]) + val).to_list()
    want = (pd.Series([1, 2, 3]) + val).tolist()
    assert [float(x) for x in got] == [float(x) for x in want], f"s + {label}"


# --- V-axis: integer boundaries (exact, no truncation — 2^53 is the trap) ----
@pytest.mark.parametrize("label,val", I.V_INT, ids=[l for l, _ in I.V_INT])
def test_int_boundary_exact(label, val):
    s = _col([val])
    assert s.dtype == "int64"
    assert s.to_list()[0] == val, f"int boundary {label} truncated"


# --- V-axis: float boundaries (dtype + isna only for nan) -------------------
@pytest.mark.parametrize("label,val", I.V_FLOAT, ids=[l for l, _ in I.V_FLOAT])
def test_float_boundary(label, val):
    s = _col([val, 1.0])
    assert s.dtype == "float64"
    # only nan is NA; ±inf / -0 / subnormal / big are ordinary values (§4.5).
    assert s.isna().to_list()[0] is (isinstance(val, float) and math.isnan(val))


# --- V-axis: boundary semantics in reductions -------------------------------
def test_boundary_reduction_semantics():
    # 2^53 sum stays exact (int64 path, not a lossy float accumulation).
    assert _col([2 ** 53, 1]).sum() == 2 ** 53 + 1
    # ±inf are values: max sees +inf, min sees -inf; nan is skipped.
    s = _col([1.0, float("inf"), float("-inf"), float("nan")])
    assert s.max() == float("inf") and s.min() == float("-inf")
    assert s.isna().to_list() == [False, False, False, True]


# --- F32: int with no exact volas dtype must RAISE (decision: C4, not lossy) --
# 2^63 / 2^63+1 currently both collapse to 9.223372036854776e+18 (silent
# precision loss). pandas avoids loss via uint64/object; volas has neither, so
# per C4 it must raise (suggest explicit dtype='float64'). xfail(strict).
@pytest.mark.parametrize("label,val", I.V_INT_OVERFLOW, ids=[l for l, _ in I.V_INT_OVERFLOW])
@pytest.mark.xfail(reason="F32: int > i64::MAX should raise (C4), volas silently demotes to lossy f64", strict=True)
def test_int_overflow_raises(label, val):
    with pytest.raises((OverflowError, ValueError)):
        _col([val])


# F41 (decision: C4): Decimal has no volas dtype (no object); silent -> float64
# loses the exact-decimal intent. Must raise. xfail(strict).
@pytest.mark.xfail(reason="F41: Decimal should raise (C4 no object), volas silently -> float64", strict=True)
def test_decimal_construct_raises():
    from decimal import Decimal
    with pytest.raises((TypeError, ValueError)):
        _col([Decimal("1.5")])


# --- V_STR: string value boundaries (now incl comma/quote/newline) -----------
# str construction is sound: every boundary (incl RFC-4180 traps) constructs as
# `str`, and sort/unique/compare/isna behave correctly. `.str` accessor is
# intentionally absent (string manipulation out-of-scope). Pinned vs regression.
@pytest.mark.parametrize("label,val", I.V_STR, ids=[l for l, _ in I.V_STR])
def test_str_boundary_construct(label, val):
    assert _col([val]).dtype == "str"
    assert _col([val]).to_list()[0] == val          # value preserved exactly


def test_str_boundary_semantics():
    s = _col(["b", "a", "", "b", "日本"])
    assert s.sort_values().to_list() == ["", "a", "b", "b", "日本"]  # empty first, unicode ordered
    assert sorted(s.unique().to_list()) == ["", "a", "b", "日本"]
    assert _col(["", "a"]).isna().to_list() == [False, False]      # empty string is NOT NA
    assert (_col(["a", "b"]) == "a").to_list() == [True, False]


def test_str_accessor_out_of_scope():
    """`.str` (string-manipulation accessor) is deliberately not implemented —
    string ops are out-of-scope for a typed numeric/quant surface."""
    assert not hasattr(_col(["a"]), "str")
