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


# --- OBS: int > i64::MAX has no volas int dtype (no uint64) -> lossy f64 ------
def test_int_overflow_is_lossy_float_obs():
    """OBS (owner-confirm pending, not asserted as a bug): a python int beyond
    i64::MAX falls to float64 (lossy) since volas has no uint64; pandas uses
    uint64. C4 (no silent precision loss) would prefer an error. Pinned visible."""
    assert _col([2 ** 63]).dtype == "float64"
