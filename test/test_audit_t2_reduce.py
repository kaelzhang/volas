"""Systematic audit — T2 (reduce): Series reductions to a scalar.

  sum mean prod min max median count nunique · idxmax idxmin · any all · describe

Value is first-class (P6): the reductions are checked exactly against the
integer-valued N-basis with pandas NA-skip semantics (skipna defaults True, so a
hole is dropped, not propagated). The all-NA / empty edges have defined
identities (sum=0, prod=1, count=0) and missing-valued mean/min/max.

Cell IDs:  T2.<reduction>/D=<d>/N=<n>
"""

from __future__ import annotations

import math

import pytest

import volas
from . import audit_dims as A

_D = ("f64", "f32", "i64", "i32")

# Exact reduction values over the [1,2,3] basis, NA skipped (# pandas skipna).
_VALUES = {
    "sum": {"N0": 6, "N1": 4}, "prod": {"N0": 6, "N1": 3},
    "mean": {"N0": 2.0, "N1": 2.0}, "median": {"N0": 2.0, "N1": 2.0},
    "min": {"N0": 1, "N1": 1}, "max": {"N0": 3, "N1": 3},
    "count": {"N0": 3, "N1": 2}, "nunique": {"N0": 3, "N1": 2},
}
# All-NA (N2) and empty (N3): the algebraic identities are defined; mean/min/max
# /median are missing.
_DEFINED_EDGE = {"sum": 0, "prod": 1, "count": 0, "nunique": 0}
_MISSING_EDGE = ("mean", "min", "max", "median")


def _is_missing(x):
    return x is volas.NA or (isinstance(x, float) and math.isnan(x))


@pytest.mark.parametrize("n", ("N0", "N1"))
@pytest.mark.parametrize("d", _D)
@pytest.mark.parametrize("reduction", list(_VALUES))
def test_reduce_values(reduction, d, n):
    got = getattr(A.series(d, n), reduction)()
    assert float(got) == float(_VALUES[reduction][n]), f"T2.{reduction}/D={d}/N={n}"


@pytest.mark.parametrize("n", ("N2", "N3"))
@pytest.mark.parametrize("reduction", list(_DEFINED_EDGE) + list(_MISSING_EDGE))
def test_reduce_allna_empty(reduction, n):
    got = getattr(A.series("i64", n), reduction)()
    if reduction in _DEFINED_EDGE:
        assert float(got) == float(_DEFINED_EDGE[reduction]), f"T2.{reduction}/N={n}"
    else:
        assert _is_missing(got), f"T2.{reduction}/N={n} should be missing, got {got!r}"


# A reduction with no surviving value returns a *missing numpy scalar*
# (np.float64 nan). This is C2-consistent for the float-typed results
# (mean/median are always float -> NaN-as-NA). Whether int min/max should
# instead yield a typed NA (volas.NA) rather than float nan is an owner decision
# (the contract has reductions return numpy scalars, and numpy int has no NA) —
# F12 in findings-ledger, tracked as open, NOT asserted as a bug.
@pytest.mark.parametrize("reduction", _MISSING_EDGE)
def test_reduce_allna_is_missing_numpy_scalar(reduction):
    got = getattr(A.series("i64", "N2"), reduction)()
    assert isinstance(got, float) and math.isnan(got)


# --- idxmax / idxmin: the index *label* of the extreme (value-faithful) -----
@pytest.mark.parametrize("n", ("N0", "N1"))
@pytest.mark.parametrize("d", _D)
def test_idxmax_idxmin(d, n):
    s = A.series(d, n)               # values 1,(NA),3 at labels 0,1,2
    assert s.idxmax() == 2           # label of the max (3)
    assert s.idxmin() == 0           # label of the min (1)


# --- any / all on bool ------------------------------------------------------
def test_any_all():
    def b(x):  # reductions return a numpy bool scalar
        return bool(x)
    assert b(volas.DataFrame({"x": [True, False, True]})["x"].any()) is True
    assert b(volas.DataFrame({"x": [True, False, True]})["x"].all()) is False
    assert b(volas.DataFrame({"x": [False, False]})["x"].any()) is False
    assert b(volas.DataFrame({"x": [True, True]})["x"].all()) is True
    # skipna default drops the hole.
    assert b(volas.DataFrame({"x": [True, None, True]})["x"].all()) is True


# --- describe: the standard 8-stat summary ---------------------------------
def test_describe_shape():
    d = A.series("f64", "N0").describe()
    assert isinstance(d, volas.Series)
    assert len(d) == 8                       # count mean std min 25% 50% 75% max
    assert d.loc["count"] == 3.0
    assert d.loc["min"] == 1.0 and d.loc["max"] == 3.0


# --- typed-order reductions: min/max/idxmax are ORDER-based, so they serve
# str (lexicographic) and datetime (instant order) — not the f64 funnel. The
# idxmax-datetime case is the historical P1-02 incident site; pinned here.
@pytest.mark.parametrize("d,lo,hi", [
    ("str", "a", "c"),
    ("datetime", volas.Timestamp("2021-01-01"), volas.Timestamp("2021-01-03")),
])
@pytest.mark.parametrize("n", ("N0", "N1"))
def test_typed_order_min_max_idx(d, lo, hi, n):
    s = A.series(d, n)                  # basis a/b/c · 2021-01-01..03, NA skipped
    assert s.min() == lo and s.max() == hi
    assert s.idxmin() == 0 and s.idxmax() == 2
