"""Systematic audit — T5 (na-mask): isna / notna.

isna is the NA *oracle* the rest of the audit leans on, so it must detect every
missing form — validity-NA, in-band float NaN, datetime NaT — while treating
±inf as ordinary values (SPEC §4.5: `isna(inf)==False` is pinned here).

Cell IDs:  T5.isna/D=<d>/N=<n> · T5.isna.boundary
"""

from __future__ import annotations

import pytest

import volas
from . import audit_dims as A

# the NA mask each (d, n) fixture is built to carry.
_MASK = {
    "N0": [False, False, False], "N1": [False, True, False],
    "N2": [True, True, True], "N3": [], "N4": [False],
}


@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_isna_matches_state(d, n):
    s = A.series(d, n)
    assert s.isna().to_list() == _MASK[n], f"T5.isna/D={d}/N={n}"
    assert s.isna().dtype == "bool"


@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_notna_is_complement(d, n):
    s = A.series(d, n)
    isna, notna = s.isna().to_list(), s.notna().to_list()
    assert notna == [not m for m in isna], f"T5.notna/D={d}/N={n}"


def test_isna_boundaries():
    # in-band float NaN is NA; ±inf are ordinary finite-direction values, not NA.
    b = volas.DataFrame({"x": [float("inf"), float("-inf"), float("nan"), 1.0]})["x"]
    assert b.isna().to_list() == [False, False, True, False]   # # pandas / §4.5
    # datetime NaT is NA.
    assert A.series("datetime", "N1").isna().to_list() == [False, True, False]
