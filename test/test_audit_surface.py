"""Systematic audit — global surface alignment (P8 §6.8, Step-2 disposition as tests).

The surface differential `pub(pd.X) − pub(volas.X)` is dispositioned in
`alignment-disposition-2026-06-11.md`. Here the `align` backlog (what volas
SHOULD have — owner-confirmed B-cluster + datetime + DataFrame reductions) is
pinned as strict-xfail so implementing any one flips loudly; a coarse drift
tripwire forces re-disposition when pandas's surface or volas's changes.

`out-of-scope` (IO / viz / MultiIndex / method-arithmetic aliases / rolling-ewm
-apply-groupby) is deliberately NOT enumerated as gaps — see the disposition doc.
Timestamp's surface is fully name-frozen in test_audit_datetime.py.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import volas

_pub = lambda o: {n for n in dir(o) if not n.startswith("_")}
_S = volas.DataFrame({"x": [1.0, 2.0, 3.0]})["x"]
_DF = volas.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

# owner-confirmed `align` (disposition §2 B-cluster + datetime symmetry).
# Note: `pct_change` is NOT here — owner ruled it out-of-scope (the `change`
# directive already computes it; one indicator path). `dt` stays backlog but its
# API shape is open (owner dislikes pandas's Series-only .dt asymmetry; volas
# should expose datetime components symmetrically on Series AND DataFrame).
_SERIES_IMPLEMENTED = {
    "tz_localize", "tz_convert",                                # F27, landed
    "value_counts", "mode", "isin", "between", "replace",
    "nlargest", "nsmallest", "drop_duplicates", "duplicated",
    "is_monotonic_increasing", "is_monotonic_decreasing", "is_unique",
    "reset_index", "sort_index", "rename", "copy", "to_frame", "to_dict", "items",
    "iat", "at",
    "rolling", "ewm", "expanding", "interpolate",
}
# `dt` stays the one backlog item BY OWNER DECISION: pandas's Series-only .dt
# asymmetry is disliked; volas will expose datetime components symmetrically on
# Series AND DataFrame — the API shape is owner-TBD, so it is not implemented yet.
_SERIES_BACKLOG = {"dt"}
_DATAFRAME_IMPLEMENTED = {
    "sum", "mean", "min", "max", "prod", "var", "std", "median", "quantile",
    "idxmax", "idxmin", "all", "any", "nunique",
    "value_counts", "isin", "replace", "nlargest", "nsmallest",
    "drop_duplicates", "duplicated", "mode",
    "rolling", "ewm", "expanding", "interpolate",
}


@pytest.mark.parametrize("m", sorted(_SERIES_IMPLEMENTED))
def test_series_align_implemented(m):
    assert hasattr(_S, m)


@pytest.mark.parametrize("m", sorted(_DATAFRAME_IMPLEMENTED))
def test_dataframe_align_implemented(m):
    assert hasattr(_DF, m)


def test_datetime_component_access_deferred_by_owner():
    """`dt` is deliberately NOT implemented yet (owner decision): pandas's
    Series-only `.dt` makes Series and DataFrame usage inconsistent; volas will
    expose datetime components SYMMETRICALLY on both — the API shape is
    owner-TBD. This pins the deferral so it stays a conscious decision."""
    assert sorted(_SERIES_BACKLOG) == ["dt"]
    assert not hasattr(_S, "dt")


def _missing_hash(pd_obj, vol_obj):
    names = sorted(_pub(pd_obj) - _pub(vol_obj))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def test_surface_drift_snapshot():
    """Name-set snapshot (not just count, per review): the SHA of the sorted set
    of pandas members volas lacks. A +1/-1 swap (a method removed AND another
    added) leaves the count unchanged but changes the set -> trips here, forcing
    re-disposition. (pandas 3.0.x pinned; rerun the differential to see the diff.)"""
    snap = {
        "Series": _missing_hash(pd.Series(dtype="float64"), _S),
        "DataFrame": _missing_hash(pd.DataFrame(), _DF),
    }
    assert snap == {"Series": "3cd8c4b87aca0eda", "DataFrame": "c4718f392b1c1266"}, \
        f"surface name-set drift (recompute differential to locate): {snap}"
