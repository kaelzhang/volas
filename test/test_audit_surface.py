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
_SERIES_ALIGN = {
    "dt", "tz_localize", "tz_convert",                          # datetime (F22/F27)
    "value_counts", "mode", "isin", "between", "replace",
    "nlargest", "nsmallest", "drop_duplicates", "duplicated",
    "is_monotonic_increasing", "is_monotonic_decreasing", "is_unique",
    "reset_index", "sort_index", "rename", "copy", "to_frame", "to_dict", "items",
    "iat", "at",
}
_DATAFRAME_ALIGN = {
    # F8: core per-column reductions (volas has only count/sem/skew/kurt/describe)
    "sum", "mean", "min", "max", "prod", "var", "std", "median", "quantile",
    "idxmax", "idxmin", "all", "any", "nunique",
    # B-cluster utilities (pct_change excluded -> directive `change`)
    "value_counts", "isin", "replace", "nlargest", "nsmallest",
    "drop_duplicates", "duplicated", "mode",
}


@pytest.mark.parametrize("m", sorted(_SERIES_ALIGN))
@pytest.mark.xfail(reason="P8 align-backlog: owner-confirmed Series API not yet implemented", strict=True)
def test_series_align_backlog(m):
    assert hasattr(_S, m)


@pytest.mark.parametrize("m", sorted(_DATAFRAME_ALIGN))
@pytest.mark.xfail(reason="P8 align-backlog: owner-confirmed DataFrame API not yet implemented", strict=True)
def test_dataframe_align_backlog(m):
    assert hasattr(_DF, m)


def test_align_backlog_is_actually_missing():
    """Sanity: every `align` member is genuinely absent today (else move it)."""
    s_present = [m for m in _SERIES_ALIGN if hasattr(_S, m)]
    d_present = [m for m in _DATAFRAME_ALIGN if hasattr(_DF, m)]
    # surfaced via the strict-xfail above too, but asserted here as one clear line.
    assert not s_present and not d_present, f"already present: Series={s_present} DataFrame={d_present}"


def test_surface_drift_tripwire():
    """Coarse anchor (pandas 3.0.x pinned): the count of pandas members volas
    lacks. A new pandas method or a volas add/remove shifts it -> re-disposition.
    (Full name-freeze for Series/DataFrame is a deferred follow-up; the align set
    above + this tripwire catch the changes that matter.)"""
    missing = {
        "Series": len(_pub(pd.Series(dtype="float64")) - _pub(_S)),
        "DataFrame": len(_pub(pd.DataFrame()) - _pub(_DF)),
    }
    assert missing == {"Series": 144, "DataFrame": 155}, f"surface drift: {missing}"
