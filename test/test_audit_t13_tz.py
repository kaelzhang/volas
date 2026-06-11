"""Systematic audit — T13 (tz): tz_localize / tz_convert / .tz.

A tz-aware frame must be distinguishable from a naive one, and the round-trip
localize -> convert must carry the zone faithfully.

Cell IDs:  T13.localize/<zone> · T13.convert · T13.naive
"""

from __future__ import annotations

import pytest

import volas


def _dt_frame():
    df = volas.DataFrame({"v": [1.0, 2.0], "t": ["2021-01-01", "2021-01-02"]})
    df["t"] = volas.to_datetime(df["t"])
    return df.set_index("t")


def test_naive_has_no_tz():
    assert _dt_frame().tz is None


def test_localize_non_utc_reports_zone():
    loc = _dt_frame().tz_localize("America/New_York")
    assert loc.tz == "America/New_York"


def test_convert_changes_zone():
    loc = _dt_frame().tz_localize("America/New_York")
    assert loc.tz_convert("Asia/Shanghai").tz == "Asia/Shanghai"


# F13 (findings-ledger): tz_localize('UTC').tz returns None — UTC is conflated
# with naive. tz_convert still works on the localized frame (so it *is* tz-aware
# internally), and a non-UTC zone reports correctly; only the .tz surface drops
# UTC. xfail(strict).
def test_localize_utc_reports_zone():
    loc = _dt_frame().tz_localize("UTC")
    assert loc.tz == "UTC"


# --- the tz model behavioural census (decision 5 = pandas semantics) ---------
def _ns(df):
    return df.reset_index().iloc[:, 0].to_numpy()[0].astype("datetime64[ns]").astype("int64")


def test_localize_anchors_instant():
    """tz_localize: naive -> aware ANCHORS the stored UTC instant (NY 00:00 ->
    05:00Z, +5h EST) — i.e. it changes storage, it is not display-only."""
    naive = _dt_frame()                                  # naive wall-clock 2021-01-01 00:00
    ny = naive.tz_localize("America/New_York")
    assert _ns(ny) == _ns(naive) + 5 * 3600 * 10**9      # +5h to the UTC instant


def test_convert_preserves_instant_shifts_wallclock():
    """tz_convert: aware -> aware keeps the UTC instant, only the display moves."""
    ny = _dt_frame().tz_localize("America/New_York")
    sh = ny.tz_convert("Asia/Shanghai")
    assert _ns(sh) == _ns(ny)                            # instant unchanged
    assert sh.tz == "Asia/Shanghai"


# F-new (decision 5): tz_convert on a *naive* frame must raise (pandas does) —
# currently volas silently re-labels without shifting. xfail(strict).
def test_naive_tz_convert_raises():
    with pytest.raises((TypeError, ValueError)):
        _dt_frame().tz_convert("Asia/Shanghai")


# F25, column-level disposition (D3 waiver): a tz lives on the INDEX (or a
# scalar Timestamp), never on a value column — so to_datetime over offset
# strings yields the correct ABSOLUTE instants (01:00Z == 09:00+08:00) as a
# zone-less column. The zone is retained where D3 places it: the scalar
# (Timestamp('...+08:00').tz == '+08:00', F25-scalar, fixed) and the index
# (set_index + tz_localize/_set_index_tz). pandas instead grows a per-column
# tz dtype — a documented divergence.
def test_offset_string_column_is_correct_instant():
    s = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01 09:00:00+08:00"]})["t"])
    assert s.to_list()[0] == volas.Timestamp("2021-01-01 09:00:00+08:00")  # same instant
    assert "01:00" in str(s.to_list()[0])     # rendered as the UTC wall (no column tz)


# --- V-axis: DST gap / fold (the hardest tz edges) --------------------------
def _at(t):
    df = volas.DataFrame({"v": [1.0]})
    df["t"] = volas.to_datetime(volas.DataFrame({"t": [t]})["t"])
    return df.set_index("t")


def test_dst_gap_nonexistent_rejected():
    """A wall-clock in the spring-forward gap doesn't exist → localize raises
    (volas matches pandas; symmetric with C4 fail-loud)."""
    with pytest.raises((ValueError, KeyError)):
        _at("2021-03-14 02:30:00").tz_localize("America/New_York")  # 02:00->03:00 EDT


# F33 (findings-ledger): a wall-clock in the fall-back fold is AMBIGUOUS (occurs
# twice). pandas demands explicit disambiguation (raises without ambiguous=);
# volas silently picks one — asymmetric with the gap guard above, against
# fail-loud. xfail(strict).
def test_dst_fold_ambiguous_rejected():
    with pytest.raises((ValueError, KeyError)):
        _at("2021-11-07 01:30:00").tz_localize("America/New_York")  # 02:00->01:00 EST
