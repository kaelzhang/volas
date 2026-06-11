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
@pytest.mark.xfail(reason="F13: tz_localize('UTC').tz is None (UTC conflated with naive)", strict=True)
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
@pytest.mark.xfail(reason="naive.tz_convert must raise (pandas TypeError), volas silently relabels", strict=True)
def test_naive_tz_convert_raises():
    with pytest.raises((TypeError, ValueError)):
        _dt_frame().tz_convert("Asia/Shanghai")


# F25 (findings-ledger): constructing from a '+08:00' offset string parses to UTC
# but DROPS the tz and shows the wrong naive wall-clock (01:00 not 09:00).
@pytest.mark.xfail(reason="F25: +08:00 offset string drops tz, shows wrong naive wall-clock", strict=True)
def test_offset_string_keeps_tz():
    s = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01 09:00:00+08:00"]})["t"])
    # the wall-clock should remain 09:00 with an attached zone, not collapse to 01:00.
    assert "09:00" in str(s.to_list()[0])


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
@pytest.mark.xfail(reason="F33: ambiguous DST-fold time silently resolved (gap rejects, fold doesn't)", strict=True)
def test_dst_fold_ambiguous_rejected():
    with pytest.raises((ValueError, KeyError)):
        _at("2021-11-07 01:30:00").tz_localize("America/New_York")  # 02:00->01:00 EST
