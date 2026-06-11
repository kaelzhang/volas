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
