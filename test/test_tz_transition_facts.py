"""Timezone facts come from the zone's transition data, never from heuristics.

`dst()` is the zone's own DST component at the instant (southern-hemisphere
summers, a 30-minute DST, Dublin's negative winter DST), and the first real
instant of a wall-clock period that starts inside a DST gap is the transition
instant itself, at second precision. zoneinfo / pandas 3.0.3 are the oracles.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import volas


@pytest.mark.parametrize("zone", [
    "Australia/Sydney", "Australia/Lord_Howe", "Europe/Dublin", "America/New_York",
    "Asia/Shanghai", "Europe/London",
])
@pytest.mark.parametrize("when", ["2024-01-15 12:00", "2024-07-15 12:00"])
def test_dst_is_the_zone_dst_component(zone, when):
    want = dt.datetime.fromisoformat(when).replace(tzinfo=ZoneInfo(zone)).dst()
    assert volas.Timestamp(when, tz=zone).dst() == want
    assert pd.Timestamp(when, tz=zone).dst() == want


def test_period_start_inside_a_gap_is_the_transition_instant():
    # Monrovia moved from -00:44:30 to UTC at 1972-01-07 00:00 local: that wall
    # clock does not exist; the day's first real instant is 00:44:30Z
    ns = np.datetime64("1972-01-07T00:44:40", "ns").astype("int64")
    df = volas.DataFrame({"close": [1.0]}, index=np.array([ns], dtype="datetime64[ns]"))
    df = df.tz_localize("UTC").tz_convert("Africa/Monrovia")
    label = df.cumulate("1d").index[0].value
    assert label == pd.Timestamp("1972-01-07 00:44:30", tz="UTC").value
