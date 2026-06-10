"""Contract D2 — i64::MIN is the NaT sentinel, not a constructable instant. A
missing instant is volas.NA, so the public time paths must reject raw i64::MIN
rather than turn it into a bucketable timestamp or a Timestamp whose repr says
'NaT' while its .year/.month/.day expose a real 1677 civil date (internally
inconsistent)."""

import pytest
import volas

NAT = -9223372036854775808  # i64::MIN


def test_timestamp_from_nat_sentinel_raises():
    with pytest.raises(Exception):
        volas.Timestamp(NAT)


def test_timeframe_unify_nat_raises():
    with pytest.raises(Exception):
        volas.TimeFrame.D1.unify(NAT)


# --- regression: real timestamps unaffected ----------------------------------

def test_valid_timestamp_still_works():
    t = volas.Timestamp(0)
    assert (t.year, t.month, t.day) == (1970, 1, 1)
    t2 = volas.Timestamp("2021-03-04 09:30:00")
    assert (t2.year, t2.month, t2.day) == (2021, 3, 4)


def test_valid_unify_still_works():
    a = volas.TimeFrame.D1.unify("2021-01-01 09:30:00")
    b = volas.TimeFrame.D1.unify("2021-01-01 16:00:00")
    c = volas.TimeFrame.D1.unify("2021-01-02 09:30:00")
    assert a == b and a != c  # same day -> same key
