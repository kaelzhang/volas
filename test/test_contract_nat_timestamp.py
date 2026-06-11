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


def test_timeframe_nat_bar_rejected():
    # unify was removed from the public surface (§6.6); the NaT guard is
    # observed through the real cumulation entry instead.
    df = volas.DataFrame({"close": [1.0]})
    df["t"] = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01"]})["t"])
    with pytest.raises(Exception):
        volas.DataFrame({"close": [1.0, 2.0]}, time_frame="1d")


# --- regression: real timestamps unaffected ----------------------------------

def test_valid_timestamp_still_works():
    t = volas.Timestamp(0)
    assert (t.year, t.month, t.day) == (1970, 1, 1)
    t2 = volas.Timestamp("2021-03-04 09:30:00")
    assert (t2.year, t2.month, t2.day) == (2021, 3, 4)


def test_same_day_bars_share_a_bucket():
    # the day-bucket invariant unify used to expose, observed through cumulate:
    # two intraday bars on the same day fold to one daily row, a third day adds one.
    def daily(ts, vals):
        df = volas.DataFrame({"close": vals, "t": list(ts)})
        df["t"] = volas.to_datetime(df["t"])
        return df.set_index("t").cumulate("1d")
    same_day = daily(["2021-01-01 09:30:00", "2021-01-01 16:00:00"], [1.0, 2.0])
    assert same_day.shape[0] == 1                    # same day -> same bucket
    two_days = daily(["2021-01-01 09:30:00", "2021-01-02 09:30:00"], [1.0, 2.0])
    assert two_days.shape[0] == 2                    # next day -> a new bucket
