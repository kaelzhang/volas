"""One instant <-> civil / text boundary, owned by the timezone.

Formatting keeps the nanoseconds and the offset of the instant, parsing reads
every offset form the formatter writes (second-precision offsets included),
offsets follow one strict grammar, a value that carries a zone keeps it, and a
float epoch converts by one exact rule everywhere. pandas 3.0.3 is the oracle.
"""

import datetime as dt
import os
import tempfile
from fractions import Fraction
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import volas

T = volas.Timestamp


# --- strftime ---------------------------------------------------------------------

@pytest.mark.parametrize("tz", [None, "+08:00", "UTC", "America/New_York", "Asia/Kolkata"])
@pytest.mark.parametrize("fmt", ["%f", "%z", "%Z", "%s", "%Y-%m-%d %H:%M:%S.%f%z", "%%f %%z"])
def test_strftime_matches_pandas(tz, fmt):
    for when in ["2024-01-01 12:00:00.123456789", "1900-01-01 00:00:00"]:
        p = pd.Timestamp(when, tz=tz)
        # pandas' %s reads the wall clock in the HOST's zone; the oracle is the
        # instant's real epoch second (the wall clock as UTC for a naive value)
        want = str(p.value // 10**9) if fmt == "%s" else p.strftime(fmt)
        assert T(when, tz=tz).strftime(fmt) == want, (when, tz, fmt)


def test_series_dt_strftime_keeps_sub_seconds_and_rejects_bad_formats():
    s = volas.DataFrame({"t": np.array(["2024-01-01T12:00:00.123456789"], dtype="datetime64[ns]")})["t"]
    assert s.dt.strftime("%S.%f").to_list() == ["00.123456"]
    assert s.dt.strftime("%z|%Z").to_list() == ["+0000|UTC"]  # a column holds UTC instants
    with pytest.raises(ValueError, match="invalid strftime"):
        s.dt.strftime("%Q")


# --- offsets: one strict grammar, second precision --------------------------------

@pytest.mark.parametrize("spec", ["+-1:00", "+01:-30", "+-600000:00", "+1日", "+24:00", "+05:60", "+1:5", "+ 5"])
def test_malformed_offsets_are_rejected(spec):
    with pytest.raises(ValueError):
        T(0, tz=spec)


@pytest.mark.parametrize("spec, secs", [
    ("+05:21:10", 19270), ("+052110", 19270), ("-0930", -34200), ("+8", 28800), ("+08", 28800),
])
def test_offset_forms(spec, secs):
    t = T(0, tz=spec)
    assert t.utcoffset() == dt.timedelta(seconds=secs)


def test_second_precision_offsets_round_trip():
    k = T("1900-01-01", tz="Asia/Kolkata")
    assert k.isoformat() == pd.Timestamp("1900-01-01", tz="Asia/Kolkata").isoformat()
    assert T(k.isoformat()).value == k.value
    assert T(k.isoformat()).tz == "+05:21:10"
    assert T(0, tz="+05:21:10").tz == "+05:21:10"


def test_sub_second_tzinfo_offsets_are_rejected_not_truncated():
    half = dt.timezone(dt.timedelta(seconds=30.5))
    with pytest.raises(ValueError):
        T("2024-01-01", tz=half)
    with pytest.raises(ValueError):
        T(dt.datetime(2024, 1, 1, tzinfo=half))
    whole = dt.timezone(dt.timedelta(seconds=30))
    assert T(dt.datetime(2024, 1, 1, tzinfo=whole)).value == pd.Timestamp(dt.datetime(2024, 1, 1, tzinfo=whole)).value


# --- values that carry a zone keep it ------------------------------------------------

def test_timestamp_from_an_aware_value_keeps_its_zone():
    ny = T("2024-01-01 09:00", tz="America/New_York")
    assert T(ny).tz == "America/New_York" and T(ny).hour == 9
    assert T(ny).tz_convert("UTC").value == ny.value
    aware = dt.datetime(2024, 1, 1, 9, tzinfo=ZoneInfo("America/New_York"))
    assert T(aware).tz == "America/New_York" and T(aware).value == ny.value
    p = pd.Timestamp("2024-01-01 09:00:00.000000123", tz="Asia/Shanghai")
    assert T(p).tz == "Asia/Shanghai" and T(p).value == p.value
    assert T(dt.datetime(2024, 1, 1, 9)).tz is None


def test_aware_to_pydatetime_is_aware():
    ny = T(1730611800 * 10**9, tz="America/New_York")  # the first 01:30 (EDT)
    d = ny.to_pydatetime()
    # (an ambiguous wall clock never compares equal across zones, PEP 495)
    assert d.tzinfo is not None and d.timestamp() == ny.value / 10**9
    assert d.utcoffset() == pd.Timestamp(ny.value, tz="America/New_York").utcoffset()
    later = T(ny.value + 3600 * 10**9, tz="America/New_York").to_pydatetime()  # the repeated 01:30
    assert later.utcoffset() == dt.timedelta(hours=-5) and later.fold == 1
    assert T("2024-01-01 12:00:00.123456789").to_pydatetime() == dt.datetime(2024, 1, 1, 12, 0, 0, 123456)


# --- awareness-aware comparison, hash consistent with datetime -----------------------

def test_naive_and_aware_never_compare_equal_or_ordered():
    naive, aware = T(0), T(0, tz="UTC")
    assert (naive == aware) is False and (naive != aware) is True
    with pytest.raises(TypeError, match="tz-naive and tz-aware"):
        naive < aware
    with pytest.raises(TypeError):
        aware >= np.datetime64(0, "ns")
    assert (naive == np.datetime64(0, "ns")) is True


def test_comparison_with_non_instants_follows_pandas():
    t = T("2024-01-01")
    assert (t == None) is False and (t != None) is True  # noqa: E711
    assert (t == dt.date(2024, 1, 1)) is False
    for other in [None, dt.date(2024, 1, 2)]:
        with pytest.raises(TypeError):
            t < other


def test_string_operands_carry_their_awareness():
    # a datetime string stays a comparable operand (volas contract); its offset
    # suffix makes it aware, so the naive / aware rule applies to it too
    naive, aware = T("2024-01-01 09:00"), T("2024-01-01 09:00", tz="+08:00")
    assert naive == "2024-01-01 09:00" and aware == "2024-01-01 09:00+08:00"
    assert (aware == "2024-01-01 09:00") is False
    with pytest.raises(TypeError, match="tz-naive and tz-aware"):
        naive < "2024-01-01 10:00+08:00"


def test_hash_agrees_with_datetime_equality():
    for t in [T("2024-01-01"), T("2024-01-01 09:00", tz="Asia/Shanghai"), T("2024-01-01 00:00:00.000001")]:
        d = t.to_pydatetime()
        assert t == d and hash(t) == hash(d) and d in {t}
    odd = T(1)  # a sub-microsecond instant hashes like pandas (by value)
    assert hash(odd) == hash(pd.Timestamp(1))


# --- one float-epoch rule -------------------------------------------------------------

@pytest.mark.parametrize("value, unit", [
    (1704067200.000001, "s"), (1577836800.25, "s"), (1704067200.5, "s"), (1.5, "ms"),
    (-1.5, "s"), (0.5, "ns"), (1.5, "ns"), (2.5, "ns"), (123.456789, "us"), (1e-12, "s"),
])
def test_float_epoch_is_the_exact_nearest_nanosecond(value, unit):
    # the float's exact binary value × the unit, rounded half-to-even — pandas'
    # result for every coarser unit; pandas truncates a fractional unit='ns' input
    # instead (a documented divergence: volas rounds every unit the same way)
    exact = round(Fraction(value) * {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit])
    got = volas.to_datetime([value], unit=unit).to_numpy().view("i8")[0]
    assert got == exact
    if unit != "ns":
        assert got == pd.to_datetime([value], unit=unit).asi8[0]


def test_read_csv_float_epoch_index_keeps_the_fraction():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.csv")
        with open(p, "w") as f:
            f.write("ts,x\n1704067200.25,1\n1704067200.75,2\n")
        df = volas.read_csv(p, index_col="ts", date_unit="s")
        assert df.index.view("i8").tolist() == [1704067200250000000, 1704067200750000000]


# --- CSV: tz tagging and offset-preserving output ---------------------------------------

@pytest.mark.parametrize("tz", ["+08:00", "America/New_York", "UTC"])
def test_read_csv_tz_tags_the_index(tz):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        with open(p, "w") as f:
            f.write("time,x\n2024-01-01 12:00:00,1\n")
        df = volas.read_csv(p, index_col="time", tz=tz)
        assert df.tz == tz
        assert df.index.view("i8")[0] == pd.Timestamp("2024-01-01 12:00:00", tz=tz).value


def test_to_csv_writes_the_offset_of_an_aware_index():
    utc = pd.DatetimeIndex(["2024-11-03 05:30", "2024-11-03 06:30"]).tz_localize("UTC")
    pdf = pd.DataFrame({"x": [1, 2]}, index=utc.tz_convert("America/New_York"))
    df = volas.DataFrame.from_pandas(pdf)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "o.csv")
        df.to_csv(p)
        text = open(p).read()
    assert "2024-11-03 01:30:00-04:00" in text and "2024-11-03 01:30:00-05:00" in text
