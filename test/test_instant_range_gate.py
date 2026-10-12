"""The instant axis has ONE checked range gate.

Every path that builds a datetime64[ns] instant — a parsed string, an epoch
number, a NumPy ``datetime64`` / ``timedelta64`` in any unit, an Arrow
timestamp / date, offset or frequency arithmetic, a replaced component — lands
inside ``[-2**63 + 1, 2**63 - 1]`` or raises. Nothing wraps into a far-away
instant, saturates to the range edge, or collapses onto the ``NaT`` sentinel
(``-2**63``), which is a missing value, never a constructable instant.
"""


import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import volas

MAX = 2**63 - 1
MIN = -(2**63) + 1  # the smallest instant; -2**63 is the NaT sentinel
DAY = 86_400 * 10**9


def _col(values):
    return volas.DataFrame({"t": values})["t"]


# --- fixed-offset arithmetic --------------------------------------------------

def test_fixed_offset_civil_view_of_the_last_instant():
    t = volas.Timestamp(MAX, tz="+08:00")
    assert t.isoformat() == "2262-04-12T07:47:16.854775807+08:00"
    assert (t.year, t.month, t.day, t.hour) == (2262, 4, 12, 7)
    assert volas.Timestamp(MIN, tz="-08:00").year == 1677


def test_fixed_offset_wall_past_the_range_raises():
    # 2262-04-11 23:00 at -08:00 is 2262-04-12 07:00Z, past the last instant
    with pytest.raises(OverflowError):
        volas.Timestamp("2262-04-11 23:00", tz="-08:00")
    with pytest.raises(OverflowError):
        volas.Timestamp("1677-09-21 00:30", tz="+08:00")


# --- frequency arithmetic -----------------------------------------------------

@pytest.mark.parametrize("value, op, freq", [
    (MAX, "ceil", "D"),
    (MIN, "floor", "D"),
    (MAX, "round", "D"),
    (0, "floor", "106752D"),           # the span itself exceeds the range
    (0, "floor", "36028797018963968s"),
])
def test_rounding_past_the_range_raises(value, op, freq):
    with pytest.raises(OverflowError):
        getattr(volas.Timestamp(value), op)(freq)


def test_rounding_near_the_range_edge_is_exact():
    unit = 100_000 * DAY
    assert volas.Timestamp(6 * 10**18).round("100000D").value == unit
    assert volas.Timestamp(MAX).floor("D").value == MAX - MAX % DAY
    assert volas.Timestamp(MIN).ceil("D").value == MIN - MIN % DAY + DAY


def test_series_dt_rounding_past_the_range_raises():
    s = _col(np.array([MAX], dtype="datetime64[ns]"))
    with pytest.raises(OverflowError):
        s.dt.ceil("D")
    with pytest.raises(OverflowError):
        s.dt.round("D")
    assert s.dt.floor("D").to_numpy().view("i8")[0] == MAX - MAX % DAY


# --- replace ------------------------------------------------------------------

def test_replace_rejects_components_that_do_not_fit():
    t = volas.Timestamp("2024-01-01")
    with pytest.raises(ValueError):
        t.replace(year=2024 + 2**32)
    with pytest.raises(ValueError):
        t.replace(month=2**32 + 1)


def test_replace_past_the_range_raises():
    with pytest.raises(OverflowError):
        volas.Timestamp("2000-01-11 23:47:16.999999999").replace(year=2262, month=4)


# --- epoch floats and the sentinel ---------------------------------------------

def test_float_epoch_bounds_are_exclusive_and_never_nat():
    with pytest.raises(OverflowError):
        volas.to_datetime([float(2**63)], unit="ns")
    with pytest.raises(OverflowError):
        volas.to_datetime([-float(2**63)], unit="ns")


def test_integer_epoch_onto_the_sentinel_raises():
    with pytest.raises(OverflowError):
        volas.to_datetime(np.array([-(2**63)], dtype="int64"), unit="ns")
    with pytest.raises(OverflowError):
        volas.to_datetime(np.array([10**10], dtype="int64"), unit="s")


# --- string parse ---------------------------------------------------------------

def test_first_instant_round_trips_through_its_text():
    t = volas.Timestamp(MIN)
    assert volas.Timestamp(str(t)).value == MIN


def test_parse_onto_the_sentinel_raises():
    with pytest.raises(OverflowError):
        volas.Timestamp("1677-09-21T00:12:43.145224192Z")


# --- NumPy units ---------------------------------------------------------------

def test_numpy_scalar_units_scale_checked():
    assert volas.Timestamp(np.datetime64("2024-01-01", "D")).value == 1_704_067_200 * 10**9
    assert volas.Timestamp(np.datetime64(1000, "ps")).value == 1
    assert volas.Timestamp(np.datetime64("2024-03", "M")) == volas.Timestamp("2024-03-01")
    assert volas.Timestamp(np.datetime64("2024", "Y")) == volas.Timestamp("2024-01-01")
    with pytest.raises(OverflowError):
        volas.Timestamp(np.datetime64("2500-01-01", "D"))
    with pytest.raises(ValueError):
        volas.Timestamp(np.datetime64(1001, "ps"))  # not a whole nanosecond


def test_numpy_timedelta_operand_scales_checked():
    t = volas.Timestamp(0)
    assert (t + np.timedelta64(2, "D")).value == 2 * DAY
    assert (t - np.timedelta64(3, "us")).value == -3000
    with pytest.raises(OverflowError):
        t + np.timedelta64(106752, "D")
    with pytest.raises(ValueError):
        t + np.timedelta64(1, "M")  # a calendar month has no fixed length


def test_nat_timedelta_operand_raises():
    with pytest.raises(ValueError):
        volas.Timestamp("2024-01-01") + np.timedelta64("NaT")
    with pytest.raises(ValueError):
        volas.Timestamp("2024-01-01") - np.timedelta64("NaT", "s")


def test_numpy_datetime_columns_scale_checked():
    a = np.array(["2024-01-01", "NaT"], dtype="datetime64[s]")
    s = _col(a)
    assert s.to_numpy().view("i8")[0] == 1_704_067_200 * 10**9
    assert s.to_list()[1] is volas.NA
    with pytest.raises(OverflowError):
        _col(np.array(["3000-01-01"], dtype="datetime64[s]"))
    with pytest.raises(OverflowError):
        volas.DataFrame({"t": np.array(["3000-01-01"], dtype="datetime64[D]")})
    with pytest.raises(ValueError):
        _col(np.array([1001], dtype="datetime64[ps]"))


def test_loc_numpy_key_out_of_range_raises():
    idx = np.array(["1999-01-01", "2000-01-01"], dtype="datetime64[ns]")
    df = volas.DataFrame({"x": [1, 2]}, index=idx)
    assert df.loc[np.datetime64("1999-01-01", "D")]["x"] == 1
    with pytest.raises(OverflowError):
        df.loc[np.datetime64("2584-01-01", "D")]


def test_from_pandas_coarse_unit_columns_scale_checked():
    ok = pd.DataFrame({"t": np.array(["2024-01-01"], dtype="datetime64[s]")})
    assert volas.DataFrame.from_pandas(ok)["t"].to_numpy().view("i8")[0] == 1_704_067_200 * 10**9
    bad = pd.DataFrame({"t": np.array(["3000-01-01"], dtype="datetime64[s]")})
    with pytest.raises(OverflowError):
        volas.DataFrame.from_pandas(bad)
    bad_index = pd.DataFrame({"x": [1]}, index=pd.DatetimeIndex(np.array(["3000-01-01"], dtype="datetime64[s]")))
    with pytest.raises(OverflowError):
        volas.DataFrame.from_pandas(bad_index)


# --- Arrow ----------------------------------------------------------------------

@pytest.mark.parametrize("arr", [
    pa.array([10**10], type=pa.timestamp("s")),
    pa.array([2**31 - 1], type=pa.date32()),
    pa.array([2**62], type=pa.date64()),
])
def test_arrow_coarse_overflow_raises_never_nat(arr):
    with pytest.raises(ValueError, match="outside the datetime64"):
        volas.Series.from_arrow(arr)


def test_arrow_sentinel_collision_raises():
    with pytest.raises(ValueError, match="outside the datetime64"):
        volas.Series.from_arrow(pa.array([-(2**63)], type=pa.timestamp("ns")))
    with pytest.raises(ValueError, match="outside the datetime64"):
        volas.Series.from_arrow(pa.array([1, -(2**63)], type=pa.timestamp("ns")))
    # a real Arrow null stays the missing value
    s = volas.Series.from_arrow(pa.array([1, None], type=pa.timestamp("s")))
    assert s.to_list()[1] is volas.NA

