"""Contract O1 (resolved -> B) — a datetime CELL scalar is a `volas.Timestamp`
(matching the index-label scalar type R6 and pandas, which returns a Timestamp
for `s.iloc[0]`); a NaT cell is `volas.NA` (the unified missing singleton, C2).
Timestamp is a complete scalar: it converts to np.datetime64 / int-ns / datetime
and supports timedelta arithmetic, so the rare scalar-datetime workflows stay
ergonomic without exposing np.datetime64 at the cell boundary."""

import datetime as _dt

import numpy as np
import pytest
import volas
from volas import DataFrame


def _dt_series():
    return DataFrame({"t": ["2021-03-04 09:30:00", "2021-03-05 16:00:00"]}).astype(
        {"t": "datetime64[ns]"}
    )["t"]


def test_cell_is_timestamp_iloc_and_tolist():
    s = _dt_series()
    assert isinstance(s.iloc[0], volas.Timestamp)
    assert isinstance(s.to_list()[0], volas.Timestamp)
    assert s.iloc[0].year == 2021 and s.iloc[0].month == 3 and s.iloc[0].day == 4


def test_row_cell_is_timestamp():
    df = DataFrame({"t": ["2021-03-04 09:30:00"], "x": [1.0]}).astype({"t": "datetime64[ns]"})
    assert isinstance(df.iloc[0]["t"], volas.Timestamp)


def test_nat_cell_is_na():
    s = DataFrame({"t": np.array(["2021-01-01", "NaT"], dtype="datetime64[ns]")})["t"]
    assert s.iloc[0].year == 2021          # present -> Timestamp
    assert s.iloc[1] is volas.NA           # NaT -> volas.NA
    assert s.isna().to_list() == [False, True]


def test_timestamp_conversions():
    ts = _dt_series().iloc[0]
    # to_numpy() is a datetime64 SCALAR — a scalar class converts to a scalar,
    # matching the stub's `-> np.datetime64` (not a 1-element ndarray).
    assert isinstance(ts.to_numpy(), np.datetime64)
    assert ts.to_numpy() == np.datetime64("2021-03-04T09:30:00")
    assert ts.value == np.datetime64("2021-03-04T09:30:00", "ns").astype("int64")
    assert ts.to_pydatetime() == _dt.datetime(2021, 3, 4, 9, 30, 0)


def test_timestamp_timedelta_arithmetic():
    s = _dt_series()
    t0, t1 = s.iloc[0], s.iloc[1]
    # Timestamp - timedelta -> Timestamp
    earlier = t0 - np.timedelta64(30, "m")
    assert isinstance(earlier, volas.Timestamp) and earlier.hour == 9 and earlier.minute == 0
    # Timestamp + timedelta -> Timestamp
    later = t0 + np.timedelta64(1, "h")
    assert later.hour == 10 and later.minute == 30
    # Timestamp - Timestamp -> np.timedelta64
    delta = t1 - t0
    assert isinstance(delta, np.timedelta64)


# --- checked arithmetic (D2): overflow / NaT-sentinel results raise -----------
# wrapping arithmetic used to fold i64::MAX + 1 onto i64::MIN — an object that
# rendered as 'NaT' yet exposed a real 1677 civil date, reopening the exact
# inconsistency the constructor's raw-NaT rejection closed.

I64_MAX = 2**63 - 1
I64_MIN = -(2**63)


def test_timestamp_add_overflow_raises():
    with pytest.raises(OverflowError):
        volas.Timestamp(I64_MAX) + 1


def test_timestamp_sub_to_nat_sentinel_raises():
    # (i64::MIN + 1) - 1 == i64::MIN exactly — no wrap, but the result IS the NaT
    # sentinel, which must not become a constructed Timestamp.
    with pytest.raises(OverflowError):
        volas.Timestamp(I64_MIN + 1) - 1


def test_timestamp_difference_overflow_raises():
    # MAX - (MIN + 1) overflows i64; the wrapped result used to be a tiny -2ns.
    with pytest.raises(OverflowError):
        volas.Timestamp(I64_MAX) - volas.Timestamp(I64_MIN + 1)


def test_timestamp_arithmetic_normal_values_unaffected():
    ts = volas.Timestamp("2021-01-01 10:00")          # HH:MM parse (minute form)
    assert (ts + np.timedelta64(1, "h")).hour == 11
    assert (ts - np.timedelta64(30, "m")).minute == 30
    assert (volas.Timestamp("2021-01-02") - volas.Timestamp("2021-01-01")) == np.timedelta64(1, "D")
