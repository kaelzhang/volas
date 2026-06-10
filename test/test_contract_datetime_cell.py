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
