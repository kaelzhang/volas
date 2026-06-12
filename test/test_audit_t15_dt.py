"""Systematic audit — T15 (dt): the `Series.dt` datetime accessor.

Owner ruling 2026-06-12: fully pandas-aligned, Series-only (no DataFrame.dt).
Every member is differential-tested against pandas on the same instants; the
NaT row pins volas's one deliberate divergence — a missing element yields NA in
EVERY component (# C2), where numpy-backed pandas degrades (float64 + NaN for
ints, False for predicates, NaT-as-False truthiness).

Deliberately absent members (contract waivers, pinned below):
  date / time / timetz / to_pydatetime — object-returning (# C3)
  to_period / freq                     — period dtype out-of-scope
  tz_localize / tz_convert             — a tz lives on the index/scalar (# D3)

Cell IDs:  T15.<member> · T15.guard · T15.waiver
"""

from __future__ import annotations

import pandas as pd
import pytest

import volas

NA = volas.NA

_STRS = ["2021-01-04 09:31:30", None, "2024-02-29 23:59:59.123456789",
         "2021-12-31 00:00:00", "2020-04-01 12:00:00"]


def _pair():
    v = volas.to_datetime(volas.DataFrame({"t": _STRS})["t"])
    p = pd.Series(pd.to_datetime(_STRS, format="mixed"))
    return v, p


_INT_PROPS = ("year", "month", "day", "hour", "minute", "second", "microsecond",
              "nanosecond", "dayofweek", "day_of_week", "weekday", "dayofyear",
              "day_of_year", "quarter", "days_in_month", "daysinmonth")
_BOOL_PROPS = ("is_month_start", "is_month_end", "is_quarter_start",
               "is_quarter_end", "is_year_start", "is_year_end", "is_leap_year")


@pytest.mark.parametrize("prop", _INT_PROPS)
def test_dt_int_component(prop):
    v, p = _pair()
    got, want = getattr(v.dt, prop), getattr(p.dt, prop).tolist()
    assert got.dtype == "int64"                       # native-NA int (# C2)
    for g, w, m in zip(got.to_list(), want, got.isna().to_list()):
        if m:
            assert g is NA                            # NaT -> NA (vs pandas NaN)
        else:
            assert int(g) == int(w), f"T15.{prop}"


@pytest.mark.parametrize("prop", _BOOL_PROPS)
def test_dt_bool_predicate(prop):
    v, p = _pair()
    got, want = getattr(v.dt, prop), getattr(p.dt, prop).tolist()
    assert got.dtype == "bool"
    for i, (g, w, m) in enumerate(zip(got.to_list(), want, got.isna().to_list())):
        if m:
            assert g is NA   # divergence pin: pandas numpy-bool says False at NaT
        else:
            assert g is w, f"T15.{prop}[{i}]"


@pytest.mark.parametrize("meth", ("day_name", "month_name"))
def test_dt_name(meth):
    v, p = _pair()
    got, want = getattr(v.dt, meth)(), getattr(p.dt, meth)().tolist()
    assert got.dtype == "str"
    for g, w, m in zip(got.to_list(), want, got.isna().to_list()):
        assert (g is NA) if m else (g == w)


@pytest.mark.parametrize("meth,freq", [("floor", "15min"), ("floor", "D"),
                                       ("ceil", "h"), ("round", "s"), ("round", "min")])
def test_dt_floor_ceil_round(meth, freq):
    v, p = _pair()
    got = getattr(v.dt, meth)(freq)
    want = getattr(p.dt, meth)(freq).tolist()
    assert got.dtype == "datetime64[ns]"
    norm = lambda x: "<NA>" if str(x) == "NaT" else str(x)  # decision 2: NA prints <NA>
    assert [norm(x) for x in got.to_list()] == [norm(x) for x in want]


def test_dt_normalize():
    v, p = _pair()
    norm = lambda x: "<NA>" if str(x) == "NaT" else str(x)
    assert [norm(x) for x in v.dt.normalize().to_list()] == \
           [norm(x) for x in p.dt.normalize().tolist()]


def test_dt_invalid_freq_raises():
    v, _ = _pair()
    with pytest.raises(ValueError):
        v.dt.floor("2 weeks")


def test_dt_strftime():
    v, p = _pair()
    got = v.dt.strftime("%Y/%m/%d %H:%M:%S")
    want = p.dt.strftime("%Y/%m/%d %H:%M:%S").tolist()
    assert got.dtype == "str"
    for g, w, m in zip(got.to_list(), want, got.isna().to_list()):
        assert (g is NA) if m else (g == w)


def test_dt_strftime_invalid_format_raises():
    v, _ = _pair()
    with pytest.raises(ValueError):
        v.dt.strftime("%Q")


def test_dt_isocalendar():
    v, p = _pair()
    got, want = v.dt.isocalendar(), p.dt.isocalendar()
    assert list(got.columns) == ["year", "week", "day"]
    for col in ("year", "week", "day"):
        for g, w, m in zip(got[col].to_list(), want[col].tolist(),
                           got[col].isna().to_list()):
            assert (g is NA) if m else (int(g) == int(w))


def test_dt_tz_and_unit():
    v, _ = _pair()
    assert v.dt.tz is None          # a tz lives on the index/scalar (# D3)
    assert v.dt.unit == "ns"        # the single storage unit (# D1)


def test_dt_preserves_name_and_index():
    df = volas.DataFrame({"t": ["2021-01-01", "2021-01-02"], "v": [10, 20]})
    df["t"] = volas.to_datetime(df["t"])
    s = df.set_index("v")["t"]
    out = s.dt.year
    assert out.name == "t"
    assert list(out.index) == list(s.index) == [10, 20]


def test_dt_guard_non_datetime():
    """`.dt` on a non-datetime Series raises AttributeError (# pandas)."""
    for col in ([1.0], [1], ["a"], [True]):
        with pytest.raises(AttributeError):
            volas.DataFrame({"x": col})["x"].dt


def test_dt_waived_members_absent():
    """The contract-waived pandas members are deliberately NOT exposed:
    object-returning (C3), period (out-of-scope), column tz (D3)."""
    v, _ = _pair()
    for m in ("date", "time", "timetz", "to_pydatetime", "to_period", "freq",
              "tz_localize", "tz_convert"):
        assert not hasattr(v.dt, m), m
