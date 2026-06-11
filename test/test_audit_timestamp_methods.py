"""Systematic audit — the F21 Timestamp surface, behaviour-checked vs pandas.

The 43 align-backlog methods landed; each is value-diffed against pandas on a
fixed instant (P8 layer 2: behaviour, not just existence).
"""

from __future__ import annotations

import pandas as pd
import pytest

import volas

_V = "2021-06-15 09:30:45"
_t = lambda: volas.Timestamp(_V)
_p = lambda: pd.Timestamp(_V)


def test_components_match_pandas():
    t, p = _t(), _p()
    for attr in ("microsecond", "nanosecond", "quarter", "dayofweek", "day_of_week",
                 "dayofyear", "day_of_year", "week", "weekofyear",
                 "days_in_month", "daysinmonth", "is_month_start", "is_month_end",
                 "is_quarter_start", "is_quarter_end", "is_year_start",
                 "is_year_end", "is_leap_year"):
        assert getattr(t, attr) == getattr(p, attr), attr
    assert t.isoweekday() == p.isoweekday()
    assert tuple(t.isocalendar()) == tuple(p.isocalendar())
    assert t.day_name() == p.day_name()
    assert t.month_name() == p.month_name()


def test_edges_quarter_leap_month_end():
    for v, checks in {
        "2020-02-29": dict(is_month_end=True, is_leap_year=True, days_in_month=29),
        "2021-04-01": dict(is_quarter_start=True, quarter=2),
        "2021-12-31": dict(is_year_end=True, is_quarter_end=True),
        "2021-01-01": dict(is_year_start=True, dayofyear=1),
    }.items():
        t = volas.Timestamp(v)
        for k, want in checks.items():
            assert getattr(t, k) == want, f"{v}.{k}"


def test_extract_and_interop():
    t, p = _t(), _p()
    assert t.date() == p.date()
    assert t.time() == p.time()
    assert t.timestamp() == p.timestamp()
    assert t.isoformat() == p.isoformat()
    assert t.to_datetime64() == p.to_datetime64()
    assert t.unit == "ns"
    assert t.as_unit("ns").value == t.value
    with pytest.raises(ValueError):
        t.as_unit("s")                       # ns-only: no silent precision change


def test_replace():
    t = _t().replace(year=2022, minute=0)
    assert (t.year, t.month, t.minute, t.second) == (2022, 6, 0, 45)
    assert _t().replace().value == _t().value           # no-arg = identity


@pytest.mark.parametrize("freq", ["D", "h", "min", "15min", "s"])
def test_floor_ceil_round_match_pandas(freq):
    t, p = _t(), _p()
    assert t.floor(freq).value == p.floor(freq).value, f"floor {freq}"
    assert t.ceil(freq).value == p.ceil(freq).value, f"ceil {freq}"
    assert t.round(freq).value == p.round(freq).value, f"round {freq}"


def test_normalize_is_midnight():
    n = _t().normalize()
    assert (n.hour, n.minute, n.second) == (0, 0, 0)
    assert (n.year, n.month, n.day) == (2021, 6, 15)


def test_scalar_tz_localize_convert():
    naive = _t()
    assert naive.tz is None and naive.tzname() is None and naive.utcoffset() is None
    ny = naive.tz_localize("America/New_York")
    assert ny.tz == "America/New_York"
    assert ny.value == naive.value + 4 * 3600 * 10 ** 9      # EDT anchor shifts +4h
    sh = ny.tz_convert("Asia/Shanghai")
    assert sh.value == ny.value                              # instant kept
    assert (sh.hour - ny.hour) % 24 == 12                    # wall moves NY->SH
    with pytest.raises(TypeError):
        naive.tz_convert("UTC")                              # naive can't convert
    with pytest.raises(TypeError):
        ny.tz_localize("UTC")                                # aware can't re-anchor
    assert ny.astimezone("UTC").value == ny.value


def test_tzinfo_objects_and_dst():
    import datetime as dt
    ny = _t().tz_localize("America/New_York")
    assert ny.utcoffset() == dt.timedelta(hours=-4)          # EDT in June
    assert ny.tzname() == "America/New_York"
    assert ny.dst() == dt.timedelta(hours=1)                 # summer DST
    assert str(ny.tzinfo) == "America/New_York"              # zoneinfo object
    utc = _t().tz_localize("UTC")
    assert utc.tzinfo is dt.timezone.utc


def test_now_today():
    a = volas.Timestamp.now()
    assert a.tz is None and a.year >= 2026
    b = volas.Timestamp.today("UTC")
    assert b.tz == "UTC"
