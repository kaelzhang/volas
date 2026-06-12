"""Value-category taxonomy for the per-parameter census (SPEC §6.8 / P8).

"All categories" of a parameter is *defined here* — the one hand-authored piece
of the generated-completeness method, deliberately small, reused across APIs
(Timestamp / to_datetime / column construction / scalar operands), and grown by
accretion: every newly-discovered category (e.g. `tz=8`) is added once and every
future sweep covers it automatically.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

import volas

try:
    from zoneinfo import ZoneInfo
    _ZONEINFO = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    _ZONEINFO = None

_TS = "2021-06-15 09:30:00"          # the canonical instant every form must express


def datetime_scalars():
    """(label, value) — every way to express one datetime *scalar*."""
    return [
        ("py-datetime", datetime(2021, 6, 15, 9, 30)),
        ("py-date", date(2021, 6, 15)),
        ("py-datetime-aware", datetime(2021, 6, 15, 9, 30, tzinfo=timezone(timedelta(hours=8)))),
        ("np-datetime64", np.datetime64("2021-06-15T09:30:00", "ns")),
        ("pd-Timestamp", pd.Timestamp(_TS)),
        ("volas-Timestamp", volas.Timestamp(_TS)),
        ("int-ns", pd.Timestamp(_TS).value),
        ("str-date", "2021-06-15"),
        ("str-datetime", _TS),
        ("str-iso-T", "2021-06-15T09:30:00"),
        ("str-time-only", "09:30"),
        ("str-offset", "2021-06-15 09:30:00+08:00"),
        ("None", None),
        ("pd-NaT", pd.NaT),
        ("np-NaT", np.datetime64("NaT")),
    ]


def tz_values():
    """(label, value) — every way to express a timezone parameter."""
    out = [
        ("str-iana", "Asia/Shanghai"),
        ("str-utc", "UTC"),
        ("str-offset", "+08:00"),
        ("int-offset", 8),
        ("tzinfo-obj", timezone(timedelta(hours=8))),
        ("None", None),
        ("empty", ""),
        ("invalid", "Not/AZone"),
    ]
    if _ZONEINFO is not None:
        out.insert(1, ("zoneinfo", _ZONEINFO))
    return out


# V-axis value boundaries (SPEC §4.5) — all five vocabularies (was only INT/FLOAT).
V_TS = [
    ("epoch", "1970-01-01"), ("min+1", "1677-09-22"), ("max", "2262-04-11"),
    ("leap-day", "2020-02-29"), ("dst-gap", "2021-03-14 02:30:00"),
    ("dst-fold", "2021-11-07 01:30:00"),
]
V_STR = [
    ("empty", ""), ("whitespace", "  "), ("digit", "123"), ("unicode", "日本語"),
    ("emoji", "x🎉y"), ("long", "x" * 1000),
    ("comma", "a,b"), ("quote", 'q"x'), ("newline", "l1\nl2"),   # RFC-4180 / to_csv traps
]
V_IDX = [
    ("in-range", 1), ("zero", 0), ("last", -1), ("out-of-range", 99),
    ("neg-out", -99),
]


def numeric_scalars():
    """(label, value) — every way to express the numeric scalar 10 (operands)."""
    return [
        ("py-int", 10),
        ("py-float", 10.0),
        ("py-bool", True),                 # 1
        ("np-int64", np.int64(10)),
        ("np-int32", np.int32(10)),
        ("np-float64", np.float64(10.0)),
        ("np-float32", np.float32(10.0)),
        ("np-bool", np.bool_(True)),
        ("decimal", Decimal("10")),
    ]


def numeric_list_inputs():
    """(label, list) — element-type forms for column construction of value 5."""
    return [
        ("py-int", [5]),
        ("py-float", [5.0]),
        ("np-int64", [np.int64(5)]),
        ("np-int32", [np.int32(5)]),
        ("np-float64", [np.float64(5.0)]),
        ("np-bool", [np.bool_(True)]),
        ("mixed-int-float", [5, 5.0]),
        ("np-array-i64", np.array([5], dtype="int64")),
        ("np-array-f64", np.array([5.0], dtype="float64")),
    ]


# V-axis value boundaries (SPEC §4.5), per dtype.
V_INT = [
    ("zero", 0), ("one", 1), ("neg-one", -1),
    ("2**53", 2 ** 53), ("2**53+1", 2 ** 53 + 1),
    ("i64-max", 9223372036854775807), ("i64-min+1", -9223372036854775807),
    ("i64-min", -9223372036854775808),
    ("i32-max", 2147483647), ("i32-min", -2147483648),
]
# Integers with NO exact volas dtype (no uint64 / object) — must error (C4), not
# silently demote to lossy float64. F32. Kept separate from V_INT (which stays
# in-range / exact).
V_INT_OVERFLOW = [
    ("2**63", 2 ** 63), ("2**63+1", 2 ** 63 + 1), ("2**64", 2 ** 64),
]
V_FLOAT = [
    ("zero", 0.0), ("neg-zero", -0.0), ("inf", float("inf")),
    ("-inf", float("-inf")), ("nan", float("nan")),
    ("2**53+1", float(2 ** 53 + 1)), ("subnormal", 5e-324), ("big", 1e308),
]
