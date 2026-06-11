"""Value-category taxonomy for the per-parameter census (SPEC §6.8 / P8).

"All categories" of a parameter is *defined here* — the one hand-authored piece
of the generated-completeness method, deliberately small, reused across APIs
(Timestamp / to_datetime / column construction / scalar operands), and grown by
accretion: every newly-discovered category (e.g. `tz=8`) is added once and every
future sweep covers it automatically.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import volas

_TS = "2021-06-15 09:30:00"          # the canonical instant every form must express


def datetime_scalars():
    """(label, value) — every way to express one datetime *scalar*."""
    return [
        ("py-datetime", datetime(2021, 6, 15, 9, 30)),
        ("py-date", date(2021, 6, 15)),
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
    ]


def tz_values():
    """(label, value) — every way to express a timezone parameter."""
    return [
        ("str-iana", "Asia/Shanghai"),
        ("str-offset", "+08:00"),
        ("int-offset", 8),
        ("tzinfo-obj", timezone(timedelta(hours=8))),
        ("None", None),
        ("invalid", "Not/AZone"),
    ]
