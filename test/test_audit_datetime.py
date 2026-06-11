"""Systematic audit — datetime (P8 §6.8 generated-completeness, applied).

This is the first subject built the *new* way (SPEC P8): coverage is generated,
not hand-listed.

  Layer 1 — surface differential: reflect pd.Timestamp's member set; every
            member volas lacks must be dispositioned align/out-of-scope
            (`alignment-disposition-2026-06-11.md`). A new pandas method or a
            volas removal trips the meta-test, forcing a disposition.
  Layer 2 — per-parameter × value-category census: drive construction and the
            tz parameter from the value-category taxonomy (audit_irep), diff
            vs pandas. Intentional divergences are waived; the rest are findings.

Findings pinned here: F23–F31 (datetime-audit-plan §4).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import volas
from . import audit_irep as I

_pub = lambda o: {n for n in dir(o) if not n.startswith("_")}

# --- Layer 1: Timestamp surface disposition (alignment-disposition §1) -------
# `align`: volas SHOULD have these (pandas-parity, real value) — currently
# missing → backlog. `out-of-scope`: deliberately not implemented.
_ALIGN = {
    # tz
    "tz_localize", "tz_convert", "astimezone", "utcoffset", "tzname", "dst", "tzinfo",
    # rounding (bar alignment)
    "floor", "ceil", "round", "normalize",
    # sub-second components (storage is ns)
    "microsecond", "nanosecond",
    # calendar predicates
    "quarter", "dayofweek", "day_of_week", "dayofyear", "day_of_year", "week",
    "weekofyear", "days_in_month", "daysinmonth", "is_month_start", "is_month_end",
    "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end",
    "is_leap_year", "isocalendar", "isoweekday", "day_name", "month_name",
    # extract / replace
    "date", "time", "replace",
    # interop / precision
    "timestamp", "to_datetime64", "isoformat", "as_unit", "unit",
    # convenience constructors
    "now", "today",
}
_OUT_OF_SCOPE = {
    "to_period", "to_julian_date", "toordinal", "fromordinal", "fromisocalendar",
    "fromisoformat", "fromtimestamp", "utcfromtimestamp", "utcnow", "combine",
    "ctime", "timetuple", "utctimetuple", "timetz", "strptime", "resolution",
    "asm8", "fold", "max", "min",
}


def test_timestamp_surface_dispositioned():
    """P8 layer 1: every pd.Timestamp member volas lacks is dispositioned."""
    missing = _pub(pd.Timestamp("2021-01-01")) - _pub(volas.Timestamp("2021-01-01"))
    classified = _ALIGN | _OUT_OF_SCOPE
    unknown = missing - classified
    orphan = classified - missing
    assert not unknown, f"undispositioned missing Timestamp members (P8 §6.8): {sorted(unknown)}"
    assert not orphan, f"disposition names a non-missing member: {sorted(orphan)}"


@pytest.mark.parametrize("m", sorted(_ALIGN))
@pytest.mark.xfail(reason="P8 align-backlog: pandas-parity datetime API not yet implemented", strict=True)
def test_timestamp_align_backlog(m):
    """Each `align` member SHOULD exist; strict-xfail flips loudly when added."""
    assert hasattr(volas.Timestamp("2021-06-15"), m)


def test_timestamp_out_of_scope_absent():
    """`out-of-scope` members are deliberately not implemented (not gaps)."""
    t = volas.Timestamp("2021-06-15")
    present = [m for m in _OUT_OF_SCOPE if hasattr(t, m)]
    assert not present, f"out-of-scope members unexpectedly present — reclassify: {present}"


# --- Layer 2: construction census (every input value-category) --------------
# disposition: ok=accept & match pandas · F<n>=finding. Findings use strict-xfail
# markers (NOT imperative pytest.xfail) and the body asserts the *target*
# behaviour, so a volas fix XPASSes and flips the marker loudly (review feedback).
# All construction categories now FIXED (F16/F23/F24/F25 landed): natural
# datetime scalars are accepted, None/NaT raise a clean ValueError (no NaT
# scalar, decision 2), an offset string keeps its zone, time-only means today.
_CONSTRUCT = {
    "py-datetime": "ok", "py-date": "ok", "py-datetime-aware": "ok",
    "pd-Timestamp": "ok", "volas-Timestamp": "ok",
    "str-time-only": "today", "str-offset": "tz-kept",
    "None": "na-raises", "pd-NaT": "na-raises", "np-NaT": "na-raises",
    "np-datetime64": "ok", "int-ns": "ok", "str-date": "ok",
    "str-datetime": "ok", "str-iso-T": "ok",
}


@pytest.mark.parametrize("label,val", I.datetime_scalars(),
                         ids=[l for l, _ in I.datetime_scalars()])
def test_timestamp_construct_census(label, val):
    disp = _CONSTRUCT[label]
    if disp == "na-raises":                 # None / NaT -> clean ValueError (no NaT scalar)
        with pytest.raises(ValueError):
            volas.Timestamp(val)
    elif disp == "tz-kept":                 # F25: offset string keeps its zone -> wall-clock 9, not UTC 1
        t = volas.Timestamp(val)
        assert t.hour == 9 and t.tz == "+08:00"
        assert t.value == pd.Timestamp(val).value          # same absolute instant
    elif disp == "today":                   # F24: time-only means today at that wall-clock
        t = volas.Timestamp("09:30")
        assert (t.hour, t.minute) == (9, 30)               # date is "today" (env-dependent)
    elif label == "volas-Timestamp":        # pandas can't parse a volas Timestamp;
        assert volas.Timestamp(val).value == val.value     # identity round-trip
    else:                                   # accept + match the pandas instant
        assert volas.Timestamp(val).value == pd.Timestamp(val).value


# --- Layer 2: tz-parameter census (every tz value-category) -----------------
# tz param now FIXED (F29/F47): tzinfo/zoneinfo objects are accepted, the empty
# string errors; tz=8 (int) stays rejected (out-of-scope: ambiguous, use '+08:00').
_TZ = {
    "str-iana": "ok", "str-utc": "ok", "str-offset": "ok", "None": "ok",
    "tzinfo-obj": "ok", "zoneinfo": "ok",
    "int-offset": "raise", "empty": "raise", "invalid": "raise",
}


@pytest.mark.parametrize("label,tz", I.tz_values(), ids=[l for l, _ in I.tz_values()])
def test_timestamp_tz_param_census(label, tz):
    base = "2021-06-15 09:30:00"
    if _TZ[label] == "raise":
        with pytest.raises((TypeError, ValueError, KeyError)):
            volas.Timestamp(base, tz=tz)
    else:                                                # accept + match pandas instant
        assert volas.Timestamp(base, tz=tz).value == pd.Timestamp(base, tz=tz).value


# --- Layer 2: column-construction census (F20 — your [datetime.now()] bug) ---
# DataFrame({'t': [<scalar>]}) per input category. pandas infers datetime64 from
# a list of datetime-ish scalars; volas rejects every natural form (only a
# prebuilt np.array works). list[str] -> str is pandas-consistent (waiver: both
# keep it as a string column unless explicitly to_datetime'd).
# F20 FIXED: a list of natural datetime scalars infers a datetime column.
_COLUMN = {
    "py-datetime": "datetime", "np-datetime64": "datetime", "pd-Timestamp": "datetime",
    "volas-Timestamp": "datetime", "str-datetime": "waiver-str",
}


@pytest.mark.parametrize("label", list(_COLUMN), ids=list(_COLUMN))
def test_datetime_column_construct_census(label):
    val = dict(I.datetime_scalars())[label]
    if _COLUMN[label] == "waiver-str":      # str list stays str in BOTH (no auto-datetime) — consistent
        assert volas.DataFrame({"t": [val]})["t"].dtype == "str"
    else:                                   # accepted + inferred datetime64
        assert str(volas.DataFrame({"t": [val]})["t"].dtype).startswith("datetime64")
    # a None slot in a datetime list is the missing instant
    if _COLUMN[label] == "datetime":
        s = volas.DataFrame({"t": [val, None]})["t"]
        assert s.isna().to_list() == [False, True]


# --- Layer 2: to_datetime parser census (the primary ingestion entry) -------
_TO_DT = {
    "list[str-date]": (["2021-01-01", "2021-06-15"], "ok"),
    "list[str-datetime]": (["2021-01-01 09:30:00"], "ok"),
    "list[str-iso-T]": (["2021-01-01T09:30:00"], "ok"),
    "list[str-with-NA]": (["2021-01-01", None], "ok"),
    "list[int-ns]": ([1609459200000000000], "ok"),
    "list[py-datetime]": ([datetime(2021, 1, 1)], "ok"),   # F20 fixed
    "list[str-mixed-fmt]": (["2021-01-01", "2021/06/15"], "lenient"),
}


@pytest.mark.parametrize("label,xs", [(l, x) for l, (x, _) in _TO_DT.items()],
                         ids=list(_TO_DT))
def test_to_datetime_census(label, xs):
    # ns (D1; pandas us is fine). `lenient` (mixed formats pandas 2.0+ rejects)
    # volas parses -> documented divergence, pinned.
    out = volas.to_datetime(volas.DataFrame({"t": xs})["t"])
    assert str(out.dtype).startswith("datetime64")


# --- Layer 2: scalar-operand I-rep census (F15: `Timestamp == <scalar>`) -----
# F15 FIXED: pandas / stdlib datetime operands compare correctly.
_OPERANDS = ["volas-Timestamp", "np-datetime64", "str-datetime", "pd-Timestamp", "py-datetime"]


@pytest.mark.parametrize("label", _OPERANDS, ids=_OPERANDS)
def test_timestamp_compare_operand_census(label):
    rhs = dict(I.datetime_scalars())[label]
    assert bool(volas.Timestamp("2021-06-15 09:30:00") == rhs) is True
