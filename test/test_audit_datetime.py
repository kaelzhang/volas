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
_CONSTRUCT = {
    "py-datetime": "F23", "py-date": "F23", "py-datetime-aware": "F23",
    "pd-Timestamp": "F23", "volas-Timestamp": "F23",
    "str-time-only": "F24", "str-offset": "F25",
    "None": "F16", "pd-NaT": "F16", "np-NaT": "natok",   # np-NaT already -> clean ValueError ✓
    "np-datetime64": "ok", "int-ns": "ok", "str-date": "ok",
    "str-datetime": "ok", "str-iso-T": "ok",
}


def _construct_params():
    for label, val in I.datetime_scalars():
        disp = _CONSTRUCT[label]
        marks = [pytest.mark.xfail(reason=f"{disp}: construction diverges (findings-ledger)", strict=True)] \
            if disp.startswith("F") else []
        yield pytest.param(label, val, id=label, marks=marks)


@pytest.mark.parametrize("label,val", list(_construct_params()))
def test_timestamp_construct_census(label, val):
    disp = _CONSTRUCT[label]
    if disp in ("F16", "natok"):            # None / NaT -> clean ValueError (no NaT scalar)
        with pytest.raises(ValueError):     # natok already correct (no mark); F16 leaks KeyError (xfail)
            volas.Timestamp(val)
    elif disp == "F25":                     # offset string -> tz-aware wall-clock hour kept (9, not UTC 1)
        assert volas.Timestamp(val).hour == 9
    else:                                   # ok / F23 / F24 -> accept + match pandas instant
        assert volas.Timestamp(val).value == pd.Timestamp(val).value


# --- Layer 2: tz-parameter census (every tz value-category) -----------------
_TZ = {
    "str-iana": "ok", "str-utc": "ok", "str-offset": "ok", "None": "ok",
    "int-offset": "out-of-scope",          # tz=8 ambiguous -> use '+08:00'
    "tzinfo-obj": "F29", "zoneinfo": "F29",  # align: tzinfo/zoneinfo objects should work
    "empty": "F47", "invalid": "raise",    # invalid zone -> error; empty silently accepted (F47)
}
_TZ_REASON = {
    "F29": "F29: tz= rejects non-str tzinfo/zoneinfo (only str accepted)",
    "F47": "F47: empty-string tz silently accepted (should error, C4)",
}


def _tz_params():
    for label, tz in I.tz_values():
        disp = _TZ[label]
        marks = [pytest.mark.xfail(reason=_TZ_REASON[disp], strict=True)] if disp in _TZ_REASON else []
        yield pytest.param(label, tz, id=label, marks=marks)


@pytest.mark.parametrize("label,tz", list(_tz_params()))
def test_timestamp_tz_param_census(label, tz):
    base = "2021-06-15 09:30:00"
    if _TZ[label] in ("raise", "out-of-scope", "F47"):   # should reject
        with pytest.raises((TypeError, ValueError, KeyError)):
            volas.Timestamp(base, tz=tz)
    else:                                                # ok / F29 -> accept + match pandas instant
        assert volas.Timestamp(base, tz=tz).value == pd.Timestamp(base, tz=tz).value


# --- Layer 2: column-construction census (F20 — your [datetime.now()] bug) ---
# DataFrame({'t': [<scalar>]}) per input category. pandas infers datetime64 from
# a list of datetime-ish scalars; volas rejects every natural form (only a
# prebuilt np.array works). list[str] -> str is pandas-consistent (waiver: both
# keep it as a string column unless explicitly to_datetime'd).
_COLUMN = {
    "py-datetime": "F20", "np-datetime64": "F20", "pd-Timestamp": "F20",
    "volas-Timestamp": "F20", "str-datetime": "waiver-str",
}


def _column_params():
    for label, disp in _COLUMN.items():
        marks = [pytest.mark.xfail(reason="F20: rejects a list of natural datetime scalars", strict=True)] \
            if disp == "F20" else []
        yield pytest.param(label, id=label, marks=marks)


@pytest.mark.parametrize("label", list(_column_params()))
def test_datetime_column_construct_census(label):
    val = dict(I.datetime_scalars())[label]
    if _COLUMN[label] == "waiver-str":      # str list stays str in BOTH (no auto-datetime) — consistent
        assert volas.DataFrame({"t": [val]})["t"].dtype == "str"
    else:                                   # F20 -> should accept + infer datetime64
        assert str(volas.DataFrame({"t": [val]})["t"].dtype).startswith("datetime64")


# --- Layer 2: to_datetime parser census (the primary ingestion entry) -------
_TO_DT = {
    "list[str-date]": (["2021-01-01", "2021-06-15"], "ok"),
    "list[str-datetime]": (["2021-01-01 09:30:00"], "ok"),
    "list[str-iso-T]": (["2021-01-01T09:30:00"], "ok"),
    "list[str-with-NA]": (["2021-01-01", None], "ok"),
    "list[int-ns]": ([1609459200000000000], "ok"),
    "list[py-datetime]": ([datetime(2021, 1, 1)], "F20"),
    "list[str-mixed-fmt]": (["2021-01-01", "2021/06/15"], "lenient"),
}


def _todt_params():
    for label, (xs, disp) in _TO_DT.items():
        marks = [pytest.mark.xfail(reason="F20: to_datetime rejects a list of python datetimes", strict=True)] \
            if disp == "F20" else []
        yield pytest.param(label, xs, id=label, marks=marks)


@pytest.mark.parametrize("label,xs", list(_todt_params()))
def test_to_datetime_census(label, xs):
    # ns (D1; pandas us is fine). F20 cells rejected today -> xfail; `lenient`
    # (mixed formats pandas 2.0+ rejects) volas parses -> documented divergence.
    out = volas.to_datetime(volas.DataFrame({"t": xs})["t"])
    assert str(out.dtype).startswith("datetime64")


# --- Layer 2: scalar-operand I-rep census (F15: `Timestamp == <scalar>`) -----
_OPERANDS = {
    "volas-Timestamp": "ok", "np-datetime64": "ok", "str-datetime": "ok",
    "pd-Timestamp": "F15", "py-datetime": "F15",
}


def _operand_params():
    for label, disp in _OPERANDS.items():
        marks = [pytest.mark.xfail(reason="F15: == pandas/stdlib datetime scalar leaks label KeyError", strict=True)] \
            if disp == "F15" else []
        yield pytest.param(label, id=label, marks=marks)


@pytest.mark.parametrize("label", list(_operand_params()))
def test_timestamp_compare_operand_census(label):
    rhs = dict(I.datetime_scalars())[label]
    # volas / numpy / string operands compare correctly (same instant -> True);
    # F15 cells (pd.Timestamp / stdlib datetime) leak today -> xfail, flip on fix.
    assert bool(volas.Timestamp("2021-06-15 09:30:00") == rhs) is True
