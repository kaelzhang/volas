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
# disposition per category: ok=accept & match pandas instant · waiver=intended
# divergence · F<n>=align-finding (should accept, currently rejects/wrong).
_CONSTRUCT = {
    "py-datetime": "F23", "py-date": "F23", "pd-Timestamp": "F23",
    "volas-Timestamp": "F23", "str-time-only": "F24", "str-offset": "F25",
    "None": "F16",         # decision 2 target = clean ValueError; currently KeyError leak
    "np-datetime64": "ok", "int-ns": "ok", "str-date": "ok",
    "str-datetime": "ok", "str-iso-T": "ok",
}
_CONSTRUCT_XFAIL = {"F16", "F23", "F24", "F25"}


@pytest.mark.parametrize("label,val", I.datetime_scalars(),
                         ids=[l for l, _ in I.datetime_scalars()])
def test_timestamp_construct_census(label, val):
    disp = _CONSTRUCT[label]
    if disp in _CONSTRUCT_XFAIL:
        pytest.xfail(f"{disp}: volas construction diverges from pandas/target for {label}")
    # ok: volas accepts and pins the same absolute instant as pandas.
    assert volas.Timestamp(val).value == pd.Timestamp(val).value, f"{label} instant"


# --- Layer 2: tz-parameter census (every tz value-category) -----------------
_TZ = {
    "str-iana": "ok", "str-offset": "ok", "None": "ok",
    "int-offset": "out-of-scope",     # tz=8 ambiguous -> use '+08:00' (disposition C)
    "tzinfo-obj": "F29",              # align: tzinfo objects should be accepted
    "invalid": "raise",               # bad zone -> error (not a silent pass)
}


@pytest.mark.parametrize("label,tz", I.tz_values(), ids=[l for l, _ in I.tz_values()])
def test_timestamp_tz_param_census(label, tz):
    disp = _TZ[label]
    base = "2021-06-15 09:30:00"
    if disp == "ok":
        assert volas.Timestamp(base, tz=tz).value == pd.Timestamp(base, tz=tz).value
    elif disp == "raise":
        with pytest.raises((ValueError, KeyError, TypeError)):
            volas.Timestamp(base, tz=tz)
    elif disp == "out-of-scope":
        # tz=8 is intentionally unsupported; assert it's rejected, not silently wrong.
        with pytest.raises((TypeError, ValueError)):
            volas.Timestamp(base, tz=tz)
    else:  # F29 align-finding: tzinfo object should work but doesn't
        pytest.xfail(f"{disp}: volas tz= rejects {label} (only str accepted)")


# --- Layer 2: column-construction census (F20 — your [datetime.now()] bug) ---
# DataFrame({'t': [<scalar>]}) per input category. pandas infers datetime64 from
# a list of datetime-ish scalars; volas rejects every natural form (only a
# prebuilt np.array works). list[str] -> str is pandas-consistent (waiver: both
# keep it as a string column unless explicitly to_datetime'd).
_COLUMN = {
    "py-datetime": "F20", "np-datetime64": "F20", "pd-Timestamp": "F20",
    "volas-Timestamp": "F20", "str-datetime": "waiver-str",
}


@pytest.mark.parametrize("label", list(_COLUMN), ids=list(_COLUMN))
def test_datetime_column_construct_census(label):
    val = dict(I.datetime_scalars())[label]
    disp = _COLUMN[label]
    if disp == "F20":
        pytest.xfail("F20: volas rejects a list of natural datetime scalars (pandas infers datetime64)")
    # waiver-str: a list of date strings stays a str column in BOTH volas and
    # pandas (no auto-datetime inference on construction) — consistent, not a bug.
    assert volas.DataFrame({"t": [val]})["t"].dtype == "str"
