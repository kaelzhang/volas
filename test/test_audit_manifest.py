"""Systematic audit — §6.1 public-API manifest + completeness meta-test.

Every public symbol (class member or top-level) MUST have a disposition:
  - a SUBJECT id (T0..T13 / "Row") -> it is audited by that subject's matrix;
  - "ignore:<reason>"             -> deliberately not matrixed (protocol/trivial);
  - "remove"                      -> §6.6, should be deleted from the public surface.

The meta-test asserts runtime-surface == classified-surface (no unknown symbol,
no orphan classification), so a new/renamed public API can never silently escape
the audit. This is the external anchor that makes "complete coverage" provable
rather than asserted.
"""

from __future__ import annotations

import inspect

import volas
import volas_rs

# --- shared operator / protocol dispositions -------------------------------
_COMPARE = {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"}
_ARITH = {
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__",
}
_LOGIC = {"__and__", "__or__", "__xor__", "__invert__", "__rand__", "__ror__", "__rxor__"}


def _ops(*sets, subject):
    return {name: subject for s in sets for name in s}


# --- the manifest: {class: {member: disposition}} --------------------------
CLASSIFICATION: dict[str, dict[str, str]] = {
    "DataFrame": {
        **_ops(_COMPARE, subject="T3"),
        "__new__": "T9", "__getitem__": "T7", "__setitem__": "T7",
        "__hash__": "ignore:object identity", "__bool__": "T0", "__len__": "T0",
        "__contains__": "T0", "__iter__": "T0", "__repr__": "T0", "__str__": "T0",
        "_repr_html_": "T0", "to_string": "T0",
        "abs": "T1", "round": "T1", "clip": "T1", "cumsum": "T1", "cummax": "T1",
        "cummin": "T1", "cumprod": "T1", "shift": "T1", "diff": "T1", "ffill": "T1", "bfill": "T1",
        "count": "T2", "sem": "T2", "skew": "T2", "kurt": "T2", "describe": "T2",
        # F8 closed: the core per-column reductions, landed.
        **{m: "T2" for m in ("sum", "mean", "min", "max", "prod", "var", "std",
                             "median", "quantile", "idxmax", "idxmin", "all",
                             "any", "nunique")},
        **{m: "T6" for m in ("value_counts", "mode", "nlargest", "nsmallest",
                             "drop_duplicates", "duplicated")},
        **{m: "T3" for m in ("isin", "replace")},
        **{m: "T1" for m in ("interpolate", "rolling", "expanding", "ewm")},
        "corr": "T3", "cov": "T3",
        "fillna": "T4", "where": "T4", "mask": "T4", "dropna": "T4",
        "isna": "T5", "notna": "T5",
        "sort_index": "T6", "rank": "T6", "drop": "T6", "reset_index": "T6", "set_index": "T6",
        "iloc": "T7", "loc": "T7", "at": "T7", "iat": "T7",
        "to_numpy": "T8", "to_pandas": "T8", "to_csv": "T8", "astype": "T8",
        "fill_into": "T8",
        "to_arrow": "T8", "from_arrow": "T8", "from_pandas": "T8", "__arrow_c_stream__": "T8",
        "exec": "T11",
        "cumulate": "T10", "append": "T10", "fulfill": "T10",
        "tz": "T13", "tz_localize": "T13", "tz_convert": "T13",
        "_set_index_tz": "ignore:interop-internal (from_pandas tz tagging)",
        "copy": "T0", "equals": "T0", "rename": "T0",
        "columns": "T0", "shape": "T0", "index": "T0", "dtypes": "T0",
        "is_computed": "T0", "ready": "T0",
        "_physical_height": "ignore:windowed-internal (memory-bound test hook)",
        "head": "T6", "tail": "T6",
        # decision 1 (2026-06-11) REVERSED the §6.6 removal: get_column is the
        # compute-free safe accessor — df[name] EXECUTES a directive when the
        # name is not an existing column (an injection surface for external
        # names); get_column only fetches, KeyError otherwise. Kept + documented.
        "get_column": "T7",
    },
    "Series": {
        **_ops(_COMPARE, _ARITH, _LOGIC, subject="T3"),
        "__new__": "ignore:no public Series constructor", "__getitem__": "T7", "__setitem__": "T7",
        "__hash__": "ignore:object identity", "__bool__": "T0", "__len__": "T0",
        "__array__": "T8", "__repr__": "T0", "__str__": "T0", "to_string": "T0",
        "abs": "T1", "round": "T1", "clip": "T1", "cumsum": "T1", "cummax": "T1",
        "cummin": "T1", "cumprod": "T1", "shift": "T1", "diff": "T1", "ffill": "T1", "bfill": "T1",
        "acos": "T1", "asin": "T1", "atan": "T1", "ceil": "T1", "cos": "T1", "cosh": "T1",
        "exp": "T1", "floor": "T1", "ln": "T1", "log10": "T1", "sin": "T1", "sinh": "T1",
        "sqrt": "T1", "tan": "T1", "tanh": "T1",
        "sum": "T2", "mean": "T2", "prod": "T2", "min": "T2", "max": "T2", "var": "T2",
        "std": "T2", "median": "T2", "sem": "T2", "skew": "T2", "kurt": "T2", "count": "T2",
        "nunique": "T2", "any": "T2", "all": "T2", "idxmax": "T2", "idxmin": "T2",
        "quantile": "T2", "describe": "T2",
        "corr": "T3", "cov": "T3",
        "fillna": "T4", "where": "T4", "mask": "T4", "dropna": "T4",
        "isna": "T5", "notna": "T5",
        "sort_values": "T6", "rank": "T6", "unique": "T6", "head": "T6", "tail": "T6",
        "iloc": "T7", "loc": "T7",
        "to_numpy": "T8", "to_list": "T8", "astype": "T8",
        "to_arrow": "T8", "from_arrow": "T8",
        "__arrow_c_array__": "T8", "__arrow_c_schema__": "T8",
        "__dlpack__": "T8", "__dlpack_device__": "T8",
        "name": "T0", "dtype": "T0", "index": "T0", "shape": "T0", "equals": "T0",
        "tz": "T13", "tz_localize": "T13", "tz_convert": "T13",
        "dt": "T15",
        # the owner-confirmed align cluster, landed:
        **{m: "T6" for m in ("value_counts", "mode", "nlargest", "nsmallest",
                             "drop_duplicates", "duplicated", "sort_index",
                             "is_monotonic_increasing", "is_monotonic_decreasing",
                             "is_unique")},
        **{m: "T3" for m in ("isin", "between", "replace")},
        **{m: "T1" for m in ("interpolate",)},
        **{m: "T0" for m in ("reset_index", "rename", "copy", "to_frame",
                             "to_dict", "items")},
        "iat": "T7", "at": "T7",
        **{m: "T1" for m in ("rolling", "expanding", "ewm")},
    },
    "Row": {
        "__new__": "ignore:no public Row constructor",
        "__getitem__": "Row", "name": "Row", "to_numpy": "Row", "to_dict": "Row",
        "__repr__": "T0", "__str__": "T0", "to_string": "T0",
    },
    "Timestamp": {
        **_ops(_COMPARE, subject="T12"),
        "__add__": "T12", "__sub__": "T12", "__radd__": "T12", "__rsub__": "T12",
        "__new__": "T9", "__hash__": "T12", "__repr__": "T0", "__str__": "T0",
        "year": "T12", "month": "T12", "day": "T12", "hour": "T12", "minute": "T12",
        "second": "T12", "weekday": "T12", "strftime": "T12",
        "to_numpy": "T12", "to_pydatetime": "T12", "value": "T12", "tz": "T12",
        # F21 align backlog, landed (pandas-parity datetime surface):
        **{m: "T12" for m in (
            "microsecond", "nanosecond", "quarter", "dayofweek", "day_of_week",
            "isoweekday", "dayofyear", "day_of_year", "week", "weekofyear",
            "isocalendar", "days_in_month", "daysinmonth", "is_month_start",
            "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start",
            "is_year_end", "is_leap_year", "day_name", "month_name", "date",
            "time", "replace", "floor", "ceil", "round", "normalize",
            "tz_localize", "tz_convert", "astimezone", "utcoffset", "tzname",
            "dst", "tzinfo", "timestamp", "to_datetime64", "isoformat", "unit",
            "as_unit", "now", "today",
        )},
    },
    "TimeFrame": {
        "__new__": "ignore:no public TimeFrame constructor",
        "__repr__": "T0", "__str__": "T0",
        **{p: "T10" for p in ("s1", "m1", "m3", "m5", "m15", "m30", "H1", "H2", "H4",
                              "H6", "H8", "H12", "D1", "D3", "W1", "M1", "Y1")},
    },
    "NAType": {
        "__new__": "ignore:singleton", "__bool__": "T0", "__repr__": "T0",
    },
    "DataFrameILoc": {"__getitem__": "T7", "__setitem__": "T7", "__new__": "ignore:accessor"},
    "DataFrameLoc": {"__getitem__": "T7", "__setitem__": "T7", "__new__": "ignore:accessor"},
    "DataFrameIat": {"__getitem__": "T7", "__setitem__": "T7", "__new__": "ignore:accessor"},
    "DataFrameAt": {"__getitem__": "T7", "__setitem__": "T7", "__new__": "ignore:accessor"},
    "SeriesILoc": {"__getitem__": "T7", "__new__": "ignore:accessor"},
    "SeriesLoc": {"__getitem__": "T7", "__new__": "ignore:accessor"},
    # window aggregator results (like the indexers: in the stub for typing, not
    # in the runtime __all__ — reached only through rolling()/expanding()/ewm()).
    "Rolling": {**{m: "T14" for m in ("count", "nunique", "sum", "mean", "median", "min", "max", "var", "std", "sem", "skew", "kurt", "quantile", "rank", "first", "last")},
                "corr": "T14", "cov": "T14", "__new__": "ignore:accessor"},
    "Expanding": {**{m: "T14" for m in ("count", "nunique", "sum", "mean", "median", "min", "max", "var", "std", "sem", "skew", "kurt", "quantile", "rank", "first", "last")},
                  "corr": "T14", "cov": "T14", "__new__": "ignore:accessor"},
    "Ewm": {**{m: "T14" for m in ("mean", "sum", "var", "std", "corr", "cov")},
            "__new__": "ignore:accessor"},
    "DatetimeAccessor": {
        **{m: "T15" for m in (
            "year", "month", "day", "hour", "minute", "second", "microsecond",
            "nanosecond", "dayofweek", "day_of_week", "weekday", "dayofyear",
            "day_of_year", "quarter", "days_in_month", "daysinmonth",
            "is_month_start", "is_month_end", "is_quarter_start",
            "is_quarter_end", "is_year_start", "is_year_end", "is_leap_year",
            "day_name", "month_name", "strftime", "normalize", "floor", "ceil",
            "round", "isocalendar", "tz", "unit",
        )},
        "__new__": "ignore:accessor",
    },
    "RollingFrame": {**{m: "T14" for m in ("count", "nunique", "sum", "mean", "median", "min", "max", "var", "std", "sem", "skew", "kurt", "quantile", "rank", "first", "last")}, "__new__": "ignore:accessor"},
    "ExpandingFrame": {**{m: "T14" for m in ("count", "nunique", "sum", "mean", "median", "min", "max", "var", "std", "sem", "skew", "kurt", "quantile", "rank", "first", "last")}, "__new__": "ignore:accessor"},
    "EwmFrame": {**{m: "T14" for m in ("mean", "sum", "var", "std")},
                 "__new__": "ignore:accessor"},
    "DirectiveError": {}, "DirectiveSyntaxError": {}, "DirectiveValueError": {},
}

TOPLEVEL = {
    "read_csv": "T8", "to_datetime": "T8",
    "directive_stringify": "T11", "directive_lookback": "T11",
    "__version__": "ignore:package metadata",
    "NA": "ignore:singleton instance — NAType audited as a class",
}

_DUNDER_KEEP = set(_COMPARE) | set(_ARITH) | set(_LOGIC) | {
    "__getitem__", "__setitem__", "__bool__", "__len__", "__contains__", "__iter__",
    "__array__", "__hash__", "__repr__", "__str__", "__new__",
    # Arrow PyCapsule protocol — public interop surface, audited like __array__.
    "__arrow_c_array__", "__arrow_c_schema__", "__arrow_c_stream__",
    # DLPack protocol — zero-copy numeric export, audited like __array__.
    "__dlpack__", "__dlpack_device__",
}


def _public_members(cls) -> set[str]:
    out = set()
    for n in vars(cls):
        if n.startswith("__") and n not in _DUNDER_KEEP:
            continue
        out.add(n)
    return out


def _resolve(name: str):
    return getattr(volas_rs, name, None) or getattr(volas, name, None)


def test_every_public_symbol_has_a_disposition():
    """No unknown public symbol, no orphan classification (§6.1)."""
    problems = []
    for cls_name, mapping in CLASSIFICATION.items():
        cls = _resolve(cls_name)
        assert cls is not None, f"manifest names a missing class {cls_name}"
        runtime = _public_members(cls)
        classified = set(mapping)
        for unknown in runtime - classified:
            problems.append(f"UNKNOWN public symbol {cls_name}.{unknown} — classify it")
        for orphan in classified - runtime:
            problems.append(f"ORPHAN classification {cls_name}.{orphan} — not on the runtime surface")
    # top-level: every non-class name in __all__ (functions + value singletons)
    top_runtime = {n for n in volas.__all__ if not inspect.isclass(getattr(volas, n))}
    for unknown in top_runtime - set(TOPLEVEL):
        problems.append(f"UNKNOWN top-level {unknown}")
    for orphan in set(TOPLEVEL) - top_runtime:
        problems.append(f"ORPHAN top-level {orphan}")
    assert not problems, "manifest drift:\n  " + "\n  ".join(problems)


def test_no_unclassified_class_on_the_surface():
    """Every public class is in the manifest — including the REACHED-only
    classes (accessor / window result types) that never appear in `__all__`.
    The universe is reflected from the extension module itself, so a newly
    registered pyclass cannot escape by being forgotten here (the
    DatetimeAccessor gap, self-audit 2026-06-12)."""
    universe = {n for n in volas.__all__ if inspect.isclass(getattr(volas, n))}
    universe |= {n for n in dir(volas_rs)
                 if not n.startswith("_") and inspect.isclass(getattr(volas_rs, n))}
    for cls_name in sorted(universe | {"NAType"}):
        assert cls_name in CLASSIFICATION, f"public class {cls_name} not in manifest"


def test_removals_landed_and_reversal_kept():
    """§6.6 dispositions, final state: unify / tolist are REMOVED from the
    runtime surface; get_column was REVERSED to keep (decision 1: the
    compute-free safe accessor vs df[name]'s directive execution)."""
    import volas
    assert not hasattr(volas.TimeFrame, "unify")
    assert not hasattr(volas.DataFrame({"x": [1]})["x"], "tolist")
    assert hasattr(volas.DataFrame({"x": [1]}), "get_column")
    assert CLASSIFICATION["DataFrame"]["get_column"] == "T7"
