"""Shared dimension vocabulary + fixture factory for the systematic API audit.

The audit treats the bug space as a Cartesian product
``API × inherent-data-state × parameter-config``; this module builds the
``inherent-data-state`` axis — every dtype (D) in every NA-combination state (N)
— so each ``test_audit_*`` module only declares its parameter axis (P) and its
oracle table, and the parametrization enumerates the full product.

Dimensions (see the audit method report `tasks/04/audits/`):
  D — the 7 storage dtypes: f64 f32 i64 i32 bool str datetime
  N — NA-combination state: N0 dense · N1 partial-NA · N2 all-NA · N3 empty · N4 single
"""

from __future__ import annotations

import numpy as np

import volas
from volas import DataFrame

NA = volas.NA

# --- D: the storage dtypes --------------------------------------------------
DTYPES = ("f64", "f32", "i64", "i32", "bool", "str", "datetime")
NUMERIC = ("f64", "f32", "i64", "i32", "bool")  # the f64-funnel-eligible dtypes
_DTYPE_STR = {
    "f64": "float64", "f32": "float32", "i64": "int64", "i32": "int32",
    "bool": "bool", "str": "str", "datetime": "datetime64[ns]",
}

# --- N: the NA-combination states -------------------------------------------
NA_STATES = ("N0", "N1", "N2", "N3", "N4")

# Three present, distinct, ordered values per dtype (the N0/N2/N3/N4 basis); the
# middle slot is the one a partial-NA (N1) state holes out.
_PRESENT = {
    "f64": [1.0, 2.0, 3.0],
    "f32": [1.0, 2.0, 3.0],
    "i64": [1, 2, 3],
    "i32": [1, 2, 3],
    "bool": [True, False, True],
    "str": ["a", "b", "c"],
    "datetime": ["2021-01-01", "2021-01-02", "2021-01-03"],
}


def _datetime_series(values):
    """A datetime Series from strings (None -> NaT)."""
    arr = np.array(
        [np.datetime64(v, "ns") if v is not None else np.datetime64("NaT") for v in values],
        dtype="datetime64[ns]",
    )
    return DataFrame({"x": arr})["x"]


def _typed(dtype: str, py_values: list):
    """A volas Series of exactly `dtype` built from a python list (None = NA)."""
    if dtype == "datetime":
        return _datetime_series(py_values)
    s = DataFrame({"x": py_values})["x"]  # infers f64 / i64 / bool / str (None-aware)
    want = _DTYPE_STR[dtype]
    return s if s.dtype == want else s.astype(want)


def series(dtype: str, na: str):
    """A volas Series of `dtype` in NA-state `na` (one of NA_STATES).

    N0 dense (3) · N1 middle held out (3) · N2 all-NA (3) · N3 empty · N4 single.
    """
    present = _PRESENT[dtype]
    if na == "N0":
        return _typed(dtype, list(present))
    if na == "N1":
        return _typed(dtype, [present[0], None, present[2]])
    if na == "N2":
        # all-NA, dtype-preserving: a typed all-None list collapses to float (or
        # rejects NaN->int), so hole out a dense column with an all-False `where`
        # (the contract-basic NA fill — audited independently in test_audit_t4).
        dense = _typed(dtype, list(present))
        return dense.where(DataFrame({"m": [False, False, False]})["m"])
    if na == "N3":
        return _typed(dtype, [])
    if na == "N4":
        return _typed(dtype, [present[0]])
    raise ValueError(f"unknown NA-state {na!r}")  # pragma: no cover


def frame(dtype: str, na: str):
    """A single-column DataFrame of `dtype` in NA-state `na`."""
    s = series(dtype, na)
    return DataFrame({"x": s})


# --- F: frame composition (the DataFrame-level axis, SPEC §4.4) -------------
FRAMES = ("single", "homogeneous", "numeric_str", "with_datetime")


def wide_frame(kind: str):
    """A DataFrame whose column composition exercises F.

    single — one numeric column · homogeneous — several same-dtype numeric ·
    numeric_str — numeric + str (the object export path) · with_datetime —
    numeric + a datetime column (mixed-kind export).
    """
    a = [1.0, 2.0, 3.0]
    if kind == "single":
        return DataFrame({"a": a})
    if kind == "homogeneous":
        return DataFrame({"a": a, "b": [4.0, 5.0, 6.0]})
    if kind == "numeric_str":
        return DataFrame({"a": a, "s": ["x", "y", "z"]})
    if kind == "with_datetime":
        return DataFrame({"a": a, "t": _datetime_series(["2021-01-01", "2021-01-02", "2021-01-03"])})
    raise ValueError(f"unknown frame kind {kind!r}")  # pragma: no cover


def bool_mask(bits, with_na: bool = False):
    """A boolean Series mask; `with_na=True` injects a volas.NA at index 1."""
    vals = [bool(b) for b in bits]
    if with_na and len(vals) > 1:
        vals = [vals[0], None] + vals[2:]
    return DataFrame({"m": vals})["m"]
