"""Systematic audit — T7 (index): iloc / loc / at / iat / getitem / setitem.

The high-value invariants here are the *container* law (C1: the selection shape
determines the return type) and the *bounds guard* (V_IDX: out-of-range raises a
clean IndexError, never a panic — P7), plus setitem *post-state* (a mutation
touches exactly its target).

Cell IDs:  T7.<recv>.<accessor>[/D=<d>/N=<n>] · T7.bounds/<recv> · T7.setitem/<mode>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import volas
from . import audit_dims as A


def _s():
    return volas.DataFrame({"x": [10.0, 20.0, 30.0]})["x"]


def _df():
    return volas.DataFrame({"a": [10.0, 20.0, 30.0], "b": [40.0, 50.0, 60.0]})


# --- C1: the selection shape determines the container ----------------------
def test_container_shape_law():
    s, df = _s(), _df()
    assert isinstance(s.iloc[1], (float, np.floating))        # scalar
    assert isinstance(s.iloc[0:2], volas.Series)              # 1-D slice -> Series
    assert isinstance(df.iloc[1], volas.Row)                  # one row -> Row
    assert isinstance(df.iloc[1, 0], (float, np.floating))    # cell -> scalar
    assert isinstance(df.iloc[0:2], volas.DataFrame)          # row slice -> DataFrame
    assert isinstance(df["a"], volas.Series)                  # one column -> Series
    assert isinstance(df[["a", "b"]], volas.DataFrame)        # column list -> DataFrame
    m = volas.DataFrame({"m": [True, False, True]})["m"]
    assert isinstance(s[m], volas.Series)                     # masked Series -> Series
    assert isinstance(df[m], volas.DataFrame)                 # masked frame -> DataFrame


# --- positional value + NA element (D×N) -----------------------------------
@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_iloc_present_values(d, n):
    s = A.series(d, n)
    for pos, m in enumerate(s.isna().to_list()):
        if m:
            continue
        got = s.iloc[pos]
        assert got is not volas.NA
        if d in ("f64", "f32", "i64", "i32", "bool"):
            assert float(got) == float(s.to_list()[pos])


# The NA *element* surface is dtype-specific by the C2 contract: a float hole
# reads back as np.nan (NaN-as-NA), every other dtype as the volas.NA singleton.
@pytest.mark.parametrize("d", A.DTYPES)
def test_iloc_na_element_surface(d):
    s = A.series(d, "N1")           # [v0, NA, v2]; position 1 is NA
    elem = s.iloc[1]
    if d in ("f64", "f32"):
        assert isinstance(elem, (float, np.floating)) and math.isnan(elem)   # # C2 NaN-as-NA
    else:
        assert elem is volas.NA                                # validity / NaT


# --- bounds guard (V_IDX): clean IndexError, never a panic (P7) -------------
def test_bounds_guard_series():
    s = _s()
    for bad in (3, 99, -4, -99):
        with pytest.raises(IndexError):
            s.iloc[bad]


def test_bounds_guard_frame():
    df = _df()
    for bad in (3, 99, -4):
        with pytest.raises(IndexError):
            df.iloc[bad]
    with pytest.raises(IndexError):
        df.iloc[0, 9]                 # column out of range


def test_negative_index_wraps():
    s = _s()
    assert s.iloc[-1] == s.iloc[2]
    assert s.iloc[-3] == s.iloc[0]


# --- label / scalar accessors ----------------------------------------------
def test_label_and_scalar_accessors():
    s, df = _s(), _df()
    assert s.loc[1] == 20.0               # label on the RangeIndex
    assert df.iat[1, 0] == 20.0           # positional cell
    assert df.at[1, "a"] == 20.0          # label cell
    assert df.iloc[1, 1] == 50.0


# --- setitem post-state -----------------------------------------------------
def test_setitem_element_post_state():
    s = _s()
    s[1] = 99.0
    assert s.to_list() == [10.0, 99.0, 30.0]    # only position 1 changed


def test_setitem_column_post_state():
    df = _df()
    df["c"] = volas.DataFrame({"c": [7.0, 8.0, 9.0]})["c"]
    assert list(df.columns) == ["a", "b", "c"]
    assert df["c"].to_list() == [7.0, 8.0, 9.0]
    assert df["a"].to_list() == [10.0, 20.0, 30.0]   # siblings untouched
    df["a"] = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]   # replace existing
    assert df["a"].to_list() == [1.0, 2.0, 3.0]
    assert list(df.columns) == ["a", "b", "c"]       # no duplicate column


# --- boolean-mask filtering -------------------------------------------------
def test_boolean_mask_filter():
    s, df = _s(), _df()
    m = volas.DataFrame({"m": [True, False, True]})["m"]
    assert s[m].to_list() == [10.0, 30.0]
    assert df[m].shape == (2, 2)
    assert df[m]["a"].to_list() == [10.0, 30.0]
