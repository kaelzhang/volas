"""Systematic audit — T0 (meta) + Row: identity / shape / copy / equals / repr.

These are the metadata and presentation surface: shape/columns/dtype, the
copy & equals identity laws (E5/equals), rename, is_computed (directive vs plain
column classification), and the Row accessor. repr is checked as a non-crashing,
content-bearing snapshot across D×N (P7: never a panic).

Cell IDs:  T0.<attr> · T0.repr/D=<d>/N=<n> · Row.<accessor>
"""

from __future__ import annotations

import numpy as np
import pytest

import volas
from . import audit_dims as A


# --- shape / columns / dtype / index ---------------------------------------
def test_shape_columns_dtypes():
    df = volas.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert df["a"].dtype == "float64" and df["b"].dtype == "int64"
    assert len(df) == 2
    assert ("a" in df) and ("z" not in df)


# --- equals / copy identity laws -------------------------------------------
def test_equals_and_copy():
    df = volas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert df.equals(df.copy()) is True                  # reflexive over a copy
    assert df.equals(volas.DataFrame({"a": [1.0, 9.0], "b": [3.0, 4.0]})) is False
    cp = df.copy()
    cp["a"] = volas.DataFrame({"a": [7.0, 8.0]})["a"]
    assert df["a"].to_list() == [1.0, 2.0]               # copy is independent


def test_rename():
    df = volas.DataFrame({"a": [1.0], "b": [2.0]})
    assert list(df.rename({"a": "A"}).columns) == ["A", "b"]
    assert list(df.columns) == ["a", "b"]                # original untouched


# --- is_computed: directive (computed) vs plain column ----------------------
def test_is_computed():
    df = volas.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert df.is_computed("close") is False              # plain, supplied per bar
    _ = df["ma:3"]                                        # materialize a directive column
    assert df.is_computed("ma:3") is True                 # cached directive (computed)
    assert df.is_computed("close") is False
    with pytest.raises(KeyError):
        df.is_computed("nope")                            # not a column at all
    with pytest.raises(KeyError):
        df.is_computed("ma:99")                           # valid directive, but not materialized


# --- repr: content-bearing, never panics (D×N) -----------------------------
@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_repr_is_safe_and_nonempty(d, n):
    df = A.frame(d, n)
    r = repr(df)
    assert isinstance(r, str) and "x" in r               # the column name appears
    assert isinstance(str(df["x"]), str)


# --- Row accessor -----------------------------------------------------------
def test_row_getitem_name_dict():
    r = volas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).iloc[0]
    assert r["a"] == 1.0 and r["b"] == 3.0
    assert r.name == 0                                   # the row's index label
    assert r.to_dict() == {"a": 1.0, "b": 3.0}


# F14 (FIXED): Row.to_numpy() returns the 1-D record (n,), like pandas
# df.iloc[0].to_numpy() — was a 2-D (1, n) frame export.
def test_row_to_numpy_is_1d():
    r = volas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).iloc[0]
    assert np.asarray(r.to_numpy()).shape == (2,)
