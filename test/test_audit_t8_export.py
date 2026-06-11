"""Systematic audit — T8 (boundary): the export surface.

  Series.to_list / Series.to_numpy / __array__ / DataFrame.to_numpy

These leave the volas type-system, so the internal "no object dtype" lock (C3)
does NOT bind here — an object array at the boundary is legitimate (owner ruling,
P2-01: to_numpy/to_pandas exist to produce system-external objects). Oracle is
`# pandas` for the export shape/values.

  * to_list — fully contract-clear: a python list, missing -> the volas.NA
    singleton, length preserved. Asserted across the whole D×N matrix.
  * to_numpy / __array__ — typed array, values preserved, for the no-NA states;
    the NA *representation* (object-with-NA vs float-with-NaN per source dtype)
    is an un-ruled boundary design choice -> backlog meta-test, not guessed.

Cell IDs:  TOLIST:<dtype>/N=<n> · TONUMPY:<dtype>/N=<n> · TONUMPY.frame/F=<f>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import volas
from . import audit_dims as A

_NO_NA = ("N0", "N4")  # dense, single — present values, nothing missing


def _present_values(d: str, n: str):
    """The non-NA values, in order, that audit_dims.series(d, n) carries."""
    p = A._PRESENT[d]
    return {
        "N0": [p[0], p[1], p[2]],
        "N1": [p[0], p[2]],     # middle held out
        "N2": [], "N3": [],
        "N4": [p[0]],
    }[n]


# --- to_list: dtype-specific NA representation, full D×N --------------------
# Per the C2 contract the NA *surface* is dtype-specific by design: float holes
# stay np.nan (NaN-as-NA), every other dtype surfaces the volas.NA singleton
# (validity bitmap / NaT). isna() unifies them; the scalar surface does not.
@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_to_list_matrix(d, n):
    s = A.series(d, n)
    lst = s.to_list()
    assert isinstance(lst, list)
    assert len(lst) == len(s)
    # missing surfaces as nan for float (# C2 NaN-as-NA), else as volas.NA.
    if d in ("f64", "f32"):
        miss = [isinstance(x, float) and math.isnan(x) for x in lst]
    else:
        miss = [x is volas.NA for x in lst]
    assert miss == s.isna().to_list(), f"TOLIST:{d}/{n} NA surface"
    # present numeric values survive exactly (the integer-valued N-basis).
    if d in ("f64", "f32", "i64", "i32", "bool"):
        present = [float(x) for x, m in zip(lst, s.isna().to_list()) if not m]
        assert present == [float(v) for v in _present_values(d, n)]


# --- to_numpy: typed array + values, no-NA states --------------------------
@pytest.mark.parametrize("n", _NO_NA)
@pytest.mark.parametrize("d", A.DTYPES)
def test_to_numpy_series_no_na(d, n):
    s = A.series(d, n)
    arr = s.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(s),)
    if d == "datetime":
        assert arr.dtype.kind == "M"          # datetime64, value detail in T12
    elif d in ("f64", "f32", "i64", "i32", "bool"):
        assert [float(x) for x in arr.tolist()] == [float(v) for v in A._PRESENT[d][: len(s)]]
    else:  # str
        assert list(arr.tolist()) == list(A._PRESENT[d][: len(s)])


def test_to_numpy_empty():
    for d in A.DTYPES:
        arr = A.series(d, "N3").to_numpy()
        assert isinstance(arr, np.ndarray) and arr.shape == (0,)


@pytest.mark.parametrize("d", A.DTYPES)
def test_array_protocol_matches_to_numpy(d):
    """np.asarray(series) ≡ series.to_numpy() (the __array__ hook)."""
    s = A.series(d, "N0")
    assert np.asarray(s).tolist() == s.to_numpy().tolist()


def test_to_numpy_na_representation():
    """The NA-bearing export representation, per the NA-model interop ruling:
    float keeps float64+NaN; int demotes to float64+NaN (numpy int has no NA);
    BOOL exports as an OBJECT array (True/nan/False — F17, float64 would destroy
    the bool identity); str is object with None; datetime keeps NaT."""
    import math
    for n in ("N1", "N2"):
        assert A.series("f64", n).to_numpy().dtype == np.float64
        assert A.series("i64", n).to_numpy().dtype == np.float64    # NaN-demoted
        b = A.series("bool", n).to_numpy()
        assert b.dtype == object, "F17: bool+NA exports as object"
        if n == "N1":
            assert b[0] is True and math.isnan(b[1]) and b[2] is True
        assert A.series("str", n).to_numpy().dtype == object
        assert str(A.series("datetime", n).to_numpy().dtype).startswith("datetime64")


# --- DataFrame.to_numpy: the F (frame-composition) axis, no NA -------------
@pytest.mark.parametrize("f", A.FRAMES)
def test_to_numpy_frame(f):
    df = A.wide_frame(f)
    arr = df.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, len(df.columns))
    if f == "single":
        assert arr.tolist() == [[1.0], [2.0], [3.0]]
    elif f == "homogeneous":
        assert arr.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    else:
        # mixed-kind -> a boundary object array (C3 does not bind on export).
        assert arr.dtype == object
        assert [row[0] for row in arr.tolist()] == [1.0, 2.0, 3.0]
