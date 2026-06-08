"""Multi-dtype storage (float32 / int32) parity with pandas 3.0.

Columns can be stored as f32 / i32 (from numpy arrays, the `dtype=` constructor,
or `astype`); element access and reductions return dtype-faithful numpy scalars
(np.float32 / np.int32), and the dtype-preserving transforms keep the narrow type.
Indicator computation is unaffected (f32/i32 convert to f64 at the kernel edge).
"""

import numpy as np
import pandas as pd

import volas


def _f32(values):
    return volas.DataFrame({"a": np.array(values, dtype=np.float32)})["a"]


def _i32(values):
    return volas.DataFrame({"a": np.array(values, dtype=np.int32)})["a"]


# --- storage + ingestion ----------------------------------------------------

def test_storage_from_numpy_arrays():
    assert _f32([1.5, 2.5]).dtype == "float32"
    assert _i32([3, 4]).dtype == "int32"


def test_constructor_dtype_param():
    d = volas.DataFrame({"a": [0.0], "b": [0]}, dtype="float32")
    assert d["a"].dtype == "float32" and d["b"].dtype == "float32"
    p = pd.DataFrame({"a": [0.0], "b": [0]}, dtype="float32")
    assert [d["a"].dtype, d["b"].dtype] == [str(t) for t in p.dtypes]


def test_astype_to_f32_i32():
    s = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]
    assert s.astype("float32").dtype == "float32"
    assert s.astype("int32").dtype == "int32"
    import pytest
    with pytest.raises(Exception):
        volas.DataFrame({"a": [2.5]})["a"].astype("int32")  # non-integral


# --- element access -> dtype-faithful numpy scalars -------------------------

def test_element_access_numpy_dtype():
    assert isinstance(_f32([1.5, 2.5])[0], np.float32)
    assert isinstance(_i32([3, 4])[0], np.int32)


# --- reductions: pandas dtype rules -----------------------------------------

def test_f32_reductions_stay_float32():
    s = _f32([1.5, 2.5, 3.5])
    for op in ("sum", "min", "max", "mean", "std"):
        assert isinstance(getattr(s, op)(), np.float32), op


def test_i32_reduction_dtypes():
    s = _i32([3, 1, 4, 1, 5])
    assert isinstance(s.sum(), np.int64)  # int32 sum promotes to int64 (numpy)
    assert isinstance(s.prod(), np.int64)
    assert isinstance(s.min(), np.int32)  # min/max keep int32
    assert isinstance(s.max(), np.int32)
    assert isinstance(s.mean(), np.float64)
    # values match pandas
    p = pd.Series([3, 1, 4, 1, 5], dtype="int32")
    assert s.sum() == p.sum() and s.min() == p.min()


# --- transforms preserve the narrow dtype -----------------------------------

def test_transforms_preserve_narrow_dtype():
    f, i = _f32([1.5, 2.5, 3.5]), _i32([3, 1, 4])
    assert f.cumsum().dtype == "float32" and f.abs().dtype == "float32"
    assert f.round(0).dtype == "float32" and f.clip(2.0, 3.0).dtype == "float32"
    assert (f + f).dtype == "float32" and (f * f).dtype == "float32"
    assert f.where(f > 2, 0.0).dtype == "float32"
    assert i.cumsum().dtype == "int32" and i.abs().dtype == "int32"
    assert (i + i).dtype == "int32"
    # mixed promotes (numpy): f32 + f64 -> f64; f32 / f32 -> f64 (true division)
    f64 = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]
    assert (f + f64).dtype == "float64"
    assert (f / f).dtype == "float64"


def test_to_numpy_keeps_dtype():
    assert str(_f32([1.5]).to_numpy().dtype) == "float32"
    assert str(_i32([3]).to_numpy().dtype) == "int32"


# --- indicators still work on a narrow-dtype frame (convert to f64) ----------

def test_indicator_on_f32_frame():
    df = volas.DataFrame({"close": np.arange(50, dtype=np.float32)})
    ma = df["ma:5"]
    assert ma.dtype == "float64"  # indicator computes in f64
    np.testing.assert_allclose(np.asarray(ma.to_numpy())[5:10], np.arange(3, 8, dtype=float))
