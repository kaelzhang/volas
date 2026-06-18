"""Arrow C-Data / C-Stream interop, masked NumPy export, and the integer-NA guard.

The zero-copy guarantees are proven at the Rust level (`volas-arrow`); here we check
the Python surface: the PyCapsule protocols, `from_arrow` / `to_arrow`, the
`to_numpy(masked=True)` pair, and the pandas-aligned raise on an integer cast of NA.
"""

import numpy as np
import pandas as pd
import pytest

import volas

pa = pytest.importorskip("pyarrow")


def s(values, dtype=None):
    """A single-column Series helper."""
    df = volas.DataFrame({"c": values})
    col = df["c"]
    return col.astype(dtype) if dtype else col


# --- integer cast of missing values must raise (pandas-aligned) ---------------

def test_to_numpy_int_dtype_on_na_raises():
    with pytest.raises(ValueError, match="missing"):
        s([1, None, 3]).to_numpy(dtype="int64")


def test_to_numpy_float_nan_to_int_raises():
    with pytest.raises(ValueError, match="missing"):
        s([1.0, float("nan")]).to_numpy(dtype="int64")


def test_to_numpy_dense_int_and_float_cast_still_work():
    assert s([1, 2, 3]).to_numpy(dtype="int64").dtype == np.int64
    assert s([1, None, 3]).to_numpy(dtype="float64")[1] != s([1, None, 3]).to_numpy(dtype="float64")[1]  # NaN


def test_array_protocol_dtype_honours_the_int_na_guard():
    # np.asarray(series, dtype=...) routes through __array__
    assert np.asarray(s([1, 2, 3]), dtype="int64").dtype == np.int64
    with pytest.raises(ValueError, match="missing"):
        np.asarray(s([1, None, 3]), dtype="int64")


def test_frame_to_numpy_int_on_na_raises_but_datetime_exempt():
    with pytest.raises(ValueError, match="missing"):
        volas.DataFrame({"a": [1, None], "b": [2, 3]}).to_numpy(dtype="int64")
    # a datetime NaT keeps its documented i64::MIN sentinel export (no raise)
    dt = volas.DataFrame({"t": pd.to_datetime(["2020-01-01", "NaT"])})
    out = dt.to_numpy(dtype="int64").ravel()
    assert out[1] == np.iinfo(np.int64).min


# --- to_numpy(masked=True) -> (values, mask) ----------------------------------

@pytest.mark.parametrize(
    "values,dtype,want_dtype",
    [
        ([1, None, 3], None, np.int64),
        ([1.5, float("nan"), 3.5], None, np.float64),
        ([True, False, None], None, np.bool_),
    ],
)
def test_masked_keeps_native_dtype(values, dtype, want_dtype):
    vals, mask = s(values, dtype).to_numpy(masked=True)
    assert vals.dtype == want_dtype
    assert mask.dtype == np.bool_
    assert mask.tolist() == [v is None or (isinstance(v, float) and np.isnan(v)) for v in values]


def test_masked_str_is_object_with_none():
    vals, mask = s(["a", None, "c"]).to_numpy(masked=True)
    assert list(vals) == ["a", None, "c"]
    assert mask.tolist() == [False, True, False]


def test_masked_datetime_and_narrow_dtypes():
    vals, mask = volas.DataFrame({"t": pd.to_datetime(["2020-01-01", "NaT"])})["t"].to_numpy(masked=True)
    assert vals.dtype == np.dtype("datetime64[ns]")
    assert mask.tolist() == [False, True]
    # narrow int32 / float32 storage arms, exercised through astype
    vi, mi = s([1, None, 3]).astype("int32").to_numpy(masked=True)
    assert vi.dtype == np.int32 and mi.tolist() == [False, True, False]
    vf, mf = s([1.5, None]).astype("float32").to_numpy(masked=True)
    assert vf.dtype == np.float32 and mf.tolist() == [False, True]


def test_masked_with_dtype_casts_values():
    vals, mask = s([1, None, 3]).to_numpy(dtype="float64", masked=True)
    assert vals.dtype == np.float64
    assert mask.tolist() == [False, True, False]


# --- Series Arrow PyCapsule protocol ------------------------------------------

def test_series_arrow_roundtrip_and_names():
    src = pa.array([10, 20, 30], type=pa.int64())
    out = volas.Series.from_arrow(src, name="x")
    assert out.to_list() == [10, 20, 30]
    assert pa.array(out).to_pylist() == [10, 20, 30]
    assert volas.Series.from_arrow(src).to_arrow().to_pylist() == [10, 20, 30]


def test_series_arrow_nullable_and_string_roundtrip():
    assert volas.Series.from_arrow(pa.array([1, None, 3])).to_list() == [1, volas.NA, 3]
    assert volas.Series.from_arrow(pa.array(["a", None, "c"])).to_list() == ["a", volas.NA, "c"]
    assert volas.Series.from_arrow(pa.array([1.5, None])).to_list()[0] == 1.5


def test_series_arrow_schema_capsule():
    cap = volas.DataFrame({"a": [1, 2]})["a"].__arrow_c_schema__()
    assert "arrow_schema" in repr(cap)


def test_series_zero_copy_through_import_export():
    src = pa.array([7, 8, 9, 10], type=pa.int64())
    reexport = pa.array(volas.Series.from_arrow(src))
    assert src.buffers()[1].address == reexport.buffers()[1].address


# --- DataFrame Arrow C-Stream protocol ----------------------------------------

def test_frame_arrow_roundtrip():
    df = volas.DataFrame({"i": [1, 2, 3], "s": ["a", "b", "c"], "f": [1.5, 2.5, 3.5]})
    tbl = pa.table(df)
    assert tbl.column_names == ["i", "s", "f"]
    back = volas.DataFrame.from_arrow(tbl)
    assert back["i"].to_list() == [1, 2, 3]
    assert back["s"].to_list() == ["a", "b", "c"]
    assert df.to_arrow().num_rows == 3


def test_frame_from_multi_chunk_table():
    chunked = pa.table({"v": pa.chunked_array([[1, 2], [3, 4]])})
    assert volas.DataFrame.from_arrow(chunked)["v"].to_list() == [1, 2, 3, 4]


def test_frame_arrow_nullable_roundtrip():
    df = volas.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    back = volas.DataFrame.from_arrow(pa.table(df))
    assert back["a"].to_list() == [1, volas.NA, 3]
    assert back["b"].to_list() == ["x", volas.NA, "z"]


# --- error paths --------------------------------------------------------------

def test_from_arrow_rejects_unsupported_type():
    # a date32 array has no volas column type
    with pytest.raises(ValueError):
        volas.Series.from_arrow(pa.array([1, 2], type=pa.date32()))


def test_from_arrow_rejects_wrong_capsule():
    class Bogus:
        def __arrow_c_array__(self, requested_schema=None):
            # a tuple of non-capsules / wrong names
            cap = volas.DataFrame({"a": [1]})["a"].__arrow_c_schema__()
            return (cap, cap)  # both "arrow_schema" — the array slot name is wrong

    with pytest.raises((ValueError, TypeError)):
        volas.Series.from_arrow(Bogus())
