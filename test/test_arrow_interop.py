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


# --- to_numpy(na_value=...) : the pandas-standard NA fill ---------------------

def test_na_value_with_int_dtype_keeps_dtype_and_fills():
    out = s([1, None, 3]).to_numpy(dtype="int64", na_value=0)
    assert out.dtype == np.int64
    assert out.tolist() == [1, 0, 3]


def test_na_value_default_dtype_keeps_float_like_pandas():
    # without an explicit dtype, an int column with NA still exports float64 (the
    # NA-model default); na_value just replaces the NaN — matching pandas.
    out = s([1, None, 3]).to_numpy(na_value=-1)
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, -1.0, 3.0]


def test_na_value_preserves_large_int_exactly():
    # the native (non-float-funnel) path keeps a value past 2**53 exact under na_value
    out = s([2**53 + 1, None]).to_numpy(dtype="int64", na_value=0)
    assert out[0] == 2**53 + 1


@pytest.mark.parametrize(
    "values,dtype,na_value,want",
    [
        ([1.5, float("nan")], None, 0.0, [1.5, 0.0]),         # float NaN filled
        (["a", None, "c"], None, "X", ["a", "X", "c"]),        # str object array filled
        ([1, None, 3], "int32", 9, [1, 9, 3]),                 # narrow int dtype
    ],
)
def test_na_value_across_dtypes(values, dtype, na_value, want):
    out = s(values, dtype).to_numpy(na_value=na_value)
    assert list(out) == want


def test_na_value_no_missing_is_a_plain_export():
    out = s([1, 2, 3]).to_numpy(dtype="int64", na_value=0)
    assert out.dtype == np.int64 and out.tolist() == [1, 2, 3]


def test_frame_na_value_fills_int_and_default():
    df = volas.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    assert df.to_numpy(dtype="int64", na_value=0).tolist() == [[1, 4], [0, 5], [3, 6]]
    out = df.to_numpy(na_value=-1)
    assert out.dtype == np.float64
    assert out.tolist() == [[1.0, 4.0], [-1.0, 5.0], [3.0, 6.0]]


def test_to_numpy_no_longer_accepts_masked():
    with pytest.raises(TypeError):
        s([1, 2, 3]).to_numpy(masked=True)


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


# --- DLPack zero-copy export --------------------------------------------------

@pytest.mark.parametrize(
    "values,dtype,want",
    [
        ([1.0, 2.0, 3.0], None, np.float64),
        ([1, 2, 3], None, np.int64),
        ([1, 2, 3], "float32", np.float32),
        ([1, 2, 3], "int32", np.int32),
        ([True, False, True], None, np.bool_),
    ],
)
def test_dlpack_dense_numeric_roundtrips(values, dtype, want):
    out = np.from_dlpack(s(values, dtype))
    assert out.dtype == want
    assert out.tolist() == values


def test_dlpack_device_is_cpu():
    assert s([1.0, 2.0]).__dlpack_device__() == (1, 0)


def test_dlpack_is_zero_copy():
    src = pa.array([5, 6, 7, 8], type=pa.int64())
    arr = np.from_dlpack(volas.Series.from_arrow(src))
    assert arr.ctypes.data == src.buffers()[1].address


def test_dlpack_unconsumed_capsule_frees_cleanly():
    # creating and dropping the capsule without a consumer must run the deleter
    for _ in range(200):
        cap = s([1.0, 2.0, 3.0]).__dlpack__()
        del cap


@pytest.mark.parametrize(
    "series,exc",
    [
        (s([1, None, 3]), ValueError),                       # int + NA: no DLPack null mask
        (s(["a", "b"]), ValueError),                         # str: no DLPack dtype
        (volas.DataFrame({"t": pd.to_datetime(["2020-01-01", "NaT"])})["t"], ValueError),  # datetime
    ],
)
def test_dlpack_rejects_unsupported(series, exc):
    with pytest.raises(exc):
        series.__dlpack__()
    # float NaN, by contrast, is an in-band value and exports fine
    np.from_dlpack(s([1.5, float("nan"), 2.5]))


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
