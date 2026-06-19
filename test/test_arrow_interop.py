"""Arrow C-Data / C-Stream interop, DLPack export, NumPy export, and the NA guard.

The zero-copy guarantees are proven at the Rust level (`volas-arrow`); here we check
the Python surface: the Arrow PyCapsule protocols, `from_arrow` / `to_arrow`, DLPack
(`__dlpack__` versioned read-only / copy / device validation), `to_numpy(na_value=...)`,
and the pandas-aligned raise on an integer cast of NA.
"""

from datetime import date
from decimal import Decimal

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
    "series",
    [
        s([1, None, 3]),                       # int + NA: no DLPack null mask
        s(["a", "b"]),                         # str: no DLPack dtype
        volas.DataFrame({"t": pd.to_datetime(["2020-01-01", "NaT"])})["t"],  # datetime
    ],
)
def test_dlpack_rejects_unsupported(series):
    # "cannot export to DLPack" is a BufferError per the Array API (matching the
    # device/stream rejections), not a ValueError.
    with pytest.raises(BufferError):
        series.__dlpack__()
    # float NaN, by contrast, is an in-band value and exports fine
    np.from_dlpack(s([1.5, float("nan"), 2.5]))


# --- error paths --------------------------------------------------------------

def test_from_arrow_rejects_unsupported_type():
    # a duration column has no volas column type (date32 / decimal / narrow ints ARE
    # supported now)
    with pytest.raises(ValueError):
        volas.Series.from_arrow(pa.array([1, 2], type=pa.duration("s")))


def test_from_arrow_extended_types():
    # decimal -> f64 (lossy), narrow/unsigned int -> int64, date32 -> ns datetime
    assert volas.Series.from_arrow(
        pa.array([Decimal("1.50"), None], type=pa.decimal128(18, 2))
    ).to_list()[0] == 1.5
    assert volas.Series.from_arrow(pa.array([1, 2], type=pa.uint32())).to_list() == [1, 2]
    assert volas.Series.from_arrow(pa.array([1, 2], type=pa.int16())).to_list() == [1, 2]
    d = volas.Series.from_arrow(pa.array([date(2020, 1, 2), None], type=pa.date32())).to_list()
    assert d[1] is volas.NA


def test_from_arrow_dictionary_and_decimal256():
    # categorical (dictionary-encoded) strings — how parquet / pandas `category` arrive
    d = pa.array(["AAPL", "MSFT", "AAPL"]).dictionary_encode()
    assert volas.Series.from_arrow(d).to_list() == ["AAPL", "MSFT", "AAPL"]
    # 256-bit decimal -> f64 (lossy), like decimal128
    dec = pa.array([Decimal("1.50"), None], type=pa.decimal256(20, 2))
    assert volas.Series.from_arrow(dec).to_list()[0] == 1.5


def test_from_arrow_string_view_and_null():
    # canonical data defaults string columns to string_view — DataFrame.from_arrow must
    # take it directly (no manual cast(pa.string()) workaround).
    tbl = pa.table({"symbol": pa.array(["AAPL", None, "MSFT"], type=pa.string_view())})
    assert volas.DataFrame.from_arrow(tbl)["symbol"].to_list() == ["AAPL", volas.NA, "MSFT"]
    # a typeless all-null column imports as an all-NA column (length preserved)
    n = volas.Series.from_arrow(pa.array([None, None, None], type=pa.null()))
    assert len(n) == 3 and all(v is volas.NA or v != v for v in n.to_list())


def test_from_arrow_rejects_wrong_capsule():
    class Bogus:
        def __arrow_c_array__(self, requested_schema=None):
            # a tuple of non-capsules / wrong names
            cap = volas.DataFrame({"a": [1]})["a"].__arrow_c_schema__()
            return (cap, cap)  # both "arrow_schema" — the array slot name is wrong

    with pytest.raises((ValueError, TypeError)):
        volas.Series.from_arrow(Bogus())


# --- __array__ copy semantics (numpy 2.0) + DLPack protocol (versioned/read-only) ---

def test_array_protocol_copy_false_raises():
    # numpy 2.0: copy=False means "must not copy"; to_numpy always copies, so raise.
    s = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]
    with pytest.raises(ValueError):
        np.array(s, copy=False)
    assert np.asarray(s).tolist() == [1.0, 2.0, 3.0]
    assert np.array(s, copy=True).tolist() == [1.0, 2.0, 3.0]


def test_dlpack_default_view_is_read_only():
    # the borrowed view is exported read-only (versioned DLPack flag) so a consumer
    # cannot write through it into volas's buffer and bypass copy-on-write.
    s = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]
    arr = np.from_dlpack(s)
    assert not arr.flags.writeable


def test_dlpack_copy_true_is_independent_and_writable():
    s = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]
    arr = np.from_dlpack(s, copy=True)
    assert arr.flags.writeable
    arr[0] = 999.0
    assert s.to_list()[0] == 1.0  # the frame is untouched (copy was independent)


def test_dlpack_rejects_non_cpu_device_and_stream():
    s = volas.DataFrame({"a": [1.0, 2.0]})["a"]
    with pytest.raises(BufferError):
        s.__dlpack__(dl_device=(2, 0))  # a non-CPU device cannot be honored
    with pytest.raises(BufferError):
        s.__dlpack__(stream=5)  # a CPU export takes no stream


def test_dlpack_unversioned_borrow_is_copied_not_an_alias():
    # A pre-1.0 (unversioned) DLPack consumer cannot be handed a read-only flag, so a
    # zero-copy borrow would alias volas's buffer (and a writing consumer would bypass
    # copy-on-write). The unversioned path must therefore hand back an independent copy —
    # observable here as a *different data pointer* from the versioned zero-copy borrow.
    sv = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]

    class OldConsumer:  # ignores max_version → forces the legacy "dltensor" capsule
        def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
            return sv.__dlpack__()

        def __dlpack_device__(self):
            return sv.__dlpack_device__()

    borrowed = np.from_dlpack(sv)  # modern numpy → versioned, zero-copy view of the buffer
    copied = np.from_dlpack(OldConsumer())  # unversioned → forced independent copy
    assert copied.ctypes.data != borrowed.ctypes.data  # distinct memory: it was copied
    assert copied.tolist() == [1.0, 2.0, 3.0]  # with the same values


def test_dlpack_unversioned_copy_false_is_refused():
    sv = volas.DataFrame({"a": [1.0, 2.0]})["a"]
    with pytest.raises(BufferError):
        sv.__dlpack__(copy=False)  # no max_version → cannot serve a safe zero-copy borrow


# --- low-level versioned-capsule inspection (flags + empty data pointer) -------

import ctypes  # noqa: E402


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_int32), ("minor", ctypes.c_int32)]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", _DLVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


_FLAG_READ_ONLY = 1 << 0
_FLAG_IS_COPIED = 1 << 1


def _versioned_tensor(capsule):
    # read (without consuming) the versioned managed tensor the capsule wraps
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = get(capsule, b"dltensor_versioned")
    return _DLManagedTensorVersioned.from_address(ptr)


def test_dlpack_versioned_flags_distinguish_borrow_and_copy():
    sv = volas.DataFrame({"a": [1.0, 2.0, 3.0]})["a"]
    borrow = sv.__dlpack__(max_version=(1, 0))
    t = _versioned_tensor(borrow)
    assert t.flags & _FLAG_READ_ONLY and not (t.flags & _FLAG_IS_COPIED)
    copied = sv.__dlpack__(max_version=(1, 0), copy=True)
    t2 = _versioned_tensor(copied)
    assert (t2.flags & _FLAG_IS_COPIED) and not (t2.flags & _FLAG_READ_ONLY)


def test_dlpack_empty_column_has_null_data_pointer():
    empty = volas.DataFrame({"a": [1.0]})["a"].iloc[1:1]  # a size-0 float column
    cap = empty.__dlpack__(max_version=(1, 0))
    t = _versioned_tensor(cap)
    assert not t.dl_tensor.data  # size-0 → NULL data pointer (ctypes reads NULL as None/0)
