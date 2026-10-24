//! Coerce Python objects into volas columns / dtypes for construction.


use numpy::{Element, PyReadonlyArray1};
use std::borrow::Cow;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use volas_core::{
    datetime, Column, DType,
};

#[allow(unused_imports)]
use crate::*;

/// Borrow a contiguous input, or copy a strided input in logical element order.
pub(crate) fn numpy_values<'a, T: Element + Copy>(a: &'a PyReadonlyArray1<'_, T>) -> Cow<'a, [T]> {
    match a.as_slice() {
        Ok(values) => Cow::Borrowed(values),
        Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
    }
}

/// The `(unit, count)` of a NumPy datetime64 (`kind == "M"`) / timedelta64
/// (`kind == "m"`) array or scalar (`np.datetime_data`), or `None` for any other
/// object — the one place NumPy's time-unit vocabulary is read.
pub(crate) fn numpy_time_unit(obj: &Bound<'_, PyAny>, kind: &str) -> PyResult<Option<(String, i64)>> {
    let Ok(dtype) = obj.getattr("dtype") else { return Ok(None) };
    if dtype.getattr("kind").and_then(|k| k.extract::<String>()).ok().as_deref() != Some(kind) {
        return Ok(None);
    }
    let numpy = obj.py().import("numpy")?;
    Ok(Some(numpy.call_method1("datetime_data", (dtype,))?.extract()?))
}

/// A NumPy datetime64 / timedelta64 SCALAR of `kind` as `(unit, count, raw)` —
/// its raw count in its native unit (`NaT` is `i64::MIN`); `None` for any other
/// object. The core unit owner then scales `raw` (checked).
pub(crate) fn numpy_time_value(
    obj: &Bound<'_, PyAny>,
    kind: &str,
) -> Option<PyResult<(String, i64, i64)>> {
    let read = || -> PyResult<Option<(String, i64, i64)>> {
        let Some((unit, count)) = numpy_time_unit(obj, kind)? else { return Ok(None) };
        let raw = obj.call_method1("astype", ("int64",))?.call_method0("item")?.extract::<i64>()?;
        Ok(Some((unit, count, raw)))
    };
    read().transpose()
}

/// Parse a pandas-style dtype string to a volas [`DType`].
pub(crate) fn parse_dtype(s: &str) -> PyResult<DType> {
    Ok(match s {
        "float" | "float64" | "float_" | "double" | "f64" => DType::F64,
        "float32" | "single" | "f32" => DType::F32,
        "int" | "int64" | "int_" | "long" | "i64" => DType::I64,
        "int32" | "i32" => DType::I32,
        "bool" | "boolean" => DType::Bool,
        "str" | "string" => DType::Utf8,
        "datetime" | "datetime64" | "datetime64[ns]" => DType::Datetime,
        // C3: volas has no object dtype. Reject pandas's catch-all spelling at the
        // boundary rather than silently aliasing it to str (which would stringify
        // numbers and reintroduce the object vocabulary into the API).
        "object" | "O" => {
            return Err(PyValueError::new_err(
                "volas has no object dtype; use \"str\" for text columns",
            ))
        }
        _ => return Err(PyValueError::new_err(format!("unknown dtype {s:?}"))),
    })
}

/// The epoch unit a `datetime64[...]` dtype string implies, or `None` when `s` is
/// not a datetime dtype. Bare `datetime` / `datetime64` / `datetime64[ns]` mean
/// nanoseconds; `datetime64[s|ms|us]` carry their own unit (pandas-aligned).
pub(crate) fn datetime_unit_of(s: &str) -> Option<&'static str> {
    match s {
        "datetime" | "datetime64" | "datetime64[ns]" => Some("ns"),
        "datetime64[s]" => Some("s"),
        "datetime64[ms]" => Some("ms"),
        "datetime64[us]" => Some("us"),
        _ => None,
    }
}

/// A NumPy masked array (`numpy.ma.MaskedArray`): its mask marks missing cells,
/// so they become volas NA in the column built from the underlying data — the mask
/// is never dropped (the masked slots' data values are unspecified). `None` for
/// anything that is not a masked array.
pub(crate) fn masked_array_column(v: &Bound<'_, PyAny>) -> PyResult<Option<Column>> {
    if !v.hasattr("mask")? {
        return Ok(None);
    }
    let ma = v.py().import("numpy.ma")?;
    if !v.is_instance(&ma.getattr("MaskedArray")?)? {
        return Ok(None);
    }
    let col = pyany_to_column(&v.getattr("data")?)?;
    let mask = ma.call_method1("getmaskarray", (v,))?.call_method0("tolist")?;
    let missing: Vec<bool> = mask.extract()?;
    Ok(Some(col.with_missing_mask(&missing).map_err(pyerr)?))
}

/// The interop-private masked construction channel: a 3-tuple
/// `(values, missing_mask, dtype)` builds a TYPED column with a validity bitmap
/// in one step — `from_pandas`'s faithful inverse of `to_pandas`'s
/// `IntegerArray(values, mask)` export (F9/F10: a valueless or Int32 nullable
/// column keeps its declared dtype instead of decaying to float64/int64).
fn masked_typed_column(t: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Option<Column>> {
    if t.len() != 3 {
        return Ok(None);
    }
    let Ok(dtype) = t.get_item(2)?.extract::<String>() else {
        return Ok(None);
    };
    let Ok(mask) = t.get_item(1)?.extract::<Vec<bool>>() else {
        return Ok(None);
    };
    let dt = parse_dtype(&dtype)?;
    let vals = t.get_item(0)?;
    let dense = match dt {
        DType::I64 => Column::i64(vals.extract()?),
        DType::I32 => Column::i32(vals.extract()?),
        DType::Bool => Column::bool(vals.extract()?),
        DType::Utf8 => Column::str(vals.extract()?),
        DType::F64 => Column::f64(vals.extract()?),
        _ => {
            return Err(PyTypeError::new_err(
                "masked construction supports int64/int32/bool/str/float64",
            ))
        }
    };
    // mask = True means missing (pandas .isna()); a mask of another length raises
    Ok(Some(dense.with_missing_mask(&mask).map_err(pyerr)?))
}

pub(crate) fn pyany_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(t) = v.downcast::<pyo3::types::PyTuple>() {
        if let Some(c) = masked_typed_column(t)? {
            return Ok(c);
        }
        // any other tuple is a sequence of values: the list rules (incl. the
        // exactness guards) apply unchanged
        return pyany_to_column(PyList::new(v.py(), t)?.as_any());
    }
    // A volas Series is an internal column, not a numpy boundary value: clone its
    // Column directly (preserving dtype + volas.NA). Without this it would fall
    // through to the array path below, invoking Series.__array__ and re-importing
    // the lossy float64 export — contract C1/C2 (parity with `df['x'] = series`).
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data.clone());
    }
    if let Some(col) = masked_array_column(v)? {
        return Ok(col);
    }
    // A numpy datetime64 array -> a Datetime column (carried as epoch-ns). This is the
    // pandas-aligned ingestion (`pd.DataFrame` accepts datetime64 arrays) and the path
    // `from_pandas` relies on. The raw values are read in their NATIVE unit (the int64
    // view) and scaled by the core unit owner — checked, exact for sub-ns units — never
    // by NumPy's `astype('datetime64[ns]')`, which wraps out-of-range values silently.
    if let Some((unit, count)) = numpy_time_unit(v, "M")? {
        let view = v.call_method1("view", ("int64",))?;
        let a = view.extract::<PyReadonlyArray1<i64>>()?;
        let ns = datetime::datetime64_values_to_ns(&numpy_values(&a), &unit, count).map_err(pyerr)?;
        return Ok(Column::datetime(ns));
    }
    if let Some(col) = exact_array_kind(v)? {
        return Ok(col);
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<f64>>() {
        return Ok(Column::f64(numpy_values(&a).into_owned()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<f32>>() {
        return Ok(Column::f32(numpy_values(&a).into_owned()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i64>>() {
        return Ok(Column::i64(numpy_values(&a).into_owned()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i32>>() {
        return Ok(Column::i32(numpy_values(&a).into_owned()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<bool>>() {
        return Ok(Column::bool(numpy_values(&a).into_owned()));
    }
    // A Python list infers its dtype like pandas/NumPy: all-bool -> bool, all-int
    // -> int64, else float64 (empty -> float64). Bool / int are tried before float
    // because a strict bool/int extraction won't match a float list.
    if let Ok(vv) = v.extract::<Vec<bool>>() {
        if !vv.is_empty() {
            return Ok(Column::bool(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<i64>>() {
        if !vv.is_empty() {
            return Ok(Column::i64(vv));
        }
    }
    // Values with NO exact volas representation must error (C4), never silently
    // demote to lossy float64:
    //  - F32: a python int beyond i64 (no uint64 dtype; 2^63 and 2^63+1 would
    //    collapse to the same f64).
    //  - F41: a Decimal (exact-decimal intent; no object dtype to keep it).
    // And a list of datetime-like scalars builds a datetime column (F20) rather
    // than falling through to the generic error.
    if let Ok(list) = v.downcast::<PyList>() {
        if let Some(col) = list_special_cases(list)? {
            return Ok(col);
        }
    }
    if let Ok(vv) = v.extract::<Vec<f64>>() {
        return Ok(Column::f64(vv));
    }
    if let Ok(vv) = v.extract::<Vec<String>>() {
        return Ok(Column::str(vv));
    }
    // A list containing `None` carries missing values: infer the present values'
    // dtype (bool -> int -> float, like the dense case) and mark each `None` cell
    // `volas.NA`. int/bool keep their dtype (pandas upcasts to float/object).
    if let Ok(vv) = v.extract::<Vec<Option<bool>>>() {
        if vv.iter().any(Option::is_some) {
            return Ok(Column::bool_from_options(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<Option<i64>>>() {
        if vv.iter().any(Option::is_some) {
            return Ok(Column::i64_from_options(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<Option<f64>>>() {
        // a float carries missing in-band as NaN (also the all-`None` fallback).
        return Ok(Column::f64(vv.iter().map(|x| x.unwrap_or(f64::NAN)).collect()));
    }
    if let Ok(vv) = v.extract::<Vec<Option<String>>>() {
        return Ok(Column::str_from_options(vv));
    }
    // A list may carry the `volas.NA` symbol itself (not `None`) — normalise it to
    // `None` and retry, so `to_list()` output round-trips back into a frame.
    if let Ok(list) = v.downcast::<PyList>() {
        let py = v.py();
        let na_obj = na(py);
        let na_bound = na_obj.bind(py);
        if list.iter().any(|item| item.is(na_bound)) {
            let items: Vec<Bound<'_, PyAny>> = list
                .iter()
                .map(|item| if item.is(na_bound) { py.None().into_bound(py) } else { item })
                .collect();
            let normalized = PyList::new(py, items)?;
            return pyany_to_column(normalized.as_any());
        }
    }
    Err(PyTypeError::new_err(
        "column values must be a 1-D numeric array, a list of numbers, or a list of strings",
    ))
}

/// The NumPy array kinds the typed fast paths cannot take exactly (contract #20):
/// an unsigned array widens to int64 when every value fits (`uint64` past
/// `i64::MAX` raises — volas has no uint64), an `object` array is read as the list
/// of its Python values (so the list guards for Decimal / big ints / mixed
/// int+float apply), and a complex array raises (no complex dtype). `None` for
/// every other value.
fn exact_array_kind(v: &Bound<'_, PyAny>) -> PyResult<Option<Column>> {
    let Ok(dtype) = v.getattr("dtype") else {
        return Ok(None);
    };
    let (Ok(kind), Ok(size)) = (
        dtype.getattr("kind").and_then(|k| k.extract::<String>()),
        dtype.getattr("itemsize").and_then(|k| k.extract::<usize>()),
    ) else {
        return Ok(None);
    };
    match kind.as_str() {
        "c" => Err(PyTypeError::new_err(
            "complex values have no volas dtype; take .real / .imag / abs() explicitly",
        )),
        "O" => pyany_to_column(&v.call_method0("tolist")?).map(Some),
        "u" if size < 8 => pyany_to_column(&v.call_method1("astype", ("int64",))?).map(Some),
        "u" => {
            let a = v.extract::<PyReadonlyArray1<u64>>()?;
            let vals = numpy_values(&a)
                .iter()
                .map(|&x| i64::try_from(x).map_err(|_| int64_overflow()))
                .collect::<PyResult<Vec<i64>>>()?;
            Ok(Some(Column::i64(vals)))
        }
        _ => Ok(None),
    }
}

/// The error for an integer past int64 (volas has no uint64 / big-int dtype).
fn int64_overflow() -> PyErr {
    PyOverflowError::new_err(
        "integer does not fit int64 (volas has no uint64); pass floats explicitly \
         if lossy float64 is acceptable",
    )
}

/// One Python scalar as a 1-row [`Column`] of EXACTLY `dtype` for the bar-append
/// path (`df.append(dict)` / `Row` / the index label): the same lossless
/// target-typed conversion as every other scalar write ([`scalar_to_column`],
/// contract #20) — one rule, so a dict append and an `iat` write agree.
pub(crate) fn bar_scalar_to_column(v: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Column> {
    crate::frame_index::scalar_to_column(v, dtype)
}

/// The list shapes that must NOT fall through to the lossy-float path: a list of
/// datetime-like scalars (F20 -> a Datetime column), an out-of-i64 integer (F32
/// -> OverflowError), or a Decimal (F41 -> TypeError). Returns `Ok(None)` when
/// the list is none of these and ordinary inference should continue.
fn list_special_cases(list: &Bound<'_, PyList>) -> PyResult<Option<Column>> {
    let mut all_datetime = !list.is_empty();
    let mut any_present = false;
    let (mut ints, mut any_float) = (Vec::new(), false);
    for item in list.iter() {
        if item.get_type().name()? == "Decimal" {
            return Err(PyTypeError::new_err(
                "Decimal has no exact volas dtype (no object dtype); convert explicitly \
                 (e.g. float(x) for a float64 column) if lossy float is acceptable",
            ));
        }
        if item.is_instance_of::<pyo3::types::PyComplex>() {
            return Err(PyTypeError::new_err(
                "complex values have no volas dtype; take .real / .imag / abs() explicitly",
            ));
        }
        // An integer element (a Python int or a NumPy integer) that does not fit
        // i64 has no exact volas representation (no uint64) -> error per item, so
        // a valid [int, None, ...] list with in-range ints is untouched.
        if let Some(i) = int_item(&item) {
            ints.push(i.ok_or_else(int64_overflow)?);
        } else if item.is_instance_of::<pyo3::types::PyFloat>() || type_name_starts(&item, &["float"]) {
            any_float = true; // a Python float or a NumPy floating scalar
        }
        if item.is_none() {
            // None is a missing slot in either interpretation.
        } else {
            any_present = true;
            if !is_datetime_like(&item) {
                all_datetime = false;
            }
        }
    }
    if all_datetime && any_present {
        // F20: a list of datetime scalars (python datetime/date, pd.Timestamp,
        // volas.Timestamp, np.datetime64) infers a datetime column, like pandas.
        // None becomes the NaT slot (D2 sentinel).
        let mut ns = Vec::with_capacity(list.len());
        for item in list.iter() {
            if item.is_none() {
                ns.push(i64::MIN);
            } else {
                ns.push(crate::scalar::parse_ts(&item)?);
            }
        }
        return Ok(Some(Column::datetime(ns)));
    }
    // Integers mixed with floats infer float64 (like pandas): through the lossless
    // owner, so an integer float64 cannot hold raises instead of rounding (#20).
    if any_float {
        Column::i64(ints).convert_lossless(DType::F64).map_err(pyerr)?;
    }
    Ok(None)
}

/// `Some(Some(i))` for an integer item that fits i64, `Some(None)` for one past
/// it, `None` for a non-integer (bools are not counted as integers here).
fn int_item(item: &Bound<'_, PyAny>) -> Option<Option<i64>> {
    let integral = item.is_instance_of::<pyo3::types::PyInt>()
        || type_name_starts(item, &["int", "uint"]); // a NumPy integer scalar
    (integral && item.extract::<bool>().is_err()).then(|| item.extract::<i64>().ok())
}

/// Whether the item's type name starts with one of `prefixes` (NumPy scalar types
/// such as `int64` / `uint64` / `float32`, which are not Python int / float).
fn type_name_starts(item: &Bound<'_, PyAny>, prefixes: &[&str]) -> bool {
    item.get_type()
        .name()
        .is_ok_and(|n| n.to_str().is_ok_and(|n| prefixes.iter().any(|p| n.starts_with(p))))
}

/// Whether a scalar is datetime-like: python datetime/date (and subclasses, e.g.
/// pd.Timestamp), volas.Timestamp, or a numpy datetime64 scalar.
fn is_datetime_like(item: &Bound<'_, PyAny>) -> bool {
    if item.is_instance_of::<pyo3::types::PyDate>()
        || item.is_instance_of::<pyo3::types::PyDateTime>()
        || item.extract::<PyRef<crate::scalar::PyTimestamp>>().is_ok()
    {
        return true;
    }
    item.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "M")
        .unwrap_or(false)
}
