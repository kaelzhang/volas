//! Coerce Python objects into volas columns / dtypes for construction.


use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use volas_core::{
    Column, DType, Validity,
};

#[allow(unused_imports)]
use crate::*;

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

pub(crate) fn pyany_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    // A volas Series is an internal column, not a numpy boundary value: clone its
    // Column directly (preserving dtype + volas.NA). Without this it would fall
    // through to the array path below, invoking Series.__array__ and re-importing
    // the lossy float64 export — contract C1/C2 (parity with `df['x'] = series`).
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data.clone());
    }
    // A numpy datetime64 array -> a Datetime column (carried as epoch-ns). This is the
    // pandas-aligned ingestion (`pd.DataFrame` accepts datetime64 arrays) and the no-copy
    // path `from_pandas` relies on. Normalise any datetime64[unit] to ns, then take its
    // int64 view (datetime64[ns] is int64 epoch-ns underneath).
    if v.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "M")
        .unwrap_or(false)
    {
        let ns = v.call_method1("astype", ("datetime64[ns]",))?;
        let view = ns.call_method1("view", ("int64",))?;
        let a = view.extract::<PyReadonlyArray1<i64>>()?;
        return Ok(Column::datetime(a.as_slice()?.to_vec())); // int64 ns view == epoch-ns
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<f64>>() {
        return Ok(Column::f64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<f32>>() {
        return Ok(Column::f32(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i64>>() {
        return Ok(Column::i64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i32>>() {
        return Ok(Column::i32(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<bool>>() {
        return Ok(Column::bool(a.as_slice()?.to_vec()));
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
            return Ok(option_bool_column(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<Option<i64>>>() {
        if vv.iter().any(Option::is_some) {
            return Ok(option_i64_column(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<Option<f64>>>() {
        // a float carries missing in-band as NaN (also the all-`None` fallback).
        return Ok(Column::f64(vv.iter().map(|x| x.unwrap_or(f64::NAN)).collect()));
    }
    if let Ok(vv) = v.extract::<Vec<Option<String>>>() {
        let validity = Validity::from_valid_iter(vv.len(), vv.iter().map(Option::is_some));
        return Ok(Column::str_with(vv.into_iter().map(Option::unwrap_or_default).collect(), validity));
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

/// Build a `Bool` column from `Option`s, marking `None` cells `volas.NA`.
pub(crate) fn option_bool_column(vv: Vec<Option<bool>>) -> Column {
    let validity = Validity::from_valid_iter(vv.len(), vv.iter().map(Option::is_some));
    Column::bool_with(vv.iter().map(|x| x.unwrap_or(false)).collect(), validity)
}

/// Build an `I64` column from `Option`s, marking `None` cells `volas.NA`.
pub(crate) fn option_i64_column(vv: Vec<Option<i64>>) -> Column {
    let validity = Validity::from_valid_iter(vv.len(), vv.iter().map(Option::is_some));
    Column::i64_with(vv.iter().map(|x| x.unwrap_or(0)).collect(), validity)
}
