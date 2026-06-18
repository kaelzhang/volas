//! The Arrow PyCapsule interface (`__arrow_c_array__` / `__arrow_c_schema__`) and
//! `from_arrow` / `to_arrow`. All conversion lives in `volas-arrow`; this module only
//! moves the C-Data structs in and out of PyCapsules and hands them across.

use std::ffi::{c_void, CString};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods, PyTuple};
use volas_arrow::{
    column_from_c_capsules, column_to_c_data, column_to_c_schema, columns_from_c_stream,
    columns_to_c_stream,
};
use volas_core::Column;

const ARRAY_NAME: &str = "arrow_array";
const SCHEMA_NAME: &str = "arrow_schema";
const STREAM_NAME: &str = "arrow_array_stream";

fn arrow_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("Arrow C-Data interface error: {e}"))
}

fn capsule<'py, T: Send + 'static>(
    py: Python<'py>,
    value: T,
    name: &str,
) -> PyResult<Bound<'py, PyCapsule>> {
    PyCapsule::new(py, value, Some(CString::new(name).expect("capsule name has no NUL")))
}

/// `__arrow_c_schema__`: a lone `arrow_schema` capsule describing the column's dtype.
pub(crate) fn column_c_schema<'py>(
    py: Python<'py>,
    col: &Column,
) -> PyResult<Bound<'py, PyCapsule>> {
    capsule(py, column_to_c_schema(col).map_err(arrow_err)?, SCHEMA_NAME)
}

/// `__arrow_c_array__`: the `(schema, array)` capsule pair. Data is shared zero-copy;
/// each capsule's destructor releases its struct if the consumer never takes it.
pub(crate) fn column_c_array<'py>(
    py: Python<'py>,
    col: &Column,
) -> PyResult<Bound<'py, PyTuple>> {
    let (array, schema) = column_to_c_data(col).map_err(arrow_err)?;
    let schema_cap = capsule(py, schema, SCHEMA_NAME)?;
    let array_cap = capsule(py, array, ARRAY_NAME)?;
    PyTuple::new(py, [schema_cap.into_any(), array_cap.into_any()])
}

/// Build a `Column` from any object exposing the `__arrow_c_array__` protocol
/// (a pyarrow `Array`, a polars `Series`, …) — the import half of `from_arrow`.
pub(crate) fn column_from_arrow_obj(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    let pair = obj.call_method0("__arrow_c_array__")?;
    let (schema_cap, array_cap) = pair.extract::<(Bound<'_, PyCapsule>, Bound<'_, PyCapsule>)>()?;
    check_name(&schema_cap, SCHEMA_NAME)?;
    check_name(&array_cap, ARRAY_NAME)?;
    let array_ptr = array_cap.pointer();
    let schema_ptr = schema_cap.pointer() as *const c_void;
    // SAFETY: the capsules are a valid, not-yet-consumed C-Data pair (their names were
    // checked); `schema_cap` / `array_cap` stay alive across the call below.
    let col = unsafe { column_from_c_capsules(array_ptr, schema_ptr) }.map_err(arrow_err)?;
    drop((schema_cap, array_cap));
    Ok(col)
}

/// `__arrow_c_stream__`: one `arrow_array_stream` capsule yielding the frame as a
/// single `RecordBatch` (every column shared zero-copy).
pub(crate) fn frame_c_stream<'py>(
    py: Python<'py>,
    names: &[String],
    cols: &[Column],
) -> PyResult<Bound<'py, PyCapsule>> {
    capsule(py, columns_to_c_stream(names, cols).map_err(arrow_err)?, STREAM_NAME)
}

/// `(names, columns)` from any object exposing `__arrow_c_stream__` (a pyarrow
/// `Table` / `RecordBatchReader`, a polars `DataFrame`, …) — the import half of the
/// frame `from_arrow`.
pub(crate) fn frame_from_arrow_obj(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Vec<Column>)> {
    let cap = obj
        .call_method0("__arrow_c_stream__")?
        .extract::<Bound<'_, PyCapsule>>()?;
    check_name(&cap, STREAM_NAME)?;
    let ptr = cap.pointer();
    // SAFETY: a valid, not-yet-consumed `arrow_array_stream` capsule (name checked);
    // `cap` stays alive across the call.
    let out = unsafe { columns_from_c_stream(ptr) }.map_err(arrow_err)?;
    drop(cap);
    Ok(out)
}

fn check_name(cap: &Bound<'_, PyCapsule>, expected: &str) -> PyResult<()> {
    match cap.name()? {
        Some(n) if n.to_bytes() == expected.as_bytes() => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "expected an Arrow '{expected}' capsule, got {other:?}"
        ))),
    }
}
