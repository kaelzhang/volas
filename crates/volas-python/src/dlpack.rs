//! **DLPack** export (`__dlpack__` / `__dlpack_device__`) for dense numeric columns,
//! so NumPy / PyTorch / JAX can borrow a volas buffer with no copy.
//!
//! All DLPack ABI / capsule unsafety is confined to this module. Two managed-tensor
//! flavours are produced, picked by the consumer's `max_version`:
//! - **versioned** (`"dltensor_versioned"`, DLPack ≥ 1.0) — carries a `read-only` flag,
//!   so a borrowed view cannot be written through into volas's buffer (which would
//!   bypass copy-on-write and corrupt aliasing columns);
//! - **unversioned** (`"dltensor"`) — the legacy fallback for a consumer that did not
//!   negotiate ≥ 1.0; it has no read-only flag, so a borrow is forced to a **copy** (a
//!   pre-1.0 consumer never gets a writable alias into the frame's buffer).
//!
//! `copy=True` returns an **independent** owned copy (writable, flagged `IS_COPIED`); a
//! non-CPU `dl_device` or a `stream` is refused with `BufferError` (CPU-only producer).

use std::any::Any;
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::Arc;

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use volas_core::{Buffer, Column, ColumnBorrowExport, Element, StrBuffer, Validity};

/// (`kDLCPU`, device 0) — the only device volas data lives on.
pub(crate) use crate::dlpack_tensor::DEVICE_CPU;

const KDL_INT: u8 = 0;
const KDL_FLOAT: u8 = 2;
const KDL_BOOL: u8 = 6;

use crate::dlpack_tensor::*;

/// Capsule destructor for the unversioned tensor: free it only while still un-taken
/// (named `"dltensor"`); once a consumer renames it to `"used_dltensor"` it owns it.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = ffi::PyCapsule_GetName(capsule);
    if name.is_null() || CStr::from_ptr(name) != c"dltensor" {
        return;
    }
    let tensor = ffi::PyCapsule_GetPointer(capsule, name) as *mut DLManagedTensor;
    if !tensor.is_null() {
        if let Some(d) = (*tensor).deleter {
            d(tensor);
        }
    }
}

/// Capsule destructor for the versioned tensor (named `"dltensor_versioned"`).
unsafe extern "C" fn capsule_destructor_versioned(capsule: *mut ffi::PyObject) {
    let name = ffi::PyCapsule_GetName(capsule);
    if name.is_null() || CStr::from_ptr(name) != c"dltensor_versioned" {
        return;
    }
    let tensor = ffi::PyCapsule_GetPointer(capsule, name) as *mut DLManagedTensorVersioned;
    if !tensor.is_null() {
        if let Some(d) = (*tensor).deleter {
            d(tensor);
        }
    }
}

/// `(data ptr, dtype, element count, keep-alive)` for a DLPack-exportable column. With
/// `copy`, the buffer is materialised into a fresh owned allocation (an independent,
/// writable view); otherwise it is borrowed (the keep-alive shares the frame's buffer).
/// Errors for a dtype DLPack cannot carry (str / datetime) or an integer/bool column
/// with missing values (DLPack has no null mask; a float `NaN` is in-band, so it is fine).
fn dl_parts(
    col: &Column,
    copy: bool,
) -> PyResult<(*mut c_void, DLDataType, usize, Arc<dyn Any + Send + Sync>)> {
    col.export(DlExport { copy })
}

struct DlExport { copy: bool }
type DlParts = PyResult<(*mut c_void, DLDataType, usize, Arc<dyn Any + Send + Sync>)>;
impl DlExport {
    fn parts<T: Element + Clone + Send + Sync + 'static>(self, buffer: &Buffer<T>, code: u8, bits: u8) -> DlParts {
        let owned;
        let b = if self.copy { owned = Buffer::from_vec(buffer.to_vec()); &owned } else { buffer };
        let data = if b.is_empty() { ptr::null_mut() } else { b.as_ptr() as *mut c_void };
        Ok((data, DLDataType { code, bits, lanes: 1 }, b.len(), b.keepalive()))
    }
    fn present(self, valid: &Validity) -> PyResult<Self> {
        if valid.has_nulls() {
            return Err(PyBufferError::new_err("cannot export a column with missing values via DLPack (DLPack has no null mask) — fill the NA first"));
        }
        Ok(self)
    }
}
impl ColumnBorrowExport for DlExport {
    type Output = DlParts;
    fn f64(self, v: &Buffer<f64>) -> DlParts { self.parts(v, KDL_FLOAT, 64) }
    fn f32(self, v: &Buffer<f32>) -> DlParts { self.parts(v, KDL_FLOAT, 32) }
    fn i64(self, v: &Buffer<i64>, valid: &Validity) -> DlParts { self.present(valid)?.parts(v, KDL_INT, 64) }
    fn i32(self, v: &Buffer<i32>, valid: &Validity) -> DlParts { self.present(valid)?.parts(v, KDL_INT, 32) }
    fn bool(self, v: &Buffer<bool>, valid: &Validity) -> DlParts { self.present(valid)?.parts(v, KDL_BOOL, 8) }
    fn str(self, _: &StrBuffer, _: &Validity) -> DlParts { Err(unsupported("str")) }
    fn datetime(self, _: &Buffer<i64>) -> DlParts { Err(unsupported("datetime64[ns]")) }
}
fn unsupported(dtype: &str) -> PyErr {
    PyBufferError::new_err(format!("cannot export a {dtype} column via DLPack (only numeric / bool dtypes are supported)"))
}

/// The `__dlpack__` payload: a `"dltensor(_versioned)"` PyCapsule over the column.
///
/// `max_version` selects the flavour (≥ 1.0 → versioned, read-only unless `copy`);
/// `dl_device` other than CPU, or a non-`None` `stream`, is refused (`BufferError`).
pub(crate) fn column_to_dlpack<'py>(
    py: Python<'py>,
    col: &Column,
    max_version: Option<(i32, i32)>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
    stream: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(dev) = dl_device {
        if dev != DEVICE_CPU {
            return Err(PyBufferError::new_err(format!(
                "volas exports DLPack on CPU ({DEVICE_CPU:?}) only; cannot honor device {dev:?}"
            )));
        }
    }
    if stream {
        return Err(PyBufferError::new_err(
            "volas's CPU DLPack export takes no stream (pass stream=None)",
        ));
    }
    let versioned = matches!(max_version, Some((major, _)) if major >= 1);
    // An unversioned (pre-1.0) tensor has no read-only flag, so a borrow would be a
    // writable alias into volas's buffer (bypassing copy-on-write). Force a copy on that
    // path; if the consumer explicitly forbade copying, we cannot serve it safely.
    if !versioned && copy == Some(false) {
        return Err(PyBufferError::new_err(
            "cannot lend a zero-copy DLPack view to a pre-1.0 consumer (no read-only flag); \
             negotiate max_version >= (1, 0), or drop copy=False",
        ));
    }
    let do_copy = copy == Some(true) || !versioned;
    let (data, dtype, len, keepalive) = dl_parts(col, do_copy)?;
    // Native metadata is built after the optional data copy, in one owned block.
    unsafe {
        let ptr = managed_tensor(data, dtype, len, keepalive, versioned, do_copy);
        let (name, dtor): (&CStr, ffi::PyCapsule_Destructor) = if versioned {
            (c"dltensor_versioned", capsule_destructor_versioned)
        } else {
            (c"dltensor", capsule_destructor)
        };
        let cap = ffi::PyCapsule_New(ptr, name.as_ptr(), Some(dtor));
        if cap.is_null() {
            // reclaim, don't leak
            if versioned {
                deleter_versioned(ptr as *mut DLManagedTensorVersioned);
            } else {
                deleter(ptr as *mut DLManagedTensor);
            }
            return Err(PyErr::fetch(py));
        }
        Bound::from_owned_ptr_or_err(py, cap)
    }
}
