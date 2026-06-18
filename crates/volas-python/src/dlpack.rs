//! Zero-copy **DLPack** export (`__dlpack__` / `__dlpack_device__`) for dense numeric
//! columns, so NumPy / PyTorch / JAX can borrow a volas buffer with no copy.
//!
//! All DLPack ABI / capsule unsafety is confined to this module. The producer hands
//! out an unversioned `"dltensor"` capsule; per the protocol, a consumer renames it to
//! `"used_dltensor"` and then owns the tensor — so the capsule destructor frees the
//! tensor only when it was *never* consumed (still named `"dltensor"`).

use std::any::Any;
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use volas_core::Column;

/// (`kDLCPU`, device 0) — the only device volas data lives on.
pub(crate) const DEVICE_CPU: (i32, i32) = (1, 0);

const KDL_INT: u8 = 0;
const KDL_FLOAT: u8 = 2;
const KDL_BOOL: u8 = 6;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Owns everything the borrowed `DLTensor` points at: the keep-alive holding the volas
/// allocation, and the boxed `shape`.
struct Manager {
    _keepalive: Arc<dyn Any + Send + Sync>,
    _shape: Box<[i64; 1]>,
}

/// Frees a `DLManagedTensor` and its `Manager` (the keep-alive + shape). Called by the
/// consumer once it is done, or by [`capsule_destructor`] for an unconsumed capsule.
unsafe extern "C" fn deleter(tensor: *mut DLManagedTensor) {
    if tensor.is_null() {
        return;
    }
    let ctx = (*tensor).manager_ctx as *mut Manager;
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
    drop(Box::from_raw(tensor));
}

/// Capsule destructor: if the capsule still holds an un-taken `"dltensor"`, free it;
/// once a consumer renames it to `"used_dltensor"`, ownership has moved and we do nothing.
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

/// `(data ptr, dtype, element count, keep-alive)` for a DLPack-exportable column, or a
/// `ValueError` for a dtype DLPack cannot carry (str / datetime) or an integer/bool
/// column with missing values (DLPack has no null mask; float `NaN` is in-band, so it
/// is fine).
fn dl_parts(col: &Column) -> PyResult<(*mut c_void, DLDataType, usize, Arc<dyn Any + Send + Sync>)> {
    macro_rules! parts {
        ($buf:expr, $code:expr, $bits:expr) => {
            Ok((
                $buf.as_ptr() as *mut c_void,
                DLDataType { code: $code, bits: $bits, lanes: 1 },
                $buf.len(),
                $buf.keepalive(),
            ))
        };
    }
    let na = || {
        PyValueError::new_err(
            "cannot export a column with missing values via DLPack (DLPack has no null \
             mask) — fill the NA first",
        )
    };
    match col {
        Column::F64(b) => parts!(b, KDL_FLOAT, 64),
        Column::F32(b) => parts!(b, KDL_FLOAT, 32),
        Column::I64(_, v) | Column::I32(_, v) | Column::Bool(_, v) if v.has_nulls() => Err(na()),
        Column::I64(b, _) => parts!(b, KDL_INT, 64),
        Column::I32(b, _) => parts!(b, KDL_INT, 32),
        Column::Bool(b, _) => parts!(b, KDL_BOOL, 8),
        Column::Str(..) | Column::Datetime(..) => Err(PyValueError::new_err(format!(
            "cannot export a {} column via DLPack (only numeric / bool dtypes are supported)",
            col.dtype()
        ))),
    }
}

/// A `"dltensor"` PyCapsule borrowing the column's buffer (the `__dlpack__` payload).
pub(crate) fn column_to_dlpack<'py>(py: Python<'py>, col: &Column) -> PyResult<Bound<'py, PyAny>> {
    let (data, dtype, len, keepalive) = dl_parts(col)?;
    let shape = Box::new([len as i64]);
    let shape_ptr = shape.as_ptr() as *mut i64;
    let manager = Box::new(Manager { _keepalive: keepalive, _shape: shape });
    let tensor = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device: DLDevice { device_type: DEVICE_CPU.0, device_id: DEVICE_CPU.1 },
            ndim: 1,
            dtype,
            shape: shape_ptr,
            strides: ptr::null_mut(), // contiguous
            byte_offset: 0,
        },
        manager_ctx: Box::into_raw(manager) as *mut c_void,
        deleter: Some(deleter),
    });
    let tensor_ptr = Box::into_raw(tensor) as *mut c_void;
    // SAFETY: `tensor_ptr` is a freshly boxed `DLManagedTensor` with a valid deleter; the
    // capsule takes the pointer and its destructor reclaims it if never consumed.
    unsafe {
        let cap = ffi::PyCapsule_New(tensor_ptr, c"dltensor".as_ptr(), Some(capsule_destructor));
        if cap.is_null() {
            deleter(tensor_ptr as *mut DLManagedTensor); // reclaim, don't leak
            return Err(PyErr::fetch(py));
        }
        Bound::from_owned_ptr_or_err(py, cap)
    }
}
