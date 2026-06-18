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
use volas_core::{Buffer, Column};

/// (`kDLCPU`, device 0) — the only device volas data lives on.
pub(crate) const DEVICE_CPU: (i32, i32) = (1, 0);

const KDL_INT: u8 = 0;
const KDL_FLOAT: u8 = 2;
const KDL_BOOL: u8 = 6;
/// `DLPACK_FLAG_BITMASK_READ_ONLY` — set on a borrowed (non-copy) versioned export.
const DLPACK_FLAG_READ_ONLY: u64 = 1;
/// `DLPACK_FLAG_BITMASK_IS_COPIED` — set when the producer materialised an owned copy,
/// the standard signal a consumer reads to tell a copy from a view.
const DLPACK_FLAG_IS_COPIED: u64 = 1 << 1;

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

#[repr(C)]
#[derive(Clone, Copy)]
struct DLPackVersion {
    major: i32,
    minor: i32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

/// Owns everything the `DLTensor` points at: the keep-alive holding the volas (or, for a
/// copy, a fresh) allocation, and the boxed `shape`.
struct Manager {
    _keepalive: Arc<dyn Any + Send + Sync>,
    _shape: Box<[i64; 1]>,
}

/// Reclaim a `Manager` (keep-alive + shape) from a raw `manager_ctx`.
unsafe fn drop_manager(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx as *mut Manager));
    }
}

unsafe extern "C" fn deleter(tensor: *mut DLManagedTensor) {
    if tensor.is_null() {
        return;
    }
    drop_manager((*tensor).manager_ctx);
    drop(Box::from_raw(tensor));
}

unsafe extern "C" fn deleter_versioned(tensor: *mut DLManagedTensorVersioned) {
    if tensor.is_null() {
        return;
    }
    drop_manager((*tensor).manager_ctx);
    drop(Box::from_raw(tensor));
}

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
    macro_rules! parts {
        ($buf:expr, $code:expr, $bits:expr) => {{
            let b = if copy { Buffer::from_vec($buf.to_vec()) } else { $buf.clone() };
            // DLPack requires a size-0 tensor's data pointer to be NULL; a Rust empty
            // slice's pointer is a non-null dangling address, so normalise it here.
            let data = if b.len() == 0 { ptr::null_mut() } else { b.as_ptr() as *mut c_void };
            Ok((data, DLDataType { code: $code, bits: $bits, lanes: 1 }, b.len(), b.keepalive()))
        }};
    }
    let na = || {
        PyBufferError::new_err(
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
        Column::Str(..) | Column::Datetime(..) => Err(PyBufferError::new_err(format!(
            "cannot export a {} column via DLPack (only numeric / bool dtypes are supported)",
            col.dtype()
        ))),
    }
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
    let shape = Box::new([len as i64]);
    let shape_ptr = shape.as_ptr() as *mut i64;
    let manager_ctx = Box::into_raw(Box::new(Manager { _keepalive: keepalive, _shape: shape }))
        as *mut c_void;
    let dl_tensor = DLTensor {
        data,
        device: DLDevice { device_type: DEVICE_CPU.0, device_id: DEVICE_CPU.1 },
        ndim: 1,
        dtype,
        shape: shape_ptr,
        strides: ptr::null_mut(), // contiguous
        byte_offset: 0,
    };
    // A consumer that negotiated DLPack ≥ 1.0 gets the versioned tensor (read-only for a
    // borrow); otherwise the legacy unversioned one (always a copy, see above).
    // SAFETY: the boxed tensor carries a valid deleter; the capsule takes the pointer and
    // its destructor reclaims it (tensor + Manager) if the consumer never takes it.
    unsafe {
        let (ptr, name, dtor): (*mut c_void, &CStr, ffi::PyCapsule_Destructor) = if versioned {
            let t = Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx,
                deleter: Some(deleter_versioned),
                flags: if do_copy { DLPACK_FLAG_IS_COPIED } else { DLPACK_FLAG_READ_ONLY },
                dl_tensor,
            });
            (Box::into_raw(t) as *mut c_void, c"dltensor_versioned", capsule_destructor_versioned)
        } else {
            let t = Box::new(DLManagedTensor { dl_tensor, manager_ctx, deleter: Some(deleter) });
            (Box::into_raw(t) as *mut c_void, c"dltensor", capsule_destructor)
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
