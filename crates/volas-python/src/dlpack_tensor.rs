//! Native DLPack metadata owner. Buffer preparation and explicit copies precede it.
use std::any::Any;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

/// CPU is the only supported export device.
pub(crate) const DEVICE_CPU: (i32, i32) = (1, 0);

/// `DLPACK_FLAG_BITMASK_READ_ONLY` — set on a borrowed (non-copy) versioned export.
const DLPACK_FLAG_READ_ONLY: u64 = 1;
/// `DLPACK_FLAG_BITMASK_IS_COPIED` — set when the producer materialised an owned copy,
/// the standard signal a consumer reads to tell a copy from a view.
const DLPACK_FLAG_IS_COPIED: u64 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DLDevice {
    pub(crate) device_type: i32,
    pub(crate) device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DLDataType {
    pub(crate) code: u8,
    pub(crate) bits: u8,
    pub(crate) lanes: u16,
}

#[repr(C)]
pub(crate) struct DLTensor {
    pub(crate) data: *mut c_void,
    pub(crate) device: DLDevice,
    pub(crate) ndim: i32,
    pub(crate) dtype: DLDataType,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) byte_offset: u64,
}

#[repr(C)]
pub(crate) struct DLManagedTensor {
    pub(crate) dl_tensor: DLTensor,
    pub(crate) manager_ctx: *mut c_void,
    pub(crate) deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DLPackVersion {
    pub(crate) major: i32,
    pub(crate) minor: i32,
}

#[repr(C)]
pub(crate) struct DLManagedTensorVersioned {
    pub(crate) version: DLPackVersion,
    pub(crate) manager_ctx: *mut c_void,
    pub(crate) deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub(crate) flags: u64,
    pub(crate) dl_tensor: DLTensor,
}

/// One allocation owns the ABI tensor, its inline shape and the shared buffer.
/// `tensor` is first, so the ABI pointer is also the owning allocation's pointer.
#[repr(C)]
pub(crate) struct ExportedTensor<T> {
    pub(crate) tensor: T,
    pub(crate) _keepalive: Arc<dyn Any + Send + Sync>,
    pub(crate) shape: [i64; 1],
}

pub(crate) unsafe extern "C" fn deleter(tensor: *mut DLManagedTensor) {
    if !tensor.is_null() { drop(Box::from_raw(tensor.cast::<ExportedTensor<DLManagedTensor>>())); }
}

pub(crate) unsafe extern "C" fn deleter_versioned(tensor: *mut DLManagedTensorVersioned) {
    if !tensor.is_null() { drop(Box::from_raw(tensor.cast::<ExportedTensor<DLManagedTensorVersioned>>())); }
}

/// Allocate the complete managed-tensor metadata block, after preparing its buffer.
pub(crate) fn managed_tensor(data: *mut c_void, dtype: DLDataType, len: usize,
    keepalive: Arc<dyn Any + Send + Sync>, versioned: bool, do_copy: bool) -> *mut c_void {
    let dl_tensor = DLTensor {
        data,
        device: DLDevice { device_type: DEVICE_CPU.0, device_id: DEVICE_CPU.1 },
        ndim: 1,
        dtype,
        shape: ptr::null_mut(), // filled after the owning allocation is stable
        strides: ptr::null_mut(), // contiguous
        byte_offset: 0,
    };
    if versioned {
        let mut t = Box::new(ExportedTensor { tensor: DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: ptr::null_mut(),
            deleter: Some(deleter_versioned),
            flags: if do_copy { DLPACK_FLAG_IS_COPIED } else { DLPACK_FLAG_READ_ONLY },
            dl_tensor,
        }, _keepalive: keepalive, shape: [len as i64] });
        t.tensor.dl_tensor.shape = t.shape.as_mut_ptr();
        Box::into_raw(t) as *mut c_void
    } else {
        let mut t = Box::new(ExportedTensor {
            tensor: DLManagedTensor { dl_tensor, manager_ctx: ptr::null_mut(), deleter: Some(deleter) },
            _keepalive: keepalive, shape: [len as i64],
        });
        t.tensor.dl_tensor.shape = t.shape.as_mut_ptr();
        Box::into_raw(t) as *mut c_void
    }
}
