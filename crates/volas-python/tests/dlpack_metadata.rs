//! Exercise the Python exporter’s actual native metadata owner without Python allocations.
#[path = "../src/dlpack_tensor.rs"]
mod dlpack_tensor;
#[path = "../../volas-core/tests/support/allocation.rs"]
mod allocation;
use dlpack_tensor::*;
use std::any::Any;
use std::ffi::c_void;
use std::sync::Arc;

#[test]
fn dlpack_metadata_budget_excludes_input_preparation_and_explicit_copies() {
    for (versioned, copied) in [(false, true), (true, false), (true, true)] {
        for len in [0, 3, 100_000] {
            let source = Arc::new(vec![7.0_f64; len]);
            // Legacy exports and copy=True prepare their independent data before metadata.
            let prepared = if copied { Arc::new(source.as_ref().clone()) } else { Arc::clone(&source) };
            let weak = Arc::downgrade(&prepared);
            let data = if len == 0 { std::ptr::null_mut() } else { prepared.as_ptr() as *mut c_void };
            let keepalive: Arc<dyn Any + Send + Sync> = prepared;
            let dtype = DLDataType { code: 2, bits: 64, lanes: 1 };
            let (pointer, calls, _) = allocation::measure(|| managed_tensor(data, dtype, len, keepalive, versioned, copied));
            assert_eq!(calls, 1, "versioned={versioned}, copied={copied}, len={len}");
            drop(source);
            assert!(weak.upgrade().is_some());
            unsafe {
                let tensor = if versioned {
                    let t = &*pointer.cast::<DLManagedTensorVersioned>();
                    assert_eq!(t.flags, if copied { 2 } else { 1 });
                    &t.dl_tensor
                } else {
                    &(*pointer.cast::<DLManagedTensor>()).dl_tensor
                };
                assert_eq!(tensor.data, data);
                assert_eq!(*tensor.shape, len as i64);
                if len > 0 { assert_eq!(*tensor.data.cast::<f64>(), 7.0); }
                if versioned { deleter_versioned(pointer.cast()); } else { deleter(pointer.cast()); }
            }
            assert!(weak.upgrade().is_none(), "the consumer deleter releases the prepared buffer");
        }
    }
    // Defensive null deleters must also be no-ops.
    unsafe { deleter(std::ptr::null_mut()); deleter_versioned(std::ptr::null_mut()); }
}
