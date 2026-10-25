//! A known dense buffer exports in constant space, sharing its data.
#[path = "support/allocation.rs"]
mod allocation;
use volas_core::{Column, DType};

#[test]
fn c_data_float_export_keeps_values_and_owner() {
    for dtype in [DType::F64, DType::F32] {
        for values in [vec![-0., f64::INFINITY, f64::NEG_INFINITY], vec![1., f64::NAN, 3.]] {
            let expected = Column::f64(values.clone()).cast(dtype).unwrap();
            let col = Column::f64(values).cast(dtype).unwrap();
            let (array, schema) = volas_arrow::column_to_c_data(&col).unwrap();
            drop(col);
            // SAFETY: the fresh C-Data pair owns its buffers even after the source drops.
            let back = unsafe { volas_arrow::column_from_c_data(array, &schema) }.unwrap();
            assert!(back.equals(&expected));
            assert_eq!(back.get_f64(0).to_bits(), expected.get_f64(0).to_bits());
            assert_eq!(back.has_nan(), expected.has_nan());
        }
    }
}

#[test]
fn dense_float_export_does_not_allocate_a_bitmap() {
    for dtype in [DType::F64, DType::F32] {
        let c = Column::f64(vec![1.; 100_000]).cast(dtype).unwrap();
        assert!(!c.has_nan());
        let (array, _, bytes) = allocation::measure(|| volas_arrow::column_to_arrow(&c));
        assert_eq!(array.null_count(), 0);
        assert!(array.nulls().is_none());
        assert!(bytes < 1024, "{dtype}: {bytes} bytes");
        let (direct, calls, bytes) = allocation::measure(|| volas_arrow::column_to_c_data(&c));
        let (indirect, old_calls, old_bytes) = allocation::measure(|| {
            arrow_array::ffi::to_ffi(&volas_arrow::column_to_arrow(&c).to_data())
        });
        assert!(direct.is_ok() && indirect.is_ok());
        assert_eq!(calls + 1, old_calls);
        assert!(bytes < old_bytes);
    }
}

#[test]
fn unknown_and_foreign_nan_facts_are_conservative() {
    use std::{
        ptr::NonNull,
        sync::Arc,
    };
    use volas_core::Buffer;
    let owned = Column::f64(vec![-0., 1.]);
    assert_eq!(owned.known_has_nan(), None);
    assert_eq!(volas_arrow::column_to_arrow(&owned).null_count(), 0);
    assert_eq!(owned.known_has_nan(), Some(false));
    let memory = Arc::new(vec![-0f64, f64::NAN]);
    let ptr = NonNull::new(memory.as_ptr() as *mut f64).unwrap();
    // SAFETY: the guard owns an immutable allocation for the entire borrow lifetime.
    let c = Column::from_f64_buffer(unsafe { Buffer::from_foreign(ptr, 2, memory.clone()) });
    assert_eq!(volas_arrow::column_to_arrow(&c).null_count(), 1);
    assert_eq!(c.known_has_nan(), None);
    let a = volas_arrow::column_to_arrow(&c);
    assert_eq!(a.null_count(), 1);
    assert!(a.is_null(1));
    drop(a);
    assert_eq!(c.known_has_nan(), None);
    let filled = c.fillna(7.).unwrap();
    assert_eq!(filled.as_f64().unwrap()[0].to_bits(), (-0f64).to_bits());
    assert_eq!(filled.as_f64().unwrap()[1], 7.);
    assert_eq!(volas_arrow::column_to_arrow(&c).null_count(), 1);
    assert_eq!(c.fillna(7.).unwrap().get_f64(1), 7.);
    assert_eq!(c.known_has_nan(), None);
}
