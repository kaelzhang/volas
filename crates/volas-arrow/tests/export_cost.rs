//! A known dense buffer exports in constant space, sharing its data.
#[path = "support/allocation.rs"]
mod allocation;
use volas_core::{Column, DType};
#[test]
fn dense_float_export_does_not_allocate_a_bitmap() {
    for dtype in [DType::F64, DType::F32] {
        let c = Column::f64(vec![1.; 100_000]).cast(dtype).unwrap();
        assert!(!c.has_nan());
        let (array, _, bytes) = allocation::measure(|| volas_arrow::column_to_arrow(&c));
        assert_eq!(array.null_count(), 0);
        assert!(array.nulls().is_none());
        assert!(bytes < 1024, "{dtype}: {bytes} bytes");
    }
}

#[test]
fn unknown_and_foreign_nan_facts_are_conservative() {
    use std::{
        ptr::NonNull,
        sync::{Arc, Mutex},
    };
    use volas_core::Buffer;
    let owned = Column::f64(vec![-0., 1.]);
    assert_eq!(owned.known_has_nan(), None);
    assert_eq!(volas_arrow::column_to_arrow(&owned).null_count(), 0);
    assert_eq!(owned.known_has_nan(), Some(false));
    let memory = Arc::new(Mutex::new(vec![-0f64, 1.]));
    let ptr = NonNull::new(memory.lock().unwrap().as_mut_ptr()).unwrap();
    // SAFETY: the guard keeps the allocation alive and fixed in size. The
    // producer writes only between calls, while no borrowed slice is live.
    let c = Column::F64(unsafe { Buffer::from_foreign(ptr, 2, memory.clone()) });
    assert_eq!(volas_arrow::column_to_arrow(&c).null_count(), 0);
    assert_eq!(c.known_has_nan(), None);
    memory.lock().unwrap()[1] = f64::NAN;
    let a = volas_arrow::column_to_arrow(&c);
    assert_eq!(a.null_count(), 1);
    assert!(a.is_null(1));
    drop(a);
    assert_eq!(c.known_has_nan(), None);
    let filled = c.fillna(7.).unwrap();
    assert_eq!(filled.as_f64().unwrap()[0].to_bits(), (-0f64).to_bits());
    assert_eq!(filled.as_f64().unwrap()[1], 7.);
    memory.lock().unwrap()[1] = 2.;
    assert_eq!(volas_arrow::column_to_arrow(&c).null_count(), 0);
    assert_eq!(c.fillna(7.).unwrap().get_f64(1), 2.);
    assert_eq!(c.known_has_nan(), None);
}
