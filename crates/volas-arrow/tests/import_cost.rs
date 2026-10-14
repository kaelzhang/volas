//! Snapshot imports allocate each final value buffer once, including multi-batch input.
#[path = "support/allocation.rs"]
mod allocation;
use std::{ffi::c_void, sync::Arc};
use arrow_array::{Float64Array, RecordBatch, RecordBatchIterator};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use volas_arrow::{column_from_arrow_with_copy, frame_from_c_stream_with_copy};

#[test]
fn one_final_allocation_per_dense_or_nullable_float_column() {
    const N: usize = 100_000;
    for nullable in [false, true] {
        let a: arrow_array::ArrayRef = if nullable {
            Arc::new(Float64Array::from((0..N).map(|i| (i != 42).then_some(i as f64)).collect::<Vec<_>>()))
        } else { Arc::new(Float64Array::from(vec![1.; N])) };
        let (c, _, bytes) = allocation::measure(|| column_from_arrow_with_copy(&a, true).unwrap());
        assert!((N * 8..N * 8 + 1024).contains(&bytes), "{bytes}");
        assert_eq!(c.len(), N);
    }
}

#[test]
fn stream_copies_each_batch_directly_into_the_final_buffer() {
    const N: usize = 100_000;
    let batches = (0..5).map(|_| {
        RecordBatch::try_from_iter(vec![("close", Arc::new(Float64Array::from(vec![1.; N])) as arrow_array::ArrayRef)]).unwrap()
    }).collect::<Vec<_>>();
    let schema = batches[0].schema();
    let mut stream = FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new(
        batches.into_iter().map(Ok), schema,
    )));
    let ((_, cols, height), _, bytes) = allocation::measure(|| {
        // SAFETY: this test owns a fresh, valid stream.
        unsafe { frame_from_c_stream_with_copy(&mut stream as *mut _ as *mut c_void, true) }.unwrap()
    });
    assert_eq!(height, N * 5);
    assert_eq!(cols[0].len(), height);
    assert!((height * 8..height * 8 + 100_000).contains(&bytes), "{bytes}");
}

#[test]
fn default_dense_import_allocation_is_independent_of_row_count() {
    for n in [1_999, 1_000_000] {
        let a: arrow_array::ArrayRef = Arc::new(Float64Array::from(vec![1.; n]));
        let ptr = a.as_any().downcast_ref::<Float64Array>().unwrap().values().as_ptr();
        let (column, _, bytes) = allocation::measure(|| volas_arrow::column_from_arrow(&a).unwrap());
        assert_eq!(column.as_f64().unwrap().as_ptr(), ptr);
        assert!(bytes < 1024, "{n} rows allocated {bytes} bytes");
        let names = vec!["close".into()];
        let mut stream = volas_arrow::columns_to_c_stream(&names, &[column]).unwrap();
        let ((_, columns, height), _, bytes) = allocation::measure(|| {
            // SAFETY: a fresh stream backed by immutable Arrow storage.
            unsafe { volas_arrow::frame_from_c_stream(&mut stream as *mut _ as *mut c_void) }.unwrap()
        });
        assert_eq!(height, n);
        assert_eq!(columns[0].as_f64().unwrap().as_ptr(), ptr);
        assert!(bytes < 16_384, "{n} rows allocated {bytes} bytes");
    }
}
