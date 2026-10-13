//! Snapshot imports allocate each final value buffer once, including multi-batch input.
#[path = "support/allocation.rs"]
mod allocation;
use std::{ffi::c_void, sync::Arc};
use arrow_array::{Float64Array, RecordBatch, RecordBatchIterator};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use volas_arrow::{column_from_arrow, frame_from_c_stream};

#[test]
fn one_final_allocation_per_dense_or_nullable_float_column() {
    const N: usize = 100_000;
    for nullable in [false, true] {
        let a: arrow_array::ArrayRef = if nullable {
            Arc::new(Float64Array::from((0..N).map(|i| (i != 42).then_some(i as f64)).collect::<Vec<_>>()))
        } else { Arc::new(Float64Array::from(vec![1.; N])) };
        let (c, _, bytes) = allocation::measure(|| column_from_arrow(&a).unwrap());
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
        unsafe { frame_from_c_stream(&mut stream as *mut _ as *mut c_void) }.unwrap()
    });
    assert_eq!(height, N * 5);
    assert_eq!(cols[0].len(), height);
    assert!((height * 8..height * 8 + 100_000).contains(&bytes), "{bytes}");
}
