//! `Column` → Arrow `ArrayRef`. Data buffers are published zero-copy; `bool` and the
//! null bitmap are repacked (see the crate-level zero-copy contract).

use std::sync::Arc;

use arrow_array::types::{
    Float32Type, Float64Type, Int32Type, Int64Type, TimestampNanosecondType,
};
use arrow_array::{ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray};
use arrow_buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use volas_core::{Buffer, Column};

use crate::{arrow_buffer_of, null_buffer_of};

/// Build a volas `Column` as an Arrow array. Never copies the data buffer (numeric /
/// string / datetime); `bool` and any null bitmap are the only repacks.
pub fn column_to_arrow(col: &Column) -> ArrayRef {
    match col {
        // Floats carry missing as in-band `NaN` (no null bitmap), a lossless round-trip.
        Column::F64(v) => Arc::new(PrimitiveArray::<Float64Type>::new(scalar(v), None)),
        Column::F32(v) => Arc::new(PrimitiveArray::<Float32Type>::new(scalar(v), None)),
        Column::I64(v, val) => {
            Arc::new(PrimitiveArray::<Int64Type>::new(scalar(v), null_buffer_of(val, v.len())))
        }
        Column::I32(v, val) => {
            Arc::new(PrimitiveArray::<Int32Type>::new(scalar(v), null_buffer_of(val, v.len())))
        }
        Column::Bool(v, val) => {
            // volas keeps one byte per bool; Arrow one bit — repack (a copy).
            let bits = BooleanBuffer::from_iter(v.iter().copied());
            Arc::new(BooleanArray::new(bits, null_buffer_of(val, v.len())))
        }
        Column::Str(v, val) => {
            let (offsets, data) = v.buffers();
            let off = ScalarBuffer::new(arrow_buffer_of(offsets), 0, offsets.len());
            // SAFETY: a StrBuffer's offsets are monotonic with `last == data.len()`, and
            // its bytes are valid UTF-8 — so both unchecked builders' invariants hold and
            // export stays O(1) (no offset/UTF-8 re-scan).
            let offsets = unsafe { OffsetBuffer::new_unchecked(off) };
            let array = unsafe {
                LargeStringArray::new_unchecked(offsets, arrow_buffer_of(data), null_buffer_of(val, v.len()))
            };
            Arc::new(array)
        }
        // `i64::MIN` is the NaT sentinel → an Arrow null; the in-band value is masked.
        Column::Datetime(v) => {
            let nulls = v
                .contains(&i64::MIN)
                .then(|| NullBuffer::from_iter(v.iter().map(|&x| x != i64::MIN)));
            Arc::new(PrimitiveArray::<TimestampNanosecondType>::new(scalar(v), nulls))
        }
    }
}

/// A volas numeric [`Buffer<T>`] as a zero-copy Arrow [`ScalarBuffer`].
fn scalar<T: arrow_buffer::ArrowNativeType>(v: &Buffer<T>) -> ScalarBuffer<T> {
    ScalarBuffer::new(arrow_buffer_of(v), 0, v.len())
}
