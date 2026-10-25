//! `Column` → Arrow `ArrayRef`. Data buffers are published zero-copy; `bool` and the
//! null bitmap are repacked (see the crate-level zero-copy contract).

use volas_core::ColumnView;
use std::sync::Arc;

use arrow_array::types::{
    Float32Type, Float64Type, Int32Type, Int64Type, TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray};
use arrow_data::ArrayData;
use arrow_buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use volas_core::{Buffer, Column, Tz};

use crate::{arrow_buffer_of, null_buffer_of};

/// Build a volas `Column` as an Arrow array. Never copies the data buffer (numeric /
/// string / datetime); `bool` and any null bitmap are the only repacks.
pub fn column_to_arrow(col: &Column) -> ArrayRef {
    match col.view() {
        // A missing float is an in-band `NaN`; export it as a real Arrow null (built from
        // the NaN positions) so int and float "missing" are consistent at the boundary.
        // The data buffer is still shared zero-copy; only NaN-bearing columns pay the
        // O(n) scan + the small bitmap (dense float stays null-free).
        ColumnView::F64(v) => Arc::new(float64(v)),
        ColumnView::F32(v) => Arc::new(float32(v)),
        ColumnView::I64(v, val) => {
            Arc::new(PrimitiveArray::<Int64Type>::new(scalar(v), null_buffer_of(val, v.len())))
        }
        ColumnView::I32(v, val) => {
            Arc::new(PrimitiveArray::<Int32Type>::new(scalar(v), null_buffer_of(val, v.len())))
        }
        ColumnView::Bool(v, val) => {
            // volas keeps one byte per bool; Arrow one bit — repack (a copy).
            let bits = BooleanBuffer::from_iter(v.iter().copied());
            Arc::new(BooleanArray::new(bits, null_buffer_of(val, v.len())))
        }
        ColumnView::Str(v, val) => {
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
        // The instants are UTC, so the type states it: `timestamp[ns, tz=UTC]`.
        ColumnView::Datetime(v) => {
            let nulls = nulls_from_validity(v.iter().map(|&x| x != i64::MIN));
            Arc::new(
                PrimitiveArray::<TimestampNanosecondType>::new(scalar(v), nulls)
                    .with_timezone(Tz::DATETIME_COLUMN.name()),
            )
        }
    }
}

/// C-Data export consumes the float array directly: no temporary `ArrayRef`
/// allocation or shared-buffer clone just to borrow its `ArrayData` and drop it.
pub(crate) fn column_to_data(col: &Column) -> ArrayData {
    match col.view() {
        ColumnView::F64(v) => float64(v).into_data(),
        ColumnView::F32(v) => float32(v).into_data(),
        _ => column_to_arrow(col).to_data(),
    }
}

#[inline(always)]
fn float64(v: &Buffer<f64>) -> PrimitiveArray<Float64Type> {
    let nulls = v.has_nan().then(|| NullBuffer::new(BooleanBuffer::from_iter(
        v.iter().map(|x| !x.is_nan()),
    )));
    PrimitiveArray::new(scalar(v), nulls)
}

#[inline(always)]
fn float32(v: &Buffer<f32>) -> PrimitiveArray<Float32Type> {
    let nulls = v.has_nan().then(|| NullBuffer::new(BooleanBuffer::from_iter(
        v.iter().map(|x| !x.is_nan()),
    )));
    PrimitiveArray::new(scalar(v), nulls)
}

/// Build a null buffer from a per-element validity flag in a **single pass**: collect the
/// bits while noting whether any were null, then discard the bitmap when the column is
/// dense (no nulls → Arrow `None`). Replaces the any-then-rebuild double scan.
fn nulls_from_validity(valid: impl Iterator<Item = bool>) -> Option<NullBuffer> {
    let mut any_null = false;
    let bits = BooleanBuffer::from_iter(valid.inspect(|&v| any_null |= !v));
    any_null.then(|| NullBuffer::new(bits))
}

/// A volas numeric [`Buffer<T>`] as a zero-copy Arrow [`ScalarBuffer`].
fn scalar<T: arrow_buffer::ArrowNativeType + volas_core::Element>(v: &Buffer<T>) -> ScalarBuffer<T> {
    ScalarBuffer::new(arrow_buffer_of(v), 0, v.len())
}
