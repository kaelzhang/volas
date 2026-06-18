//! Arrow `ArrayRef` → `Column`. Numeric / string / ns-datetime data is borrowed
//! zero-copy ([`Buffer::Borrowed`](volas_core::Buffer)); `bool`, narrow `Utf8` offsets,
//! and non-ns timestamps are the only repacks.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use volas_core::{Buffer, Column, StrBuffer};

use crate::{borrow_primitive, validity_of};

/// Import an Arrow array as a volas `Column`. Borrows the data buffer where the
/// physical layouts match; copies only `bool`, `Utf8` (i32 → i64 offsets), and
/// timestamps coarser than nanoseconds.
pub fn column_from_arrow(src: &ArrayRef) -> Result<Column, ArrowError> {
    let len = src.len();
    let col = match src.data_type() {
        // Floats: missing is in-band NaN. Dense → borrow; an external null bitmap
        // collapses to NaN (a copy, the only lossy-free way into volas's float model).
        DataType::Float64 => {
            let a = src.as_primitive::<Float64Type>();
            match a.nulls() {
                None => Column::F64(borrow_primitive(a.values(), src)),
                Some(_) => Column::F64(Buffer::from_vec(
                    (0..len).map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) }).collect(),
                )),
            }
        }
        DataType::Float32 => {
            let a = src.as_primitive::<Float32Type>();
            match a.nulls() {
                None => Column::F32(borrow_primitive(a.values(), src)),
                Some(_) => Column::F32(Buffer::from_vec(
                    (0..len).map(|i| if a.is_null(i) { f32::NAN } else { a.value(i) }).collect(),
                )),
            }
        }
        DataType::Int64 => {
            let a = src.as_primitive::<Int64Type>();
            Column::I64(borrow_primitive(a.values(), src), validity_of(a.nulls(), len))
        }
        DataType::Int32 => {
            let a = src.as_primitive::<Int32Type>();
            Column::I32(borrow_primitive(a.values(), src), validity_of(a.nulls(), len))
        }
        DataType::Boolean => {
            // Arrow packs one bit per value; volas one byte — unpack (a copy).
            let a = src.as_boolean();
            let vals: Vec<bool> = a.values().iter().collect();
            Column::Bool(Buffer::from_vec(vals), validity_of(a.nulls(), len))
        }
        DataType::LargeUtf8 => {
            let a = src.as_string::<i64>();
            let offsets = borrow_primitive(a.value_offsets(), src);
            let data = borrow_primitive(a.value_data(), src);
            Column::Str(StrBuffer::from_buffers(offsets, data), validity_of(a.nulls(), len))
        }
        DataType::Utf8 => {
            // 32-bit offsets must widen to volas's i64 (a copy of the small offset
            // array); the UTF-8 bytes are still borrowed zero-copy.
            let a = src.as_string::<i32>();
            let offsets = Buffer::from_vec(a.value_offsets().iter().map(|&o| o as i64).collect());
            let data = borrow_primitive(a.value_data(), src);
            Column::Str(StrBuffer::from_buffers(offsets, data), validity_of(a.nulls(), len))
        }
        // Datetimes land on volas's nanosecond grid; `null` → the `i64::MIN` NaT
        // sentinel. The ns case is borrowed when dense; coarser units rescale (a copy).
        DataType::Timestamp(unit, _) => datetime_from_timestamp(src, *unit, len),
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "volas cannot import an Arrow {other:?} column"
            )))
        }
    };
    Ok(col)
}

/// Timestamp → `Datetime` (ns). Borrows the buffer for a dense nanosecond array;
/// otherwise rescales / fills the NaT sentinel into a fresh buffer.
fn datetime_from_timestamp(src: &ArrayRef, unit: TimeUnit, len: usize) -> Column {
    let scale = match unit {
        TimeUnit::Nanosecond => 1,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Second => 1_000_000_000,
    };
    macro_rules! filled {
        ($ty:ty) => {{
            let a = src.as_primitive::<$ty>();
            Buffer::from_vec(
                (0..len)
                    .map(|i| if a.is_null(i) { i64::MIN } else { a.value(i) * scale })
                    .collect(),
            )
        }};
    }
    let buf = match unit {
        TimeUnit::Nanosecond => {
            let a = src.as_primitive::<TimestampNanosecondType>();
            match a.nulls() {
                None => return Column::Datetime(borrow_primitive(a.values(), src)),
                Some(_) => Buffer::from_vec(
                    (0..len).map(|i| if a.is_null(i) { i64::MIN } else { a.value(i) }).collect(),
                ),
            }
        }
        TimeUnit::Microsecond => filled!(TimestampMicrosecondType),
        TimeUnit::Millisecond => filled!(TimestampMillisecondType),
        TimeUnit::Second => filled!(TimestampSecondType),
    };
    Column::Datetime(buf)
}
