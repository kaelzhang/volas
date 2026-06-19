//! Arrow `ArrayRef` → `Column`. Numeric / string / ns-datetime data is borrowed
//! zero-copy ([`Buffer::Borrowed`](volas_core::Buffer)); `bool`, narrow `Utf8` offsets,
//! and non-ns timestamps are the only repacks.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal128Type, Float32Type, Float64Type, Int16Type, Int32Type,
    Int64Type, Int8Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
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
    // A narrow / unsigned integer Arrow array widened to volas's i64 (a copy). The
    // `as i64` cast wraps a `UInt64` past `i64::MAX` — the usable-not-lossless contract.
    macro_rules! widen_int {
        ($ty:ty) => {{
            let a = src.as_primitive::<$ty>();
            Column::I64(
                Buffer::from_vec((0..len).map(|i| a.value(i) as i64).collect()),
                validity_of(a.nulls(), len),
            )
        }};
    }
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
            let raw = a.value_offsets();
            let (first, last) = (raw[0], raw[raw.len() - 1]);
            // A canonical (un-sliced) array already satisfies StrBuffer's invariant, so its
            // offsets borrow zero-copy. A sliced array's offsets start at `first != 0` (and
            // its data keeps the sliced-away cells) — re-base the small offset array to 0 and
            // borrow only the live `[first, last)` byte span, restoring the invariant.
            let (offsets, data) = if first == 0 && last as usize == a.value_data().len() {
                (borrow_primitive(raw, src), borrow_primitive(a.value_data(), src))
            } else {
                let offsets = Buffer::from_vec(raw.iter().map(|&o| o - first).collect());
                let data = borrow_primitive(&a.value_data()[first as usize..last as usize], src);
                (offsets, data)
            };
            Column::Str(StrBuffer::from_buffers(offsets, data), validity_of(a.nulls(), len))
        }
        DataType::Utf8 => {
            // 32-bit offsets must widen to volas's i64 (a copy of the small offset array);
            // re-base to 0 in the same pass (a no-op for an un-sliced array). The UTF-8 bytes
            // stay borrowed zero-copy over the live `[first, last)` span.
            let a = src.as_string::<i32>();
            let raw = a.value_offsets();
            let (first, last) = (raw[0], raw[raw.len() - 1]);
            let offsets = Buffer::from_vec(raw.iter().map(|&o| (o - first) as i64).collect());
            let data = borrow_primitive(&a.value_data()[first as usize..last as usize], src);
            Column::Str(StrBuffer::from_buffers(offsets, data), validity_of(a.nulls(), len))
        }
        // Narrow / unsigned integers widen to volas's i64 (a copy). `u8`/`u16`/`u32` and
        // the signed widths always fit; only `UInt64` can exceed `i64::MAX` (handled below).
        DataType::Int8 => widen_int!(Int8Type),
        DataType::Int16 => widen_int!(Int16Type),
        DataType::UInt8 => widen_int!(UInt8Type),
        DataType::UInt16 => widen_int!(UInt16Type),
        DataType::UInt32 => widen_int!(UInt32Type),
        // A `UInt64` past `i64::MAX` has no lossless i64 image — fail loud rather than
        // wrap to a negative (a value-range corruption that still looks like a valid int).
        // A null slot carries no value, so its physical bits are never range-checked.
        DataType::UInt64 => {
            let a = src.as_primitive::<UInt64Type>();
            let widened: Result<Vec<i64>, ArrowError> = (0..len)
                .map(|i| {
                    let v = a.value(i);
                    if a.is_valid(i) && v > i64::MAX as u64 {
                        Err(ArrowError::InvalidArgumentError(format!(
                            "Arrow UInt64 value {v} exceeds i64::MAX; volas has no unsigned \
                             64-bit column dtype — narrow it upstream first"
                        )))
                    } else {
                        Ok(v as i64)
                    }
                })
                .collect();
            Column::I64(Buffer::from_vec(widened?), validity_of(a.nulls(), len))
        }
        // Exact decimals map to f64 — **lossy** past ~15 significant digits; for exact
        // prices keep the column as a string upstream. `scale` divides the integer mantissa.
        DataType::Decimal128(_, scale) => {
            let a = src.as_primitive::<Decimal128Type>();
            let div = 10f64.powi(*scale as i32);
            Column::F64(Buffer::from_vec(
                (0..len)
                    .map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) as f64 / div })
                    .collect(),
            ))
        }
        // Plain dates land on the ns datetime grid (start-of-day). Date32 = days,
        // Date64 = milliseconds; out-of-range → NaT.
        DataType::Date32 => {
            let a = src.as_primitive::<Date32Type>();
            Column::Datetime(Buffer::from_vec(
                (0..len)
                    .map(|i| {
                        if a.is_null(i) {
                            i64::MIN
                        } else {
                            (a.value(i) as i64).checked_mul(86_400_000_000_000).unwrap_or(i64::MIN)
                        }
                    })
                    .collect(),
            ))
        }
        DataType::Date64 => {
            let a = src.as_primitive::<Date64Type>();
            Column::Datetime(Buffer::from_vec(
                (0..len)
                    .map(|i| {
                        if a.is_null(i) {
                            i64::MIN
                        } else {
                            a.value(i).checked_mul(1_000_000).unwrap_or(i64::MIN)
                        }
                    })
                    .collect(),
            ))
        }
        // Datetimes land on volas's nanosecond grid; `null` → the `i64::MIN` NaT
        // sentinel. The ns case is borrowed when dense; coarser units rescale (a copy).
        DataType::Timestamp(unit, _) => datetime_from_timestamp(src, *unit, len),
        // Categorical (dictionary-encoded) data — common for parquet string columns
        // (symbols, venues, sides). Decode to its dense value type and re-import (→ Str /
        // I64 / …); a null key stays NA.
        DataType::Dictionary(_, value_type) => {
            let dense = arrow_cast::cast(src, value_type)?;
            column_from_arrow(&dense)?
        }
        // 256-bit exact decimal → f64 (lossy past ~15 digits), like `Decimal128` above.
        DataType::Decimal256(_, _) => {
            let dense = arrow_cast::cast(src, &DataType::Float64)?;
            column_from_arrow(&dense)?
        }
        // Arrow `string_view` (Utf8View) — a view layout volas cannot borrow into its
        // contiguous StrBuffer; materialise it to LargeUtf8 (one copy — the same one the
        // manual `cast(pa.string())` workaround pays) and re-import.
        DataType::Utf8View => {
            let dense = arrow_cast::cast(src, &DataType::LargeUtf8)?;
            column_from_arrow(&dense)?
        }
        // A typeless all-null column (Arrow `Null`) carries no dtype; land it as an all-NA
        // f64 column — volas's neutral missing-value carrier (in-band NaN).
        DataType::Null => Column::f64(vec![f64::NAN; len]),
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
                    .map(|i| {
                        // overflow (a coarse-unit instant outside the ns-representable
                        // range) collapses to the NaT sentinel rather than panicking /
                        // wrapping.
                        if a.is_null(i) {
                            i64::MIN
                        } else {
                            a.value(i).checked_mul(scale).unwrap_or(i64::MIN)
                        }
                    })
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
