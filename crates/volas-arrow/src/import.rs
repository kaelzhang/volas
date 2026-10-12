//! Arrow `ArrayRef` → `Column`. Numeric / ns-datetime data is borrowed zero-copy
//! ([`Buffer::Borrowed`](volas_core::Buffer)); `bool`, strings (copied and validated,
//! see `copy_strings`) and non-ns timestamps are repacked.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal128Type, Decimal256Type, Float32Type, Float64Type, Int16Type, Int32Type,
    Int64Type, Int8Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use volas_core::{datetime, decimal, Buffer, Column, StrBuffer, Validity};

use crate::{borrow_primitive, validity_of};

/// Import an Arrow array as a volas `Column`. Borrows a numeric / datetime data
/// buffer where the physical layouts match; copies `bool`, strings, and timestamps
/// coarser than nanoseconds.
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
            let offsets = a.value_offsets().to_vec();
            copy_strings(offsets, a.value_data(), validity_of(a.nulls(), len))?
        }
        DataType::Utf8 => {
            // 32-bit offsets widen to volas's i64 in the same copying pass.
            let a = src.as_string::<i32>();
            let offsets = a.value_offsets().iter().map(|&o| o as i64).collect();
            copy_strings(offsets, a.value_data(), validity_of(a.nulls(), len))?
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
        // Exact decimals map to the f64 nearest to each value (`volas_core::decimal`) —
        // **lossy** past ~15 significant digits; for exact prices keep the column as a
        // string upstream. A value past the f64 range is an error, never `inf`.
        DataType::Decimal128(_, scale) => {
            let a = src.as_primitive::<Decimal128Type>();
            decimals(len, |i| a.is_null(i), |i| decimal::scaled_to_f64(a.value(i), *scale as i32))?
        }
        // Plain dates land on the ns datetime grid (start-of-day). Date32 = days,
        // Date64 = milliseconds; a date outside the instant range raises.
        DataType::Date32 => {
            let a = src.as_primitive::<Date32Type>();
            rescaled(len, |i| (!a.is_null(i)).then(|| a.value(i) as i64), "D")?
        }
        DataType::Date64 => {
            let a = src.as_primitive::<Date64Type>();
            rescaled(len, |i| (!a.is_null(i)).then(|| a.value(i)), "ms")?
        }
        // Datetimes land on volas's nanosecond grid; `null` → the NaT sentinel. The
        // ns case is borrowed when dense; coarser units rescale (a copy).
        DataType::Timestamp(unit, _) => datetime_from_timestamp(src, *unit, len)?,
        // Categorical (dictionary-encoded) data — common for parquet string columns
        // (symbols, venues, sides). Decode to its dense value type and re-import (→ Str /
        // I64 / …); a null key stays NA.
        DataType::Dictionary(_, value_type) => {
            let dense = arrow_cast::cast(src, value_type)?;
            column_from_arrow(&dense)?
        }
        // 256-bit exact decimal → the nearest f64, like `Decimal128` above.
        DataType::Decimal256(_, scale) => {
            let a = src.as_primitive::<Decimal256Type>();
            decimals(len, |i| a.is_null(i), |i| match a.value(i).to_i128() {
                Some(m) => decimal::scaled_to_f64(m, *scale as i32),
                None => decimal::scaled_text_to_f64(&a.value(i).to_string(), *scale as i32),
            })?
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

/// The live byte span `[first, last)` a string array's end offsets name, checked to
/// lie inside its `data_len`-byte value buffer before anything is sliced by it.
fn live_span(first: i64, last: i64, data_len: usize) -> Result<(usize, usize), ArrowError> {
    match (usize::try_from(first), usize::try_from(last)) {
        (Ok(f), Ok(l)) if f <= l && l <= data_len => Ok((f, l)),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "invalid string buffer: offsets span {first}..{last} lies outside the \
             {data_len}-byte data buffer"
        ))),
    }
}

/// A `Str` column **copied** out of an Arrow string array: its (widened) `offsets`
/// and the live `[first, last)` span of `bytes`, re-based to 0 into volas-owned
/// buffers, then validated by the string buffer's owner
/// (`StrBuffer::try_from_buffers`) — malformed offsets or a non-UTF-8 present cell
/// are an import error, a non-UTF-8 null slot becomes an empty span. Strings are
/// never borrowed: the unchecked string readers rely on the checked invariant,
/// which a producer rewriting its buffers after import would otherwise break.
fn copy_strings(
    mut offsets: Vec<i64>,
    bytes: &[u8],
    validity: Validity,
) -> Result<Column, ArrowError> {
    if offsets.len() <= 1 {
        // zero cells: nothing to read (a zero-length slice's one offset may point
        // anywhere in its parent)
        return Ok(Column::Str(StrBuffer::from_vec(Vec::new()), validity));
    }
    let (first, last) = live_span(offsets[0], offsets[offsets.len() - 1], bytes.len())?;
    let base = first as i64;
    offsets.iter_mut().for_each(|o| *o = o.wrapping_sub(base));
    let sb = StrBuffer::try_from_buffers(
        Buffer::from_vec(offsets),
        Buffer::from_vec(bytes[first..last].to_vec()),
        &validity,
    )
    .map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?;
    Ok(Column::Str(sb, validity))
}

/// A `Datetime` column from `len` raw values in `unit` (`None` = an Arrow null →
/// NaT). Every present value passes the instant gate: a value outside the
/// datetime64[ns] range — or one landing on the NaT sentinel — is an import
/// error, never a silently missing cell.
fn rescaled(
    len: usize,
    value: impl Fn(usize) -> Option<i64>,
    unit: &str,
) -> Result<Column, ArrowError> {
    (0..len)
        .map(|i| match value(i) {
            None => Ok(datetime::NAT),
            Some(v) => datetime::datetime64_to_ns(v, unit, 1),
        })
        .collect::<volas_core::Result<Vec<i64>>>()
        .map(|v| Column::Datetime(Buffer::from_vec(v)))
        .map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))
}

/// Timestamp → `Datetime` (ns). Borrows the buffer for a dense nanosecond array
/// (after checking no value collides with the NaT sentinel); otherwise rescales /
/// fills the sentinel into a fresh buffer through [`rescaled`].
fn datetime_from_timestamp(
    src: &ArrayRef,
    unit: TimeUnit,
    len: usize,
) -> Result<Column, ArrowError> {
    macro_rules! filled {
        ($ty:ty, $unit:literal) => {{
            let a = src.as_primitive::<$ty>();
            rescaled(len, |i| (!a.is_null(i)).then(|| a.value(i)), $unit)
        }};
    }
    match unit {
        TimeUnit::Nanosecond => {
            let a = src.as_primitive::<TimestampNanosecondType>();
            match a.nulls() {
                None if !a.values().contains(&datetime::NAT) => {
                    Ok(Column::Datetime(borrow_primitive(a.values(), src)))
                }
                _ => filled!(TimestampNanosecondType, "ns"),
            }
        }
        TimeUnit::Microsecond => filled!(TimestampMicrosecondType, "us"),
        TimeUnit::Millisecond => filled!(TimestampMillisecondType, "ms"),
        TimeUnit::Second => filled!(TimestampSecondType, "s"),
    }
}

/// A decimal column as f64 (`null` → NaN, volas's float missing value), each value read
/// by `value`; a value past the f64 range is an error naming its row.
fn decimals(
    len: usize,
    is_null: impl Fn(usize) -> bool,
    value: impl Fn(usize) -> Option<f64>,
) -> Result<Column, ArrowError> {
    let out = (0..len)
        .map(|i| {
            if is_null(i) {
                return Ok(f64::NAN);
            }
            value(i).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "Arrow decimal at row {i} lies past the float64 range"
                ))
            })
        })
        .collect::<Result<Vec<f64>, ArrowError>>()?;
    Ok(Column::F64(Buffer::from_vec(out)))
}
