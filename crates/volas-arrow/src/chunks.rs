//! Multi-batch imports write directly into the final owned buffers. No Arrow
//! concatenation buffer is needed before dtype or missing-value conversion.

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use crate::rows::{cell, dtype, valid};
use volas_core::{datetime, decimal, Buffer, Column, StrBufferBuilder, Validity};

pub(crate) fn owned_chunks(src: &[ArrayRef]) -> Result<Column, ArrowError> {
    // The stream schema guarantees a common dtype. Length was checked by its owner.
    let len = src.iter().map(|a| a.len()).sum();
    let validity = || Validity::from_valid_iter(len, src.iter().flat_map(|a| {
        (0..a.len()).map(|i| valid(a, i))
    }));
    macro_rules! values {
        ($ty:ty, $f:expr, $missing:expr) => {{
            let mut out = Vec::with_capacity(len);
            for s in src {
                out.extend((0..s.len()).map(|i| match cell(s, i) {
                    Some((a, i)) => ($f)(a.as_primitive::<$ty>(), i),
                    None => $missing,
                }));
            }
            out
        }};
    }
    macro_rules! native {
        ($ty:ty) => {{
            let mut out = Vec::with_capacity(len);
            for s in src {
                if matches!(s.data_type(), DataType::Dictionary(..)) {
                    out.extend((0..s.len()).map(|i| match cell(s, i) {
                        Some((a, i)) => a.as_primitive::<$ty>().value(i),
                        None => Default::default(),
                    }));
                } else { out.extend_from_slice(s.as_primitive::<$ty>().values()); }
            }
            Buffer::from_vec(out)
        }};
    }
    macro_rules! float {
        ($ty:ty, $nan:expr) => {{
            let mut out = Vec::with_capacity(len);
            for s in src {
                if matches!(s.data_type(), DataType::Dictionary(..)) {
                    out.extend((0..s.len()).map(|i| match cell(s, i) {
                        Some((a, i)) if a.is_valid(i) => a.as_primitive::<$ty>().value(i),
                        _ => $nan,
                    }));
                    continue;
                }
                let a = s.as_primitive::<$ty>();
                if a.nulls().is_none() {
                    out.extend_from_slice(a.values());
                } else {
                    out.extend((0..a.len()).map(|i| if a.is_null(i) { $nan } else { a.value(i) }));
                }
            }
            Buffer::from_vec(out)
        }};
    }
    macro_rules! integer {
        ($ty:ty) => { Column::from_i64_buffer(Buffer::from_vec(values!($ty, |a: &arrow_array::PrimitiveArray<$ty>, i| a.value(i) as i64, 0)), validity()).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))? };
    }
    macro_rules! instant {
        ($ty:ty, $unit:expr) => {{
            let mut out = Vec::with_capacity(len);
            for s in src {
                for i in 0..s.len() {
                    out.push(match cell(s, i) {
                        Some((a, i)) if a.is_valid(i) => {
                            datetime::datetime64_to_ns(a.as_primitive::<$ty>().value(i) as i64, $unit, 1)
                                .map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?
                        }
                        _ => datetime::NAT,
                    });
                }
            }
            Column::from_datetime_buffer(Buffer::from_vec(out))
        }};
    }
    Ok(match dtype(src[0].data_type()) {
        DataType::Float64 => Column::from_f64_buffer(float!(Float64Type, f64::NAN)),
        DataType::Float32 => Column::from_f32_buffer(float!(Float32Type, f32::NAN)),
        DataType::Int64 => Column::from_i64_buffer(native!(Int64Type), validity()).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?,
        DataType::Int32 => Column::from_i32_buffer(native!(Int32Type), validity()).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?,
        DataType::Int8 => integer!(Int8Type),
        DataType::Int16 => integer!(Int16Type),
        DataType::UInt8 => integer!(UInt8Type),
        DataType::UInt16 => integer!(UInt16Type),
        DataType::UInt32 => integer!(UInt32Type),
        DataType::UInt64 => {
            let mut out = Vec::with_capacity(len);
            for s in src {
                for i in 0..s.len() {
                    let Some((a, i)) = cell(s, i) else { out.push(0); continue; };
                    let a = a.as_primitive::<UInt64Type>();
                    let v = a.value(i);
                    if a.is_valid(i) && v > i64::MAX as u64 {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "Arrow UInt64 value {v} exceeds i64::MAX; volas has no unsigned \
                             64-bit column dtype — narrow it upstream first"
                        )));
                    }
                    out.push(v as i64);
                }
            }
            Column::from_i64_buffer(Buffer::from_vec(out), validity()).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?
        }
        DataType::Boolean => {
            let mut out = Vec::with_capacity(len);
            for s in src {
                if matches!(s.data_type(), DataType::Dictionary(..)) {
                    out.extend((0..s.len()).map(|i| cell(s, i).is_some_and(|(a, i)| a.as_boolean().value(i))));
                } else { out.extend(s.as_boolean().values().iter()); }
            }
            Column::from_bool_buffer(Buffer::from_vec(out), validity()).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => instant!(TimestampNanosecondType, "ns"),
        DataType::Timestamp(TimeUnit::Microsecond, _) => instant!(TimestampMicrosecondType, "us"),
        DataType::Timestamp(TimeUnit::Millisecond, _) => instant!(TimestampMillisecondType, "ms"),
        DataType::Timestamp(TimeUnit::Second, _) => instant!(TimestampSecondType, "s"),
        DataType::Date32 => instant!(Date32Type, "D"),
        DataType::Date64 => instant!(Date64Type, "ms"),
        DataType::Decimal128(_, scale) => Column::from_f64_buffer(Buffer::from_vec(values!(Decimal128Type,
            |a: &arrow_array::Decimal128Array, i| if a.is_null(i) { f64::NAN } else {
                decimal::scaled_to_f64(a.value(i), *scale as i32).expect("Arrow decimal fits f64")
            }, f64::NAN))),
        DataType::Decimal256(_, scale) => Column::from_f64_buffer(Buffer::from_vec(values!(Decimal256Type,
            |a: &arrow_array::Decimal256Array, i| if a.is_null(i) { f64::NAN } else {
                match a.value(i).to_i128() {
                    Some(m) => decimal::scaled_to_f64(m, *scale as i32),
                    None => decimal::scaled_text_to_f64(&a.value(i).to_string(), *scale as i32),
                }.expect("Arrow decimal fits f64")
            }, f64::NAN))),
        DataType::Utf8 => strings::<i32>(src, len)?,
        DataType::LargeUtf8 => strings::<i64>(src, len)?,
        DataType::Null => Column::f64(vec![f64::NAN; len]),
        DataType::Utf8View => string_views(src, len)?,
        other => return Err(ArrowError::NotYetImplemented(format!(
            "volas cannot import an Arrow {other:?} column"
        ))),
    })
}

fn strings<O: arrow_array::OffsetSizeTrait>(src: &[ArrayRef], len: usize) -> Result<Column, ArrowError> {
    for s in src {
        let mut leaf = s;
        while matches!(leaf.data_type(), DataType::Dictionary(..)) {
            leaf = leaf.as_any_dictionary().values();
        }
        if leaf.is_empty() { continue; }
        let a = leaf.as_string::<O>();
        let offsets = a.value_offsets();
        let first = offsets[0].to_usize();
        let last = offsets[offsets.len() - 1].to_usize();
        let (Some(first), Some(last)) = (first, last) else {
            return Err(ArrowError::InvalidArgumentError("string offsets outside data buffer".into()));
        };
        if first > last || last > a.value_data().len() {
            return Err(ArrowError::InvalidArgumentError("string offsets outside data buffer".into()));
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(ArrowError::InvalidArgumentError("string offsets are not monotonic".into()));
        }
    }
    let value = |s, i| -> Result<&str, ArrowError> {
        let Some((a, i)) = cell(s, i) else { return Ok(""); };
        if a.is_null(i) { return Ok(""); }
        let a = a.as_string::<O>();
        let offsets = a.value_offsets();
        let bytes = &a.value_data()[offsets[i].as_usize()..offsets[i + 1].as_usize()];
        std::str::from_utf8(bytes).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("invalid string buffer: invalid UTF-8: {e}"))
        })
    };
    let mut bytes = 0;
    for s in src { for i in 0..s.len() { bytes += value(s, i)?.len(); } }
    let mut builder = StrBufferBuilder::with_capacities(len, bytes);
    for s in src { for i in 0..s.len() { builder.push(value(s, i)?); } }
    let validity = Validity::from_valid_iter(len, src.iter().flat_map(|a| {
        (0..a.len()).map(|i| valid(a, i))
    }));
    Column::from_str_buffer(builder.finish(), validity).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))
}

fn string_views(src: &[ArrayRef], len: usize) -> Result<Column, ArrowError> {
    let mut bytes = 0;
    for s in src {
        for i in 0..s.len() {
            if let Some((a, i)) = cell(s, i).filter(|(a, i)| a.is_valid(*i)) {
                bytes += a.as_string_view().value(i).len();
            }
        }
    }
    let mut builder = StrBufferBuilder::with_capacities(len, bytes);
    for s in src {
        for i in 0..s.len() {
            builder.push(match cell(s, i).filter(|(a, i)| a.is_valid(*i)) {
                Some((a, i)) => a.as_string_view().value(i),
                None => "",
            });
        }
    }
    Column::from_str_buffer(builder.finish(), Validity::from_valid_iter(len,
        src.iter().flat_map(|a| (0..a.len()).map(|i| valid(a, i))))).map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))
}
