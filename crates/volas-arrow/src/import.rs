//! Arrow imports form a stable snapshot before any fact, cache or state is built.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type, TimestampNanosecondType};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use volas_core::{datetime, Column};

use crate::{borrow_primitive, validity_of};

/// Import an Arrow array into an owned snapshot, fusing dtype and validity conversion.
pub fn column_from_arrow(src: &ArrayRef) -> Result<Column, ArrowError> {
    column_from_arrow_with_copy(src, true)
}

/// Import with an explicit ownership policy. With `copy=false`, the producer and
/// every alias must never write borrowed memory again while imported columns or
/// views survive. Layouts needing conversion still copy.
pub fn column_from_arrow_with_copy(src: &ArrayRef, copy: bool) -> Result<Column, ArrowError> {
    if !copy {
        let col = match src.data_type() {
            DataType::Float64 if src.nulls().is_none() => {
                Column::F64(borrow_primitive(src.as_primitive::<Float64Type>().values(), src))
            }
            DataType::Float32 if src.nulls().is_none() => {
                Column::F32(borrow_primitive(src.as_primitive::<Float32Type>().values(), src))
            }
            DataType::Int64 => Column::I64(
                borrow_primitive(src.as_primitive::<Int64Type>().values(), src),
                validity_of(src.nulls(), src.len()),
            ),
            DataType::Int32 => Column::I32(
                borrow_primitive(src.as_primitive::<Int32Type>().values(), src),
                validity_of(src.nulls(), src.len()),
            ),
            DataType::Timestamp(TimeUnit::Nanosecond, _) if src.nulls().is_none() => {
                let a = src.as_primitive::<TimestampNanosecondType>();
                if a.values().contains(&datetime::NAT) {
                    // The instant owner rejects a present value colliding with NaT.
                    return crate::chunks::owned_chunks(std::slice::from_ref(src));
                }
                Column::Datetime(borrow_primitive(a.values(), src))
            }
            _ => return crate::chunks::owned_chunks(std::slice::from_ref(src)),
        };
        return Ok(col);
    }
    crate::chunks::owned_chunks(std::slice::from_ref(src))
}
