//! Dictionary keys select producer cells directly; never materialize a decoded
//! Arrow column before writing the final snapshot.
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;

pub(crate) fn dtype(mut dt: &DataType) -> &DataType {
    while let DataType::Dictionary(_, values) = dt { dt = values; }
    dt
}

/// Resolve a logical row to its leaf array and physical offset. A missing key
/// has no payload to read; a leaf null keeps its physical payload for native ints.
pub(crate) fn cell(mut a: &ArrayRef, mut i: usize) -> Option<(&ArrayRef, usize)> {
    while let DataType::Dictionary(key, _) = a.data_type() {
        if a.is_null(i) { return None; }
        let dict = a.as_any_dictionary();
        macro_rules! key {
            ($ty:ty) => { dict.keys().as_primitive::<$ty>().value(i) as usize };
        }
        i = match key.as_ref() {
            DataType::Int8 => key!(Int8Type),
            DataType::Int16 => key!(Int16Type),
            DataType::Int32 => key!(Int32Type),
            DataType::Int64 => key!(Int64Type),
            DataType::UInt8 => key!(UInt8Type),
            DataType::UInt16 => key!(UInt16Type),
            DataType::UInt32 => key!(UInt32Type),
            DataType::UInt64 => key!(UInt64Type),
            _ => unreachable!("Arrow dictionary keys are integer typed"), // LCOV_EXCL_LINE
        };
        a = dict.values();
    }
    Some((a, i))
}

pub(crate) fn valid(a: &ArrayRef, i: usize) -> bool {
    cell(a, i).is_some_and(|(a, i)| a.is_valid(i))
}
