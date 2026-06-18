//! The Arrow **C-Data interface** for a `Column`: produce / consume the `FFI_Arrow*`
//! structs the PyCapsule protocol (`__arrow_c_array__`) moves across the boundary.
//! All arrow-rs use stays here — the Python layer only wraps these structs in capsules.

use std::ffi::c_void;

use arrow_array::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{make_array, Array};
use arrow_schema::ArrowError;
use volas_core::Column;

use crate::{column_from_arrow, column_to_arrow};

/// A column as the two owned Arrow C-Data structs (array + schema). The data buffer is
/// shared (no copy) — the structs carry a release callback that drops the keep-alive.
pub fn column_to_c_data(col: &Column) -> Result<(FFI_ArrowArray, FFI_ArrowSchema), ArrowError> {
    to_ffi(&column_to_arrow(col).to_data())
}

/// Just the C-Data schema for a column (the `__arrow_c_schema__` half).
pub fn column_to_c_schema(col: &Column) -> Result<FFI_ArrowSchema, ArrowError> {
    FFI_ArrowSchema::try_from(column_to_arrow(col).data_type())
}

/// A column from an Arrow C-Data pair: consumes `array`, reads `schema`. The data
/// buffer is borrowed through (no copy) for the dtypes whose layouts match.
///
/// # Safety
/// `array` / `schema` must be a valid, not-yet-released Arrow C-Data pair (the
/// contract the exporting producer guarantees).
pub unsafe fn column_from_c_data(
    array: FFI_ArrowArray,
    schema: &FFI_ArrowSchema,
) -> Result<Column, ArrowError> {
    column_from_arrow(&make_array(from_ffi(array, schema)?))
}

/// A column from the two PyCapsule payload pointers of the `__arrow_c_array__`
/// protocol. Moves the array out of its capsule — leaving an empty (already-released)
/// struct so the capsule's own destructor becomes a no-op — and borrows the schema,
/// which its capsule still owns. Keeps the whole FFI dance inside this crate so the
/// Python layer hands over only raw `c_void` pointers.
///
/// # Safety
/// `array` / `schema` must be the live capsule payloads of a valid C-Data pair
/// (named `"arrow_array"` / `"arrow_schema"`), not yet consumed.
pub unsafe fn column_from_c_capsules(
    array: *mut c_void,
    schema: *const c_void,
) -> Result<Column, ArrowError> {
    let array = std::ptr::replace(array as *mut FFI_ArrowArray, FFI_ArrowArray::empty());
    column_from_c_data(array, &*(schema as *const FFI_ArrowSchema))
}
