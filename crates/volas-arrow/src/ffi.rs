//! The Arrow **C-Data interface** for a `Column`: produce / consume the `FFI_Arrow*`
//! structs the PyCapsule protocol (`__arrow_c_array__`) moves across the boundary.
//! All arrow-rs use stays here — the Python layer only wraps these structs in capsules.

use std::ffi::{c_char, c_void, CStr};

use arrow_array::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{make_array, Array, OffsetSizeTrait};
use arrow_data::ArrayData;
use arrow_schema::{ArrowError, DataType};
use volas_core::Column;

use crate::{column_from_arrow_with_copy, column_to_arrow};

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
/// buffer is copied into an owned snapshot.
///
/// # Safety
/// `array` / `schema` must be a valid, not-yet-released Arrow C-Data pair (the
/// contract the exporting producer guarantees).
pub unsafe fn column_from_c_data(
    array: FFI_ArrowArray,
    schema: &FFI_ArrowSchema,
) -> Result<Column, ArrowError> {
    column_from_c_data_with_copy(array, schema, true)
}

/// Import a C-Data pair with an explicit ownership policy.
///
/// # Safety
/// The pair must be valid and live. For `copy=false`, the producer and every alias
/// must never write borrowed memory again while imported columns or views survive.
pub unsafe fn column_from_c_data_with_copy(
    array: FFI_ArrowArray,
    schema: &FFI_ArrowSchema,
    copy: bool,
) -> Result<Column, ArrowError> {
    let data = from_ffi(array, schema)?;
    validate_imported(&data)?;
    column_from_arrow_with_copy(&make_array(data), copy)
}

/// Check foreign C-Data before any of it is read: arrow-rs imports it unchecked, but a
/// producer's buffers are untrusted. Layout, buffer sizes and the null count are
/// checked for every array (children included), and values in full — except a string
/// array's UTF-8, which the volas string buffer checks itself on import: Arrow leaves a
/// null slot's bytes unspecified, while arrow-rs's full check would reject them. Its
/// offsets are still checked here (in range, monotonic), so nothing — concatenation
/// included — slices by a malformed offset.
pub(crate) fn validate_imported(data: &ArrayData) -> Result<(), ArrowError> {
    if data.is_empty() && matches!(data.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
        // Valid Arrow, and never read: a zero-length slice keeps its parent's offset,
        // which arrow-rs's C-Data import pairs with an empty value buffer.
        return Ok(());
    }
    data.validate()?;
    data.validate_nulls()?;
    match data.data_type() {
        DataType::Utf8 => string_offsets_in_range::<i32>(data)?,
        DataType::LargeUtf8 => string_offsets_in_range::<i64>(data)?,
        _ => data.validate_values()?,
    }
    data.child_data().iter().try_for_each(validate_imported)
}

/// A string array's offsets (its `len + 1` window) are monotonic and inside the value
/// buffer. Assumes `validate()` passed (the offset buffer holds the window).
fn string_offsets_in_range<O: OffsetSizeTrait>(data: &ArrayData) -> Result<(), ArrowError> {
    let window = &data.buffers()[0].typed_data::<O>()[data.offset()..=data.offset() + data.len()];
    let limit = data.buffers()[1].len();
    let mut prev = 0usize;
    for (i, o) in window.iter().enumerate() {
        match o.to_usize() {
            Some(o) if o <= limit && (i == 0 || prev <= o) => prev = o,
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "invalid string buffer: offset {i} ({o:?}) is out of range or not monotonic"
                )))
            }
        }
    }
    Ok(())
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
    column_from_c_capsules_with_copy(array, schema, true)
}

/// Consume live capsule payloads with an explicit ownership policy.
///
/// # Safety
/// Payloads must satisfy `column_from_c_capsules`'s safety condition. For
/// `copy=false`, all producer aliases must remain immutable for the borrow lifetime.
pub unsafe fn column_from_c_capsules_with_copy(
    array: *mut c_void,
    schema: *const c_void,
    copy: bool,
) -> Result<Column, ArrowError> {
    // A capsule pair may already have been consumed (by volas or another library):
    // a released struct's pointers dangle, so check `release` before reading either.
    if array.is_null() || (*(array as *const FFI_ArrowArray)).is_released() {
        return Err(released("array"));
    }
    check_schema(schema)?;
    let array = std::ptr::replace(array as *mut FFI_ArrowArray, FFI_ArrowArray::empty());
    column_from_c_data_with_copy(array, &*(schema as *const FFI_ArrowSchema), copy)
}

fn released(what: &str) -> ArrowError {
    ArrowError::CDataInterface(format!(
        "the Arrow {what} was already released (a capsule can be consumed only once)"
    ))
}

/// The head of the C ABI `ArrowSchema` struct (the Arrow C Data Interface spec's
/// stable layout): what must be checked before arrow-rs reads a foreign schema.
#[repr(C)]
struct RawSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut RawSchema,
    dictionary: *mut RawSchema,
    release: Option<unsafe extern "C" fn(*mut RawSchema)>,
    private_data: *mut c_void,
}

/// A foreign `ArrowSchema` is live (not released) and well-formed where arrow-rs
/// would otherwise panic: a non-null UTF-8 `format`, UTF-8 `name`s, and the same
/// for every child and dictionary schema (the depth is bounded by the producer's
/// nesting; a cycle would be a malformed producer).
///
/// # Safety
/// `schema` is null or points to an `ArrowSchema` struct.
pub(crate) unsafe fn check_schema(schema: *const c_void) -> Result<(), ArrowError> {
    let s = schema as *const RawSchema;
    if s.is_null() || (*s).release.is_none() {
        return Err(released("schema"));
    }
    check_schema_fields(s)
}

unsafe fn check_schema_fields(s: *const RawSchema) -> Result<(), ArrowError> {
    let malformed = |what: &str| ArrowError::CDataInterface(format!("malformed Arrow schema: {what}"));
    let s = &*s;
    if s.format.is_null() || CStr::from_ptr(s.format).to_str().is_err() {
        return Err(malformed("the format is missing or not UTF-8"));
    }
    if !s.name.is_null() && CStr::from_ptr(s.name).to_str().is_err() {
        return Err(malformed("a field name is not UTF-8"));
    }
    if s.n_children < 0 || (s.n_children > 0 && s.children.is_null()) {
        return Err(malformed("the children are inconsistent"));
    }
    for i in 0..s.n_children as usize {
        let child = *s.children.add(i);
        if child.is_null() {
            return Err(malformed("a child schema is null"));
        }
        check_schema_fields(child)?;
    }
    if !s.dictionary.is_null() {
        check_schema_fields(s.dictionary)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;

    unsafe extern "C" fn noop(_: *mut RawSchema) {}

    fn raw(format: *const c_char, name: *const c_char) -> RawSchema {
        RawSchema {
            format,
            name,
            metadata: std::ptr::null(),
            flags: 0,
            n_children: 0,
            children: null_mut(),
            dictionary: null_mut(),
            release: Some(noop),
            private_data: null_mut(),
        }
    }

    fn check(s: &RawSchema) -> String {
        // SAFETY: `s` is a live, fully initialised RawSchema.
        match unsafe { check_schema(s as *const RawSchema as *const c_void) } {
            Ok(()) => "ok".into(),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn foreign_schemas_are_checked_before_arrow_reads_them() {
        let (fmt, bad) = (c"l".as_ptr(), c"\xff".as_ptr());
        assert_eq!(check(&raw(fmt, c"x".as_ptr())), "ok");
        // SAFETY: `noop` ignores its argument (it only stands in for a live release).
        unsafe { noop(null_mut()) };
        // SAFETY: a null pointer is rejected before any read.
        assert!(unsafe { check_schema(std::ptr::null()) }.unwrap_err().to_string().contains("released"));
        let mut released = raw(fmt, std::ptr::null());
        released.release = None;
        assert!(check(&released).contains("released"));
        assert!(check(&raw(std::ptr::null(), std::ptr::null())).contains("format"));
        assert!(check(&raw(bad, std::ptr::null())).contains("format"));
        assert!(check(&raw(fmt, bad)).contains("name"));
        let mut negative = raw(fmt, std::ptr::null());
        negative.n_children = -1;
        assert!(check(&negative).contains("children"));
        let mut missing = raw(fmt, std::ptr::null());
        missing.n_children = 1;
        assert!(check(&missing).contains("children"));
        let mut slots: [*mut RawSchema; 1] = [null_mut()];
        let mut null_child = raw(fmt, std::ptr::null());
        null_child.n_children = 1;
        null_child.children = slots.as_mut_ptr();
        assert!(check(&null_child).contains("child schema is null"));
        let mut child = raw(bad, std::ptr::null());
        // SAFETY: `children` points at the live one-slot array above.
        unsafe { *null_child.children = &mut child };
        assert!(check(&null_child).contains("format"));
        let mut dict_value = raw(bad, std::ptr::null());
        let mut with_dict = raw(fmt, std::ptr::null());
        with_dict.dictionary = &mut dict_value;
        assert!(check(&with_dict).contains("format"));
    }
}
