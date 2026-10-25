//! volas-arrow: a zero-copy bridge between volas `Column`s and Arrow arrays.
//!
//! arrow-rs lives **only** in this crate — `volas-core` stays dependency-light and
//! the Python layer composes this bridge with the C-Data interface.
//!
//! ## Ownership contract
//! Arrow data is immutable. Imports borrow matching numeric / dense ns-datetime
//! data by default; `*_with_copy(..., true)` takes an owned snapshot. The precondition is
//! that the producer and every alias never write borrowed memory again while any
//! imported column or view survives. Conversion layouts still copy.
//! Exports share read-only data buffers; bool and validity are repacked.

#[cfg(test)]
use volas_core::ColumnView;
use std::any::Any;
use std::panic::RefUnwindSafe;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_buffer::{Buffer as ArrowBuffer, NullBuffer};
use volas_core::{Buffer, Validity};

/// Default ownership policy for all Arrow-family imports, including Python bindings.
pub const DEFAULT_IMPORT_COPY: bool = false;

mod chunks;
mod export;
mod ffi;
mod import;
mod rows;
mod stream;

pub use export::column_to_arrow;
pub use ffi::{
    column_from_c_capsules, column_from_c_capsules_with_copy, column_from_c_data,
    column_from_c_data_with_copy, column_to_c_data, column_to_c_schema,
};
pub use import::{column_from_arrow, column_from_arrow_with_copy};
pub use stream::{
    columns_from_c_stream, columns_from_c_stream_with_copy, columns_to_c_stream,
    frame_from_c_stream, frame_from_c_stream_with_copy, frame_to_c_stream,
};
// Re-exported so the Python layer wraps these in PyCapsules without depending on
// arrow-rs directly — the bridge keeps arrow-rs entirely within this crate.
pub use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
pub use arrow_array::ffi_stream::FFI_ArrowArrayStream;

/// Wraps a volas keep-alive (`Arc<dyn Any>`) so it satisfies Arrow's `Allocation`
/// bound. The handle is only ever held to defer the drop of the backing
/// allocation — never observed after a panic — so asserting `RefUnwindSafe` is sound.
struct KeepAlive(#[allow(dead_code)] Arc<dyn Any + Send + Sync>);
impl RefUnwindSafe for KeepAlive {}

/// Wraps a source Arrow array as an opaque `Send + Sync` guard, the keep-alive a
/// [`Buffer::Borrowed`] holds so the imported pointer stays valid.
struct Hold(#[allow(dead_code)] ArrayRef);

/// Publish a volas [`Buffer<T>`] to Arrow as a borrowed allocation — no copy; the
/// buffer's `Arc` keep-alive backs the returned Arrow buffer.
fn arrow_buffer_of<T: volas_core::Element + Send + Sync + 'static>(buf: &Buffer<T>) -> ArrowBuffer {
    let bytes = std::mem::size_of_val(buf.as_slice());
    let ptr = NonNull::new(buf.as_ptr() as *mut u8).unwrap_or(NonNull::dangling());
    // SAFETY: `ptr` covers `bytes` initialised bytes kept alive by `owner` (the
    // buffer's `Arc`); Arrow only reads them and drops `owner` when done.
    unsafe { ArrowBuffer::from_custom_allocation(ptr, bytes, Arc::new(KeepAlive(buf.keepalive()))) }
}

/// A volas [`Validity`] as an Arrow null buffer: `None` when dense (no nulls). The
/// bitmap is rebuilt (≤ `n/8` bytes) rather than shared — volas stores it as `u64`
/// words behind an `Arc<Bitmap>` that Arrow cannot borrow directly.
fn null_buffer_of(validity: &Validity, len: usize) -> Option<NullBuffer> {
    validity
        .has_nulls()
        .then(|| NullBuffer::from_iter((0..len).map(|i| validity.is_valid(i))))
}

/// Import an Arrow null buffer into a volas [`Validity`] (copies the small bitmap).
fn validity_of(nulls: Option<&NullBuffer>, len: usize) -> Validity {
    match nulls {
        None => Validity::dense(),
        Some(nb) => Validity::from_valid_iter(len, (0..len).map(|i| nb.is_valid(i))),
    }
}

/// View an imported Arrow primitive slice as a zero-copy [`Buffer::Borrowed`], guarded
/// by `src` (the source array, kept alive for the buffer's lifetime).
fn borrow_primitive<T: volas_core::Element + Send + Sync + 'static>(values: &[T], src: &ArrayRef) -> Buffer<T> {
    let ptr = NonNull::new(values.as_ptr() as *mut T).unwrap_or(NonNull::dangling());
    // SAFETY: `values` points into `src`'s buffer; `Hold(src)` keeps it alive for as
    // long as the returned `Buffer`; the caller explicitly accepts the producer
    // immutability precondition.
    unsafe { Buffer::from_foreign(ptr, values.len(), Arc::new(Hold(src.clone()))) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};
    use arrow_array::{
        Array, BooleanArray, Float64Array, Int64Array, LargeStringArray, StringArray,
        TimestampNanosecondArray,
    };
    use volas_core::{Column, Validity};

    fn na_at(len: usize, na: &[usize]) -> Validity {
        Validity::from_valid_iter(len, (0..len).map(|i| !na.contains(&i)))
    }

    #[test]
    fn f64_dense_roundtrip_is_zero_copy_both_ways() {
        // A NaN-free float column round-trips zero-copy in both directions (no null
        // bitmap, so import borrows). A column WITH a missing value is covered by
        // `float_missing_exports_as_arrow_null` (import materialises, by design).
        let col = Column::f64(vec![1.0, 2.0, 3.0]);
        let src_ptr = col.as_f64().unwrap().as_ptr();
        let arr = column_to_arrow(&col);
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.as_primitive::<Float64Type>().values().as_ptr(), src_ptr); // export shares
        let arrow_ptr = arr.as_primitive::<Float64Type>().values().as_ptr();
        let back = column_from_arrow_with_copy(&arr, false).unwrap();
        assert_eq!(back.as_f64().unwrap().as_ptr(), arrow_ptr); // import borrows
        assert_eq!(back.as_f64().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn float_missing_exports_as_arrow_null() {
        // A missing float (in-band NaN) must cross as a real Arrow null — visible to
        // `is_null` / `null_count` — so int and float "missing" are consistent at the
        // boundary (not a NaN value that looks present to a null-aware consumer).
        let arr = column_to_arrow(&Column::f64(vec![1.0, f64::NAN, 3.0]));
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1) && !arr.is_null(0));
        // and it round-trips back to NaN-as-missing
        let back = column_from_arrow(&arr).unwrap();
        assert!(back.as_f64().unwrap()[1].is_nan());
        assert_eq!(back.as_f64().unwrap()[0], 1.0);
        // dense floats stay null-free
        assert_eq!(column_to_arrow(&Column::f64(vec![1.0, 2.0])).null_count(), 0);
        // same for f32 (export null + the f32-with-nulls import branch)
        let f32 = column_to_arrow(&Column::f32(vec![1.5, f32::NAN]));
        assert_eq!(f32.null_count(), 1);
        match column_from_arrow(&f32).unwrap().view() {
            ColumnView::F32(b) => assert!(b[1].is_nan() && b[0] == 1.5),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
        // a dense (null-free) f32 imports via the borrow branch
        let dense_f32: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![1.0f32, 2.0]));
        match column_from_arrow(&dense_f32).unwrap().view() {
            ColumnView::F32(b) => assert_eq!(&b[..], &[1.0, 2.0]),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn i64_with_nulls_roundtrips() {
        let col = Column::i64_with(vec![10, 0, 30], na_at(3, &[1])).unwrap();
        let arr = column_to_arrow(&col);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(column_from_arrow(&arr).unwrap(), col);
        // a dense int column exports its buffer and re-imports it with no copy
        let dense = Column::i64_with(vec![7, 8], Validity::dense()).unwrap();
        let exported = column_to_arrow(&dense);
        let arrow_ptr = exported.as_primitive::<Int64Type>().values().as_ptr();
        assert_eq!(dense.as_i64().unwrap().as_ptr(), arrow_ptr);
        assert_eq!(column_from_arrow_with_copy(&exported, false).unwrap().as_i64().unwrap().as_ptr(), arrow_ptr);
    }

    #[test]
    fn bool_roundtrips_through_bitpacking() {
        let col = Column::bool_with(vec![true, false, true], na_at(3, &[2])).unwrap();
        assert_eq!(column_from_arrow(&column_to_arrow(&col)).unwrap(), col);
    }

    #[test]
    fn str_roundtrip_borrows_the_byte_buffer() {
        let col = Column::str_with(vec!["alpha".into(), "".into(), "ω".into()], na_at(3, &[1])).unwrap();
        let data_ptr = match col.view() {
            ColumnView::Str(sb, _) => sb.buffers().1.as_ptr(),
            _ => unreachable!(), // LCOV_EXCL_LINE
        };
        let arr = column_to_arrow(&col);
        assert_eq!(arr.as_string::<i64>().value_data().as_ptr(), data_ptr);
        assert_eq!(column_from_arrow(&arr).unwrap(), col);
    }

    #[test]
    fn datetime_nat_roundtrips() {
        let col = Column::datetime(vec![1_000, i64::MIN, 3_000]);
        let arr = column_to_arrow(&col);
        assert_eq!(arr.null_count(), 1);
        // a column's instants are UTC, and the exported type says so
        assert_eq!(
            arr.data_type(),
            &arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, Some("UTC".into()))
        );
        assert_eq!(column_from_arrow(&arr).unwrap(), col);
    }

    #[test]
    fn external_float_null_bitmap_imports_as_nan() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        let col = column_from_arrow(&arr).unwrap();
        let v = col.as_f64().unwrap();
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn external_utf8_i32_offsets_import() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "bc"]));
        assert_eq!(
            column_from_arrow(&arr).unwrap(),
            Column::str_with(vec!["a".into(), "bc".into()], Validity::dense()).unwrap()
        );
        // large_string offsets need no widening
        let large: ArrayRef = Arc::new(LargeStringArray::from(vec!["a", "bc"]));
        assert_eq!(column_from_arrow(&large).unwrap().dtype(), col_str_dtype());
    }

    fn col_str_dtype() -> volas_core::DType {
        Column::str_with(vec![], Validity::dense()).unwrap().dtype()
    }

    #[test]
    fn boolean_and_timestamp_import_paths() {
        let b: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None]));
        assert_eq!(column_from_arrow(&b).unwrap(), Column::bool_with(vec![true, false], na_at(2, &[1])).unwrap());
        let ts: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![Some(5), None]));
        assert_eq!(column_from_arrow(&ts).unwrap(), Column::datetime(vec![5, i64::MIN]));
    }

    #[test]
    fn unsupported_type_is_an_error() {
        // a Duration column has no volas column type (Date32 is now supported below)
        let d: ArrayRef = Arc::new(arrow_array::DurationSecondArray::from(vec![1, 2]));
        assert!(column_from_arrow(&d).is_err());
    }

    #[test]
    fn imports_decimal_as_f64() {
        use arrow_array::Decimal128Array;
        // decimal(18,2): 150 -> 1.50, NA, 250 -> 2.50 (lossy: maps to f64)
        let arr: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(150), None, Some(250)])
                .with_precision_and_scale(18, 2)
                .unwrap(),
        );
        let col = column_from_arrow(&arr).unwrap();
        let v = col.as_f64().unwrap();
        assert_eq!(v[0], 1.5);
        assert!(v[1].is_nan());
        assert_eq!(v[2], 2.5);
    }

    #[test]
    fn imports_narrow_and_unsigned_ints_as_i64() {
        use arrow_array::{
            Int16Array, Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
        };
        let i16: ArrayRef = Arc::new(Int16Array::from(vec![Some(1), None, Some(-3)]));
        assert_eq!(
            column_from_arrow(&i16).unwrap(),
            Column::i64_with(vec![1, 0, -3], na_at(3, &[1])).unwrap()
        );
        // every narrow / unsigned width widens to i64
        let i8: ArrayRef = Arc::new(Int8Array::from(vec![-5i8, 5]));
        assert_eq!(column_from_arrow(&i8).unwrap().as_i64().unwrap(), &[-5, 5]);
        let u8: ArrayRef = Arc::new(UInt8Array::from(vec![1u8, 255]));
        assert_eq!(column_from_arrow(&u8).unwrap().as_i64().unwrap(), &[1, 255]);
        let u16: ArrayRef = Arc::new(UInt16Array::from(vec![1u16, 60000]));
        assert_eq!(column_from_arrow(&u16).unwrap().as_i64().unwrap(), &[1, 60000]);
        let u32: ArrayRef = Arc::new(UInt32Array::from(vec![10u32, 20]));
        assert_eq!(column_from_arrow(&u32).unwrap().as_i64().unwrap(), &[10, 20]);
        let u64: ArrayRef = Arc::new(UInt64Array::from(vec![7u64, 8]));
        assert_eq!(column_from_arrow(&u64).unwrap().as_i64().unwrap(), &[7, 8]);
    }

    #[test]
    fn uint64_overflow_is_an_error_not_a_silent_wrap() {
        use arrow_array::UInt64Array;
        // within i64 range → lossless import
        let ok: ArrayRef = Arc::new(UInt64Array::from(vec![0u64, i64::MAX as u64]));
        assert_eq!(column_from_arrow(&ok).unwrap().as_i64().unwrap(), &[0, i64::MAX]);
        // past i64::MAX volas has no unsigned-64 dtype → fail loud, never wrap to negative
        let over: ArrayRef = Arc::new(UInt64Array::from(vec![1u64, i64::MAX as u64 + 1]));
        assert!(column_from_arrow(&over).is_err());
        let umax: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX]));
        assert!(column_from_arrow(&umax).is_err());
        // a null slot carries no value, so its physical bits are never range-checked
        let with_null: ArrayRef = Arc::new(UInt64Array::from(vec![None, Some(9u64)]));
        assert_eq!(
            column_from_arrow(&with_null).unwrap(),
            Column::i64_with(vec![0, 9], na_at(2, &[0])).unwrap()
        );
    }

    #[test]
    fn imports_dictionary_and_decimal256() {
        use arrow_array::types::Int32Type;
        use arrow_array::{Decimal256Array, DictionaryArray};
        use arrow_buffer::i256;
        // dictionary<int32, utf8> (categorical) decodes to a dense Str column; a null key
        // stays NA. This is how parquet / pandas `category` string columns arrive.
        let dict: DictionaryArray<Int32Type> =
            vec![Some("AAPL"), Some("MSFT"), Some("AAPL"), None].into_iter().collect();
        assert_eq!(
            column_from_arrow(&(Arc::new(dict) as ArrayRef)).unwrap(),
            Column::str_with(
                vec!["AAPL".into(), "MSFT".into(), "AAPL".into(), "".into()],
                na_at(4, &[3]),
            ).unwrap()
        );
        // decimal256(_, 2): 150 -> 1.50 (lossy → f64), NA stays NaN
        let dec = Decimal256Array::from(vec![Some(i256::from_i128(150)), None])
            .with_precision_and_scale(20, 2)
            .unwrap();
        let v = column_from_arrow(&(Arc::new(dec) as ArrayRef)).unwrap();
        let f = v.as_f64().unwrap();
        assert_eq!(f[0], 1.5);
        assert!(f[1].is_nan());
    }

    #[test]
    fn imports_utf8view_and_null() {
        use arrow_array::{NullArray, StringViewArray};
        // Arrow `string_view` (Utf8View) materialises to a dense Str column; a null stays NA.
        let sv: ArrayRef = Arc::new(StringViewArray::from(vec![Some("AAPL"), None, Some("MSFT")]));
        assert_eq!(
            column_from_arrow(&sv).unwrap(),
            Column::str_with(vec!["AAPL".into(), "".into(), "MSFT".into()], na_at(3, &[1])).unwrap(),
        );
        // a typeless `Null` column carries no dtype → an all-NA f64 column.
        let nulls: ArrayRef = Arc::new(NullArray::new(3));
        match column_from_arrow(&nulls).unwrap().view() {
            ColumnView::F64(b) => assert!(b.len() == 3 && b.iter().all(|x| x.is_nan())),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn imports_date32_and_date64_as_ns_datetime() {
        use arrow_array::{Date32Array, Date64Array};
        // Date32 = days since epoch: day 1 -> 86_400 s -> 86_400_000_000_000 ns
        let d32: ArrayRef = Arc::new(Date32Array::from(vec![Some(1), None]));
        assert_eq!(
            column_from_arrow(&d32).unwrap(),
            Column::datetime(vec![86_400_000_000_000, i64::MIN])
        );
        // Date64 = ms since epoch: 2 ms -> 2_000_000 ns
        let d64: ArrayRef = Arc::new(Date64Array::from(vec![Some(2), None]));
        assert_eq!(
            column_from_arrow(&d64).unwrap(),
            Column::datetime(vec![2_000_000, i64::MIN])
        );
    }

    #[test]
    fn sliced_utf8_import_rebases_offsets_and_appends_cleanly() {
        use volas_core::StrBuffer;
        // A sliced Arrow string array's offsets no longer start at 0 (and its data buffer
        // keeps the sliced-away cells' bytes) — importing it verbatim breaks StrBuffer's
        // `offsets[0]==0 && offsets[len]==data.len()` invariant, corrupting a later append.
        let full = LargeStringArray::from(vec!["aa", "bbb", "cccc", "d"]);
        let sliced: ArrayRef = Arc::new(full.slice(1, 2)); // ["bbb", "cccc"] — a middle slice
        let col = column_from_arrow(&sliced).unwrap();
        let mut sb: StrBuffer = match col.view() {
            ColumnView::Str(sb, _) => sb.clone(),
            _ => unreachable!(), // LCOV_EXCL_LINE
        };
        assert_eq!(sb.iter().collect::<Vec<_>>(), ["bbb", "cccc"]); // reads correct
        let (offs, data) = sb.buffers();
        assert_eq!(offs.as_slice()[0], 0); // invariant: offsets re-based to 0
        assert_eq!(*offs.as_slice().last().unwrap() as usize, data.len()); // last == data.len()
        sb.extend(["X"]); // an append must not absorb the sliced-away cells' dead bytes
        assert_eq!(sb.iter().collect::<Vec<_>>(), ["bbb", "cccc", "X"]);
    }

    #[test]
    fn c_data_ffi_roundtrips_every_dtype() {
        let cols = [
            Column::f64(vec![1.0, f64::NAN, 3.0]),
            Column::i64_with(vec![10, 0, 30], na_at(3, &[1])).unwrap(),
            Column::bool_with(vec![true, false, true], na_at(3, &[2])).unwrap(),
            Column::str_with(vec!["a".into(), "".into(), "cd".into()], na_at(3, &[1])).unwrap(),
            Column::datetime(vec![1_000, i64::MIN, 3_000]),
        ];
        for col in &cols {
            let (array, schema) = column_to_c_data(col).unwrap();
            // SAFETY: `array`/`schema` are a fresh, valid C-Data pair from `to_ffi`.
            let back = unsafe { column_from_c_data(array, &schema).unwrap() };
            // floats carry NaN, which `!=` itself — compare via dtype + null structure there
            if matches!(col.view(), ColumnView::F64(_)) {
                assert!(back.as_f64().unwrap()[1].is_nan());
                assert_eq!(back.as_f64().unwrap()[0], 1.0);
            } else {
                assert_eq!(&back, col);
            }
        }
    }

    #[test]
    fn c_stream_roundtrips_a_named_frame() {
        let names = vec!["a".to_string(), "s".to_string()];
        let cols = vec![
            Column::i64_with(vec![1, 2, 3], na_at(3, &[1])).unwrap(),
            Column::str_with(vec!["x".into(), "y".into(), "z".into()], Validity::dense()).unwrap(),
        ];
        let mut stream = columns_to_c_stream(&names, &cols).unwrap();
        // SAFETY: `stream` is a fresh, valid C-Stream; the import moves it out.
        let (got_names, got_cols) = unsafe {
            columns_from_c_stream(&mut stream as *mut _ as *mut std::ffi::c_void)
        }
        .unwrap();
        assert_eq!(got_names, names);
        assert_eq!(got_cols, cols);
    }

    #[test]
    fn c_stream_export_rejects_names_it_cannot_carry() {
        let cols = vec![Column::f64(vec![1.0])];
        let err = columns_to_c_stream(&["a\0b".to_string()], &cols).err().unwrap();
        assert!(err.to_string().contains("NUL"), "{err}");
        let err = columns_to_c_stream(&[], &cols).err().unwrap();
        assert!(err.to_string().contains("0 names for 1 columns"), "{err}");
    }

    #[test]
    fn c_data_schema_matches_the_column_dtype() {
        let col = Column::i64_with(vec![1, 2], Validity::dense()).unwrap();
        let schema = column_to_c_schema(&col).unwrap();
        assert_eq!(schema.format(), "l"); // Arrow C-Data format string for int64
    }

    #[test]
    fn c_data_capsule_pointers_roundtrip() {
        // The raw `*mut c_void` capsule-import path the Python `__arrow_c_array__` glue
        // uses — exercised here so coverage does not hinge on a pyarrow-backed pytest run.
        let col = Column::i64_with(vec![10, 0, 30], na_at(3, &[1])).unwrap();
        let (mut array, schema) = column_to_c_data(&col).unwrap();
        // SAFETY: `array`/`schema` are a fresh, valid C-Data pair; the import moves `array`
        // out of its slot (leaving it empty/released) and borrows `schema`.
        let back = unsafe {
            column_from_c_capsules(
                &mut array as *mut FFI_ArrowArray as *mut std::ffi::c_void,
                &schema as *const FFI_ArrowSchema as *const std::ffi::c_void,
            )
            .unwrap()
        };
        assert_eq!(back, col);
    }

    #[test]
    fn reexporting_an_imported_column_borrows_through() {
        // import → a `Buffer::Borrowed` column → export again: the second export reads
        // the buffer's pointer + keep-alive through the *borrowed* arms, and the bytes
        // are still shared end to end (no copy was inserted anywhere).
        let src: ArrayRef = Arc::new(Int64Array::from(vec![11, 22, 33]));
        let src_ptr = src.as_primitive::<Int64Type>().values().as_ptr();
        let borrowed = column_from_arrow_with_copy(&src, false).unwrap();
        let reexported = column_to_arrow(&borrowed);
        assert_eq!(reexported.as_primitive::<Int64Type>().values().as_ptr(), src_ptr);
    }

    #[test]
    fn f32_roundtrip_is_zero_copy() {
        let col = Column::f32(vec![1.5, f32::NAN, 3.5]);
        let src = match col.view() {
            ColumnView::F32(b) => b.as_ptr(),
            _ => unreachable!(), // LCOV_EXCL_LINE
        };
        let arr = column_to_arrow(&col);
        assert_eq!(arr.as_primitive::<arrow_array::types::Float32Type>().values().as_ptr(), src);
        let back = column_from_arrow(&arr).unwrap();
        match back.view() {
            ColumnView::F32(b) => {
                assert_eq!(b[0], 1.5);
                assert!(b[1].is_nan());
            }
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
        // an external f32 null bitmap collapses to NaN
        let ext: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![Some(2.0), None]));
        match column_from_arrow(&ext).unwrap().view() {
            ColumnView::F32(b) => assert!(b[1].is_nan()),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn i32_with_nulls_roundtrips() {
        let col = Column::i32_with(vec![1, 0, 3], na_at(3, &[1])).unwrap();
        assert_eq!(column_from_arrow(&column_to_arrow(&col)).unwrap(), col);
    }

    #[test]
    fn utf8_with_nulls_imports() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("cd")]));
        assert_eq!(
            column_from_arrow(&arr).unwrap(),
            Column::str_with(vec!["a".into(), "".into(), "cd".into()], na_at(3, &[1])).unwrap(),
        );
    }

    #[test]
    fn coarser_timestamp_units_rescale_to_ns() {
        use arrow_array::{
            TimestampMicrosecondArray, TimestampMillisecondArray, TimestampSecondArray,
        };
        let us: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(2), None]));
        assert_eq!(column_from_arrow(&us).unwrap(), Column::datetime(vec![2_000, i64::MIN]));
        let ms: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![Some(3), None]));
        assert_eq!(column_from_arrow(&ms).unwrap(), Column::datetime(vec![3_000_000, i64::MIN]));
        let s: ArrayRef = Arc::new(TimestampSecondArray::from(vec![Some(4), None]));
        assert_eq!(column_from_arrow(&s).unwrap(), Column::datetime(vec![4_000_000_000, i64::MIN]));
    }

    #[test]
    fn out_of_range_dates_and_timestamps_raise_never_nat() {
        use arrow_array::{Date32Array, Date64Array, TimestampNanosecondArray, TimestampSecondArray};
        // 10^10 s · 10^9 (s→ns) overflows i64: an import error, never a silent NaT
        // (a missing cell the source never had) nor a wrapped instant.
        let ok: ArrayRef = Arc::new(TimestampSecondArray::from(vec![1_000_i64]));
        assert_eq!(column_from_arrow(&ok).unwrap(), Column::datetime(vec![1_000_000_000_000]));
        let out: Vec<ArrayRef> = vec![
            Arc::new(TimestampSecondArray::from(vec![1_000_i64, 10_000_000_000_i64])),
            Arc::new(Date32Array::from(vec![i32::MAX])),
            Arc::new(Date64Array::from(vec![i64::MAX])),
            // a present ns value on the NaT sentinel, dense (borrow path) or nullable
            Arc::new(TimestampNanosecondArray::from(vec![i64::MIN])),
            Arc::new(TimestampNanosecondArray::from(vec![None, Some(i64::MIN)])),
        ];
        for arr in out {
            let err = column_from_arrow(&arr).unwrap_err().to_string();
            assert!(err.contains("outside the datetime64[ns] range"), "{err}");
        }
    }

    #[test]
    fn empty_columns_roundtrip_both_ways() {
        for col in [
            Column::f64(vec![]),
            Column::i64_with(vec![], Validity::dense()).unwrap(),
            Column::str_with(vec![], Validity::dense()).unwrap(),
            Column::datetime(vec![]),
        ] {
            assert_eq!(column_from_arrow(&column_to_arrow(&col)).unwrap(), col);
        }
        // an externally-built empty array (whose data pointer may be null) imports cleanly
        let empty: ArrayRef = Arc::new(Float64Array::from(Vec::<f64>::new()));
        assert!(column_from_arrow(&empty).unwrap().as_f64().unwrap().is_empty());
    }

    #[test]
    fn null_slot_with_non_utf8_payload_imports_as_an_empty_span() {
        // Arrow leaves a null slot's bytes unspecified, so this is a VALID array whose
        // null slot holds non-UTF-8 bytes. Every span of the imported string buffer must
        // still be valid UTF-8 (its accessors decode spans unchecked).
        use arrow_buffer::{Buffer as AB, OffsetBuffer};
        // SAFETY: offsets are monotonic and in range; the only non-UTF-8 bytes sit in
        // the null slot, which the Arrow format leaves unspecified.
        let arr = unsafe {
            LargeStringArray::new_unchecked(
                OffsetBuffer::new(vec![0i64, 2, 3].into()),
                AB::from(b"\xff\xfea".to_vec()),
                Some(NullBuffer::from(vec![false, true])),
            )
        };
        let col = column_from_arrow(&(Arc::new(arr) as ArrayRef)).unwrap();
        let ColumnView::Str(sb, validity) = col.view() else { unreachable!() }; // LCOV_EXCL_LINE
        let (offsets, data) = sb.buffers();
        for w in offsets.as_slice().windows(2) {
            let span = &data.as_slice()[w[0] as usize..w[1] as usize];
            assert!(std::str::from_utf8(span).is_ok(), "span {w:?} is not UTF-8");
        }
        assert!(!validity.is_valid(0) && validity.is_valid(1));
        assert_eq!(sb.get(1), "a");
    }

    #[test]
    fn malformed_string_offsets_are_an_import_error() {
        use arrow_buffer::{Buffer as AB, OffsetBuffer, ScalarBuffer};
        // SAFETY (of the construction only): the deliberately non-monotonic offsets are
        // the malformed input under test; the import must reject them, never read them.
        let reversed = unsafe {
            LargeStringArray::new_unchecked(
                OffsetBuffer::new_unchecked(ScalarBuffer::from(vec![0i64, 3, 1, 3])),
                AB::from(b"abc".to_vec()),
                None,
            )
        };
        let err = column_from_arrow(&(Arc::new(reversed) as ArrayRef)).unwrap_err();
        assert!(err.to_string().contains("offsets"), "{err}");
        // a valid slot holding invalid UTF-8
        let bad = unsafe {
            LargeStringArray::new_unchecked(
                OffsetBuffer::new(vec![0i64, 2].into()),
                AB::from(b"\xff\xfe".to_vec()),
                None,
            )
        };
        let err = column_from_arrow(&(Arc::new(bad) as ArrayRef)).unwrap_err();
        assert!(err.to_string().contains("UTF-8"), "{err}");
        // end offsets naming a span outside the data buffer (checked before slicing)
        for offsets in [vec![0i32, 9], vec![2, 1]] {
            let outside = unsafe {
                StringArray::new_unchecked(
                    OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets)),
                    AB::from(b"abc".to_vec()),
                    None,
                )
            };
            let err = column_from_arrow(&(Arc::new(outside) as ArrayRef)).unwrap_err();
            assert!(err.to_string().contains("outside"), "{err}");
        }
    }

    #[test]
    fn c_stream_whose_schema_call_fails_is_an_error() {
        use arrow_array::ffi::FFI_ArrowSchema;
        use std::ffi::c_int;
        unsafe extern "C" fn fail(_: *mut FFI_ArrowArrayStream, _: *mut FFI_ArrowSchema) -> c_int {
            22
        }
        unsafe extern "C" fn release(s: *mut FFI_ArrowArrayStream) {
            (*s).release = None;
        }
        let mut stream = FFI_ArrowArrayStream::empty();
        stream.get_schema = Some(fail);
        stream.get_next = Some(no_next);
        stream.release = Some(release);
        unsafe extern "C" fn no_next(_: *mut FFI_ArrowArrayStream, _: *mut FFI_ArrowArray) -> c_int {
            23 // never called by the import: the schema call fails first
        }
        // SAFETY: `no_next` ignores its arguments.
        assert_eq!(unsafe { no_next(std::ptr::null_mut(), std::ptr::null_mut()) }, 23);
        // SAFETY: a live C-Stream struct whose callbacks are defined above.
        let err = unsafe { columns_from_c_stream(&mut stream as *mut _ as *mut std::ffi::c_void) };
        assert!(err.unwrap_err().to_string().contains("stream error code 22"));
    }
}
