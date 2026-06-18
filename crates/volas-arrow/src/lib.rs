//! volas-arrow: a zero-copy bridge between volas `Column`s and Arrow arrays.
//!
//! arrow-rs lives **only** in this crate — `volas-core` stays dependency-light and
//! the Python layer composes this bridge with the C-Data interface.
//!
//! ## Zero-copy contract
//! The **data** buffers move without a copy in both directions: a volas
//! [`Buffer<T>`](volas_core::Buffer) is published to Arrow as a borrowed allocation
//! (its `Arc` is the keep-alive), and an Arrow buffer is imported as a
//! [`Buffer::Borrowed`](volas_core::Buffer) whose guard is the source array. Two
//! representations are not bit-compatible and are repacked (a copy):
//! - **bool** — volas stores one byte per value, Arrow one *bit*;
//! - **validity** — the null bitmap (≤ `n/8` bytes, negligible beside the data).

use std::any::Any;
use std::panic::RefUnwindSafe;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_buffer::{Buffer as ArrowBuffer, NullBuffer};
use volas_core::{Buffer, Validity};

mod export;
mod ffi;
mod import;
mod stream;

pub use export::column_to_arrow;
pub use ffi::{
    column_from_c_capsules, column_from_c_data, column_to_c_data, column_to_c_schema,
};
pub use import::column_from_arrow;
pub use stream::{columns_from_c_stream, columns_to_c_stream};
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
fn arrow_buffer_of<T: Send + Sync + 'static>(buf: &Buffer<T>) -> ArrowBuffer {
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
fn borrow_primitive<T: Send + Sync + 'static>(values: &[T], src: &ArrayRef) -> Buffer<T> {
    let ptr = NonNull::new(values.as_ptr() as *mut T).unwrap_or(NonNull::dangling());
    // SAFETY: `values` points into `src`'s buffer; `Hold(src)` keeps it alive for as
    // long as the returned `Buffer` (and Arrow guarantees the bytes are immutable).
    unsafe { Buffer::from_foreign(ptr, values.len(), Arc::new(Hold(src.clone()))) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};
    use arrow_array::{
        Array, BooleanArray, Date32Array, Float64Array, Int64Array, LargeStringArray, StringArray,
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
        let back = column_from_arrow(&arr).unwrap();
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
        match column_from_arrow(&f32).unwrap() {
            Column::F32(b) => assert!(b[1].is_nan() && b[0] == 1.5),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
        // a dense (null-free) f32 imports via the borrow branch
        let dense_f32: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![1.0f32, 2.0]));
        match column_from_arrow(&dense_f32).unwrap() {
            Column::F32(b) => assert_eq!(&b[..], &[1.0, 2.0]),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn i64_with_nulls_roundtrips() {
        let col = Column::i64_with(vec![10, 0, 30], na_at(3, &[1]));
        let arr = column_to_arrow(&col);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(column_from_arrow(&arr).unwrap(), col);
        // a dense int column exports its buffer and re-imports it with no copy
        let dense = Column::i64_with(vec![7, 8], Validity::dense());
        let exported = column_to_arrow(&dense);
        let arrow_ptr = exported.as_primitive::<Int64Type>().values().as_ptr();
        assert_eq!(dense.as_i64().unwrap().as_ptr(), arrow_ptr);
        assert_eq!(column_from_arrow(&exported).unwrap().as_i64().unwrap().as_ptr(), arrow_ptr);
    }

    #[test]
    fn bool_roundtrips_through_bitpacking() {
        let col = Column::bool_with(vec![true, false, true], na_at(3, &[2]));
        assert_eq!(column_from_arrow(&column_to_arrow(&col)).unwrap(), col);
    }

    #[test]
    fn str_roundtrip_borrows_the_byte_buffer() {
        let col = Column::str_with(vec!["alpha".into(), "".into(), "ω".into()], na_at(3, &[1]));
        let data_ptr = match &col {
            Column::Str(sb, _) => sb.buffers().1.as_ptr(),
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
            Column::str_with(vec!["a".into(), "bc".into()], Validity::dense())
        );
        // large_string is the zero-copy path
        let large: ArrayRef = Arc::new(LargeStringArray::from(vec!["a", "bc"]));
        assert_eq!(column_from_arrow(&large).unwrap().dtype(), col_str_dtype());
    }

    fn col_str_dtype() -> volas_core::DType {
        Column::str_with(vec![], Validity::dense()).dtype()
    }

    #[test]
    fn boolean_and_timestamp_import_paths() {
        let b: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None]));
        assert_eq!(column_from_arrow(&b).unwrap(), Column::bool_with(vec![true, false], na_at(2, &[1])));
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
            Column::i64_with(vec![1, 0, -3], na_at(3, &[1]))
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
    fn c_data_ffi_roundtrips_every_dtype() {
        let cols = [
            Column::f64(vec![1.0, f64::NAN, 3.0]),
            Column::i64_with(vec![10, 0, 30], na_at(3, &[1])),
            Column::bool_with(vec![true, false, true], na_at(3, &[2])),
            Column::str_with(vec!["a".into(), "".into(), "cd".into()], na_at(3, &[1])),
            Column::datetime(vec![1_000, i64::MIN, 3_000]),
        ];
        for col in &cols {
            let (array, schema) = column_to_c_data(col).unwrap();
            // SAFETY: `array`/`schema` are a fresh, valid C-Data pair from `to_ffi`.
            let back = unsafe { column_from_c_data(array, &schema).unwrap() };
            // floats carry NaN, which `!=` itself — compare via dtype + null structure there
            if matches!(col, Column::F64(_)) {
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
            Column::i64_with(vec![1, 2, 3], na_at(3, &[1])),
            Column::str_with(vec!["x".into(), "y".into(), "z".into()], Validity::dense()),
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
    fn c_data_schema_matches_the_column_dtype() {
        let col = Column::i64_with(vec![1, 2], Validity::dense());
        let schema = column_to_c_schema(&col).unwrap();
        assert_eq!(schema.format(), "l"); // Arrow C-Data format string for int64
    }

    #[test]
    fn c_data_capsule_pointers_roundtrip() {
        // The raw `*mut c_void` capsule-import path the Python `__arrow_c_array__` glue
        // uses — exercised here so coverage does not hinge on a pyarrow-backed pytest run.
        let col = Column::i64_with(vec![10, 0, 30], na_at(3, &[1]));
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
        let borrowed = column_from_arrow(&src).unwrap();
        let reexported = column_to_arrow(&borrowed);
        assert_eq!(reexported.as_primitive::<Int64Type>().values().as_ptr(), src_ptr);
    }

    #[test]
    fn f32_roundtrip_is_zero_copy() {
        let col = Column::f32(vec![1.5, f32::NAN, 3.5]);
        let src = match &col {
            Column::F32(b) => b.as_ptr(),
            _ => unreachable!(), // LCOV_EXCL_LINE
        };
        let arr = column_to_arrow(&col);
        assert_eq!(arr.as_primitive::<arrow_array::types::Float32Type>().values().as_ptr(), src);
        let back = column_from_arrow(&arr).unwrap();
        match &back {
            Column::F32(b) => {
                assert_eq!(b[0], 1.5);
                assert!(b[1].is_nan());
            }
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
        // an external f32 null bitmap collapses to NaN
        let ext: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![Some(2.0), None]));
        match column_from_arrow(&ext).unwrap() {
            Column::F32(b) => assert!(b[1].is_nan()),
            _ => unreachable!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn i32_with_nulls_roundtrips() {
        let col = Column::i32_with(vec![1, 0, 3], na_at(3, &[1]));
        assert_eq!(column_from_arrow(&column_to_arrow(&col)).unwrap(), col);
    }

    #[test]
    fn utf8_with_nulls_imports() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("cd")]));
        assert_eq!(
            column_from_arrow(&arr).unwrap(),
            Column::str_with(vec!["a".into(), "".into(), "cd".into()], na_at(3, &[1])),
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
    fn coarse_timestamp_overflow_becomes_nat_not_panic() {
        use arrow_array::TimestampSecondArray;
        // 10^10 s · 10^9 (s→ns) overflows i64 — must land on the NaT sentinel, never
        // panic in the overflow-checked gate build nor wrap to a wrong instant in release.
        let arr: ArrayRef = Arc::new(TimestampSecondArray::from(vec![1_000_i64, 10_000_000_000_i64]));
        let v = column_from_arrow(&arr).unwrap();
        let got = v.as_datetime().unwrap();
        assert_eq!(got[0], 1_000_000_000_000); // 1000 s -> ns, exact
        assert_eq!(got[1], i64::MIN); // overflow -> NaT
    }

    #[test]
    fn empty_columns_roundtrip_both_ways() {
        for col in [
            Column::f64(vec![]),
            Column::i64_with(vec![], Validity::dense()),
            Column::str_with(vec![], Validity::dense()),
            Column::datetime(vec![]),
        ] {
            assert_eq!(column_from_arrow(&column_to_arrow(&col)).unwrap(), col);
        }
        // an externally-built empty array (whose data pointer may be null) imports cleanly
        let empty: ArrayRef = Arc::new(Float64Array::from(Vec::<f64>::new()));
        assert!(column_from_arrow(&empty).unwrap().as_f64().unwrap().is_empty());
    }
}
