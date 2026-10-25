//! Every Rust foreign import entry point borrows by default and offers explicit owned snapshots.
use volas_core::ColumnView;
use std::ffi::c_void;
use std::sync::Arc;
use arrow_array::{ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, TimestampNanosecondArray};
use volas_arrow::*;
use volas_core::Column;

fn borrowed(c: &Column) -> bool {
    match c.view() {
        ColumnView::F64(b) => b.is_borrowed(),
        ColumnView::F32(b) => b.is_borrowed(),
        ColumnView::I64(b, _) | ColumnView::Datetime(b) => b.is_borrowed(),
        ColumnView::I32(b, _) => b.is_borrowed(),
        _ => panic!("native primitive expected"),
    }
}

#[test]
fn all_import_entry_points_default_to_borrowed() {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![1., 2.])),
        Arc::new(Float32Array::from(vec![1., 2.])),
        Arc::new(Int64Array::from(vec![1, 2])),
        Arc::new(Int32Array::from(vec![1, 2])),
        Arc::new(TimestampNanosecondArray::from(vec![1, 2])),
    ];
    for a in arrays {
        let owned = column_from_arrow_with_copy(&a, true).unwrap();
        assert!(!borrowed(&owned));
        assert!(borrowed(&column_from_arrow(&a).unwrap()));
        let (array, schema) = column_to_c_data(&owned).unwrap();
        // SAFETY: fresh exported structs, kept alive for this call.
        assert!(borrowed(&unsafe { column_from_c_data(array, &schema) }.unwrap()));
        let (array, schema) = column_to_c_data(&owned).unwrap();
        // SAFETY: fresh exported structs; the owned source is immutable while shared.
        assert!(borrowed(&unsafe { column_from_c_data_with_copy(array, &schema, false) }.unwrap()));
        let (mut array, schema) = column_to_c_data(&owned).unwrap();
        // SAFETY: these fresh C-Data payloads live through the call; source is immutable.
        let capsule_column = unsafe {
            column_from_c_capsules(
                &mut array as *mut _ as *mut c_void,
                &schema as *const _ as *const c_void,
            )
        }.unwrap();
        assert!(borrowed(&capsule_column));
        let names = vec!["close".into()];
        let cols = vec![owned];
        let mut stream = columns_to_c_stream(&names, &cols).unwrap();
        // SAFETY: fresh stream; immutable columns stay alive through the import.
        let (_, got) = unsafe { columns_from_c_stream(&mut stream as *mut _ as *mut c_void) }.unwrap();
        assert!(borrowed(&got[0]));
        for copy in [true, false] {
            let mut stream = columns_to_c_stream(&names, &cols).unwrap();
            // SAFETY: fresh stream with immutable source columns.
            let (_, got) = unsafe {
                columns_from_c_stream_with_copy(&mut stream as *mut _ as *mut c_void, copy)
            }.unwrap();
            assert_eq!(borrowed(&got[0]), !copy);
        }
    }
}

#[test]
fn negative_foreign_string_offsets_are_rejected_before_any_read() {
    use arrow_array::LargeStringArray;
    use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
    // Deliberately malformed foreign metadata: the import gate must reject it
    // before constructing a string slice, just like a C-Data producer's input.
    let a = unsafe {
        LargeStringArray::new_unchecked(
            OffsetBuffer::new_unchecked(ScalarBuffer::from(vec![-1i64, 1])),
            Buffer::from(b"a".to_vec()), None,
        )
    };
    let error = column_from_arrow(&(Arc::new(a) as ArrayRef)).unwrap_err();
    assert!(error.to_string().contains("outside"));
}
