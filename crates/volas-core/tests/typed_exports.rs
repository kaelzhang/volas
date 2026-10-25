//! Typed dispatch preserves every storage pairing and allocation.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{Buffer, Column, ColumnExport, ColumnBorrowExport, DataFrame, Index, Series, StrBuffer, Validity};

struct Adopt;
impl ColumnExport for Adopt {
    type Output = Column;
    fn f64(self, v: Buffer<f64>) -> Column { Column::from_f64_buffer(v) }
    fn f32(self, v: Buffer<f32>) -> Column { Column::from_f32_buffer(v) }
    fn i64(self, v: Buffer<i64>, valid: &Validity) -> Column { Column::from_i64_buffer(v, valid.clone()).unwrap() }
    fn i32(self, v: Buffer<i32>, valid: &Validity) -> Column { Column::from_i32_buffer(v, valid.clone()).unwrap() }
    fn bool(self, v: Buffer<bool>, valid: &Validity) -> Column { Column::from_bool_buffer(v, valid.clone()).unwrap() }
    fn str(self, v: &StrBuffer, valid: &Validity) -> Column { Column::from_str_buffer(v.clone(), valid.clone()).unwrap() }
    fn datetime(self, v: Buffer<i64>) -> Column { Column::from_datetime_buffer(v) }
}

impl ColumnBorrowExport for Adopt {
    type Output = Column;
    fn f64(self, v: &Buffer<f64>) -> Column { Column::from_f64_buffer(v.clone()) }
    fn f32(self, v: &Buffer<f32>) -> Column { Column::from_f32_buffer(v.clone()) }
    fn i64(self, v: &Buffer<i64>, valid: &Validity) -> Column { Column::from_i64_buffer(v.clone(), valid.clone()).unwrap() }
    fn i32(self, v: &Buffer<i32>, valid: &Validity) -> Column { Column::from_i32_buffer(v.clone(), valid.clone()).unwrap() }
    fn bool(self, v: &Buffer<bool>, valid: &Validity) -> Column { Column::from_bool_buffer(v.clone(), valid.clone()).unwrap() }
    fn str(self, v: &StrBuffer, valid: &Validity) -> Column { Column::from_str_buffer(v.clone(), valid.clone()).unwrap() }
    fn datetime(self, v: &Buffer<i64>) -> Column { Column::from_datetime_buffer(v.clone()) }
}

#[test]
fn typed_exports_dispatch_all_types_without_allocating_or_changing_validity() {
    for col in [Column::f64(vec![1., 2.]), Column::f32(vec![1., 2.]),
        Column::i64_from_options([Some(1), None]), Column::i32(vec![1, 2]),
        Column::bool_from_options([None, Some(true)]),
        Column::str_from_options([Some("x".into()), None]), Column::datetime(vec![1, i64::MIN])] {
        let (borrowed, calls, bytes) = measure(|| col.export(Adopt));
        assert_eq!((calls, bytes), (0, 0));
        assert_eq!(borrowed, col);
        let (moved, calls, bytes) = measure(|| col.into_export(Adopt));
        assert_eq!((calls, bytes), (0, 0));
        assert_eq!(moved, borrowed);
    }
}

#[test]
fn column_admission_keeps_values_and_builds_the_matching_positional_index() {
    let col = Column::f64(vec![1., 2.]);
    let pointer = col.as_f64().unwrap().as_ptr();
    let series = Series::from_column(Some("x".into()), col).unwrap();
    assert_eq!(series.name(), Some("x"));
    assert_eq!(series.data().as_f64().unwrap().as_ptr(), pointer);
    assert_eq!(series.index().to_i64_labels(), [0, 1]);
    assert!(Series::from_column(None, Column::i64(vec![])).unwrap().is_empty());
    assert!(Index::positional(usize::MAX).is_err());
    assert!(DataFrame::from_columns(vec!["x".into()], vec![Column::i64(vec![1])], 2).is_err());
    let empty = DataFrame::from_columns(vec![], vec![], 3).unwrap();
    assert_eq!((empty.height(), empty.index().len()), (3, 3));
}
