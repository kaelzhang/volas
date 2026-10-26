//! The frame owns the shape proof; extraction shares it and row views inherit it.
use std::sync::Arc;
use volas_core::{Column, DataFrame, Index, IndexKind, Series, Tz, VolasError};

#[test]
fn frame_series_carries_shape_through_extraction_and_rows() {
    let frame = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![4, 5, 6])],
        Some(Index::new(Some("row".into()), IndexKind::Int64(vec![9, 2, 8])).unwrap())).unwrap();
    let series: Series = frame.series_at(0);
    assert!(Arc::ptr_eq(series.index(), frame.index()));
    assert_eq!(series.data().as_i64().unwrap().as_ptr(), frame.columns()[0].as_i64().unwrap().as_ptr());
    let name = String::from("x");
    let name_ptr = name.as_ptr();
    let sub = frame.series_rows(name, 1, 3).unwrap();
    assert_eq!(sub.name(), Some("x"));
    assert_eq!(sub.name().unwrap().as_ptr(), name_ptr, "retain the caller's owned name");
    assert_eq!(sub.data().as_i64(), Some(&[5, 6][..]));
    assert_eq!(sub.index().name(), Some("row"));
    assert_eq!(sub.index().label_at(0), frame.index().label_at(1));
    assert!(!Arc::ptr_eq(sub.index(), frame.index()));
    assert_eq!(frame.series_rows("x", 9, 20).unwrap().len(), 0);
    assert_eq!(frame.series_rows("x", 2, 1).unwrap().len(), 0);
    assert_eq!(frame.series_rows("x", 0, 99).unwrap().len(), 3);
    assert_eq!(frame.series(String::from("x")).unwrap().data(), series.data());
    assert!(matches!(frame.series("absent"), Err(VolasError::ColumnNotFound(_))));
    assert!(matches!(frame.series_rows("absent", 0, 1), Err(VolasError::ColumnNotFound(_))));
    assert!(matches!(Series::new(None, Column::i64(vec![1]), Arc::clone(frame.index())), Err(VolasError::Shape(_))));
}

#[test]
fn frame_series_rows_keep_missing_cells_and_datetime_labels() {
    let frame = DataFrame::new(vec!["x".into()],
        vec![Column::i64_from_options([Some(7), None, Some(9)])],
        Some(Index::new(Some("time".into()), IndexKind::Datetime(vec![1, 2, 3], Tz::Naive)).unwrap())).unwrap();
    let sub = frame.series_rows("x", 1, 3).unwrap();
    assert_eq!(sub.len(), 2);
    assert!(!sub.data().is_valid(0));
    assert!(sub.data().is_valid(1));
    assert_eq!(sub.data().as_i64().unwrap()[1], 9);
    assert_eq!(sub.index().label_at(0), frame.index().label_at(1));
    assert_eq!(sub.index().name(), Some("time"));
    assert_eq!(frame.height(), 3);
}
