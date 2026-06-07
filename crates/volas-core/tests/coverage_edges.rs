use volas_core::{Column, DataFrame};

fn frame_with_computed_f64() -> DataFrame {
    let mut df = DataFrame::new(
        vec!["value".into()],
        vec![Column::f64(vec![1.0, 2.0, 3.0])],
        None,
    )
    .unwrap();
    df.set_computed("value", "ma:2".into(), 1);
    df
}

#[test]
fn computed_scalar_update_refreshes_valid_rows_and_reports_errors() {
    let mut df = frame_with_computed_f64();
    df.update_computed_f64_value("value", 1, 8.0).unwrap();
    assert_eq!(
        df.column("value").unwrap().as_f64().unwrap(),
        &[1.0, 8.0, 3.0]
    );
    assert_eq!(df.computed_columns()[0].1.valid_rows, 3);

    df.update_computed_f64_value("value", 99, 9.0).unwrap();
    assert_eq!(
        df.column("value").unwrap().as_f64().unwrap(),
        &[1.0, 8.0, 3.0]
    );

    assert!(df.update_computed_f64_value("missing", 0, 1.0).is_err());

    let mut bool_df = DataFrame::new(
        vec!["flag".into()],
        vec![Column::bool(vec![true, false])],
        None,
    )
    .unwrap();
    bool_df.set_computed("flag", "value > 0".into(), 0);
    assert!(bool_df.update_computed_f64_value("flag", 0, 1.0).is_err());
}

#[test]
fn append_missing_rejects_columns_without_placeholder() {
    let mut int_col = Column::i64(vec![1, 2]);
    assert!(int_col.append_missing(1).is_err());

    let mut str_col = Column::str(vec!["a".into()]);
    assert!(str_col.append_missing(1).is_err());
}
