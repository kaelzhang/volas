use volas_core::{Column, DataFrame, History, Reach};

fn frame_with_computed_f64() -> DataFrame {
    let mut df = DataFrame::new(
        vec!["value".into()],
        vec![Column::f64(vec![1.0, 2.0, 3.0])],
        None,
    )
    .unwrap();
    df.set_computed("value", Reach { lookback: 1, resume: 0, continuation: 1, history: History::Window });
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

    // a row past the frame is an error, never a silently dropped write
    assert!(df.update_computed_f64_value("value", 99, 9.0).is_err());
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
    bool_df.set_computed("flag", Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    assert!(bool_df.update_computed_f64_value("flag", 0, 1.0).is_err());
}

#[test]
fn every_dtype_takes_missing_placeholders_and_a_tail_write() {
    let present = |c: &Column| (0..c.len()).map(|i| c.is_valid(i)).collect::<Vec<_>>();
    let cols = [
        (Column::f64(vec![1.0, 2.0]), Column::f64(vec![7.0, f64::NAN])),
        (Column::f32(vec![1.0, 2.0]), Column::f32(vec![7.0, f32::NAN])),
        (Column::i64(vec![1, 2]), Column::i64_from_options([Some(7), None])),
        (Column::i32(vec![1, 2]), Column::i32(vec![7, 8])),
        (Column::bool(vec![true, false]), Column::bool_from_options([Some(true), None])),
        (Column::str(vec!["a".into(), "b".into()]), Column::str_from_options([Some("z".into()), None])),
        (Column::datetime(vec![1, 2]), Column::datetime(vec![7, i64::MIN])),
    ];
    for (mut col, tail) in cols {
        col.append_na(2);
        assert_eq!(present(&col), [true, true, false, false], "{}", col.dtype());
        col.write_rows(2, &tail).unwrap();
        let want_last = tail.is_valid(1);
        assert_eq!(present(&col), [true, true, true, want_last], "{}", col.dtype());
        col.write_rows(3, &tail).unwrap(); // a tail running past the end writes what fits
        assert_eq!(col.len(), 4);
        assert!(col.write_rows(0, &Column::f64(vec![0.0])).is_err() == (col.dtype() != volas_core::DType::F64));
    }
    let mut s = Column::str(vec!["a".into(), "b".into(), "c".into()]);
    s.write_rows(1, &Column::str(vec!["x".into()])).unwrap();
    assert_eq!((s.str_at(1), s.str_at(2)), (Some("x"), Some("c"))); // later rows kept
}

#[test]
fn a_cached_int_directive_appends_and_refreshes() {
    let mut df = DataFrame::new(
        vec!["close".into(), "5".into()],
        vec![Column::f64(vec![1.0, 2.0]), Column::i64(vec![5, 5])],
        None,
    )
    .unwrap();
    df.set_computed("5", Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    let bar = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![3.0])], None).unwrap();
    df.append(&bar).unwrap();
    assert!(!df.column("5").unwrap().is_valid(2), "a stale placeholder is missing");
    df.update_computed_tail("5", 2, &Column::i64(vec![5])).unwrap();
    assert_eq!(df.column("5").unwrap(), &Column::i64(vec![5, 5, 5]));
    assert!(df.update_computed_tail("5", 2, &Column::f64(vec![5.0])).is_err());
}
