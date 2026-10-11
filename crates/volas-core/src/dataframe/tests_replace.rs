//! A row appended under the last row's label replaces it (`append_labelled` /
//! `replace_last_row`), cell by cell, for every dtype.

use super::*;
use crate::tz::Tz;
use crate::DType;

fn every_dtype(n: usize, at: i64) -> DataFrame {
    let f = |k: usize| (k as i64 + at) as f64;
    DataFrame::new(
        ["f64", "f32", "i64", "i32", "bool", "str", "dt"].map(String::from).to_vec(),
        vec![
            Column::f64((0..n).map(f).collect()),
            Column::f32((0..n).map(|k| f(k) as f32).collect()),
            Column::i64((0..n).map(|k| k as i64 + at).collect()),
            Column::i32((0..n).map(|k| (k as i64 + at) as i32).collect()),
            Column::bool((0..n).map(|k| k % 2 == 0).collect()),
            Column::str((0..n).map(|k| format!("s{}", k as i64 + at)).collect()),
            Column::datetime((0..n).map(|k| k as i64 + at).collect()),
        ],
        Some(Index::int64((0..n as i64).map(|k| 10 * k).collect())),
    )
    .unwrap()
}

#[test]
fn every_dtype_replaces_its_last_cell() {
    let mut df = every_dtype(3, 0);
    let src = every_dtype(1, 100);
    df.replace_last_row(&src, 0).unwrap();
    assert_eq!(df.height(), 3);
    for (name, col) in df.names().iter().zip(df.columns()) {
        let want = &src.columns()[src.column_pos(name).unwrap()];
        assert!(col.slice(2, 3).equals(want), "{name}");
    }
    // the rows before are untouched
    assert!(df.column("str").unwrap().slice(0, 2).equals(&Column::str(vec!["s0".into(), "s1".into()])));
}

#[test]
fn a_missing_column_becomes_missing_and_a_mistyped_one_is_cast() {
    let mut df = every_dtype(2, 0);
    let src = DataFrame::new(vec!["f64".into()], vec![Column::i64(vec![7])], None).unwrap();
    df.replace_last_row(&src, 0).unwrap();
    assert_eq!(df.column("f64").unwrap().get_f64(1), 7.0);
    for name in ["f32", "i64", "i32", "bool", "str", "dt"] {
        assert!(!df.column(name).unwrap().is_valid(1), "{name}");
        assert!(df.column(name).unwrap().is_valid(0), "{name}");
    }
    // an uncastable value is an error
    let bad = DataFrame::new(vec!["f64".into()], vec![Column::str(vec!["x".into()])], None).unwrap();
    assert!(df.replace_last_row(&bad, 0).is_err());
}

#[test]
fn replacing_needs_a_last_row_and_a_source_row() {
    let mut empty = every_dtype(0, 0);
    let src = every_dtype(1, 0);
    assert!(empty.replace_last_row(&src, 0).is_err());
    let mut df = every_dtype(2, 0);
    assert!(df.replace_last_row(&src, 1).is_err());
    // the column-level primitives: an empty column has no last cell
    let mut col = Column::f64(vec![]);
    assert!(col.set_last_from(&Column::f64(vec![1.0]), 0).is_err());
    col.set_last_na(); // no-op
    assert!(Column::f64(vec![1.0]).set_last_from(&Column::i64(vec![1]), 0).is_err());
    let mut s = crate::strbuf::StrBuffer::from_vec(vec![]);
    s.set_last("x"); // no-op
    assert_eq!(s.len(), 0);
}

#[test]
fn set_last_na_marks_each_dtype_missing() {
    let mut df = every_dtype(2, 0);
    for col in [0, 1, 6] {
        let mut c = df.columns()[col].clone();
        c.set_last_na();
        assert!(!c.is_valid(1) || c.get_f64(1).is_nan(), "{col}");
    }
    let src = DataFrame::new(vec!["other".into()], vec![Column::f64(vec![1.0])], None).unwrap();
    df.replace_last_row(&src, 0).unwrap();
    assert!(df.column("f64").unwrap().get_f64(1).is_nan());
    assert!(df.column("f32").unwrap().get_f64(1).is_nan());
    assert_eq!(df.column("dt").unwrap().get_i64(1), i64::MIN);
    assert_eq!(df.column("dt").unwrap().dtype(), DType::Datetime);
}

fn labelled(labels: Index, v: f64) -> DataFrame {
    let n = labels.len();
    DataFrame::new(vec!["x".into()], vec![Column::f64(vec![v; n])], Some(labels)).unwrap()
}

#[test]
fn string_labels_follow_the_same_rule() {
    let mut df = labelled(Index::str(vec!["b".into(), "d".into()]), 1.0);
    df.append_labelled(&labelled(Index::str(vec!["d".into()]), 2.0)).unwrap();
    assert_eq!((df.height(), df.column("x").unwrap().get_f64(1)), (2, 2.0));
    // a label before the last that no row has is appended; one an earlier row has raises
    df.append_labelled(&labelled(Index::str(vec!["c".into()]), 3.0)).unwrap();
    assert_eq!(df.height(), 3);
    let err = df.append_labelled(&labelled(Index::str(vec!["b".into()]), 4.0)).unwrap_err();
    assert!(err.to_string().contains("\"b\""), "{err}");
    // a batch whose row precedes the one before it without repeating a label is appended
    df.append_labelled(&labelled(Index::str(vec!["z".into(), "y".into()]), 5.0)).unwrap();
    assert_eq!(df.height(), 5);
}

#[test]
fn a_datetime_index_only_increases() {
    let mut df = labelled(Index::datetime(vec![10, 20], Tz::Naive), 1.0);
    let batch = labelled(Index::datetime(vec![20, 30, 30, 40], Tz::Naive), 2.0);
    df.append_labelled(&batch).unwrap();
    assert_eq!(df.height(), 4); // 20 replaced, 30 appended then replaced, 40 appended
    // earlier than the last: an error, a new label or not, single or in a batch
    for labels in [vec![15], vec![20], vec![50, 45]] {
        let err = df.append_labelled(&labelled(Index::datetime(labels, Tz::Naive), 3.0)).unwrap_err();
        assert!(err.to_string().contains("only increases"), "{err}");
        assert_eq!(df.height(), 4);
    }
}

#[test]
fn integer_batches_replace_and_raise_atomically() {
    let mut df = labelled(Index::int64(vec![10, 20]), 1.0);
    let batch = labelled(Index::int64(vec![20, 30, 30, 25]), 2.0);
    df.append_labelled(&batch).unwrap();
    assert_eq!(df.height(), 4); // 20 replaced, 30 appended then replaced, 25 appended
    let before = df.height();
    let bad = labelled(Index::int64(vec![40, 10]), 3.0);
    let err = df.append_labelled(&bad).unwrap_err();
    assert!(err.to_string().contains("10"), "{err}");
    assert_eq!(df.height(), before);
    // an empty frame takes the rows as they come; a mismatched label kind never compares
    let mut empty = labelled(Index::int64(vec![]), 0.0);
    empty.append_labelled(&labelled(Index::int64(vec![5]), 1.0)).unwrap();
    assert_eq!(empty.height(), 1);
    assert!(!label_precedes(&Label::I64(1), &Label::Str("a".into())));
}
