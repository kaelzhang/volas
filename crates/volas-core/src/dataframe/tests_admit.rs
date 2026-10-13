//! Rows entering a frame (`admit` / `append` / `replace_last_row` / the forming-row
//! folds): a row appended under the last row's label replaces it, cell by cell, for
//! every dtype; every rejected row set leaves the frame exactly as it was.

use super::*;
use crate::tz::Tz;
use crate::{Column, DType, History, Index, Reach};

fn every_dtype(n: usize, at: i64) -> DataFrame {
    let f = |k: usize| (k as i64 + at) as f64;
    DataFrame::new(
        ["f64", "f32", "i64", "i32", "bool", "str", "dt"]
            .map(String::from)
            .to_vec(),
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

/// Everything a rejected mutation must leave as it was.
fn snapshot(df: &DataFrame) -> (Vec<String>, String, Index, usize, Vec<(String, usize)>) {
    let mut cursors: Vec<(String, usize)> = df
        .computed_columns()
        .into_iter()
        .map(|(n, m)| (n, m.valid_rows))
        .collect();
    cursors.sort();
    (
        df.names().to_vec(),
        format!("{:?}", df.columns()),
        (**df.index()).clone(),
        df.height(),
        cursors,
    )
}

fn frame(names: &[&str], cols: Vec<Column>, index: Index) -> DataFrame {
    DataFrame::new(
        names.iter().map(|s| s.to_string()).collect(),
        cols,
        Some(index),
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
    assert!(df
        .column("str")
        .unwrap()
        .slice(0, 2)
        .equals(&Column::str(vec!["s0".into(), "s1".into()])));
}

#[test]
fn a_missing_column_becomes_missing_and_a_mistyped_one_is_rejected_whole() {
    let mut df = every_dtype(2, 0);
    let src = DataFrame::new(vec!["f64".into()], vec![Column::f64(vec![7.0])], None).unwrap();
    df.replace_last_row(&src, 0).unwrap();
    assert_eq!(df.column("f64").unwrap().get_f64(1), 7.0);
    for name in ["f32", "i64", "i32", "bool", "str", "dt"] {
        assert!(!df.column(name).unwrap().is_valid(1), "{name}");
        assert!(df.column(name).unwrap().is_valid(0), "{name}");
    }
    // a value of another dtype converts losslessly (as an append does); a lossy one
    // raises before the first write, even when a column before it fits
    let before = snapshot(&df);
    let bad = DataFrame::new(
        vec!["f64".into(), "i64".into()],
        vec![Column::f64(vec![9.0]), Column::f64(vec![2.5])],
        None,
    )
    .unwrap();
    let err = df.replace_last_row(&bad, 0).unwrap_err();
    assert!(matches!(err, VolasError::DType(_)), "{err}");
    assert_eq!(snapshot(&df), before);
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
    // another dtype converts losslessly (as an append does); a lossy value raises
    let mut f = Column::f64(vec![1.0]);
    f.set_last_from(&Column::i64(vec![7]), 0).unwrap();
    assert_eq!(f, Column::f64(vec![7.0]));
    let mut i = Column::i64(vec![1]);
    let err = i.set_last_from(&Column::f64(vec![1.5]), 0).unwrap_err();
    assert!(matches!(err, VolasError::DType(_)), "{err}");
    assert_eq!(i, Column::i64(vec![1]), "unchanged");
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
    DataFrame::new(
        vec!["x".into()],
        vec![Column::f64(vec![v; n])],
        Some(labels),
    )
    .unwrap()
}

#[test]
fn string_labels_follow_the_same_rule() {
    let mut df = labelled(Index::str(vec!["b".into(), "d".into()]), 1.0);
    df.append(&labelled(Index::str(vec!["d".into()]), 2.0))
        .unwrap();
    assert_eq!((df.height(), df.column("x").unwrap().get_f64(1)), (2, 2.0));
    // a label before the last that no row has is appended; one an earlier row has raises
    df.append(&labelled(Index::str(vec!["c".into()]), 3.0))
        .unwrap();
    assert_eq!(df.height(), 3);
    let err = df
        .append(&labelled(Index::str(vec!["b".into()]), 4.0))
        .unwrap_err();
    assert!(err.to_string().contains("\"b\""), "{err}");
    // a batch whose row precedes the one before it without repeating a label is appended
    df.append(&labelled(Index::str(vec!["z".into(), "y".into()]), 5.0))
        .unwrap();
    assert_eq!(df.height(), 5);
}

#[test]
fn a_datetime_index_only_increases() {
    let mut df = labelled(Index::datetime(vec![10, 20], Tz::Naive), 1.0);
    let batch = labelled(Index::datetime(vec![20, 30, 30, 40], Tz::Naive), 2.0);
    df.append(&batch).unwrap();
    assert_eq!(df.height(), 4); // 20 replaced, 30 appended then replaced, 40 appended
                                // earlier than the last: an error, a new label or not, single or in a batch
    for labels in [vec![15], vec![20], vec![50, 45]] {
        let err = df
            .append(&labelled(Index::datetime(labels, Tz::Naive), 3.0))
            .unwrap_err();
        assert!(err.to_string().contains("only increases"), "{err}");
        assert_eq!(df.height(), 4);
    }
}

#[test]
fn integer_batches_replace_and_raise_atomically() {
    let mut df = labelled(Index::int64(vec![10, 20]), 1.0);
    let batch = labelled(Index::int64(vec![20, 30, 30, 25]), 2.0);
    df.append(&batch).unwrap();
    assert_eq!(df.height(), 4); // 20 replaced, 30 appended then replaced, 25 appended
    let before = df.height();
    let bad = labelled(Index::int64(vec![40, 10]), 3.0);
    let err = df.append(&bad).unwrap_err();
    assert!(err.to_string().contains("10"), "{err}");
    assert_eq!(df.height(), before);
    // after a smaller label, a label above the last can still be an earlier row's:
    // the index stays unique, singly and in a batch
    df.append(&labelled(Index::int64(vec![5]), 4.0)).unwrap(); // [10, 20, 30, 25, 5]
    for labels in [vec![20], vec![7, 30]] {
        let err = df.append(&labelled(Index::int64(labels), 5.0)).unwrap_err();
        assert!(err.to_string().contains("earlier row"), "{err}");
        assert_eq!(df.height(), before + 1);
    }
    // an empty frame takes the rows as they come; a mismatched label kind never compares
    let mut empty = labelled(Index::int64(vec![]), 0.0);
    empty.append(&labelled(Index::int64(vec![5]), 1.0)).unwrap();
    assert_eq!(empty.height(), 1);
    assert!(!label_precedes(&Label::I64(1), &Label::Str("a".into())));
}

#[test]
fn a_rejected_column_leaves_every_column_and_the_index_as_they_were() {
    // the last column cannot take the value: nothing before it may grow (M1)
    let mut df = every_dtype(2, 0);
    let before = snapshot(&df);
    let mut bad = every_dtype(1, 5);
    bad.set_column("dt", Column::i64(vec![1])).unwrap();
    let bad = DataFrame::new(
        bad.names().to_vec(),
        bad.columns().to_vec(),
        Some(Index::int64(vec![99])),
    )
    .unwrap();
    let err = df.append(&bad).unwrap_err();
    assert!(err.to_string().contains("column \"dt\""), "{err}");
    assert_eq!(snapshot(&df), before);
    // a lossy value raises, a lossless one of another dtype is converted
    let lossy = frame(
        &["i64"],
        vec![Column::f64(vec![2.5])],
        Index::int64(vec![99]),
    );
    assert!(matches!(df.append(&lossy), Err(VolasError::DType(_))));
    let big = frame(
        &["i32"],
        vec![Column::i64(vec![1 << 40])],
        Index::int64(vec![99]),
    );
    assert!(matches!(df.append(&big), Err(VolasError::Overflow(_))));
    assert_eq!(snapshot(&df), before);
    let exact = frame(
        &["i64", "f64"],
        vec![Column::f64(vec![3.0]), Column::i64(vec![4])],
        Index::int64(vec![99]),
    );
    df.append(&exact).unwrap();
    assert_eq!(
        (
            df.column("i64").unwrap().get_i64(2),
            df.column("f64").unwrap().get_f64(2)
        ),
        (3, 4.0)
    );
}

#[test]
fn a_rejected_index_leaves_the_columns_unchanged() {
    // the index check runs before any column grows (A-3)
    let mut df = frame(
        &["x"],
        vec![Column::f64(vec![1.0, 2.0])],
        Index::datetime(vec![1, 2], Tz::Utc),
    );
    let before = snapshot(&df);
    for bad in [
        labelled(Index::str(vec!["x".into()]), 3.0),
        labelled(Index::range(1), 3.0),
        labelled(Index::int64(vec![3]), 3.0),
        labelled(Index::datetime(vec![3], Tz::Naive), 3.0),
        labelled(Index::datetime(vec![i64::MIN], Tz::Utc), 3.0),
        labelled(Index::range(0), 3.0), // zero rows, but still the wrong kind (A-8)
    ] {
        assert!(df.append(&bad).is_err());
        assert_eq!(snapshot(&df), before);
    }
    // the same-label replacement checks the index kind too (A-6)
    let mut ints = labelled(Index::int64(vec![1, 2, 3]), 1.0);
    let before = snapshot(&ints);
    assert!(ints
        .append(&labelled(Index::datetime(vec![3], Tz::Naive), 9.0))
        .is_err());
    assert_eq!(snapshot(&ints), before);
    let mut naive = labelled(Index::datetime(vec![1, 2], Tz::Naive), 1.0);
    assert!(naive
        .append(&labelled(Index::datetime(vec![2], Tz::Utc), 9.0))
        .is_err());
    assert_eq!(naive.column("x").unwrap().get_f64(1), 1.0);
    // unlabelled rows continue the labels, never onto an existing one
    let mut unsorted = labelled(Index::int64(vec![1, 0]), 1.0);
    let err = unsorted
        .append(&labelled(Index::range(1), 2.0))
        .unwrap_err();
    assert!(err.to_string().contains("already in the index"), "{err}");
    assert_eq!(unsorted.height(), 2);
    // an empty positional frame is still positional: timestamps are another index kind
    let mut empty = labelled(Index::range(0), 0.0);
    assert!(empty
        .append(&labelled(Index::datetime(vec![5], Tz::Utc), 1.0))
        .is_err());
    assert_eq!(empty.height(), 0);
}

#[test]
fn a_rejected_batch_writes_no_row() {
    // [last label, new label] where the second row's value is lossy: the replacement of
    // the first row must not have happened either (A-2)
    let mut df = frame(
        &["s", "x"],
        vec![
            Column::str(vec!["a".into(), "b".into()]),
            Column::f64(vec![1.0, 2.0]),
        ],
        Index::int64(vec![1, 2]),
    );
    df.set_computed("x", Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    let before = snapshot(&df);
    let batch = frame(
        &["s", "x"],
        vec![Column::i64(vec![7, 8]), Column::f64(vec![9.0, 10.0])],
        Index::int64(vec![2, 3]),
    );
    assert!(df.append(&batch).is_err());
    assert_eq!(snapshot(&df), before);
}

#[test]
fn cached_directive_columns_are_never_appended() {
    let mut df = frame(
        &["open", "close", "ema:12"],
        vec![Column::f64(vec![1.0]); 3],
        Index::range(1),
    );
    df.set_computed("ema:12", Reach { lookback: 12, resume: 0, continuation: 12, history: History::Window });
    let before = snapshot(&df);
    let rows = frame(
        &["close", "ema:12"],
        vec![Column::f64(vec![2.0]); 2],
        Index::range(1),
    );
    let err = df.append(&rows).unwrap_err();
    assert_eq!(
        err.to_string(),
        "value error: cannot append columns ['ema:12']: they are cached directives that this \
         DataFrame computes itself. Select the data columns first, e.g. \
         df.append(other[['open', 'close']])."
    );
    assert_eq!(snapshot(&df), before);
    // the data columns alone are appended, the cached one goes stale
    df.append(&frame(
        &["close"],
        vec![Column::f64(vec![2.0])],
        Index::range(1),
    ))
    .unwrap();
    assert!(df.has_stale_computed());
    // looked up from either side: more cached columns than the rows carry, out of order
    let mut wide = frame(
        &["a", "b", "c1", "c2"],
        vec![Column::f64(vec![1.0]); 4],
        Index::range(1),
    );
    wide.set_computed("c1", Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    wide.set_computed("c2", Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    wide.append(&frame(
        &["b"],
        vec![Column::f64(vec![2.0])],
        Index::range(1),
    ))
    .unwrap();
    assert!(wide
        .append(&frame(
            &["c2"],
            vec![Column::f64(vec![2.0])],
            Index::range(1)
        ))
        .is_err());
    assert_eq!(wide.height(), 2);
    // names are written as Python writes them
    assert_eq!(
        py_list(&["a'b", "c\"d'", "e\\\n\u{1}"]),
        r#"["a'b", 'c"d\'', 'e\\\n\x01']"#
    );
    assert_eq!(py_list(&["\r\t"]), r#"['\r\t']"#);
}

#[test]
fn an_extra_column_is_rejected() {
    let mut df = labelled(Index::range(1), 1.0);
    let err = df
        .append(&frame(
            &["y"],
            vec![Column::f64(vec![2.0])],
            Index::range(1),
        ))
        .unwrap_err();
    assert!(
        err.to_string().contains("\"y\" is not in the target frame"),
        "{err}"
    );
}

fn ints(names: &[&str], rows: &[Vec<i64>]) -> DataFrame {
    let cols = (0..names.len())
        .map(|j| Column::i64(rows.iter().map(|r| r[j]).collect()))
        .collect();
    frame(names, cols, Index::range(rows.len()))
}

#[test]
fn a_fold_that_overflows_changes_nothing() {
    // `high` (max) folds fine, `volume` (sum) overflows: neither is written (A-4)
    let ops = [(0, 0, CombineOp::Max), (1, 1, CombineOp::Sum)];
    let mut df = ints(&["high", "volume"], &[vec![1, 1], vec![10, i64::MAX - 5]]);
    df.set_computed_states("x", None, None); // untracked: a no-op
    let before = snapshot(&df);
    let bar = ints(&["high", "volume"], &[vec![50, 6]]);
    assert!(matches!(
        df.fold_forming_row(1, &bar, 0, &ops),
        Err(VolasError::Overflow(_))
    ));
    assert_eq!(snapshot(&df), before);
    let fits = ints(&["high", "volume"], &[vec![50, 5]]);
    df.fold_forming_row(1, &fits, 0, &ops).unwrap();
    assert_eq!(
        (df.columns()[0].get_i64(1), df.columns()[1].get_i64(1)),
        (50, i64::MAX)
    );
    // a re-cumulation that overflows changes nothing either
    let before = snapshot(&df);
    let period = ints(&["high", "volume"], &[vec![3, i64::MAX], vec![4, 1]]);
    assert!(df.refold_forming_row(1, &period, &ops).is_err());
    assert_eq!(snapshot(&df), before);
    // checked with the re-sent bar standing in for the period's last bar
    let resent = ints(&["high", "volume"], &[vec![4, 0]]);
    df.check_refold(1, &period, &resent, &ops).unwrap();
    let missing = ints(&["high"], &[vec![4]]); // the period keeps its own volume
    assert!(df.check_refold(1, &period, &missing, &ops).is_err());
    assert!(df
        .check_refold(1, &ints(&["high", "volume"], &[]), &resent, &ops)
        .is_ok());
}

#[test]
fn refolding_the_forming_row_is_the_period_reduce() {
    let ops = [
        (0, 0, CombineOp::Keep),
        (1, 1, CombineOp::Max),
        (2, 2, CombineOp::Sum),
    ];
    let mut df = DataFrame::new(
        vec!["first".into(), "max".into(), "sum".into()],
        vec![
            Column::f64(vec![0.0, 0.0]),
            Column::f64(vec![0.0, 9.0]),
            Column::f64(vec![0.0, 9.0]),
        ],
        None,
    )
    .unwrap();
    df.set_computed("sum", Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    df.set_computed_states("sum", Some(vec![1.0]), Some(vec![0.5]));
    let period = DataFrame::new(
        vec!["first".into(), "max".into(), "sum".into()],
        vec![
            Column::f64(vec![4.0, 2.0, 3.0]),
            Column::f64(vec![f64::NAN, 2.0, 1.0]),
            Column::f64(vec![0.5, 0.25, 0.125]),
        ],
        None,
    )
    .unwrap();
    let before = snapshot(&df);
    df.check_refold(1, &period, &period.slice_data(2, 3), &ops).unwrap();
    assert_eq!(snapshot(&df), before); // float / NaN validation leaves cells and cache states alone
    df.refold_forming_row(1, &period, &ops).unwrap();
    assert_eq!(df.column("first").unwrap().get_f64(1), 4.0);
    // a missing input makes the period's max missing (no skipped value)
    assert!(df.column("max").unwrap().get_f64(1).is_nan());
    assert_eq!(df.column("sum").unwrap().get_f64(1), 0.875);
    // only the forming row went stale; the state before it is the column's state again
    assert_eq!(df.computed_resume_state("sum"), Some((&[0.5][..], 0)));
    // a str column folds its first / last in place at the forming (last) row
    let bars = DataFrame::new(vec!["s".into()], vec![Column::str(vec!["x".into(), "y".into()])], None).unwrap();
    let mut s = DataFrame::new(vec!["s".into()], vec![Column::str(vec!["a".into()])], None).unwrap();
    s.refold_forming_row(0, &bars, &[(0, 0, CombineOp::Replace)]).unwrap();
    assert_eq!(s.column("s").unwrap().str_at(0), Some("y"));
    assert!(s.refold_forming_row(0, &bars, &[(0, 0, CombineOp::Max)]).is_err()); // unsupported
}

#[test]
fn a_new_row_never_replaces_or_precedes_the_last() {
    let mut df = labelled(Index::datetime(vec![10, 20], Tz::Utc), 1.0);
    let before = snapshot(&df);
    for label in [20, 15] {
        let err = df
            .append_new(&labelled(Index::datetime(vec![label], Tz::Utc), 9.0))
            .unwrap_err();
        assert!(
            err.to_string().contains("does not follow the last row"),
            "{err}"
        );
        assert_eq!(snapshot(&df), before);
    }
    assert!(df
        .append_new(&labelled(Index::datetime(vec![30, 30], Tz::Utc), 9.0))
        .is_err());
    df.append_new(&labelled(Index::datetime(vec![30, 40], Tz::Utc), 9.0))
        .unwrap();
    assert_eq!(df.height(), 4);
    let mut empty = labelled(Index::datetime(vec![], Tz::Utc), 0.0);
    empty
        .append_new(&labelled(Index::datetime(vec![5], Tz::Utc), 1.0))
        .unwrap();
    assert_eq!(empty.height(), 1);
}
