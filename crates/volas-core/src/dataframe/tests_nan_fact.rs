//! A column remembers whether it holds a NaN; every way a frame's values change must
//! keep that fact right. A stale "no NaN" would let a recursive indicator skip the
//! check for missing rows and read a NaN as data, so each path is tested from a
//! column already known to hold none.

use super::*;
use crate::CombineOp;

const NAN: f64 = f64::NAN;

/// A frame whose float columns `x` (plain) and `c` (a cached directive) are both known
/// to hold no NaN.
fn known_clean() -> DataFrame {
    let mut df = DataFrame::new(
        vec!["x".into(), "c".into()],
        vec![
            Column::f64(vec![1.0, 2.0, 3.0]),
            Column::f64(vec![4.0, 5.0, 6.0]),
        ],
        None,
    )
    .unwrap();
    df.set_computed("c", Reach { lookback: 0, resume: 0, continuation: 0 });
    assert!(!df.column("x").unwrap().has_nan());
    assert!(!df.column("c").unwrap().has_nan());
    assert_eq!(df.column("x").unwrap().known_has_nan(), Some(false));
    df
}

fn x(df: &DataFrame) -> &Column {
    df.column("x").unwrap()
}

fn row(xs: f64, cs: f64) -> DataFrame {
    DataFrame::new(
        vec!["x".into(), "c".into()],
        vec![Column::f64(vec![xs]), Column::f64(vec![cs])],
        None,
    )
    .unwrap()
}

/// A row of the data column `x` alone (a cached column is never appended).
fn data_row(xs: f64) -> DataFrame {
    DataFrame::new(vec!["x".into()], vec![Column::f64(vec![xs])], None).unwrap()
}

#[test]
fn append_of_a_nan_row() {
    let mut df = known_clean();
    df.append(&data_row(NAN)).unwrap();
    assert!(x(&df).has_nan());
    // a clean row keeps the fact without a rescan
    let mut df = known_clean();
    df.append(&data_row(4.0)).unwrap();
    assert_eq!(x(&df).known_has_nan(), Some(false));
}

#[test]
fn append_by_name_pads_a_missing_column_with_nan() {
    let mut df = known_clean();
    df.invalidate_computed_on_write("c"); // `c` becomes a data column
    let only_c = DataFrame::new(vec!["c".into()], vec![Column::f64(vec![7.0])], None).unwrap();
    df.append(&only_c).unwrap(); // `x` is a plain column: padded with NA
    assert!(x(&df).has_nan());
    let mut df = known_clean();
    let only_x = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![7.0])], None).unwrap();
    df.append(&only_x).unwrap(); // `c` is cached: a stale NaN placeholder
    assert!(df.column("c").unwrap().has_nan());
    assert_eq!(x(&df).known_has_nan(), Some(false));
}

#[test]
fn fold_forming_row_writing_a_nan() {
    let mut df = known_clean();
    df.fold_forming_row(2, &row(NAN, NAN), 0, &[(0, 0, CombineOp::Replace)])
        .unwrap();
    assert!(x(&df).has_nan());
    // a running max / sum folding in a NaN too
    let mut df = known_clean();
    df.fold_forming_row(2, &row(NAN, NAN), 0, &[(0, 0, CombineOp::Sum)])
        .unwrap();
    assert!(x(&df).has_nan());
}

#[test]
fn fold_forming_row_over_the_only_nan() {
    let mut df = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![1.0, NAN])], None).unwrap();
    assert!(x(&df).has_nan());
    let src = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![5.0])], None).unwrap();
    df.fold_forming_row(1, &src, 0, &[(0, 0, CombineOp::Replace)])
        .unwrap();
    assert!(!x(&df).has_nan());
}

#[test]
fn assign_positions_writing_a_nan_and_clearing_one() {
    let mut df = known_clean();
    df.assign_positions(0, &[1], &Column::f64(vec![NAN]))
        .unwrap();
    assert!(x(&df).has_nan());
    df.assign_positions(0, &[1], &Column::f64(vec![2.0]))
        .unwrap();
    assert!(!x(&df).has_nan());
}

#[test]
fn update_computed_tail_writing_a_nan() {
    let mut df = known_clean();
    df.update_computed_tail("c", 1, &Column::f64(vec![NAN, 1.0]))
        .unwrap();
    assert!(df.column("c").unwrap().has_nan());
}

#[test]
fn update_computed_f64_value_writing_and_clearing_a_nan() {
    let mut df = known_clean();
    df.update_computed_f64_value("c", 2, NAN).unwrap();
    assert!(df.column("c").unwrap().has_nan());
    df.update_computed_f64_value("c", 2, 1.0).unwrap();
    assert!(!df.column("c").unwrap().has_nan());
    df.update_computed_f64_value("c", 0, 2.0).unwrap();
    assert_eq!(df.column("c").unwrap().known_has_nan(), Some(false));
}

#[test]
fn set_column_replacing_with_a_nan_column() {
    let mut df = known_clean();
    df.set_column("x", Column::f64(vec![1.0, NAN, 3.0]))
        .unwrap();
    assert!(x(&df).has_nan());
}

#[test]
fn slicing_and_selecting_keep_the_fact_right() {
    let df = known_clean();
    assert_eq!(x(&df.slice_data(0, 2)).known_has_nan(), Some(false));
    assert_eq!(x(&df.slice(1, 3)).known_has_nan(), Some(false));
    assert_eq!(
        x(&df.select(&["x".into()]).unwrap()).known_has_nan(),
        Some(false)
    );
    let nan = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![NAN, 1.0])], None).unwrap();
    assert!(x(&nan).has_nan());
    assert!(x(&nan.slice_data(0, 1)).has_nan());
    assert!(!x(&nan.slice_data(1, 2)).has_nan());
    assert!(x(&nan.select(&["x".into()]).unwrap()).has_nan());
}

#[test]
fn a_write_to_a_shared_copy_leaves_the_other_untouched() {
    let a = known_clean();
    let mut b = a.clone();
    b.assign_positions(0, &[0], &Column::f64(vec![NAN]))
        .unwrap();
    assert!(x(&b).has_nan());
    assert!(!x(&a).has_nan());
    let mut c = a.clone();
    c.append(&data_row(NAN)).unwrap();
    assert!(x(&c).has_nan());
    assert_eq!(x(&a).known_has_nan(), Some(false));
}

#[test]
fn an_int_column_reads_its_validity() {
    let ints = Column::i64_with(
        vec![1, 0],
        crate::Validity::from_valid_iter(2, [true, false]),
    );
    assert!(ints.has_nan());
    assert_eq!(ints.known_has_nan(), Some(true));
    assert!(!Column::i64(vec![1, 2]).has_nan());
    let f = Column::f32(vec![1.0]);
    assert!(!f.has_nan());
    assert_eq!(f.known_has_nan(), Some(false));
    assert!(Column::datetime(vec![i64::MIN]).has_nan());
    assert_eq!(Column::datetime(vec![0]).known_has_nan(), None);
    assert!(Column::str(vec!["a".into()]).has_nan());
}
