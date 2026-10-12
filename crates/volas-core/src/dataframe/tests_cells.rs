//! Cell writes land in place, all or nothing, and the forming row a batch leaves is
//! read without touching the rest of the frame.

use super::*;
use crate::{CellWrite, Label, Validity};

fn w(col: usize, rows: &[usize], values: Column) -> CellWrite {
    CellWrite { col, rows: rows.to_vec(), values }
}

/// `x` (float), `n` (int with a hole at row 0), `c` (a cached directive), 4 rows.
fn frame() -> DataFrame {
    let mut df = DataFrame::new(
        vec!["x".into(), "n".into(), "c".into()],
        vec![
            Column::f64(vec![1.0, 2.0, 3.0, 4.0]),
            Column::i64_masked(vec![0, 20, 30, 40], Validity::from_valid_iter(4, [false, true, true, true])),
            Column::f64(vec![5.0, 6.0, 7.0, 8.0]),
        ],
        None,
    )
    .unwrap();
    df.set_computed("c", Reach { lookback: 0, resume: 0, continuation: 0, recursive: false });
    df
}

fn f64s(df: &DataFrame, name: &str) -> Vec<f64> {
    df.column(name).unwrap().as_f64().unwrap().to_vec()
}

#[test]
fn a_uniquely_owned_column_is_written_where_it_lies() {
    let mut df = frame();
    let before = df.column("x").unwrap().as_f64().unwrap().as_ptr();
    let writes = df.check_writes(vec![w(0, &[2], Column::f64(vec![9.0]))]).unwrap();
    df.write_cells(writes);
    assert_eq!(df.column("x").unwrap().as_f64().unwrap().as_ptr(), before, "no copy");
    assert_eq!(f64s(&df, "x"), [1.0, 2.0, 9.0, 4.0]);
}

#[test]
fn a_shared_column_is_copied_once_and_the_holder_keeps_its_values() {
    let mut df = frame();
    let held = df.column("x").unwrap().clone();
    df.write_one(0, &[0], &Column::f64(vec![-1.0])).unwrap();
    assert_eq!(held.as_f64().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(f64s(&df, "x"), [-1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_refused_batch_leaves_the_frame_unchanged() {
    let df = frame();
    let snapshot = df.clone();
    // the first write is fine; the second is lossy: nothing lands
    let r = df.check_writes(vec![w(0, &[1], Column::f64(vec![9.0])), w(1, &[1], Column::f64(vec![2.5]))]);
    assert!(matches!(r, Err(VolasError::DType(_))));
    assert!(df.equals(&snapshot) && df.is_computed("c"));
    // each check, in order: the column, the value count, the rows
    let shape = |writes| match df.check_writes(writes) {
        Err(VolasError::Shape(m)) => m,
        other => panic!("{other:?}"),
    };
    assert!(shape(vec![w(7, &[0], Column::f64(vec![1.0]))]).contains("column position 7"));
    assert!(shape(vec![w(0, &[0, 1], Column::f64(vec![1.0, 2.0, 3.0]))]).contains("3 values to 2"));
    assert!(shape(vec![w(0, &[4], Column::f64(vec![1.0]))]).contains("row position 4"));
}

#[test]
fn a_write_selecting_no_row_is_dropped_unchecked() {
    let mut df = frame();
    // an out-of-range column and a lossy value: no row selected, nothing checked
    let writes = df
        .check_writes(vec![w(9, &[], Column::f64(vec![1.0])), w(1, &[], Column::f64(vec![2.5]))])
        .unwrap();
    df.write_cells(writes);
    assert!(df.is_computed("c"), "an empty batch stales nothing");
}

#[test]
fn landing_writes_drops_the_written_cached_column_and_stales_the_rest() {
    let mut df = frame();
    let writes = df
        .check_writes(vec![w(1, &[0, 3], Column::f64(vec![1.0, f64::NAN])), w(2, &[1, 1], Column::i64(vec![60, 61]))])
        .unwrap();
    df.write_cells(writes);
    let n = df.column("n").unwrap();
    assert!(n.is_valid(0) && !n.is_valid(3), "a filled hole and a new one");
    assert_eq!(n.get_i64(0), 1);
    assert_eq!(f64s(&df, "c"), [5.0, 61.0, 7.0, 8.0], "a later repeat wins");
    assert!(!df.is_computed("c"), "a written cached column is data");
}

#[test]
fn the_row_a_batch_leaves_is_read_without_landing_it() {
    let df = frame();
    // x: written at the row (the later repeat wins); n: broadcast; c: cached, written
    // elsewhere — data after the batch, its forming cell as it was
    let writes = df
        .check_writes(vec![
            w(0, &[3, 0, 3], Column::f64(vec![10.0, 11.0, 12.0])),
            w(1, &[3], Column::f64(vec![f64::NAN])),
            w(2, &[0], Column::f64(vec![99.0])),
        ])
        .unwrap();
    let row = writes.data_row_after(&df, 3);
    assert_eq!(row.names(), ["x", "n", "c"]);
    assert_eq!(row.height(), 1);
    assert_eq!(f64s(&row, "x"), [12.0]);
    assert!(!row.column("n").unwrap().is_valid(0));
    assert_eq!(f64s(&row, "c"), [8.0]);
    assert_eq!(f64s(&df, "x"), [1.0, 2.0, 3.0, 4.0], "the frame is untouched");
    // a cached column the batch does not write stays out of the row
    let only_x = df.check_writes(vec![w(0, &[1], Column::f64(vec![0.5]))]).unwrap();
    assert_eq!(only_x.data_row_after(&df, 3).names(), ["x", "n"]);
}

#[test]
fn a_data_row_holds_the_data_columns_at_the_row_label() {
    let df = frame();
    let row = df.data_row(2);
    assert_eq!(row.names(), ["x", "n"]);
    assert_eq!(row.index().label_at(0), Label::I64(2));
    assert_eq!(f64s(&row, "x"), [3.0]);
}
