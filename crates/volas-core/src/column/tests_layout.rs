//! The column's own invariants at its boundaries: a validity mask covers exactly
//! its values, two-column operations check their lengths, equality reads presence
//! (never a missing payload), and a write stores another dtype only losslessly.

use super::*;

fn na(len: usize, missing: &[usize]) -> Validity {
    Validity::from_valid_iter(len, (0..len).map(|i| !missing.contains(&i)))
}

#[test]
fn a_checked_constructor_rejects_a_mask_of_another_length() {
    let shape = |r: Result<Column>| matches!(r, Err(VolasError::Shape(_)));
    assert!(shape(Column::i64_with(vec![1, 2], na(3, &[1]))));
    assert!(shape(Column::i32_with(vec![1, 2, 3, 4], na(3, &[1]))));
    assert!(shape(Column::bool_with(vec![true], na(3, &[0]))));
    assert!(shape(Column::str_with(vec!["a".into()], na(2, &[1]))));
    // a dense mask fits any length; a covering bitmap is accepted
    assert_eq!(Column::i64_with(vec![1, 2], Validity::dense()).unwrap(), Column::i64(vec![1, 2]));
    let ok = Column::bool_with(vec![true, false], na(2, &[1])).unwrap();
    assert!(ok.is_valid(0) && !ok.is_valid(1));
    assert!(Validity::dense().covers(7) && na(3, &[0]).covers(3) && !na(3, &[0]).covers(4));
}

#[test]
fn a_frame_rejects_a_column_whose_mask_does_not_cover_it() {
    // Crate-local corruption is rejected defensively; external code cannot construct it.
    let forged = Column { data: ColumnData::I64(Buffer::from_vec(vec![1, 2, 3]), na(97, &[5])) };
    assert!(matches!(forged.check_layout(), Err(VolasError::Shape(_))));
    let err = crate::DataFrame::new(vec!["x".into()], vec![forged], None).unwrap_err();
    assert_eq!(
        err,
        VolasError::Shape("column \"x\": validity mask covers 97 values but the column has 3".into())
    );
    assert!(Column::f64(vec![1.0]).check_layout().is_ok());
    // nor through set_column, which leaves the frame unchanged
    let mut df = crate::DataFrame::new(vec!["a".into()], vec![Column::f64(vec![1.0, 2.0, 3.0])], None).unwrap();
    let forged = Column { data: ColumnData::I64(Buffer::from_vec(vec![1, 2, 3]), na(2, &[1])) };
    assert!(matches!(df.set_column("x", forged), Err(VolasError::Shape(_))));
    assert_eq!(df.width(), 1);
}

#[test]
fn optional_cells_build_a_column_whose_layout_holds() {
    let c = Column::i64_from_options([Some(4), None, Some(6)]);
    assert_eq!((c.len(), c.null_count(), c.get_i64(2)), (3, 1, 6));
    let b = Column::bool_from_options(vec![None, Some(true)]);
    assert_eq!((b.null_count(), b.truth_at(1)), (1, Some(true)));
    let s = Column::str_from_options([Some("a".to_string()), None]);
    assert_eq!((s.str_at(0), s.is_valid(1)), (Some("a"), false));
    for col in [c, b, s] {
        col.check_layout().unwrap();
    }
}

#[test]
fn a_missing_mask_is_checked_and_applied_in_each_dtype() {
    let mask = [false, true];
    let cols = [
        Column::f64(vec![1.0, 2.0]),
        Column::f32(vec![1.0, 2.0]),
        Column::i64(vec![1, 2]),
        Column::i32(vec![1, 2]),
        Column::bool(vec![true, true]),
        Column::str(vec!["a".into(), "b".into()]),
        Column::datetime(vec![1, 2]),
    ];
    for col in cols {
        let out = col.with_missing_mask(&mask).unwrap();
        assert_eq!((out.dtype(), out.is_valid(0), out.is_valid(1)), (col.dtype(), true, false));
        let err = col.with_missing_mask(&[false; 3]).unwrap_err();
        assert_eq!(err, VolasError::Shape("missing mask has 3 entries but the column has 2".into()));
    }
}

#[test]
fn select_checks_every_length_before_reading() {
    let a = Column::i64(vec![1, 2, 3]);
    let shape = |r: Result<Column>| matches!(r, Err(VolasError::Shape(_)));
    assert!(shape(a.select(&[true, false, true], &Column::i64(vec![9]), DType::I64)));
    assert!(shape(a.select(&[true], &Column::i64(vec![7, 8, 9]), DType::I64)));
    assert!(shape(a.select(&[true; 5], &Column::f64(vec![0.0; 3]), DType::F64)));
    assert_eq!(
        a.select(&[true, false, true], &Column::i64(vec![7, 8, 9]), DType::I64).unwrap(),
        Column::i64(vec![1, 8, 3])
    );
}

#[test]
fn equality_compares_presence_never_a_missing_payload() {
    let a = Column::i64_masked(vec![0, 5], na(2, &[0]));
    let b = Column::i64_masked(vec![7, 5], na(2, &[0]));
    assert!(a.equals(&b));
    assert!(!a.equals(&Column::i64(vec![0, 5]))); // presence differs
    assert!(!a.equals(&Column::i64(vec![0]))); // length differs
    assert!(!a.equals(&Column::f64(vec![f64::NAN, 5.0]))); // dtype differs
    let pairs = [
        (Column::i32_masked(vec![1, 2], na(2, &[1])), Column::i32_masked(vec![1, 9], na(2, &[1]))),
        (Column::bool_masked(vec![true, true], na(2, &[1])), Column::bool_masked(vec![true, false], na(2, &[1]))),
        (Column::str_masked(vec!["a".into(), "x".into()], na(2, &[1])), Column::str_masked(vec!["a".into(), "y".into()], na(2, &[1]))),
        (Column::f64(vec![1.0, f64::NAN]), Column::f64(vec![1.0, f64::NAN])),
        (Column::f32(vec![1.0, f32::NAN]), Column::f32(vec![1.0, f32::NAN])),
        (Column::datetime(vec![1, i64::MIN]), Column::datetime(vec![1, i64::MIN])),
    ];
    for (x, y) in pairs {
        assert!(x.equals(&y), "{}", x.dtype());
    }
    assert!(!Column::datetime(vec![1]).equals(&Column::datetime(vec![2])));
}

#[test]
fn scatter_stores_another_dtype_only_losslessly() {
    let big = (1i64 << 53) + 1;
    let f = Column::f64(vec![0.0, 0.0]);
    assert!(matches!(f.scattered(&[0], &Column::i64(vec![big])), Err(VolasError::DType(_))));
    assert_eq!(f.scattered(&[1], &Column::i64(vec![1 << 53])).unwrap(), Column::f64(vec![0.0, 9007199254740992.0]));
    let g = Column::f32(vec![0.0]);
    assert!(matches!(g.scattered(&[0], &Column::f64(vec![1e300])), Err(VolasError::Overflow(_))));
    assert!(matches!(g.scattered(&[0], &Column::i64(vec![(1 << 24) + 1])), Err(VolasError::DType(_))));
    let i = Column::i32(vec![0, 0]);
    assert!(matches!(i.scattered(&[0], &Column::i64(vec![1 << 40])), Err(VolasError::Overflow(_))));
    let out = i.scattered(&[0, 1], &Column::f64(vec![3.0, f64::NAN])).unwrap();
    assert_eq!((out.get_i64(0), out.is_valid(1)), (3, false));
}

#[test]
fn row_order_is_exact_and_missing_aware() {
    use std::cmp::Ordering::*;
    let big = (1i64 << 53) + 1;
    let a = Column::i64_masked(vec![big, 1, 5], na(3, &[1]));
    let b = Column::f64(vec![9007199254740992.0, 1.0, f64::NAN]);
    assert_eq!(a.order_rows(&b).unwrap(), vec![Some(Greater), None, None]);
    assert_eq!(b.order_rows(&a).unwrap(), vec![Some(Less), None, None]);
    let s = Column::str(vec!["b".into()]);
    assert_eq!(s.order_rows(&Column::str(vec!["a".into()])).unwrap(), vec![Some(Greater)]);
    assert!(matches!(s.order_rows(&a), Err(VolasError::Shape(_))));
    assert!(matches!(s.order_rows(&Column::f64(vec![1.0])), Err(VolasError::DType(_))));
}

#[test]
fn reduction_scalars_keep_bool_and_missing() {
    let bools = Column::from_scalars(&[(DType::Bool, Scalar::Bool(true)), (DType::Bool, Scalar::F64(f64::NAN))]).unwrap();
    assert_eq!((bools.dtype(), bools.truth_at(0), bools.is_valid(1)), (DType::Bool, Some(true), false));
    let mixed = Column::from_scalars(&[(DType::Bool, Scalar::Bool(true)), (DType::I64, Scalar::I64(3))]).unwrap();
    assert_eq!(mixed, Column::i64(vec![1, 3]));
    let ints = Column::from_scalars(&[(DType::I64, Scalar::F64(f64::NAN))]).unwrap();
    assert!(!ints.is_valid(0));
}

#[test]
fn default_rank_keeps_missing_and_orders_large_integers_exactly() {
    let a = 1_i64 << 53;
    let col = Column::i64_from_options([Some(a + 1), None, Some(a), Some(a + 1)]);
    for (method, ascending, pct, expected) in [
        (stats::RankMethod::Average, true, false, [2.5_f64, f64::NAN, 1.0, 2.5]),
        (stats::RankMethod::Dense, false, true, [0.5, f64::NAN, 1.0, 0.5]),
    ] {
        for (actual, expected) in col.rank(method, ascending, pct).into_iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn adopted_buffers_validate_storage_pairing_and_keep_the_allocation() {
    let i64_values = Buffer::from_vec(vec![7_i64, 9]);
    let ptr = i64_values.as_ptr();
    let ints = Column::from_i64_buffer(i64_values, na(2, &[1])).unwrap();
    let ColumnView::I64(values, validity) = ints.view() else { panic!("int64 view") };
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!((values[0], validity.null_count()), (7, 1));
    let cases = [
        Column::from_i64_buffer(Buffer::from_vec(vec![1]), na(2, &[1])),
        Column::from_i32_buffer(Buffer::from_vec(vec![1]), na(2, &[1])),
        Column::from_bool_buffer(Buffer::from_vec(vec![true]), na(2, &[1])),
        Column::from_str_buffer(StrBuffer::from_vec(vec!["x".into()]), na(2, &[1])),
    ];
    for result in cases {
        assert_eq!(result.unwrap_err(), VolasError::Shape(
            "validity mask covers 2 values but the column has 1".into()
        ));
    }
    let views = [
        Column::from_f64_buffer(Buffer::from_vec(vec![1.0])),
        Column::from_f32_buffer(Buffer::from_vec(vec![1.0])),
        Column::from_i64_buffer(Buffer::from_vec(vec![1]), Validity::dense()).unwrap(),
        Column::from_i32_buffer(Buffer::from_vec(vec![1]), Validity::dense()).unwrap(),
        Column::from_bool_buffer(Buffer::from_vec(vec![true]), Validity::dense()).unwrap(),
        Column::from_str_buffer(StrBuffer::from_vec(vec!["x".into()]), Validity::dense()).unwrap(),
        Column::from_datetime_buffer(Buffer::from_vec(vec![1])),
    ];
    for column in views {
        match column.view() {
            ColumnView::F64(v) => assert_eq!(v[0], 1.0),
            ColumnView::F32(v) => assert_eq!(v[0], 1.0),
            ColumnView::I64(v, mask) => assert_eq!((v[0], mask.null_count()), (1, 0)),
            ColumnView::I32(v, mask) => assert_eq!((v[0], mask.null_count()), (1, 0)),
            ColumnView::Bool(v, mask) => assert_eq!((v[0], mask.null_count()), (true, 0)),
            ColumnView::Str(v, mask) => assert_eq!((v.get(0), mask.null_count()), ("x", 0)),
            ColumnView::Datetime(v) => assert_eq!(v[0], 1),
        }
    }
}

#[test]
fn consuming_typed_exports_move_unique_buffers_and_convert_other_dtypes() {
    let floats = Column::f64(vec![1.0, 2.0]);
    let ptr = floats.as_f64().unwrap().as_ptr();
    let values = floats.into_f64_vec();
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!(Column::i64(vec![1]).into_f64_vec(), vec![1.0]);
    let ints = Column::i64(vec![1, 2]);
    let ptr = ints.as_i64().unwrap().as_ptr();
    let values = ints.into_i64_vec().unwrap();
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!(Column::i32(vec![2]).into_i64_vec().unwrap(), vec![2]);
    let bools = Column::bool(vec![true, false]);
    let ptr = bools.as_bool().unwrap().as_ptr();
    let values = bools.into_bool_vec().unwrap();
    assert_eq!(values.as_ptr(), ptr);
    assert!(Column::f64(vec![1.0]).into_bool_vec().is_err());
}
