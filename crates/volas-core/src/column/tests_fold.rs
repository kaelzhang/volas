//! The aggregation fold (`init_at` / `combine_at`): the support table, the
//! missing-value rule, checked integer sums and lossless sources.

use super::*;

#[test]
fn combine_at_numeric_ops_in_place() {
    // F64: Max / Min / Sum / Replace / Keep, against src = 5.0
    let src = Column::f64(vec![5.0]);
    for (op, want) in [
        (CombineOp::Max, 5.0),
        (CombineOp::Min, 3.0),
        (CombineOp::Sum, 8.0),
        (CombineOp::Replace, 5.0),
        (CombineOp::Keep, 3.0),
    ] {
        let mut c = Column::f64(vec![3.0]);
        c.combine_at(0, op, &src, 0).unwrap();
        assert_eq!(c.get_f64(0), want, "f64 {op:?}");
    }
    // a NaN on either side makes Max / Min missing (never a skipped value)
    for op in [CombineOp::Max, CombineOp::Min] {
        for (acc, x) in [(3.0, f64::NAN), (f64::NAN, 3.0)] {
            let mut c = Column::f64(vec![acc]);
            c.combine_at(0, op, &Column::f64(vec![x]), 0).unwrap();
            assert!(c.get_f64(0).is_nan(), "f64 {op:?}");
            let mut c = Column::f32(vec![acc as f32]);
            c.combine_at(0, op, &Column::f32(vec![x as f32]), 0).unwrap();
            assert!(c.get_f64(0).is_nan(), "f32 {op:?}");
        }
    }

    // F32: every op (covers the f32 kernel arms).
    for (op, want) in [
        (CombineOp::Max, 5.0),
        (CombineOp::Min, 3.0),
        (CombineOp::Sum, 8.0),
        (CombineOp::Replace, 5.0),
    ] {
        let mut c = Column::f32(vec![3.0]);
        c.combine_at(0, op, &Column::f32(vec![5.0]), 0).unwrap();
        assert_eq!(c.get_f64(0), want, "f32 {op:?}");
    }

    // Datetime + I32: every supported op.
    for (op, want) in [
        (CombineOp::Max, 5),
        (CombineOp::Min, 3),
        (CombineOp::Sum, 8),
        (CombineOp::Replace, 5),
    ] {
        if op != CombineOp::Sum {
            let mut c = Column::datetime(vec![3]);
            c.combine_at(0, op, &Column::datetime(vec![5]), 0).unwrap();
            assert_eq!(c.get_i64(0), want, "datetime {op:?}");
        }
        let mut c = Column::i32(vec![3]);
        c.combine_at(0, op, &Column::i32(vec![5]), 0).unwrap();
        assert_eq!(c.get_i64(0), want, "i32 {op:?}");
    }
    // a NaT on either side makes a datetime Max / Min NaT
    for (acc, x) in [(3, i64::MIN), (i64::MIN, 3)] {
        let mut c = Column::datetime(vec![acc]);
        c.combine_at(0, CombineOp::Max, &Column::datetime(vec![x]), 0).unwrap();
        assert_eq!(c.get_i64(0), i64::MIN);
    }
}

#[test]
fn the_fold_support_table_is_owned_by_combine_op() {
    use CombineOp::*;
    use DType::*;
    for dt in [F64, F32, I64, I32, Datetime, Bool, Utf8] {
        assert!(Keep.supports(dt) && Replace.supports(dt), "{dt}");
        assert_eq!(Max.supports(dt), dt.is_numeric() || dt == Datetime, "{dt}");
        assert_eq!(Min.supports(dt), Max.supports(dt), "{dt}");
        assert_eq!(Sum.supports(dt), dt.is_numeric(), "{dt}");
    }
    // a datetime sum / a bool max is rejected by both kernels, the cell unchanged
    let mut d = Column::datetime(vec![3]);
    let err = d.combine_at(0, Sum, &Column::datetime(vec![5]), 0).unwrap_err();
    assert_eq!(err, VolasError::DType("cannot sum a datetime64[ns] column".into()));
    assert!(d.init_at(0, Sum, &Column::datetime(vec![5]), 0).is_err());
    assert_eq!(d.get_i64(0), 3);
    let mut b = Column::bool(vec![true]);
    assert_eq!(
        b.combine_at(0, Max, &Column::bool(vec![false]), 0).unwrap_err(),
        VolasError::DType("cannot max a bool column".into())
    );
    assert_eq!(Min.unsupported(Utf8), VolasError::DType(format!("cannot min a {} column", Utf8)));
}

#[test]
fn an_int_fold_reads_presence_never_the_placeholder() {
    use CombineOp::*;
    let na = |x: i64| Column::i64_masked(vec![x], Validity::from_valid_iter(1, [false]));
    for op in [Max, Min, Sum] {
        // a missing source makes the aggregate missing, whatever its payload
        let mut c = Column::i64(vec![-5]);
        c.combine_at(0, op, &na(999), 0).unwrap();
        assert!(!c.is_valid(0), "{op:?}");
        // a missing aggregate stays missing through later present values
        c.combine_at(0, op, &Column::i64(vec![7]), 0).unwrap();
        assert!(!c.is_valid(0), "{op:?} sticky");
        let mut c = Column::i32(vec![-5]);
        c.combine_at(0, op, &na(0).convert_lossless(DType::I32).unwrap(), 0).unwrap();
        assert!(!c.is_valid(0), "i32 {op:?}");
    }
    // a source of another dtype converts losslessly: 2.0 folds as 2, 2.5 raises
    let mut c = Column::i64(vec![1]);
    c.combine_at(0, Sum, &Column::f64(vec![2.0]), 0).unwrap();
    assert_eq!(c.get_i64(0), 3);
    let err = c.combine_at(0, Max, &Column::f64(vec![2.5]), 0).unwrap_err();
    assert!(matches!(err, VolasError::DType(_)), "{err}");
    assert!(c.combine_at(0, Max, &Column::f64(vec![1e30]), 0).is_err()); // never saturated
    assert_eq!(c.get_i64(0), 3, "unchanged");
}

#[test]
fn combine_at_i64_replace_copies_validity() {
    // Replace from a present source keeps the cell present...
    let mut c = Column::i64(vec![1, 2]);
    c.combine_at(1, CombineOp::Replace, &Column::i64(vec![9]), 0).unwrap();
    assert_eq!(c.get_i64(1), 9);
    assert!(c.is_valid(1));
    // ...and from an NA source marks it missing (validity materialized).
    let na = Column::i64_masked(vec![0], Validity::from_valid_iter(1, [false]));
    let mut c = Column::i64(vec![1, 2]);
    c.combine_at(1, CombineOp::Replace, &na, 0).unwrap();
    assert!(!c.is_valid(1));
    // Sum keeps the destination present (all-valid result, like the batch reduce).
    let mut c = Column::i64(vec![1, 2]);
    c.combine_at(1, CombineOp::Sum, &Column::i64(vec![5]), 0).unwrap();
    assert_eq!(c.get_i64(1), 7);
    assert!(c.is_valid(1));
}

#[test]
fn bool_and_str_fold_their_first_and_last_in_place() {
    use CombineOp::*;
    let mut b = Column::bool(vec![false]);
    b.combine_at(0, Replace, &Column::bool(vec![true]), 0).unwrap();
    assert_eq!(b.truth_at(0), Some(true));
    b.combine_at(0, Keep, &Column::bool(vec![false]), 0).unwrap();
    assert_eq!(b.truth_at(0), Some(true));
    b.combine_at(0, Keep, &Column::bool_from_options([None]), 0).unwrap();
    assert_eq!(b.truth_at(0), None, "a missing input makes `first` missing");
    b.combine_at(0, Replace, &Column::bool(vec![true]), 0).unwrap();
    assert_eq!(b.truth_at(0), None, "and it stays missing");
    let mut s = Column::str(vec!["a".into(), "b".into()]);
    s.combine_at(1, Replace, &Column::str(vec!["z".into()]), 0).unwrap();
    assert_eq!(s.str_at(1), Some("z"));
    // a str cell is rewritten only at the end of the column
    for r in [s.combine_at(0, Replace, &Column::str(vec!["y".into()]), 0), s.init_at(0, Keep, &Column::str(vec!["y".into()]), 0)] {
        assert!(matches!(r, Err(VolasError::Shape(_))));
    }
    assert!(matches!(s.combine_at(0, Keep, &Column::str_from_options([None]), 0), Err(VolasError::Shape(_))));
    s.combine_at(1, Replace, &Column::str_from_options([None]), 0).unwrap();
    assert!(!s.is_valid(1));
    assert_eq!(Max.unsupported(DType::Bool), VolasError::DType("cannot max a bool column".into()));
    assert_eq!([Keep, Replace, Max, Min, Sum].map(CombineOp::name), ["first", "last", "max", "min", "sum"]);
}

#[test]
fn first_and_last_are_missing_when_any_input_is() {
    use CombineOp::*;
    for op in [Keep, Replace] {
        let mut c = Column::f64(vec![1.0]);
        c.combine_at(0, op, &Column::f64(vec![f64::NAN]), 0).unwrap();
        assert!(c.get_f64(0).is_nan(), "{op:?}");
        c.combine_at(0, op, &Column::f64(vec![2.0]), 0).unwrap();
        assert!(c.get_f64(0).is_nan(), "{op:?} sticky");
        let mut d = Column::datetime(vec![1]);
        d.combine_at(0, op, &Column::datetime(vec![i64::MIN]), 0).unwrap();
        assert_eq!(d.get_i64(0), i64::MIN, "{op:?}");
        let mut i = Column::i32(vec![1]);
        i.combine_at(0, op, &Column::i32_masked(vec![9], Validity::from_valid_iter(1, [false])), 0).unwrap();
        assert!(!i.is_valid(0), "{op:?}");
    }
}


#[test]
fn init_at_is_the_aggregate_of_one_value() {
    use CombineOp::*;
    // f64: the value itself — a NaN first bar is a missing aggregate, never ±inf
    for (op, x, want) in [
        (Max, 5.0, 5.0),
        (Max, f64::NAN, f64::NAN),
        (Min, f64::NAN, f64::NAN),
        (Sum, 2.5, 2.5),
        (Keep, 7.0, 7.0),
        (Replace, f64::NAN, f64::NAN),
    ] {
        let mut c = Column::f64(vec![0.0, 0.0]);
        c.init_at(1, op, &Column::f64(vec![x]), 0).unwrap();
        let got = c.get_f64(1);
        assert!(got == want || (got.is_nan() && want.is_nan()), "{op:?} {x}");
    }
    for (op, want) in [(Max, 5.0), (Min, 5.0), (Sum, 5.0), (Keep, 5.0)] {
        let mut c = Column::f32(vec![0.0]);
        c.init_at(0, op, &Column::f32(vec![5.0]), 0).unwrap();
        assert_eq!(c.get_f64(0), want, "f32 {op:?}");
    }
    // ints: the value with its presence, for every op (a missing first value is a
    // missing aggregate)
    let na = Column::i64_masked(vec![3], Validity::from_valid_iter(1, [false]));
    for op in [Keep, Replace, Max, Sum] {
        let mut c = Column::i64(vec![1]);
        c.init_at(0, op, &na, 0).unwrap();
        assert!(!c.is_valid(0), "i64 {op:?}");
        let mut c = Column::i32(vec![1]);
        c.init_at(0, op, &na, 0).unwrap(); // converted from int64: still missing
        assert!(!c.is_valid(0), "i32 {op:?}");
        c.init_at(0, op, &Column::i32(vec![4]), 0).unwrap();
        assert_eq!((c.get_i64(0), c.is_valid(0)), (4, true), "i32 {op:?}");
    }
    let mut c = Column::datetime(vec![1]);
    c.init_at(0, Min, &Column::datetime(vec![9]), 0).unwrap();
    assert_eq!(c.get_i64(0), 9);
    let mut b = Column::bool_from_options([None]);
    b.init_at(0, Keep, &Column::bool(vec![true]), 0).unwrap();
    assert_eq!(b.truth_at(0), Some(true));
    let mut s = Column::str(vec!["a".into()]);
    s.init_at(0, Replace, &Column::str_from_options([None]), 0).unwrap();
    assert!(!s.is_valid(0));
    s.init_at(0, Replace, &Column::str(vec!["b".into()]), 0).unwrap();
    assert_eq!((s.str_at(0), s.is_valid(0)), (Some("b"), true));
}

#[test]
fn an_integer_fold_never_wraps() {
    use CombineOp::*;
    let overflow = |r: Result<()>| matches!(r, Err(VolasError::Overflow(_)));
    let mut c = Column::i64(vec![i64::MAX]);
    assert!(overflow(c.combine_at(0, Sum, &Column::i64(vec![1]), 0)));
    assert_eq!(c.get_i64(0), i64::MAX, "unchanged");
    let mut c = Column::i32(vec![i32::MAX]);
    assert!(overflow(c.combine_at(0, Sum, &Column::i32(vec![1]), 0)));
    assert!(overflow(c.combine_at(0, Replace, &Column::i64(vec![1 << 40]), 0)));
    assert!(overflow(c.init_at(0, Keep, &Column::i64(vec![1 << 40]), 0)));
    let mut d = Column::datetime(vec![5]);
    d.combine_at(0, Max, &Column::datetime(vec![9]), 0).unwrap();
    assert_eq!(d.get_i64(0), 9);
}
