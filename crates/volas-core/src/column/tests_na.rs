//! Column NA / validity-behaviour tests (missing-value propagation across
//! reductions, cumulatives, elementwise ops, casts, fill, and scatter).

use super::*;

// --- NA (validity) behaviour ---------------------------------------------

fn na_i64(vals: &[i64], present: &[bool]) -> Column {
    Column::i64_with(
        vals.to_vec(),
        Validity::from_valid_iter(present.len(), present.iter().copied()),
    )
}

/// Assert a column's present/missing pattern and present values (NA -> NaN).
fn assert_na(c: &Column, expected: &[f64]) {
    let got = c.to_f64_vec();
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected) {
        if e.is_nan() {
            assert!(g.is_nan(), "expected NA, got {g}");
        } else {
            assert_eq!(g, e);
        }
    }
}

#[test]
fn na_is_valid_and_null_count() {
    let c = na_i64(&[1, 0, 3], &[true, false, true]);
    assert!(c.is_valid(0) && !c.is_valid(1) && c.is_valid(2));
    assert_eq!(c.null_count(), 1);
    // float -> NaN, datetime -> NaT (i64::MIN), str -> never missing
    let f = Column::f64(vec![1.0, f64::NAN]);
    assert!(!f.is_valid(1) && f.null_count() == 1);
    let f32c = Column::f32(vec![1.0, f32::NAN]);
    assert!(!f32c.is_valid(1) && f32c.null_count() == 1);
    let d = Column::datetime(vec![i64::MIN, 5]);
    assert!(!d.is_valid(0) && d.is_valid(1) && d.null_count() == 1);
    let s = Column::str(vec!["a".into()]);
    assert!(s.is_valid(0) && s.null_count() == 0);
    // to_f64_vec maps NA -> NaN
    assert_na(&c, &[1.0, f64::NAN, 3.0]);
}

#[test]
fn na_reductions_skip_missing() {
    let c = na_i64(&[1, 0, 3], &[true, false, true]); // 1, NA, 3
    assert_eq!(c.sum(), Scalar::I64(4));
    assert_eq!(c.prod(), Scalar::I64(3));
    assert_eq!(c.extreme(false), Scalar::I64(1));
    assert_eq!(c.extreme(true), Scalar::I64(3));
    // all-NA int -> NaN
    assert!(matches!(na_i64(&[0, 0], &[false, false]).extreme(true), Scalar::F64(x) if x.is_nan()));
    // i32 / bool skip NA, promote to i64
    let i = Column::i32_with(
        vec![5, 0, 7],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_eq!(i.sum(), Scalar::I64(12));
    assert_eq!(i.extreme(false), Scalar::I32(5));
    let b = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_eq!(b.sum(), Scalar::I64(1)); // present trues: pos0
    assert_eq!(b.extreme(true), Scalar::Bool(true)); // any present
    assert_eq!(b.extreme(false), Scalar::Bool(false)); // all present (true && false)
    assert!(matches!(
        Column::bool_with(vec![false], Validity::from_valid_iter(1, [false])).extreme(true),
        Scalar::F64(x) if x.is_nan()
    ));
    // the f64-funnel (mean / std / …) skips NA because to_f64_vec maps it to
    // NaN: here the present values are 1 and 3.
    let present: Vec<f64> = c.to_f64_vec().into_iter().filter(|x| !x.is_nan()).collect();
    assert_eq!(present, vec![1.0, 3.0]);
}

#[test]
fn na_cumulatives_propagate() {
    assert_na(
        &na_i64(&[1, 0, 3], &[true, false, true]).cumsum().unwrap(),
        &[1.0, f64::NAN, 4.0],
    );
    assert_na(
        &na_i64(&[2, 0, 3], &[true, false, true]).cumprod().unwrap(),
        &[2.0, f64::NAN, 6.0],
    );
    assert_na(
        &na_i64(&[3, 0, 1], &[true, false, true]).cummax().unwrap(),
        &[3.0, f64::NAN, 3.0],
    );
    assert_na(
        &na_i64(&[3, 0, 1], &[true, false, true]).cummin().unwrap(),
        &[3.0, f64::NAN, 1.0],
    );
    // i32
    let i = Column::i32_with(
        vec![1, 0, 3],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_na(&i.cumsum().unwrap(), &[1.0, f64::NAN, 4.0]);
    assert_na(&i.cummax().unwrap(), &[1.0, f64::NAN, 3.0]);
    assert_na(&i.cummin().unwrap(), &[1.0, f64::NAN, 1.0]);
    assert_na(&i.cumprod().unwrap(), &[1.0, f64::NAN, 3.0]);
    // bool: cumsum/cumprod -> i64; cummax = running OR, cummin = running AND
    let b = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_eq!(b.cumsum().unwrap().dtype(), DType::I64);
    assert_na(&b.cumsum().unwrap(), &[1.0, f64::NAN, 1.0]);
    assert_na(&b.cumprod().unwrap(), &[1.0, f64::NAN, 0.0]);
    assert_na(&b.cummax().unwrap(), &[1.0, f64::NAN, 1.0]);
    assert_na(&b.cummin().unwrap(), &[1.0, f64::NAN, 0.0]);
}

#[test]
fn na_elementwise_propagate() {
    assert_na(
        &na_i64(&[-1, 0, -3], &[true, false, true]).abs().unwrap(),
        &[1.0, f64::NAN, 3.0],
    );
    assert_na(
        &na_i64(&[12, 0, 28], &[true, false, true])
            .round(-1)
            .unwrap(),
        &[10.0, f64::NAN, 30.0],
    );
    assert_na(
        &na_i64(&[1, 0, 9], &[true, false, true])
            .clip(Some(2.0), Some(8.0))
            .unwrap(),
        &[2.0, f64::NAN, 8.0],
    );
    // a non-integral bound promotes int -> float; NA stays NA (NaN)
    let promoted = na_i64(&[1, 0, 9], &[true, false, true])
        .clip(Some(2.5), None)
        .unwrap();
    assert_eq!(promoted.dtype(), DType::F64);
    assert_na(&promoted, &[2.5, f64::NAN, 9.0]);
    // i32 round keeps i32 + validity
    let r = Column::i32_with(
        vec![28, 0, 12],
        Validity::from_valid_iter(3, [true, false, true]),
    )
    .round(-1)
    .unwrap();
    assert_eq!(r.dtype(), DType::I32);
    assert_na(&r, &[30.0, f64::NAN, 10.0]);
}

#[test]
fn na_binary_and_select_propagate() {
    // x ∘ NA = NA: combined validity (present only where both present)
    let a = na_i64(&[1, 0, 3], &[true, false, true]);
    let b = na_i64(&[10, 20, 0], &[true, true, false]);
    assert_na(
        &a.binary(&b, BinOp::Add).unwrap(),
        &[11.0, f64::NAN, f64::NAN],
    );
    // bool ∘ bool keeps bool, combined validity
    let bt = Column::bool_with(
        vec![true, false, true],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    let bf = Column::bool_with(
        vec![false, true, true],
        Validity::from_valid_iter(3, [true, true, false]),
    );
    assert_na(
        &bt.binary(&bf, BinOp::Add).unwrap(),
        &[1.0, f64::NAN, f64::NAN],
    ); // OR
       // bool ∘ int promotes to int (validity()'s Bool arm)
    assert_na(
        &bt.binary(&Column::i64(vec![1, 1, 1]), BinOp::Add).unwrap(),
        &[2.0, f64::NAN, 2.0],
    );
    // division funnels through f64 (NA -> NaN automatically)
    assert_na(
        &a.div(&Column::f64(vec![2.0, 2.0, 2.0])).unwrap(),
        &[0.5, f64::NAN, 1.5],
    );
    // select carries the chosen side's validity
    let other = na_i64(&[9, 9, 9], &[true, true, true]);
    let r = a.select(&[true, true, false], &other, DType::I64).unwrap();
    assert_na(&r, &[1.0, f64::NAN, 9.0]);
    // bool select target carries validity too
    let rb = bt
        .select(
            &[true, true, false],
            &Column::bool(vec![false, false, false]),
            DType::Bool,
        )
        .unwrap();
    assert_eq!(rb.dtype(), DType::Bool);
    assert_na(&rb, &[1.0, f64::NAN, 0.0]);
    // as_i32_vec's NA-placeholder branch: an i32 target with an f64 fill whose
    // NaN is selected becomes NA (value 0, masked).
    let i32c = Column::i32(vec![1, 2, 3]);
    let r32 = i32c
        .select(
            &[true, false, false],
            &Column::f64(vec![9.0, f64::NAN, 9.0]),
            DType::I32,
        )
        .unwrap();
    assert_eq!(r32.dtype(), DType::I32);
    assert_na(&r32, &[1.0, f64::NAN, 9.0]);
}

#[test]
fn na_shift_and_diff() {
    // float gap = NaN
    assert_na(
        &Column::f64(vec![1.0, 2.0, 3.0]).shift(1),
        &[f64::NAN, 1.0, 2.0],
    );
    assert_na(
        &Column::f32(vec![1.0, 2.0, 3.0]).shift(1),
        &[f64::NAN, 1.0, 2.0],
    );
    // int / bool keep their dtype with an NA gap (PDEP-16 alignment)
    let i = Column::i64(vec![1, 2, 3]);
    assert_eq!(i.shift(1).dtype(), DType::I64);
    assert_na(&i.shift(1), &[f64::NAN, 1.0, 2.0]);
    assert_na(&i.shift(-1), &[2.0, 3.0, f64::NAN]);
    assert_na(&Column::i32(vec![1, 2, 3]).shift(1), &[f64::NAN, 1.0, 2.0]);
    let b = Column::bool(vec![true, false, true]);
    assert_eq!(b.shift(1).dtype(), DType::Bool);
    assert_na(&b.shift(1), &[f64::NAN, 1.0, 0.0]);
    // datetime gap = NaT; str degrades to an all-missing float column
    assert_na(
        &Column::datetime(vec![10, 20, 30]).shift(1),
        &[f64::NAN, 10.0, 20.0],
    );
    assert_na(
        &Column::str(vec!["a".into(), "b".into()]).shift(1),
        &[f64::NAN, f64::NAN],
    );
    // shift carries a pre-existing NA; shift(0) is identity; beyond len -> all NA
    assert_na(
        &na_i64(&[1, 0, 3], &[true, false, true]).shift(1),
        &[f64::NAN, 1.0, f64::NAN],
    );
    assert_na(&i.shift(0), &[1.0, 2.0, 3.0]);
    assert_na(&i.shift(5), &[f64::NAN, f64::NAN, f64::NAN]);

    // diff is subtraction: int keeps int + NA gap, float stays float.
    assert_eq!(
        Column::i64(vec![1, 3, 6]).diff(1).unwrap().dtype(),
        DType::I64
    );
    assert_na(
        &Column::i64(vec![1, 3, 6]).diff(1).unwrap(),
        &[f64::NAN, 2.0, 3.0],
    );
    assert_na(
        &Column::f64(vec![1.0, 3.0, 6.0]).diff(1).unwrap(),
        &[f64::NAN, 2.0, 3.0],
    );
    assert_na(
        &Column::f32(vec![1.0, 3.0, 6.0]).diff(1).unwrap(),
        &[f64::NAN, 2.0, 3.0],
    );
    // negative-n diff (the backward branch of diff_kernel)
    assert_na(
        &Column::f64(vec![1.0, 3.0, 6.0]).diff(-1).unwrap(),
        &[-2.0, -3.0, f64::NAN],
    );
    // bool / str / datetime diff raises (bool-bool subtraction unsupported;
    // str/datetime not numeric / a datetime difference is a timedelta, not f64).
    assert!(Column::bool(vec![true, false, true]).diff(1).is_err());
    assert!(Column::str(vec!["a".into(), "b".into()]).diff(1).is_err());
    assert!(Column::datetime(vec![10, 25, 30]).diff(1).is_err());
}

#[test]
fn na_take_slice_append_preserve_validity() {
    let c = na_i64(&[1, 0, 3, 0, 5], &[true, false, true, false, true]); // 1,NA,3,NA,5
    assert_na(&c.slice(1, 4), &[f64::NAN, 3.0, f64::NAN]);
    assert_na(&c.take(&[4, 1, 0]), &[5.0, f64::NAN, 1.0]);
    // dense slice / take stay dense (the validity fast path)
    assert_eq!(Column::i64(vec![1, 2, 3]).slice(0, 2).null_count(), 0);
    assert_eq!(Column::i64(vec![1, 2, 3]).take(&[2, 0]).null_count(), 0);
    // append concatenates validity (NA ++ NA)
    let mut a = na_i64(&[1, 0], &[true, false]);
    a.append(&na_i64(&[0, 4], &[false, true])).unwrap();
    assert_na(&a, &[1.0, f64::NAN, f64::NAN, 4.0]);
    // dense ++ dense stays dense
    let mut d = Column::i64(vec![1, 2]);
    d.append(&Column::i64(vec![3])).unwrap();
    assert_eq!(d.null_count(), 0);
    // append_missing keeps the new bool rows dense `false` (refresh placeholder)
    let mut b = Column::bool(vec![true, false]);
    b.append_missing(2).unwrap();
    assert_eq!(b.null_count(), 0);
    assert_na(&b, &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn na_fillna() {
    // keep dtype when the fill fits; promote int -> float on a non-integral fill
    let c = na_i64(&[1, 0, 3], &[true, false, true]);
    assert_eq!(c.fillna(9.0).unwrap().dtype(), DType::I64);
    assert_na(&c.fillna(9.0).unwrap(), &[1.0, 9.0, 3.0]);
    assert_eq!(c.fillna(2.5).unwrap().dtype(), DType::F64);
    assert_na(&c.fillna(2.5).unwrap(), &[1.0, 2.5, 3.0]);
    let i32c = Column::i32_with(
        vec![1, 0, 3],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_eq!(i32c.fillna(9.0).unwrap().dtype(), DType::I32);
    assert_na(&i32c.fillna(9.0).unwrap(), &[1.0, 9.0, 3.0]);
    assert_eq!(i32c.fillna(2.5).unwrap().dtype(), DType::F64);
    // bool: a 0/1 fill keeps bool, else promote to float
    let bc = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_eq!(bc.fillna(1.0).unwrap().dtype(), DType::Bool);
    assert_na(&bc.fillna(1.0).unwrap(), &[1.0, 1.0, 0.0]);
    assert_eq!(bc.fillna(5.0).unwrap().dtype(), DType::F64);
    // float fill; a dense column is cloned unchanged
    assert_na(
        &Column::f64(vec![1.0, f64::NAN]).fillna(0.0).unwrap(),
        &[1.0, 0.0],
    );
    assert_na(
        &Column::f32(vec![1.0, f32::NAN]).fillna(0.0).unwrap(),
        &[1.0, 0.0],
    );
    assert_eq!(
        Column::i64(vec![1, 2]).fillna(9.0).unwrap(),
        Column::i64(vec![1, 2])
    );
    // a numeric fill on a non-numeric column (str / datetime) is rejected, not
    // silently funneled through f64 (which corrupted strings / lost the dtype)
    assert!(Column::datetime(vec![i64::MIN, 20]).fillna(9.0).is_err());
    assert!(Column::str_with(
        vec!["a".into(), String::new()],
        Validity::from_valid_iter(2, [true, false])
    )
    .fillna(0.0)
    .is_err());
}

#[test]
fn scatter_preserves_validity() {
    // a scatter keeps every other row's NA (regression: a scalar write used to
    // return a dense column and silently turn pre-existing NA into 0 / false)
    let c = na_i64(&[1, 0, 3], &[true, false, true]); // 1, NA, 3
    let r = c.scatter(&[0], &Column::f64(vec![9.0])).unwrap();
    assert_eq!(r.dtype(), DType::I64);
    assert!(r.is_valid(0) && !r.is_valid(1) && r.is_valid(2)); // 9, NA, 3
                                                               // writing NaN marks the position NA and keeps int (no f64 upcast)
    let r2 = c.scatter(&[2], &Column::f64(vec![f64::NAN])).unwrap();
    assert_eq!(r2.dtype(), DType::I64);
    assert!(r2.is_valid(0) && !r2.is_valid(1) && !r2.is_valid(2));
    // bool keeps its validity too
    let b = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    let rb = b.scatter(&[0], &Column::bool(vec![false])).unwrap();
    assert!(rb.dtype() == DType::Bool && rb.is_valid(0) && !rb.is_valid(1) && rb.is_valid(2));
    // a present write introduces no NA (null_count stays 0)
    assert_eq!(
        Column::i64(vec![1, 2, 3])
            .scatter(&[0], &Column::f64(vec![9.0]))
            .unwrap()
            .null_count(),
        0
    );
    // a bool source into an int column converts (pandas), keeping validity
    let rib = c.scatter(&[1], &Column::bool(vec![true])).unwrap();
    assert!(rib.dtype() == DType::I64 && rib.is_valid(1));
    // a typed-NA source (what the binding builds for None / NaN into bool) marks
    // the cell NA, keeping bool
    let bn = Column::bool(vec![true, false])
        .scatter(&[0], &Column::na_of(DType::Bool, 1))
        .unwrap();
    assert!(bn.dtype() == DType::Bool && !bn.is_valid(0) && bn.is_valid(1));
}

#[test]
fn na_of_and_select_edge_arms() {
    // `na_of` (the where/mask default `other`) for the remaining dtypes
    assert!(matches!(Column::na_of(DType::F32, 2), Column::F32(_)));
    let i32na = Column::na_of(DType::I32, 2);
    assert!(i32na.dtype() == DType::I32 && !i32na.is_valid(0) && !i32na.is_valid(1));
    let bna = Column::na_of(DType::Bool, 2);
    assert!(bna.dtype() == DType::Bool && !bna.is_valid(0));
    // `select` errors when the str / datetime arm gets a mismatched `other`
    let cond = vec![true, false];
    assert!(Column::str(vec!["a".into(), "b".into()])
        .select(&cond, &Column::i64(vec![0, 0]), DType::Utf8)
        .is_err());
    assert!(Column::datetime(vec![1, 2])
        .select(&cond, &Column::i64(vec![0, 0]), DType::Datetime)
        .is_err());
}

#[test]
fn na_fill_dir() {
    let c = na_i64(&[0, 2, 0, 0, 5], &[false, true, false, false, true]); // NA,2,NA,NA,5
    assert_na(&c.fill_dir(true), &[f64::NAN, 2.0, 2.0, 2.0, 5.0]); // ffill: leading NA stays
    assert_na(&c.fill_dir(false), &[2.0, 2.0, 5.0, 5.0, 5.0]); // bfill: trailing filled
    let i32c = Column::i32_with(
        vec![7, 0, 0],
        Validity::from_valid_iter(3, [true, false, false]),
    );
    assert_na(&i32c.fill_dir(true), &[7.0, 7.0, 7.0]);
    let bc = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, false, false]),
    );
    assert_na(&bc.fill_dir(true), &[1.0, 1.0, 1.0]);
    assert_na(
        &Column::f64(vec![1.0, f64::NAN, 3.0]).fill_dir(true),
        &[1.0, 1.0, 3.0],
    );
    assert_na(
        &Column::f32(vec![1.0, f32::NAN]).fill_dir(true),
        &[1.0, 1.0],
    );
    assert_na(
        &Column::datetime(vec![i64::MIN, 20]).fill_dir(false),
        &[20.0, 20.0],
    );
    assert_eq!(
        Column::i64(vec![1, 2]).fill_dir(true),
        Column::i64(vec![1, 2])
    ); // dense clone
    assert_eq!(
        Column::str(vec!["a".into()]).fill_dir(true),
        Column::str(vec!["a".into()])
    );
    // str + NA carries values directionally like every other dtype (regression:
    // the Str arm used to `unreachable!()` and panic on a missing cell).
    let sc = Column::str_with(
        vec!["x".into(), String::new(), String::new(), "z".into()],
        Validity::from_valid_iter(4, [true, false, false, true]),
    ); // "x", NA, NA, "z"
    assert_eq!(
        sc.fill_dir(true), // ffill -> x, x, x, z (no holes left -> dense)
        Column::str(vec!["x".into(), "x".into(), "x".into(), "z".into()])
    );
    assert_eq!(
        sc.fill_dir(false), // bfill -> x, z, z, z
        Column::str(vec!["x".into(), "z".into(), "z".into(), "z".into()])
    );
    // a leading gap stays NA on ffill (nothing to carry in)
    let ff = Column::str_with(
        vec![String::new(), "a".into()],
        Validity::from_valid_iter(2, [false, true]),
    )
    .fill_dir(true);
    assert!(!ff.is_valid(0) && ff.is_valid(1)); // NA, "a"
                                                // a fully-missing str column ffills to itself (all still NA)
    let allna = Column::str_with(
        vec![String::new(), String::new()],
        Validity::from_valid_iter(2, [false, false]),
    );
    assert_eq!(allna.fill_dir(true).null_count(), 2);
}

#[test]
fn append_na_pads_dtype_preserving() {
    // a plain column padded on append keeps its dtype and marks the new rows NA
    let mut i = Column::i64(vec![1, 2]);
    i.append_na(2);
    assert_eq!(i.dtype(), DType::I64); // no upcast to float
    assert!(i.is_valid(0) && i.is_valid(1) && !i.is_valid(2) && !i.is_valid(3));
    let mut i32c = Column::i32(vec![7]);
    i32c.append_na(1);
    assert!(i32c.is_valid(0) && !i32c.is_valid(1) && i32c.dtype() == DType::I32);
    let mut b = Column::bool(vec![true]);
    b.append_na(1);
    assert!(b.is_valid(0) && !b.is_valid(1) && b.dtype() == DType::Bool);
    let mut s = Column::str(vec!["a".into()]);
    s.append_na(1);
    assert!(s.is_valid(0) && !s.is_valid(1) && s.dtype() == DType::Utf8);
    let mut d = Column::datetime(vec![100]);
    d.append_na(1);
    assert!(d.is_valid(0) && !d.is_valid(1)); // NaT sentinel
    let mut f = Column::f64(vec![1.0]);
    f.append_na(1);
    assert!(f.is_valid(0) && !f.is_valid(1)); // NaN in-band
    let mut f32c = Column::f32(vec![1.0]);
    f32c.append_na(1);
    assert!(f32c.is_valid(0) && !f32c.is_valid(1)); // f32 NaN in-band
                                                    // an existing hole is preserved, the appended rows are added as NA
    let mut i2 = Column::i64_with(vec![1, 0], Validity::from_valid_iter(2, [true, false]));
    i2.append_na(1);
    assert!(i2.is_valid(0) && !i2.is_valid(1) && !i2.is_valid(2) && i2.null_count() == 2);
}

#[test]
fn na_logical_kleene_and_not() {
    let b = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, true, false]),
    ); // T,F,NA
    let t = Column::bool(vec![true, true, true]);
    let f = Column::bool(vec![false, false, false]);
    // Kleene AND: NA & True = NA, NA & False = False
    assert_na(&b.logical(&t, BoolOp::And), &[1.0, 0.0, f64::NAN]);
    assert_na(&b.logical(&f, BoolOp::And), &[0.0, 0.0, 0.0]);
    // Kleene OR: NA | True = True, NA | False = NA
    assert_na(&b.logical(&t, BoolOp::Or), &[1.0, 1.0, 1.0]);
    assert_na(&b.logical(&f, BoolOp::Or), &[1.0, 0.0, f64::NAN]);
    // XOR: missing if either is missing
    assert_na(&b.logical(&t, BoolOp::Xor), &[0.0, 1.0, f64::NAN]);
    // NOT propagates NA
    assert_na(&b.not(), &[0.0, 1.0, f64::NAN]);
    // a non-bool operand reads as x != 0 (present), so it never injects NA
    let i = Column::i64(vec![1, 0, 5]);
    assert_na(&i.not(), &[0.0, 1.0, 0.0]);
    assert_na(&b.logical(&i, BoolOp::And), &[1.0, 0.0, f64::NAN]);
}

#[test]
fn na_cast_int_bool_carries_validity() {
    let i = na_i64(&[1, 0, 3], &[true, false, true]); // 1, NA, 3
    assert_eq!(i.cast(DType::I32).unwrap().dtype(), DType::I32);
    assert_na(&i.cast(DType::I32).unwrap(), &[1.0, f64::NAN, 3.0]);
    assert_na(&i.cast(DType::Bool).unwrap(), &[1.0, f64::NAN, 1.0]);
    assert_na(&i.cast(DType::F64).unwrap(), &[1.0, f64::NAN, 3.0]); // int+NA -> float (NaN)
    let i32c = Column::i32_with(
        vec![1, 0, 5],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_eq!(i32c.cast(DType::I64).unwrap().dtype(), DType::I64);
    assert_na(&i32c.cast(DType::I64).unwrap(), &[1.0, f64::NAN, 5.0]);
    assert_na(&i32c.cast(DType::Bool).unwrap(), &[1.0, f64::NAN, 1.0]);
    let b = Column::bool_with(
        vec![true, false, false],
        Validity::from_valid_iter(3, [true, false, true]),
    );
    assert_na(&b.cast(DType::I64).unwrap(), &[1.0, f64::NAN, 0.0]);
    assert_na(&b.cast(DType::I32).unwrap(), &[1.0, f64::NAN, 0.0]);
    // a present out-of-range i64 -> i32 errors (a missing one never does)
    assert!(Column::i64(vec![3_000_000_000]).cast(DType::I32).is_err());
}

#[test]
fn na_str_carries_validity() {
    let s = Column::str_with(
        vec!["a".into(), String::new(), "c".into()],
        Validity::from_valid_iter(3, [true, false, true]),
    ); // a, NA, c
    assert!(s.is_valid(0) && !s.is_valid(1) && s.is_valid(2) && s.null_count() == 1);
    // shift keeps str with an NA gap: [NA, a, NA] (the trailing NA was already missing)
    let sh = s.shift(1);
    assert_eq!(sh.dtype(), DType::Utf8);
    assert!(!sh.is_valid(0) && sh.is_valid(1) && !sh.is_valid(2));
    assert_eq!(sh.as_str().unwrap()[1], "a");
    // slice / take carry validity
    assert!(!s.slice(1, 3).is_valid(0) && s.slice(1, 3).is_valid(1));
    assert!(!s.take(&[1, 0]).is_valid(0) && s.take(&[1, 0]).is_valid(1));
    // append concatenates validity
    let mut a = Column::str_with(
        vec!["x".into(), String::new()],
        Validity::from_valid_iter(2, [true, false]),
    );
    a.append(&Column::str(vec!["y".into()])).unwrap();
    assert!(a.len() == 3 && a.is_valid(0) && !a.is_valid(1) && a.is_valid(2));
    // int + NA -> str carries the missing cell
    let i = na_i64(&[1, 0, 3], &[true, false, true])
        .cast(DType::Utf8)
        .unwrap();
    assert_eq!(i.dtype(), DType::Utf8);
    assert!(!i.is_valid(1) && i.null_count() == 1);
}
