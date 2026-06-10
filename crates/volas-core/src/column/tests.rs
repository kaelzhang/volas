
use super::*;

#[test]
fn datetime_column_basics() {
    let c = Column::datetime(vec![10, 20, 30]);
    assert_eq!(c.len(), 3);
    assert_eq!(c.dtype(), DType::Datetime);
    assert_eq!(c.as_datetime().unwrap(), &[10, 20, 30]);
    assert_eq!(c.get_f64(1), 20.0);
    assert_eq!(c.to_f64_vec(), vec![10.0, 20.0, 30.0]);
    assert_eq!(c.slice(1, 3), Column::datetime(vec![20, 30]));
    assert_eq!(c.take(&[2, 0]), Column::datetime(vec![30, 10]));
}

#[test]
fn append_is_copy_on_write() {
    // A shared view must not see a later append (CoW), but an unshared column
    // grows in place.
    let mut a = Column::f64(vec![1.0, 2.0]);
    let view = a.clone(); // shares the Arc buffer
    a.append(&Column::f64(vec![3.0])).unwrap();
    assert_eq!(a.as_f64().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(view.as_f64().unwrap(), &[1.0, 2.0]); // view unchanged
}

#[test]
fn datetime_append_same_dtype_only() {
    let mut a = Column::datetime(vec![1]);
    a.append(&Column::datetime(vec![2, 3])).unwrap();
    assert_eq!(a, Column::datetime(vec![1, 2, 3]));
    assert!(a.append(&Column::i64(vec![4])).is_err());
}

#[test]
fn to_datetime_parses_strings() {
    let c = Column::str(vec!["2020-01-01".into(), "2020-01-02 03:04:05".into()]);
    let dt = c.to_datetime().unwrap();
    assert_eq!(dt.dtype(), DType::Datetime);
    assert_eq!(dt.len(), 2);
    // idempotent on an already-datetime column
    assert_eq!(dt.to_datetime().unwrap(), dt);
}

#[test]
fn to_datetime_errors() {
    assert!(Column::str(vec!["not-a-date".into()])
        .to_datetime()
        .is_err());
    assert!(Column::i64(vec![1, 2]).to_datetime().is_err());
}

#[test]
fn cast_between_dtypes_and_errors() {
    // no-op when already the target dtype
    let f = Column::f64(vec![1.0, 2.0]);
    assert_eq!(f.cast(DType::F64).unwrap(), f);

    // -> F64
    assert_eq!(
        Column::i64(vec![3]).cast(DType::F64).unwrap(),
        Column::f64(vec![3.0])
    );
    assert_eq!(
        Column::bool(vec![true, false]).cast(DType::F64).unwrap(),
        Column::f64(vec![1.0, 0.0])
    );
    // Str -> F64 PARSES (explicit astype, Q2): a valid number converts, a
    // blank cell -> NaN, and a non-empty non-numeric string raises rather
    // than funnelling silently to NaN.
    let parsed = Column::str(vec!["1.5".into(), "".into()])
        .cast(DType::F64)
        .unwrap();
    assert_eq!(parsed.dtype(), DType::F64);
    let pv = parsed.to_f64_vec();
    assert_eq!(pv[0], 1.5);
    assert!(pv[1].is_nan());
    assert!(Column::str(vec!["a".into()]).cast(DType::F64).is_err());

    // -> I64 (F64 / F32 truncate, Bool / Datetime; Str parses, see below)
    assert_eq!(
        Column::f64(vec![2.9]).cast(DType::I64).unwrap(),
        Column::i64(vec![2])
    );
    assert_eq!(
        Column::f32(vec![2.9]).cast(DType::I64).unwrap(),
        Column::i64(vec![2])
    );
    assert!(Column::f32(vec![f32::NAN]).cast(DType::I64).is_err()); // non-finite
    assert_eq!(
        Column::bool(vec![true]).cast(DType::I64).unwrap(),
        Column::i64(vec![1])
    );
    assert_eq!(
        Column::datetime(vec![5]).cast(DType::I64).unwrap(),
        Column::i64(vec![5])
    );
    assert!(Column::str(vec!["x".into()]).cast(DType::I64).is_err());

    // -> Bool (F64 / I64; Str errors)
    assert_eq!(
        Column::f64(vec![0.0, 1.5]).cast(DType::Bool).unwrap(),
        Column::bool(vec![false, true])
    );
    assert_eq!(
        Column::i64(vec![0, 2]).cast(DType::Bool).unwrap(),
        Column::bool(vec![false, true])
    );
    assert!(Column::str(vec!["x".into()]).cast(DType::Bool).is_err());

    // -> Utf8 (every source variant of to_string_vec)
    assert_eq!(
        Column::f64(vec![1.5]).cast(DType::Utf8).unwrap(),
        Column::str(vec!["1.5".into()])
    );
    assert_eq!(
        Column::i64(vec![7]).cast(DType::Utf8).unwrap(),
        Column::str(vec!["7".into()])
    );
    assert_eq!(
        Column::bool(vec![true, false]).cast(DType::Utf8).unwrap(),
        Column::str(vec!["True".into(), "False".into()])
    );
    let dt_str = Column::datetime(vec![0]).cast(DType::Utf8).unwrap();
    assert_eq!(dt_str.dtype(), DType::Utf8);
    assert_eq!(dt_str.len(), 1);

    // -> Datetime (delegates to to_datetime)
    assert_eq!(
        Column::str(vec!["2020-01-01".into()])
            .cast(DType::Datetime)
            .unwrap()
            .dtype(),
        DType::Datetime
    );
}

#[test]
fn cast_str_to_numeric_parses_blanks_and_rejects() {
    let s = |xs: &[&str]| Column::str(xs.iter().map(|x| x.to_string()).collect());

    // F32: a valid number parses, a blank cell -> NaN.
    let f32c = s(&["3.25", "  "]).cast(DType::F32).unwrap();
    assert_eq!(f32c.dtype(), DType::F32);
    let f32v = f32c.to_f64_vec();
    assert_eq!(f32v[0], 3.25);
    assert!(f32v[1].is_nan());

    // I32: a valid integer parses, a blank cell is NA (validity), and an
    // out-of-i32-range value raises.
    let i32c = s(&["5", "", "-7"]).cast(DType::I32).unwrap();
    assert_eq!(i32c.dtype(), DType::I32);
    assert_eq!(i32c.to_f64_vec()[0], 5.0);
    assert!(!i32c.is_valid(1)); // blank -> NA
    assert_eq!(i32c.to_f64_vec()[2], -7.0);
    assert!(s(&["9999999999"]).cast(DType::I32).is_err()); // > i32::MAX

    // int target rejects a non-integral / non-numeric literal (no truncation).
    assert!(s(&["1.5"]).cast(DType::I64).is_err());
    assert!(s(&["abc"]).cast(DType::I64).is_err());
    assert!(s(&["nope"]).cast(DType::F32).is_err());
}

#[test]
fn equals_treats_nan_as_equal() {
    let a = Column::f64(vec![1.0, f64::NAN]);
    let b = Column::f64(vec![1.0, f64::NAN]);
    assert!(a.equals(&b)); // NaN == NaN here ...
    assert_ne!(a, b); // ... but derived PartialEq says NaN != NaN
    assert!(!a.equals(&Column::f64(vec![1.0]))); // length mismatch
    assert!(Column::i64(vec![1, 2]).equals(&Column::i64(vec![1, 2]))); // non-F64 fallback
    assert!(!Column::i64(vec![1]).equals(&Column::str(vec!["1".into()]))); // dtype mismatch
}

#[test]
fn typed_accessors_reject_wrong_variant() {
    let f = Column::f64(vec![1.0]);
    assert!(f.as_bool().is_none());
    assert!(f.as_i64().is_none());
    assert!(f.as_str().is_none());
    assert!(f.as_datetime().is_none());
    assert!(Column::bool(vec![true]).as_f64().is_none());
    assert!(Column::f64(vec![]).is_empty());
}

#[test]
fn per_variant_get_slice_take() {
    // get_f64 across the Bool / I64 / Str / F64 arms
    assert_eq!(Column::f64(vec![2.5]).get_f64(0), 2.5);
    assert_eq!(Column::bool(vec![true, false]).get_f64(0), 1.0);
    assert_eq!(Column::i64(vec![5]).get_f64(0), 5.0);
    assert!(Column::str(vec!["x".into()]).get_f64(0).is_nan());

    // slice / take across Bool / I64 / Str
    assert_eq!(
        Column::bool(vec![true, false, true]).slice(1, 3),
        Column::bool(vec![false, true])
    );
    assert_eq!(
        Column::i64(vec![1, 2, 3]).take(&[2, 0]),
        Column::i64(vec![3, 1])
    );
    assert_eq!(
        Column::str(vec!["a".into(), "b".into(), "c".into()]).take(&[1, 2]),
        Column::str(vec!["b".into(), "c".into()])
    );
    assert_eq!(
        Column::str(vec!["a".into(), "b".into()]).slice(0, 1),
        Column::str(vec!["a".into()])
    );

    // to_f64_vec Bool / I64 arms
    assert_eq!(Column::bool(vec![true, false]).to_f64_vec(), vec![1.0, 0.0]);
    assert_eq!(Column::i64(vec![3, 4]).to_f64_vec(), vec![3.0, 4.0]);
}

#[test]
fn bool_get_false_branch_and_bool_append() {
    assert_eq!(Column::bool(vec![true, false]).get_f64(1), 0.0); // the `else { 0.0 }` arm
    let mut a = Column::bool(vec![true]);
    a.append(&Column::bool(vec![false, true])).unwrap();
    assert_eq!(a.as_bool().unwrap(), &[true, false, true]);
}

#[test]
fn epoch_to_datetime_and_to_string_vec() {
    // epoch_to_datetime over int64 and float64 epochs; non-numeric dtypes error.
    assert!(Column::i64(vec![1, 2]).epoch_to_datetime("s").is_ok());
    assert!(Column::f64(vec![1.0, 2.0]).epoch_to_datetime("s").is_ok());
    assert!(Column::bool(vec![true]).epoch_to_datetime("s").is_err());
    // epoch_to_datetime_rounded preserves a fractional second; integers agree.
    assert_eq!(
        Column::f64(vec![1.5])
            .epoch_to_datetime_rounded("s")
            .unwrap(),
        Column::datetime(vec![1_500_000_000])
    );
    assert_eq!(
        Column::f64(vec![2.0]).epoch_to_datetime("s").unwrap(),
        Column::datetime(vec![2_000_000_000])
    );
    assert_eq!(
        Column::i64(vec![3]).epoch_to_datetime_rounded("s").unwrap(),
        Column::datetime(vec![3_000_000_000])
    );
    // the error closure on each numeric arm fires on an unknown unit
    assert!(Column::i64(vec![1]).epoch_to_datetime("weeks").is_err());
    assert!(Column::f64(vec![1.0]).epoch_to_datetime("weeks").is_err());
    assert!(Column::f64(vec![1.0])
        .epoch_to_datetime_rounded("weeks")
        .is_err());
    assert!(Column::bool(vec![true])
        .epoch_to_datetime_rounded("s")
        .is_err());
    // A missing epoch maps to NaT (i64::MIN), not 1970 / an error: a float NaN
    // and an int64 NA-bit both yield NaT, in both the truncating and rounded
    // variants; a present value still converts, and a bad unit still errors.
    let fnan = Column::f64(vec![f64::NAN, 2.0])
        .epoch_to_datetime("s")
        .unwrap();
    assert!(!fnan.is_valid(0) && fnan.is_valid(1) && fnan.null_count() == 1);
    let fnan_r = Column::f64(vec![f64::NAN, 1.5])
        .epoch_to_datetime_rounded("s")
        .unwrap();
    assert!(!fnan_r.is_valid(0) && fnan_r.is_valid(1));
    let ina = Column::i64_with(vec![0, 100], Validity::from_valid_iter(2, [false, true]))
        .epoch_to_datetime("s")
        .unwrap();
    assert!(!ina.is_valid(0) && ina.is_valid(1)); // NA-bit -> NaT, not epoch 0 -> 1970
    let ina_r = Column::i64_with(vec![5, 0], Validity::from_valid_iter(2, [true, false]))
        .epoch_to_datetime_rounded("s")
        .unwrap();
    assert!(ina_r.is_valid(0) && !ina_r.is_valid(1));
    // an all-NA float column is all NaT
    assert_eq!(
        Column::f64(vec![f64::NAN, f64::NAN])
            .epoch_to_datetime("s")
            .unwrap()
            .null_count(),
        2
    );
    // to_string_vec renders each supported dtype.
    assert_eq!(
        Column::str(vec!["a".into()]).to_string_vec(),
        vec!["a".to_string()]
    );
    assert_eq!(
        Column::f64(vec![1.5]).to_string_vec(),
        vec!["1.5".to_string()]
    );
    assert_eq!(Column::i64(vec![3]).to_string_vec(), vec!["3".to_string()]);
}

#[test]
fn scatter_follows_dtype_rules() {
    // `scatter` is the single assignment primitive (a 1-element source is a
    // scalar write); it keeps the target dtype and updates validity.
    // F64 stays F64 for a number or a bool source.
    let f = Column::f64(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        f.scatter(&[1], &Column::f64(vec![9.0])).unwrap(),
        Column::f64(vec![1.0, 9.0, 3.0])
    );
    assert_eq!(
        f.scatter(&[0], &Column::bool(vec![false])).unwrap(),
        Column::f64(vec![0.0, 2.0, 3.0])
    );
    // I64 keeps int for an integral number or a bool source.
    let i = Column::i64(vec![1, 2, 3]);
    assert_eq!(
        i.scatter(&[2], &Column::f64(vec![0.0])).unwrap(),
        Column::i64(vec![1, 2, 0])
    );
    assert_eq!(
        i.scatter(&[0], &Column::bool(vec![false])).unwrap(),
        Column::i64(vec![0, 2, 3])
    );
    // I64 + NaN keeps int64, marking that cell NA (the NA model; no float upcast).
    let na = i.scatter(&[1], &Column::f64(vec![f64::NAN])).unwrap();
    assert_eq!(na.dtype(), DType::I64);
    assert!(na.is_valid(0) && !na.is_valid(1) && na.is_valid(2));
    // I64 + a non-integral number is lossy -> error.
    assert!(i.scatter(&[0], &Column::f64(vec![2.5])).is_err());
    // Bool keeps bool for a bool source; a number into bool is lossy -> error.
    let b = Column::bool(vec![true, false]);
    assert_eq!(
        b.scatter(&[1], &Column::bool(vec![true])).unwrap(),
        Column::bool(vec![true, true])
    );
    assert!(b.scatter(&[0], &Column::f64(vec![0.0])).is_err());
    // A number / datetime into a str column is unsupported -> error.
    assert!(Column::str(vec!["a".into()])
        .scatter(&[0], &Column::f64(vec![1.0]))
        .is_err());
}

#[test]
fn cumulatives_preserve_dtype() {
    // i64 stays i64, computed natively
    assert_eq!(
        Column::i64(vec![1, 2, 3, 4]).cumsum().unwrap(),
        Column::i64(vec![1, 3, 6, 10])
    );
    assert_eq!(
        Column::i64(vec![3, 1, 4, 1]).cummax().unwrap(),
        Column::i64(vec![3, 3, 4, 4])
    );
    assert_eq!(
        Column::i64(vec![3, 1, 4, 1]).cummin().unwrap(),
        Column::i64(vec![3, 1, 1, 1])
    );
    assert_eq!(
        Column::i64(vec![1, 2, 3]).cumprod().unwrap(),
        Column::i64(vec![1, 2, 6])
    );
    // f64 keeps NaN in place (compare with equals: NaN == NaN)
    assert!(Column::f64(vec![1.0, f64::NAN, 2.0, 4.0])
        .cumsum()
        .unwrap()
        .equals(&Column::f64(vec![1.0, f64::NAN, 3.0, 7.0])));
    assert!(Column::f64(vec![1.0, f64::NAN, 4.0, 2.0])
        .cummax()
        .unwrap()
        .equals(&Column::f64(vec![1.0, f64::NAN, 4.0, 4.0])));
    assert!(Column::f64(vec![3.0, f64::NAN, 1.0])
        .cummin()
        .unwrap()
        .equals(&Column::f64(vec![3.0, f64::NAN, 1.0])));
    assert!(Column::f64(vec![2.0, f64::NAN, 3.0])
        .cumprod()
        .unwrap()
        .equals(&Column::f64(vec![2.0, f64::NAN, 6.0])));
    // bool is treated as i64 (pandas bool.cumsum -> int64); str -> error
    assert_eq!(
        Column::bool(vec![true, false, true]).cumsum().unwrap(),
        Column::i64(vec![1, 1, 2])
    );
    assert!(Column::str(vec!["a".into()]).cumsum().is_err());
}

#[test]
fn abs_preserves_dtype_and_wraps() {
    assert!(Column::f64(vec![-1.0, f64::NAN, 2.0])
        .abs()
        .unwrap()
        .equals(&Column::f64(vec![1.0, f64::NAN, 2.0])));
    // abs(i64::MIN) wraps to i64::MIN (pandas / numpy)
    assert_eq!(
        Column::i64(vec![-3, 4, i64::MIN]).abs().unwrap(),
        Column::i64(vec![3, 4, i64::MIN])
    );
}

#[test]
fn round_preserves_dtype() {
    // f64 banker's, NaN passthrough
    assert!(Column::f64(vec![0.5, 1.5, 2.5, f64::NAN])
        .round(0)
        .unwrap()
        .equals(&Column::f64(vec![0.0, 2.0, 2.0, f64::NAN])));
    // i64 identity at decimals>=0; banker's-to-multiple at negative decimals
    assert_eq!(
        Column::i64(vec![7, 8]).round(0).unwrap(),
        Column::i64(vec![7, 8])
    );
    assert_eq!(
        Column::i64(vec![15, 25, 35, 45, 5]).round(-1).unwrap(),
        Column::i64(vec![20, 20, 40, 40, 0])
    );
    assert_eq!(
        Column::i64(vec![16, 13]).round(-1).unwrap(),
        Column::i64(vec![20, 10])
    ); // r>half / r<half
    assert_eq!(
        Column::i64(vec![-15, -25]).round(-1).unwrap(),
        Column::i64(vec![-20, -20])
    ); // negative
    assert_eq!(
        Column::i64(vec![123]).round(-25).unwrap(),
        Column::i64(vec![0])
    ); // 10^25 overflows -> 0
    assert_eq!(
        Column::bool(vec![true, false]).round(0).unwrap(),
        Column::bool(vec![true, false])
    ); // bool no-op
    assert!(Column::str(vec!["a".into()]).round(0).is_err());
}

#[test]
fn clip_preserves_dtype_or_promotes() {
    use DType::{F64, I64};
    // f64: both bounds, lo-only, hi-only, no bounds, NaN passthrough
    assert!(Column::f64(vec![-1.0, 1.0, 3.0, f64::NAN])
        .clip(Some(0.0), Some(2.0))
        .unwrap()
        .equals(&Column::f64(vec![0.0, 1.0, 2.0, f64::NAN])));
    assert_eq!(
        Column::f64(vec![-1.0, 5.0]).clip(Some(0.0), None).unwrap(),
        Column::f64(vec![0.0, 5.0])
    );
    assert_eq!(
        Column::f64(vec![-1.0, 5.0]).clip(None, Some(2.0)).unwrap(),
        Column::f64(vec![-1.0, 2.0])
    );
    assert_eq!(
        Column::f64(vec![1.0, 5.0]).clip(None, None).unwrap(),
        Column::f64(vec![1.0, 5.0])
    );
    // i64 with integral bounds stays int
    assert_eq!(
        Column::i64(vec![1, 5, 9])
            .clip(Some(2.0), Some(8.0))
            .unwrap(),
        Column::i64(vec![2, 5, 8])
    );
    // i64 with a non-integral bound promotes to float (pandas)
    let p = Column::i64(vec![1, 5, 9]).clip(Some(2.5), None).unwrap();
    assert_eq!(p.dtype(), F64);
    assert_eq!(p, Column::f64(vec![2.5, 5.0, 9.0]));
    let _ = I64;
    // bool stays bool: clip(F,T) no-op, clip(T,T) forces true, clip(F,F) forces false
    assert_eq!(
        Column::bool(vec![true, false])
            .clip(Some(0.0), Some(1.0))
            .unwrap(),
        Column::bool(vec![true, false])
    );
    assert_eq!(
        Column::bool(vec![true, false])
            .clip(Some(1.0), Some(1.0))
            .unwrap(),
        Column::bool(vec![true, true])
    );
    assert_eq!(
        Column::bool(vec![true, false])
            .clip(Some(0.0), Some(0.0))
            .unwrap(),
        Column::bool(vec![false, false])
    );
    assert_eq!(
        Column::bool(vec![true, false]).clip(None, None).unwrap(),
        Column::bool(vec![true, false])
    );
    assert!(Column::str(vec!["a".into()]).clip(None, None).is_err());
}

#[test]
fn select_picks_in_target_dtype() {
    let cond = [true, false, true];
    let a = Column::i64(vec![1, 2, 3]);
    // target I64: other is i64 (direct) and f64-integral (lossless narrow)
    assert_eq!(
        a.select(&cond, &Column::i64(vec![10, 20, 30]), DType::I64)
            .unwrap(),
        Column::i64(vec![1, 20, 3])
    );
    assert_eq!(
        a.select(&cond, &Column::f64(vec![10.0, 20.0, 30.0]), DType::I64)
            .unwrap(),
        Column::i64(vec![1, 20, 3])
    );
    // target F64
    assert_eq!(
        Column::f64(vec![1.0, 2.0, 3.0])
            .select(&cond, &Column::f64(vec![10.0, 20.0, 30.0]), DType::F64)
            .unwrap(),
        Column::f64(vec![1.0, 20.0, 3.0])
    );
    // as_i64_vec error: target I64 but a value is non-integral
    assert!(a
        .select(&cond, &Column::f64(vec![1.5, 2.0, 3.0]), DType::I64)
        .is_err());
}

#[test]
fn binary_and_div_dtype() {
    use DType::{F64, I64};
    let a = Column::i64(vec![5, 7]);
    let b = Column::i64(vec![2, 3]);
    assert_eq!(a.binary(&b, BinOp::Add).unwrap(), Column::i64(vec![7, 10]));
    assert_eq!(a.binary(&b, BinOp::Sub).unwrap(), Column::i64(vec![3, 4]));
    assert_eq!(a.binary(&b, BinOp::Mul).unwrap(), Column::i64(vec![10, 21]));
    // int + float -> f64
    let r = a.binary(&Column::f64(vec![2.0, 3.0]), BinOp::Add).unwrap();
    assert_eq!(r.dtype(), F64);
    assert_eq!(r, Column::f64(vec![7.0, 10.0]));
    // wrapping overflow matches pandas int64
    assert_eq!(
        Column::i64(vec![i64::MAX])
            .binary(&Column::i64(vec![1]), BinOp::Add)
            .unwrap(),
        Column::i64(vec![i64::MIN])
    );
    // div is always float
    assert_eq!(a.div(&b).unwrap().dtype(), F64);
    assert_eq!(a.div(&b).unwrap(), Column::f64(vec![2.5, 7.0 / 3.0]));
    let _ = I64;
}

#[test]
fn reductions_carry_result_dtype() {
    use Scalar::{Bool as SB, F64, I64};
    // sum / prod: float -> F64; int / bool -> I64; non-numeric -> F64 (f64 fallback)
    assert_eq!(Column::f64(vec![1.0, f64::NAN, 2.0]).sum(), F64(3.0));
    assert_eq!(Column::i64(vec![1, 2, 3]).sum(), I64(6));
    assert_eq!(Column::bool(vec![true, false, true]).sum(), I64(2));
    assert!(matches!(Column::str(vec!["a".into()]).sum(), F64(_)));
    assert_eq!(Column::f64(vec![2.0, 3.0]).prod(), F64(6.0));
    assert_eq!(Column::i64(vec![2, 3, 4]).prod(), I64(24));
    assert_eq!(Column::bool(vec![true, true]).prod(), I64(1));
    assert!(matches!(Column::str(vec!["a".into()]).prod(), F64(_)));
    // min / max keep dtype: int -> I64, bool -> Bool, float -> F64
    assert_eq!(Column::i64(vec![3, 1, 2]).extreme(false), I64(1));
    assert_eq!(Column::i64(vec![3, 1, 2]).extreme(true), I64(3));
    assert_eq!(
        Column::bool(vec![true, false, true]).extreme(false),
        SB(false)
    ); // AND
    assert_eq!(
        Column::bool(vec![true, false, true]).extreme(true),
        SB(true)
    ); // OR
    assert_eq!(Column::f64(vec![3.0, 1.0]).extreme(false), F64(1.0));
    assert!(matches!(
        Column::str(vec!["a".into()]).extreme(true),
        F64(_)
    ));
    // empty / all-missing extreme -> NaN (F64)
    assert!(matches!(Column::i64(vec![]).extreme(false), F64(x) if x.is_nan()));
    assert!(matches!(Column::bool(vec![]).extreme(true), F64(x) if x.is_nan()));
    assert!(matches!(Column::f64(vec![]).extreme(true), F64(x) if x.is_nan()));
}

#[test]
fn f32_i32_columns() {
    use Scalar::{F32, I32, I64};
    let f = Column::f32(vec![1.5, 2.5, 3.5]);
    let i = Column::i32(vec![3, 1, 4]);
    // storage basics
    assert_eq!((f.dtype(), i.dtype(), f.len()), (DType::F32, DType::I32, 3));
    assert_eq!(f.to_f64_vec(), vec![1.5, 2.5, 3.5]);
    assert_eq!(i.get_f64(0), 3.0);
    assert_eq!(f.slice(0, 2), Column::f32(vec![1.5, 2.5]));
    assert_eq!(i.take(&[2, 0]), Column::i32(vec![4, 3]));
    assert_eq!(Column::i64(vec![1, 2]).to_f32_vec(), vec![1.0_f32, 2.0]);
    assert_eq!(i.to_string_vec(), vec!["3", "1", "4"]);
    assert!(Column::f32(vec![f32::NAN]).equals(&Column::f32(vec![f32::NAN]))); // NaN == NaN
    let mut a = Column::f32(vec![1.0]);
    a.append(&Column::f32(vec![2.0])).unwrap();
    assert_eq!(a, Column::f32(vec![1.0, 2.0]));
    // cast
    assert_eq!(
        Column::f64(vec![1.5]).cast(DType::F32).unwrap(),
        Column::f32(vec![1.5])
    );
    assert_eq!(
        Column::f64(vec![3.0]).cast(DType::I32).unwrap(),
        Column::i32(vec![3])
    );
    assert!(Column::f64(vec![2.5]).cast(DType::I32).is_err()); // non-integral
    assert!(Column::f64(vec![3e9]).cast(DType::I32).is_err()); // out of range
    assert_eq!(
        f.cast(DType::F64).unwrap(),
        Column::f64(vec![1.5, 2.5, 3.5])
    );
    // reductions: f32 -> F32; i32 sum -> I64 (promotes), min -> I32
    assert_eq!(f.sum(), F32(7.5));
    assert_eq!(f.extreme(false), F32(1.5));
    assert_eq!(i.sum(), I64(8));
    assert_eq!(i.prod(), I64(12));
    assert_eq!(i.extreme(true), I32(4));
    // round / clip preserve dtype
    assert_eq!(
        Column::f32(vec![1.4, 2.6]).round(0).unwrap(),
        Column::f32(vec![1.0, 3.0])
    );
    assert_eq!(i.round(-1).unwrap().dtype(), DType::I32);
    assert_eq!(
        f.clip(Some(2.0), Some(3.0)).unwrap(),
        Column::f32(vec![2.0, 2.5, 3.0])
    );
    assert_eq!(i.clip(Some(2.0), Some(3.0)).unwrap().dtype(), DType::I32);
    // binary: same-dtype preserves
    assert_eq!(
        f.binary(&f, BinOp::Add).unwrap(),
        Column::f32(vec![3.0, 5.0, 7.0])
    );
    assert_eq!(
        i.binary(&i, BinOp::Add).unwrap(),
        Column::i32(vec![6, 2, 8])
    );
    // select (where/mask) in f32 / i32 target
    let cond = [true, false, true];
    assert_eq!(
        f.select(&cond, &Column::f32(vec![0.0, 0.0, 0.0]), DType::F32)
            .unwrap(),
        Column::f32(vec![1.5, 0.0, 3.5])
    );
    assert_eq!(
        i.select(&cond, &Column::i32(vec![0, 0, 0]), DType::I32)
            .unwrap(),
        Column::i32(vec![3, 0, 4])
    );
    // assignment (scatter, 1-elem source): f32 writes; i32 keeps the dtype (a
    // NaN write marks the cell NA, no float upcast), rejects a lossy value
    assert_eq!(
        f.scatter(&[1], &Column::f32(vec![9.0])).unwrap(),
        Column::f32(vec![1.5, 9.0, 3.5])
    );
    assert_eq!(
        i.scatter(&[1], &Column::bool(vec![true])).unwrap(),
        Column::i32(vec![3, 1, 4])
    );
    assert_eq!(
        i.scatter(&[1], &Column::f64(vec![9.0])).unwrap(),
        Column::i32(vec![3, 9, 4])
    );
    let i_na = i.scatter(&[0], &Column::f64(vec![f64::NAN])).unwrap();
    assert!(i_na.dtype() == DType::I32 && !i_na.is_valid(0) && i_na.is_valid(1));
    assert!(i.scatter(&[0], &Column::f64(vec![2.5])).is_err());
    assert_eq!(
        f.scatter(&[0], &Column::bool(vec![false])).unwrap(),
        Column::f32(vec![0.0, 2.5, 3.5])
    );
    // remaining f32/i32 arms (both directions of slice/take, the other reductions,
    // sub/mul kernels through the trait, bool->i32, append/append_missing)
    assert_eq!(f.get_f64(0), 1.5);
    assert_eq!(i.slice(1, 3), Column::i32(vec![1, 4]));
    assert_eq!(f.take(&[2, 0]), Column::f32(vec![3.5, 1.5]));
    assert_eq!(f.to_string_vec(), vec!["1.5", "2.5", "3.5"]);
    assert_eq!(f.prod(), F32(13.125));
    assert!(matches!(Column::i32(vec![]).extreme(false), Scalar::F64(x) if x.is_nan()));
    assert_eq!(
        f.binary(&f, BinOp::Sub).unwrap(),
        Column::f32(vec![0.0, 0.0, 0.0])
    );
    assert_eq!(
        f.binary(&f, BinOp::Mul).unwrap(),
        Column::f32(vec![2.25, 6.25, 12.25])
    );
    assert_eq!(
        i.binary(&i, BinOp::Sub).unwrap(),
        Column::i32(vec![0, 0, 0])
    );
    assert_eq!(
        i.binary(&i, BinOp::Mul).unwrap(),
        Column::i32(vec![9, 1, 16])
    );
    assert_eq!(
        Column::bool(vec![true, false, true])
            .binary(&i, BinOp::Add)
            .unwrap(),
        Column::i32(vec![4, 1, 5])
    );
    // as_i32_vec f64 fallback (lossless narrow + lossy error)
    assert_eq!(
        i.select(&cond, &Column::f64(vec![0.0, 0.0, 0.0]), DType::I32)
            .unwrap(),
        Column::i32(vec![3, 0, 4])
    );
    assert!(i
        .select(&cond, &Column::f64(vec![2.5, 0.0, 0.0]), DType::I32)
        .is_err());
    assert_eq!(
        f.scatter(&[0], &Column::bool(vec![true])).unwrap(),
        Column::f32(vec![1.0, 2.5, 3.5])
    );
    let mut ii = Column::i32(vec![1]);
    ii.append(&Column::i32(vec![2])).unwrap();
    assert_eq!(ii, Column::i32(vec![1, 2]));
    assert!(Column::f32(vec![1.0])
        .append(&Column::i32(vec![1]))
        .is_err());
    let mut fm = Column::f32(vec![1.0]);
    fm.append_missing(2).unwrap();
    assert!(matches!(&fm, Column::F32(v) if v.len() == 3 && v[1].is_nan()));
}

#[test]
fn bool_matches_pandas() {
    let b = || Column::bool(vec![true, false, true]);
    let c = Column::bool(vec![true, true, false]);
    // cumsum / cumprod -> int64 (counts / product)
    assert_eq!(b().cumsum().unwrap(), Column::i64(vec![1, 1, 2]));
    assert_eq!(b().cumprod().unwrap(), Column::i64(vec![1, 0, 0]));
    // cummax / cummin -> bool (running OR / AND)
    assert_eq!(b().cummax().unwrap(), Column::bool(vec![true, true, true]));
    assert_eq!(
        b().cummin().unwrap(),
        Column::bool(vec![true, false, false])
    );
    // abs -> bool (identity)
    assert_eq!(b().abs().unwrap(), b());
    // + is OR, * is AND, - is an error
    assert_eq!(
        b().binary(&c, BinOp::Add).unwrap(),
        Column::bool(vec![true, true, true])
    );
    assert_eq!(
        b().binary(&c, BinOp::Mul).unwrap(),
        Column::bool(vec![true, false, false])
    );
    assert!(b().binary(&c, BinOp::Sub).is_err());
    // bool / bool -> error; bool ∘ number promotes (bool acts as 0/1)
    assert!(b().div(&c).is_err());
    assert_eq!(
        b().binary(&Column::i64(vec![1, 1, 1]), BinOp::Add).unwrap(),
        Column::i64(vec![2, 1, 2])
    );
    let f = b()
        .binary(&Column::f64(vec![1.0, 1.0, 1.0]), BinOp::Add)
        .unwrap();
    assert_eq!(f.dtype(), DType::F64);
    // where/mask with a bool fill stays bool (Column::select Bool target)
    let cond = [true, false, true];
    assert_eq!(
        b().select(&cond, &Column::bool(vec![false, false, false]), DType::Bool)
            .unwrap(),
        Column::bool(vec![true, false, true])
    );
    assert!(Column::i64(vec![1]).as_bool_vec().is_err()); // non-bool -> error
}

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
