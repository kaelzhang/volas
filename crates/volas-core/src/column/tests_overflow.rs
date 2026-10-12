//! Checked integer arithmetic: every integer kernel raises on overflow, identically
//! in debug and release builds (never wraps, never panics), and a missing cell's
//! placeholder payload never trips the check.

use super::*;

fn overflow<T: std::fmt::Debug>(r: Result<T>) -> bool {
    matches!(r, Err(VolasError::Overflow(_)))
}

fn na_i64(v: &[i64], valid: &[bool]) -> Column {
    Column::i64_with(v.to_vec(), Validity::from_valid_iter(v.len(), valid.iter().copied()))
}

#[test]
fn binary_arithmetic_raises_on_overflow() {
    let max = Column::i64(vec![i64::MAX]);
    let min = Column::i64(vec![i64::MIN]);
    let one = Column::i64(vec![1]);
    assert!(overflow(max.binary(&one, BinOp::Add)));
    assert!(overflow(min.binary(&one, BinOp::Sub)));
    assert!(overflow(max.binary(&Column::i64(vec![2]), BinOp::Mul)));
    // i32 overflows at its own width (no silent widening)
    let m32 = Column::i32(vec![i32::MAX]);
    assert!(overflow(m32.binary(&Column::i32(vec![1]), BinOp::Add)));
    // bool counts as 0/1 and promotes, so it overflows like an int
    assert!(overflow(max.binary(&Column::bool(vec![true]), BinOp::Add)));
    // in range: exact
    assert_eq!(
        Column::i64(vec![i64::MAX - 1]).binary(&one, BinOp::Add).unwrap(),
        Column::i64(vec![i64::MAX])
    );
    // floats never overflow: IEEE infinity
    let big = Column::f64(vec![f64::MAX]);
    assert_eq!(big.binary(&big, BinOp::Add).unwrap(), Column::f64(vec![f64::INFINITY]));
}

#[test]
fn a_missing_cell_payload_never_overflows() {
    // the NA slot carries i64::MAX as its (meaningless) payload
    let a = na_i64(&[i64::MAX, 1], &[false, true]);
    let r = a.binary(&Column::i64(vec![1, 1]), BinOp::Add).unwrap();
    assert!(!r.is_valid(0));
    assert_eq!(r.as_i64().unwrap()[1], 2);
    assert!(!a.binary(&Column::i64(vec![2, 2]), BinOp::Mul).unwrap().is_valid(0));
    // reductions / cumulatives / abs skip it too
    assert_eq!(na_i64(&[i64::MAX, 1, 1], &[false, true, true]).sum().unwrap(), Scalar::I64(2));
    assert_eq!(na_i64(&[i64::MAX, 2, 3], &[false, true, true]).prod().unwrap(), Scalar::I64(6));
    assert!(na_i64(&[i64::MAX, 1, 1], &[false, true, true]).cumsum().is_ok());
    assert!(na_i64(&[i64::MAX, 2, 3], &[false, true, true]).cumprod().is_ok());
    assert!(na_i64(&[i64::MIN, -2], &[false, true]).abs().is_ok());
    assert!(Column::i32_with(vec![i32::MIN, -2], Validity::from_valid_iter(2, [false, true]))
        .abs()
        .is_ok());
}

#[test]
fn reductions_raise_on_overflow() {
    assert!(overflow(Column::i64(vec![i64::MAX, 1]).sum()));
    assert!(overflow(Column::i64(vec![i64::MAX, 2]).prod()));
    assert!(overflow(na_i64(&[i64::MAX, 1, 0], &[true, true, false]).sum()));
    assert!(overflow(na_i64(&[i64::MAX, 2, 0], &[true, true, false]).prod()));
    // int32 sums accumulate in int64 (pandas), so they do not overflow at i32
    assert_eq!(Column::i32(vec![i32::MAX, 1]).sum().unwrap(), Scalar::I64(i32::MAX as i64 + 1));
    // floats: IEEE
    assert_eq!(Column::f64(vec![f64::MAX, f64::MAX]).sum().unwrap(), Scalar::F64(f64::INFINITY));
}

#[test]
fn cumulatives_raise_on_overflow() {
    assert!(overflow(Column::i64(vec![i64::MAX, 1]).cumsum()));
    assert!(overflow(Column::i64(vec![i64::MAX, 2]).cumprod()));
    assert!(overflow(Column::i32(vec![i32::MAX, 1]).cumsum()));
    assert!(overflow(Column::i32(vec![i32::MAX, 2]).cumprod()));
    assert!(overflow(na_i64(&[i64::MAX, 0, 1], &[true, false, true]).cumsum()));
    assert!(overflow(na_i64(&[i64::MAX, 0, 2], &[true, false, true]).cumprod()));
}

#[test]
fn abs_diff_round_floordiv_raise_on_overflow() {
    assert!(overflow(Column::i64(vec![-3, i64::MIN]).abs()));
    assert!(overflow(Column::i32(vec![i32::MIN]).abs()));
    assert_eq!(Column::i64(vec![-3, 4]).abs().unwrap(), Column::i64(vec![3, 4]));
    // diff is subtraction
    assert!(overflow(Column::i64(vec![i64::MIN, 1]).diff(1)));
    assert!(overflow(Column::i64(vec![1, i64::MIN]).diff(-1)));
    // rounding to a power-of-ten multiple past the type's range
    assert!(overflow(Column::i64(vec![i64::MAX]).round(-1)));
    assert!(overflow(Column::i32(vec![i32::MAX]).round(-1)));
    assert_eq!(Column::i32(vec![15]).round(-1).unwrap(), Column::i32(vec![20]));
    // the one overflowing integer floor division
    assert!(overflow(Column::i64(vec![i64::MIN]).floordiv(&Column::i64(vec![-1]))));
    assert!(overflow(Column::i32(vec![i32::MIN]).floordiv(&Column::i32(vec![-1]))));
    assert_eq!(
        Column::i32(vec![-7]).floordiv(&Column::i32(vec![2])).unwrap(),
        Column::i32(vec![-4])
    );
}

#[test]
fn checked_element_ops_follow_the_dtype() {
    use crate::numeric::Numeric;
    assert_eq!(i64::MAX.checked_add(1), None);
    assert_eq!(Numeric::checked_sub(i64::MIN, 1), None);
    assert_eq!(Numeric::checked_mul(3_i64, 4), Some(12));
    assert_eq!(Numeric::checked_abs(i32::MIN), None);
    assert_eq!(Numeric::checked_add(1.5_f64, 1.0), Some(2.5));
    assert_eq!(Numeric::checked_sub(1.5_f32, 1.0), Some(0.5));
    assert_eq!(Numeric::checked_mul(f32::MAX, 2.0), Some(f32::INFINITY));
    assert_eq!(Numeric::checked_abs(-2.0_f64), Some(2.0));
    assert_eq!(Numeric::checked_abs(-2.0_f32), Some(2.0));
}
