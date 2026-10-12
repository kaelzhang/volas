//! Exact integer paths (contract #26): comparisons, order, reductions and fills
//! never round through float64; float64 math raises for an int64 past 2^53.

use super::*;

const T: i64 = 1 << 53;
const B: i64 = T + 1;

fn bools(c: Column) -> Vec<bool> {
    c.as_bool().unwrap().to_vec()
}

fn overflow_err<T: std::fmt::Debug>(r: Result<T>) -> bool {
    matches!(r, Err(VolasError::Overflow(_)))
}

#[test]
fn numeric_compare_is_exact() {
    let b = Column::i64(vec![B, 1]);
    assert_eq!(bools(b.compare(&Column::i64(vec![T, 1]), CmpOp::Gt).unwrap()), [true, false]);
    assert_eq!(bools(b.compare(&Column::f64(vec![T as f64, f64::NAN]), CmpOp::Ne).unwrap()), [true, true]);
    assert_eq!(bools(Column::f64(vec![T as f64, 0.5]).compare(&b, CmpOp::Lt).unwrap()), [true, true]);
    // a missing int side: only `!=` holds
    let na = Column::i64_with(vec![0, 1], Validity::from_valid_iter(2, [false, true]));
    assert_eq!(bools(na.compare(&Column::f64(vec![0.0, 1.0]), CmpOp::Eq).unwrap()), [false, true]);
    assert_eq!(bools(Column::f64(vec![0.0, 1.0]).compare(&na, CmpOp::Ne).unwrap()), [true, false]);
    assert_eq!(bools(Column::i32(vec![1]).compare(&Column::bool(vec![true]), CmpOp::Eq).unwrap()), [true]);
    assert_eq!(bools(Column::f32(vec![1.5]).compare(&Column::f64(vec![1.5]), CmpOp::Eq).unwrap()), [true]);
}

#[test]
fn monotonic_is_typed() {
    assert!(!Column::i64(vec![B, T]).is_monotonic(true));
    assert!(Column::i64(vec![B, T]).is_monotonic(false));
    assert!(Column::str(vec!["a".into(), "b".into()]).is_monotonic(true));
    assert!(!Column::f64(vec![1.0, f64::NAN]).is_monotonic(true));
    assert!(Column::datetime(vec![3]).is_monotonic(false));
}

#[test]
fn from_scalars_takes_the_supertype_losslessly() {
    let ints = Column::from_scalars(&[(DType::I64, Scalar::I64(B)), (DType::I64, Scalar::F64(f64::NAN))]).unwrap();
    assert_eq!(ints.as_i64().unwrap()[0], B);
    assert!(!ints.is_valid(1));
    let i32s = Column::from_scalars(&[(DType::I32, Scalar::I32(3))]).unwrap();
    assert_eq!(i32s, Column::i32(vec![3]));
    let mixed = Column::from_scalars(&[(DType::I64, Scalar::I64(1 << 60)), (DType::F32, Scalar::F32(0.5))]);
    assert!(mixed.unwrap().equals(&Column::f64(vec![(1u64 << 60) as f64, 0.5])));
    assert!(Column::from_scalars(&[(DType::I64, Scalar::I64(B)), (DType::F64, Scalar::F64(0.5))]).is_err());
    assert!(Column::from_scalars(&[(DType::I64, Scalar::Bool(true))]).unwrap().equals(&Column::i64(vec![1])));
    assert_eq!(Column::from_scalars(&[]).unwrap().dtype(), DType::F64);
    let dtypes = [Scalar::F64(0.0), Scalar::F32(0.0), Scalar::I64(0), Scalar::I32(0), Scalar::Bool(true)]
        .map(|s| s.dtype());
    assert_eq!(dtypes, [DType::F64, DType::F32, DType::I64, DType::I32, DType::Bool]);
}

#[test]
fn float64_math_guard() {
    let big = Column::i64(vec![B, 1]);
    assert!(overflow_err(big.to_f64_domain("mean")));
    let hidden = Column::i64_with(vec![B, 1], Validity::from_valid_iter(2, [false, true]));
    assert!(hidden.to_f64_domain("mean").is_ok()); // a missing payload is not a value
    assert_eq!(Column::i64(vec![T, -T]).to_f64_domain("mean").unwrap(), vec![T as f64, -T as f64]);
    assert!(overflow_err(big.div(&Column::i64(vec![1, 1]))));
    assert!(overflow_err(big.floordiv(&Column::i64(vec![0, 1]))));
    assert!(overflow_err(big.binary(&Column::f64(vec![0.5, 0.5]), BinOp::Add)));
    assert!(overflow_err(big.clip(Some(0.5), None)));
}

#[test]
fn float_select_converts_only_the_picked_cells() {
    let s = Column::i64(vec![B, 2]);
    // B is replaced, never converted: fine
    let r = s.select(&[false, true], &Column::f64(vec![0.5, 0.5]), DType::F64).unwrap();
    assert!(r.equals(&Column::f64(vec![0.5, 2.0])));
    // B is kept in a float64 result: inexact, raises
    assert!(s.select(&[true, true], &Column::f64(vec![0.5, 0.5]), DType::F64).is_err());
    let r = Column::f32(vec![1.5]).select(&[false], &Column::i64(vec![3]), DType::F32).unwrap();
    assert!(r.equals(&Column::f32(vec![3.0])));
    let b = Column::bool(vec![true]).select(&[true], &Column::f64(vec![0.5]), DType::F64).unwrap();
    assert!(b.equals(&Column::f64(vec![1.0])));
    let st = Column::str(vec!["a".into()]);
    assert!(st.select(&[true], &Column::f64(vec![0.5]), DType::F64).is_err());
    // fillna promoting an int column to float is lossless too
    let holes = Column::i64_with(vec![B, 0], Validity::from_valid_iter(2, [true, false]));
    assert!(holes.fillna(0.5).is_err());
}
