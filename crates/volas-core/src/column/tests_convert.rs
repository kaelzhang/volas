//! The lossless conversion owner (contract #20) and the int64 <-> float exactness
//! predicates it is built on.

use super::*;
use crate::numeric::{
    cmp_i64_f64, f64_to_f32_in_range, i64_to_f32_exact, i64_to_f64_exact, in_f64_domain,
    outside_f64_domain,
};
use std::cmp::Ordering::{Equal, Greater, Less};

const B: i64 = (1 << 53) + 1;

fn dtype_err<T: std::fmt::Debug>(r: Result<T>) -> bool {
    matches!(r, Err(VolasError::DType(_)))
}

fn overflow_err<T: std::fmt::Debug>(r: Result<T>) -> bool {
    matches!(r, Err(VolasError::Overflow(_)))
}

#[test]
fn exactness_predicates() {
    assert_eq!(i64_to_f64_exact(1 << 60), Some((1u64 << 60) as f64));
    assert_eq!(i64_to_f64_exact(B), None);
    assert_eq!(i64_to_f64_exact(i64::MAX), None); // rounds to 2^63, not an i64
    assert_eq!(i64_to_f64_exact(i64::MIN), Some(-9.223_372_036_854_776e18));
    assert_eq!(i64_to_f32_exact((1 << 24) + 1), None);
    assert_eq!(i64_to_f32_exact(i64::MAX), None);
    assert_eq!(i64_to_f32_exact(1 << 24), Some(16_777_216.0));
    assert!(in_f64_domain(1 << 53) && in_f64_domain(-(1 << 53)) && !in_f64_domain(B));
    assert!(matches!(outside_f64_domain(B, "var"), VolasError::Overflow(m) if m.starts_with("var")));
    assert_eq!(f64_to_f32_in_range(0.1), Some(0.1_f32));
    assert_eq!(f64_to_f32_in_range(1e300), None);
    assert_eq!(f64_to_f32_in_range(f64::INFINITY), Some(f32::INFINITY));
    assert!(f64_to_f32_in_range(f64::NAN).unwrap().is_nan());
}

#[test]
fn exact_int_float_ordering() {
    assert_eq!(cmp_i64_f64(B, (1u64 << 53) as f64), Some(Greater));
    assert_eq!(cmp_i64_f64(1 << 53, (1u64 << 53) as f64), Some(Equal));
    assert_eq!(cmp_i64_f64(2, 2.5), Some(Less));
    assert_eq!(cmp_i64_f64(-2, -2.5), Some(Greater));
    assert_eq!(cmp_i64_f64(i64::MAX, 9.223_372_036_854_776e18), Some(Less));
    assert_eq!(cmp_i64_f64(i64::MIN, -9.3e18), Some(Greater));
    assert_eq!(cmp_i64_f64(0, f64::INFINITY), Some(Less));
    assert_eq!(cmp_i64_f64(0, f64::NAN), None);
}

#[test]
fn convert_lossless_to_floats() {
    use DType::{F32, F64};
    let ints = Column::i64_with(vec![1, 0], Validity::from_valid_iter(2, [true, false]));
    let f = ints.convert_lossless(F64).unwrap();
    assert!(f.equals(&Column::f64(vec![1.0, f64::NAN])));
    assert!(dtype_err(Column::i64(vec![B]).convert_lossless(F64)));
    assert!(dtype_err(Column::i64(vec![(1 << 24) + 1]).convert_lossless(F32)));
    assert!(Column::bool(vec![true]).convert_lossless(F32).unwrap().equals(&Column::f32(vec![1.0])));
    // float64 -> float32 rounds precision but never overflows to inf
    assert!(overflow_err(Column::f64(vec![1e300]).convert_lossless(F32)));
    assert!(Column::f64(vec![f64::NAN]).convert_lossless(F32).unwrap().equals(&Column::f32(vec![f32::NAN])));
    assert_eq!(Column::f32(vec![0.5]).convert_lossless(F64).unwrap(), Column::f64(vec![0.5]));
    // same dtype: shared as-is
    assert_eq!(Column::f64(vec![2.0]).convert_lossless(F64).unwrap(), Column::f64(vec![2.0]));
}

#[test]
fn convert_lossless_to_ints() {
    use DType::{I32, I64};
    let f = Column::f64(vec![2.0, f64::NAN]);
    let i = f.convert_lossless(I64).unwrap();
    assert_eq!(i.as_i64().unwrap()[0], 2);
    assert!(!i.is_valid(1));
    assert!(dtype_err(Column::f64(vec![2.5]).convert_lossless(I64)));
    assert!(overflow_err(Column::f64(vec![1e20]).convert_lossless(I64)));
    assert!(overflow_err(Column::i64(vec![1 << 40]).convert_lossless(I32)));
    assert!(overflow_err(Column::f32(vec![3e9]).convert_lossless(I32)));
    assert_eq!(Column::i64(vec![7]).convert_lossless(I32).unwrap(), Column::i32(vec![7]));
    assert_eq!(Column::bool(vec![true]).convert_lossless(I64).unwrap(), Column::i64(vec![1]));
}

#[test]
fn convert_lossless_rejects_other_kinds() {
    assert!(dtype_err(Column::str(vec!["1".into()]).convert_lossless(DType::F64)));
    assert!(dtype_err(Column::i64(vec![1]).convert_lossless(DType::Bool)));
    assert!(dtype_err(Column::datetime(vec![1]).convert_lossless(DType::I32)));
    assert!(dtype_err(Column::f64(vec![1.0]).convert_lossless(DType::Utf8)));
}

#[test]
fn astype_never_saturates() {
    assert!(overflow_err(Column::f64(vec![1e20]).cast(DType::I64)));
    assert!(overflow_err(Column::f32(vec![-1e30]).cast(DType::I64)));
    assert_eq!(Column::f64(vec![-2.7]).cast(DType::I64).unwrap(), Column::i64(vec![-2]));
    assert!(dtype_err(Column::i64(vec![B]).cast(DType::F64)));
    assert!(overflow_err(Column::f64(vec![1e300]).cast(DType::F32)));
    // a float epoch past int64 is unconvertible, not saturated to i64::MAX
    assert!(Column::f64(vec![1e30]).epoch_to_datetime("ns").is_err());
    assert!(Column::f64(vec![9.3e18]).epoch_to_datetime_rounded("ns").is_err());
}
