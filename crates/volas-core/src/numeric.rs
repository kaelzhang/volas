//! The numeric element types a column can compute over (`f64`, `i64`), behind a
//! single trait so the cumulative / arithmetic / clip / select kernels are
//! written once and monomorphised per dtype (zero-cost, no f64 round-trip).
//!
//! Missing values use the f64 `NaN` sentinel — `i64` has none, so `is_missing`
//! is always false there — matching pandas 3.0's default (non-nullable) dtypes.
//! `wrapping_*` matches pandas/numpy int64 overflow (it wraps, e.g.
//! `abs(i64::MIN) == i64::MIN`) and also keeps volas's overflow-checked test
//! build from panicking.

use crate::column::Column;
use crate::dtype::DType;

/// A numeric column element type. Implemented for `f64` and `i64`; adding `i32` /
/// `f32` / `u64` later is just another `impl` plus a dispatch arm.
pub trait Numeric: Copy + PartialOrd {
    /// The column dtype backed by this element type.
    const DTYPE: DType;
    /// Additive identity (cumsum seed).
    const ZERO: Self;
    /// Multiplicative identity (cumprod seed).
    const ONE: Self;

    /// Wrap a computed buffer back into a `Column` of this element's dtype.
    fn into_column(values: Vec<Self>) -> Column;

    /// Whether this element is the missing sentinel (`NaN` for f64; never i64).
    fn is_missing(self) -> bool;
    /// Widen to f64 (lossless for both supported types within range).
    fn to_f64(self) -> f64;
    /// Narrow from f64 losslessly, or `None` when the value would not fit
    /// (non-integral / out of range / `NaN` for i64). Powers the keep-vs-promote
    /// decision in clip / where / mask / assignment.
    fn try_from_f64(x: f64) -> Option<Self>;

    /// Wrapping add (matches pandas/numpy int64 overflow; plain `+` for f64).
    fn wrapping_add(self, other: Self) -> Self;
    /// Wrapping subtract.
    fn wrapping_sub(self, other: Self) -> Self;
    /// Wrapping multiply.
    fn wrapping_mul(self, other: Self) -> Self;
    /// Wrapping absolute value (`abs(i64::MIN) == i64::MIN`, like pandas).
    fn wrapping_abs(self) -> Self;
}

impl Numeric for f64 {
    const DTYPE: DType = DType::F64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn into_column(values: Vec<Self>) -> Column {
        Column::f64(values)
    }
    #[inline]
    fn is_missing(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn try_from_f64(x: f64) -> Option<Self> {
        Some(x)
    }
    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        self - other
    }
    #[inline]
    fn wrapping_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn wrapping_abs(self) -> Self {
        self.abs()
    }
}

impl Numeric for i64 {
    const DTYPE: DType = DType::I64;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn into_column(values: Vec<Self>) -> Column {
        Column::i64(values)
    }
    #[inline]
    fn is_missing(self) -> bool {
        false
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn try_from_f64(x: f64) -> Option<Self> {
        // Lossless only: finite, integral, and within [-2^63, 2^63). `i64::MAX as
        // f64` rounds up to 2^63, so the upper bound must be exclusive.
        if x.is_finite() && x.fract() == 0.0 && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&x) {
            Some(x as i64)
        } else {
            None
        }
    }
    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        i64::wrapping_add(self, other)
    }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        i64::wrapping_sub(self, other)
    }
    #[inline]
    fn wrapping_mul(self, other: Self) -> Self {
        i64::wrapping_mul(self, other)
    }
    #[inline]
    fn wrapping_abs(self) -> Self {
        i64::wrapping_abs(self)
    }
}

// --- dtype promotion (single source of truth for pandas 3.0 result dtypes) ---

/// Result dtype of a binary arithmetic op (`+` `-` `*`) by operand *type*:
/// `int ∘ int → i64` (bool counts as int — `True + True == 2`), anything
/// involving `f64` (or a non-numeric operand) → f64. This is type-based, matching
/// pandas (`int * 2.0 → float64` even though `2.0` is integral); the value-based
/// rule is [`fits`], used by clip / where / assign. (`/` is always f64.)
pub fn binary_supertype(a: DType, b: DType) -> DType {
    let int_like = |d| matches!(d, DType::I64 | DType::Bool);
    if int_like(a) && int_like(b) {
        DType::I64
    } else {
        DType::F64
    }
}

/// Whether the scalar `x` fits `target` losslessly — the keep-vs-promote test for
/// clip / where / mask / assignment (e.g. `0.0` fits `int64`, `2.5` does not).
pub fn fits(target: DType, x: f64) -> bool {
    match target {
        DType::F64 => true,
        DType::I64 => i64::try_from_f64(x).is_some(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_element_semantics() {
        assert_eq!(f64::DTYPE, DType::F64);
        assert!(f64::NAN.is_missing() && !1.0_f64.is_missing());
        assert_eq!(2.5_f64.to_f64(), 2.5);
        assert_eq!(f64::try_from_f64(2.5), Some(2.5)); // f64 always fits
        assert_eq!(3.0_f64.wrapping_add(4.0), 7.0);
        assert_eq!((-2.0_f64).wrapping_abs(), 2.0);
    }

    #[test]
    fn i64_element_semantics() {
        assert_eq!(i64::DTYPE, DType::I64);
        assert!(!5_i64.is_missing()); // i64 has no missing
        assert_eq!(5_i64.to_f64(), 5.0);
        // lossless narrowing
        assert_eq!(i64::try_from_f64(5.0), Some(5));
        assert_eq!(i64::try_from_f64(2.5), None); // non-integral
        assert_eq!(i64::try_from_f64(f64::NAN), None); // NaN
        assert_eq!(i64::try_from_f64(1e30), None); // out of range
        assert_eq!(i64::try_from_f64(9.223372036854776e18), None); // 2^63, exclusive
        // wrapping matches pandas int64 overflow
        assert_eq!(i64::MAX.wrapping_add(1), i64::MIN);
        assert_eq!(i64::MIN.wrapping_abs(), i64::MIN);
        assert_eq!(3_i64.wrapping_mul(4), 12);
        assert_eq!(7_i64.wrapping_sub(10), -3);
    }

    #[test]
    fn promotion_rules() {
        use DType::{Bool, F64, I64, Utf8};
        assert_eq!(binary_supertype(I64, I64), I64);
        assert_eq!(binary_supertype(I64, F64), F64);
        assert_eq!(binary_supertype(F64, I64), F64);
        assert_eq!(binary_supertype(Bool, Bool), I64); // bool is int-like (True+True==2)
        assert_eq!(binary_supertype(I64, Bool), I64);
        assert_eq!(binary_supertype(I64, Utf8), F64); // non-numeric operand -> f64
        assert!(fits(I64, 3.0) && !fits(I64, 2.5)); // value-based
        assert!(fits(F64, 2.5)); // anything fits f64
        assert!(!fits(Bool, 1.0)); // not a numeric target
    }
}
