//! Column: a single typed, contiguous buffer.
//!
//! Each variant holds its buffer behind an `Arc`, so a `Column` — and the
//! `DataFrame` / `Series` that contain it — is cheap to clone: cloning shares the
//! buffer (an O(1) refcount bump, not an O(n) copy). Mutation (`append`) is
//! copy-on-write via `Arc::make_mut`: it grows the `Vec` in place when the buffer
//! is uniquely owned, and copies only when a view (another `Series`, a zero-copy
//! export) is still alive. `F64` columns use `NaN` for missing values (matching
//! stock-pandas / pandas semantics).

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::buffer::Buffer;
use crate::strbuf::StrBuffer;
use crate::datetime;
use crate::dtype::DType;
use crate::error::{Result, VolasError};
use crate::numeric::{
    binary_supertype, cmp_i64_f64, f64_to_f32_in_range, fits, i64_to_f32_exact, i64_to_f64_exact,
    in_f64_domain, outside_f64_domain, overflow, Numeric,
};
use crate::stats;
use crate::validity::Validity;

/// Run a numeric kernel over a column's element type, monomorphised per dtype
/// (`F64` / `I64`) with no f64 round-trip. `$slice` is bound to the typed slice;
/// `$body` must produce a `Column` (via [`Numeric::into_column`]). `Bool` is
/// handled per-op by the caller (pandas treats it inconsistently — `cumsum -> int`
/// but `abs/cummax -> bool`); a `Bool` / `Str` / `Datetime` column here is an error.
macro_rules! numeric_dispatch {
    ($col:expr, $slice:ident => $body:expr) => {
        match $col {
            Column { data: ColumnData::F64(buf) } => {
                let $slice: &[f64] = buf.as_slice();
                Ok($body)
            }
            Column { data: ColumnData::F32(buf) } => {
                let $slice: &[f32] = buf.as_slice();
                Ok($body)
            }
            Column { data: ColumnData::I64(buf, _) } => {
                let $slice: &[i64] = buf.as_slice();
                Ok($body)
            }
            Column { data: ColumnData::I32(buf, _) } => {
                let $slice: &[i32] = buf.as_slice();
                Ok($body)
            }
            other => Err(VolasError::DType(format!(
                "expected a numeric column, got {}",
                other.dtype()
            ))),
        }
    };
}

/// A typed, contiguous column of values. The value buffer is a [`Buffer`] —
/// owned (`Arc`-shared, cheap clone, copy-on-write mutation) or a zero-copy borrow
/// of foreign (Arrow / NumPy) memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub(crate) data: ColumnData,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ColumnData {
    /// 64-bit floats; `NaN` denotes missing.
    F64(Buffer<f64>),
    /// 32-bit floats (narrow storage); `NaN` denotes missing.
    F32(Buffer<f32>),
    /// Booleans (comparison / signal results); `Validity` marks missing cells.
    Bool(Buffer<bool>, Validity),
    /// 64-bit signed integers; `Validity` marks missing cells.
    I64(Buffer<i64>, Validity),
    /// 32-bit signed integers (narrow storage); `Validity` marks missing cells.
    I32(Buffer<i32>, Validity),
    /// UTF-8 strings; `Validity` marks missing cells.
    Str(StrBuffer, Validity),
    /// Datetimes as i64 nanoseconds since the Unix epoch (UTC-naive).
    Datetime(Buffer<i64>),
}

/// The result of a dtype-preserving scalar reduction ([`Column::sum`] etc.),
/// carrying the value in its pandas result dtype so the binding can box it as the
/// matching numpy scalar (`np.int64` / `np.float64` / `np.bool_`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// A float64 result (`mean`, an f64 column's `sum`/`min`/`max`, …).
    F64(f64),
    /// A float32 result (an f32 column's `sum`/`min`/`max`).
    F32(f32),
    /// An int64 result (an i64/bool column's `sum`/`prod`, an i64 `min`/`max`).
    I64(i64),
    /// An int32 result (an i32 column's `sum`/`prod`/`min`/`max`).
    I32(i32),
    /// A boolean result (a bool column's `min`/`max`).
    Bool(bool),
}

/// A dtype-preserving binary arithmetic op for [`Column::binary`] (pandas
/// `+ - *`). True division is always float, so it is [`Column::div`], not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
}

/// A three-valued logical op for [`Column::logical`] (pandas `&` / `|` / `^` on
/// bool columns, Kleene semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    /// Logical AND (a present `false` short-circuits to `false`).
    And,
    /// Logical OR (a present `true` short-circuits to `true`).
    Or,
    /// Logical XOR (missing if either operand is missing).
    Xor,
}

/// An element-wise comparison op for [`Column::compare`] (pandas `== != < <= > >=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// IEEE comparison of an optional order: only `!=` holds for a missing side.
    #[inline]
    fn matches_optional(self, o: Option<Ordering>) -> bool {
        o.map_or(self == CmpOp::Ne, |o| self.matches(o))
    }

    /// Whether `ordering` (the result of comparing two **present** values)
    /// satisfies this op.
    pub fn matches(self, o: Ordering) -> bool {
        match self {
            CmpOp::Eq => o == Ordering::Equal,
            CmpOp::Ne => o != Ordering::Equal,
            CmpOp::Lt => o == Ordering::Less,
            CmpOp::Le => o != Ordering::Greater,
            CmpOp::Gt => o == Ordering::Greater,
            CmpOp::Ge => o != Ordering::Less,
        }
    }
}

mod view;
pub use view::ColumnView;
mod build;
mod cast;
mod fold;
pub use fold::CombineOp;
mod ops;
mod order;
mod transform;

impl Column {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Column { data: ColumnData::F64(v) } => v.len(),
            Column { data: ColumnData::F32(v) } => v.len(),
            Column { data: ColumnData::Bool(v, _) } => v.len(),
            Column { data: ColumnData::I64(v, _) } => v.len(),
            Column { data: ColumnData::I32(v, _) } => v.len(),
            Column { data: ColumnData::Str(v, _) } => v.len(),
            Column { data: ColumnData::Datetime(v) } => v.len(),
        }
    }

    /// Whether the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The logical dtype of the column.
    pub fn dtype(&self) -> DType {
        match self {
            Column { data: ColumnData::F64(_) } => DType::F64,
            Column { data: ColumnData::F32(_) } => DType::F32,
            Column { data: ColumnData::Bool(_, _) } => DType::Bool,
            Column { data: ColumnData::I64(_, _) } => DType::I64,
            Column { data: ColumnData::I32(_, _) } => DType::I32,
            Column { data: ColumnData::Str(_, _) } => DType::Utf8,
            Column { data: ColumnData::Datetime(_) } => DType::Datetime,
        }
    }

    /// Whether the value at `i` is present (not `volas.NA`). Each dtype reads its
    /// natural missing representation: a float `NaN`, an int/bool validity mask,
    /// or a datetime `NaT` (`i64::MIN`). `i` is assumed in bounds.
    pub fn is_valid(&self, i: usize) -> bool {
        match self {
            Column { data: ColumnData::F64(v) } => !v[i].is_nan(),
            Column { data: ColumnData::F32(v) } => !v[i].is_nan(),
            Column { data: ColumnData::Bool(_, val) }
            | Column { data: ColumnData::I64(_, val) }
            | Column { data: ColumnData::I32(_, val) }
            | Column { data: ColumnData::Str(_, val) } => val.is_valid(i),
            Column { data: ColumnData::Datetime(v) } => v[i] != i64::MIN,
        }
    }

    /// Read cell `i` of an integer-like column (int / bool / datetime) as its exact
    /// `i64`, ignoring validity (callers check presence first). A float has no
    /// exact integer reading — it converts through
    /// [`convert_lossless`](Self::convert_lossless) — and a str none at all, so
    /// callers only reach this for integer-like columns.
    pub(crate) fn get_i64(&self, i: usize) -> i64 {
        match self {
            Column { data: ColumnData::I64(v, _) } => v[i],
            Column { data: ColumnData::I32(v, _) } => v[i] as i64,
            Column { data: ColumnData::Datetime(v) } => v[i],
            Column { data: ColumnData::Bool(v, _) } => v[i] as i64,
            Column { data: ColumnData::F64(_) } | Column { data: ColumnData::F32(_) } | Column { data: ColumnData::Str(..) } => {
                unreachable!("an integer reading of a {} cell", self.dtype()) // LCOV_EXCL_LINE
            }
        }
    }

    /// Debug-build layout check of a nullable column's validity against its values
    /// (compiled out of release builds; see [`Validity::debug_check`]).
    #[cfg(debug_assertions)]
    pub(crate) fn debug_check(&self, name: &str) {
        match self {
            Column { data: ColumnData::Bool(v, val) } => val.debug_check(v.len(), name),
            Column { data: ColumnData::I64(v, val) } => val.debug_check(v.len(), name),
            Column { data: ColumnData::I32(v, val) } => val.debug_check(v.len(), name),
            Column { data: ColumnData::Str(v, val) } => val.debug_check(v.len(), name),
            Column { data: ColumnData::F64(_) } | Column { data: ColumnData::F32(_) } | Column { data: ColumnData::Datetime(_) } => {}
        }
    }

    /// Count of missing (`volas.NA`) values.
    pub fn null_count(&self) -> usize {
        match self {
            Column { data: ColumnData::F64(v) } => v.iter().filter(|x| x.is_nan()).count(),
            Column { data: ColumnData::F32(v) } => v.iter().filter(|x| x.is_nan()).count(),
            Column { data: ColumnData::Bool(_, val) }
            | Column { data: ColumnData::I64(_, val) }
            | Column { data: ColumnData::I32(_, val) }
            | Column { data: ColumnData::Str(_, val) } => val.null_count(),
            Column { data: ColumnData::Datetime(v) } => v.iter().filter(|&&x| x == i64::MIN).count(),
        }
    }

    /// Whether a cell is missing, using the buffer's NaN fact for floats and
    /// the cached null count for nullable columns. Unlike `has_nan`, present
    /// strings are values here, not NaNs from a numeric conversion.
    pub fn has_missing(&self) -> bool {
        match self {
            Column { data: ColumnData::F64(v) } => v.has_nan(),
            Column { data: ColumnData::F32(v) } => v.has_nan(),
            Column { data: ColumnData::Bool(_, val) } | Column { data: ColumnData::I64(_, val) } | Column { data: ColumnData::I32(_, val) }
            | Column { data: ColumnData::Str(_, val) } => val.has_nulls(),
            Column { data: ColumnData::Datetime(v) } => v.contains(&i64::MIN),
        }
    }

    /// Attach a validity to a (nullable) column produced by a value-only kernel,
    /// so an element-wise transform carries the input's missing positions through.
    /// A float/str/datetime column is returned unchanged (its missing lives in the
    /// values, not a side mask).
    fn with_validity(self, validity: Validity) -> Column {
        match self {
            Column { data: ColumnData::I64(v, _) } => Column { data: ColumnData::I64(v, validity) },
            Column { data: ColumnData::I32(v, _) } => Column { data: ColumnData::I32(v, validity) },
            // bool transforms build their column with the mask directly; float /
            // str / datetime carry missing in the values, so leave them unchanged.
            other => other,
        }
    }

    /// The validity of a nullable (int / bool) column, else `None` (a float column
    /// carries its missing in `NaN`, not a mask).
    fn validity(&self) -> Option<&Validity> {
        match self {
            Column { data: ColumnData::Bool(_, val) }
            | Column { data: ColumnData::I64(_, val) }
            | Column { data: ColumnData::I32(_, val) }
            | Column { data: ColumnData::Str(_, val) } => Some(val),
            _ => None,
        }
    }

    /// Borrow the underlying `f64` slice, if this is an `F64` column.
    pub fn as_f64(&self) -> Option<&[f64]> {
        if let Column { data: ColumnData::F64(v) } = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `bool` slice, if this is a `Bool` column.
    pub fn as_bool(&self) -> Option<&[bool]> {
        if let Column { data: ColumnData::Bool(v, _) } = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `i64` slice, if this is an `I64` column.
    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Column { data: ColumnData::I64(v, _) } = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// The `i`-th string of a `Str` column as `&str`, if this is a `Str` column.
    pub fn str_at(&self, i: usize) -> Option<&str> {
        if let Column { data: ColumnData::Str(v, _) } = self {
            Some(v.get(i))
        } else {
            None
        }
    }

    /// Borrow the underlying epoch-ns slice, if this is a `Datetime` column.
    pub fn as_datetime(&self) -> Option<&[i64]> {
        if let Column { data: ColumnData::Datetime(v) } = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Whether the values read as `f64` ([`to_f64_vec`](Self::to_f64_vec)) hold a NaN:
    /// a float NaN, or a missing int / bool / datetime cell. For a float column this is
    /// the buffer's remembered fact ([`Buffer::has_nan`]) — one scan, then free until
    /// the values change.
    pub fn has_nan(&self) -> bool {
        match self {
            Column { data: ColumnData::F64(v) } => v.has_nan(),
            Column { data: ColumnData::F32(v) } => v.has_nan(),
            Column { data: ColumnData::Bool(_, val) } | Column { data: ColumnData::I64(_, val) } | Column { data: ColumnData::I32(_, val) } => val.has_nulls(),
            Column { data: ColumnData::Str(v, _) } => !v.is_empty(),
            Column { data: ColumnData::Datetime(v) } => v.contains(&i64::MIN),
        }
    }

    /// [`has_nan`](Self::has_nan) when it is known without a scan: always for an int /
    /// bool column (its validity says), for a float column once its buffer remembers.
    pub fn known_has_nan(&self) -> Option<bool> {
        match self {
            Column { data: ColumnData::F64(v) } => v.known_has_nan(),
            Column { data: ColumnData::F32(v) } => v.known_has_nan(),
            Column { data: ColumnData::Bool(_, val) } | Column { data: ColumnData::I64(_, val) } | Column { data: ColumnData::I32(_, val) } => {
                Some(val.has_nulls())
            }
            Column { data: ColumnData::Str(..) } | Column { data: ColumnData::Datetime(_) } => None,
        }
    }

    /// Whether an input column contains a missing computation value (including ±inf).
    pub fn has_nonfinite(&self) -> bool {
        match self {
            Column { data: ColumnData::F64(v) } => v.has_nonfinite(),
            Column { data: ColumnData::F32(v) } => v.has_nonfinite(),
            _ => self.has_nan(),
        }
    }

    /// The all-finite computation fact, if already known without scanning.
    pub fn known_has_nonfinite(&self) -> Option<bool> {
        match self {
            Column { data: ColumnData::F64(v) } => v.known_has_nonfinite(),
            Column { data: ColumnData::F32(v) } => v.known_has_nonfinite(),
            _ => self.known_has_nan(),
        }
    }

    /// Materialize the values as `f64` (`bool` -> 0.0/1.0, `i64` / `datetime` ->
    /// as f64, `str` -> NaN). Used to feed indicator kernels, which operate on
    /// `f64`.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Column { data: ColumnData::F64(v) } => v.to_vec(),
            Column { data: ColumnData::F32(v) } => v.iter().map(|&x| x as f64).collect(),
            Column { data: ColumnData::Bool(v, val) } => mask_f64(v.iter().map(|&b| if b { 1.0 } else { 0.0 }), val),
            Column { data: ColumnData::I64(v, val) } => mask_f64(v.iter().map(|&i| i as f64), val),
            Column { data: ColumnData::I32(v, val) } => mask_f64(v.iter().map(|&i| i as f64), val),
            Column { data: ColumnData::Str(v, _) } => vec![f64::NAN; v.len()],
            Column { data: ColumnData::Datetime(v) } => v
                .iter()
                .map(|&i| if i == i64::MIN { f64::NAN } else { i as f64 })
                .collect(),
        }
    }

    /// The values as `f64` for an operation computed in float64 (`op` names it in
    /// the error): mean / var / std / quantiles / moments / corr / cov / the math
    /// transforms / true division / the window aggregations. A present int64 with
    /// `|x| > 2^53` raises ([`outside_f64_domain`], contract #26) — its float64
    /// image is not the value, so the result would not be a function of the input.
    /// int32 / bool / float inputs always fit; missing cells read as `NaN`.
    pub fn to_f64_domain(&self, op: &str) -> Result<Vec<f64>> {
        self.check_f64_domain(op)?;
        Ok(self.to_f64_vec())
    }

    /// The check of [`to_f64_domain`](Self::to_f64_domain) alone, for a reader that
    /// converts later (or never, for an `F64` column it borrows).
    pub fn check_f64_domain(&self, op: &str) -> Result<()> {
        if let Column { data: ColumnData::I64(v, val) } = self {
            if let Some(x) = present(v, val).find(|&x| !in_f64_domain(x)) {
                return Err(outside_f64_domain(x, op));
            }
        }
        Ok(())
    }

    /// Value at position `i` coerced to `f64`, **ignoring validity** (a missing
    /// cell reads its raw placeholder); callers check validity first (e.g.
    /// [`truth_at`](Self::truth_at)). NumPy export goes through `to_f64_vec`.
    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            Column { data: ColumnData::F64(v) } => v[i],
            Column { data: ColumnData::F32(v) } => v[i] as f64,
            Column { data: ColumnData::Bool(v, _) } => {
                if v[i] {
                    1.0
                } else {
                    0.0
                }
            }
            Column { data: ColumnData::I64(v, _) } => v[i] as f64,
            Column { data: ColumnData::I32(v, _) } => v[i] as f64,
            Column { data: ColumnData::Str(_, _) } => f64::NAN,
            Column { data: ColumnData::Datetime(v) } => v[i] as f64,
        }
    }

    /// The column as `bool` values (a `Bool` column directly; else an error).
    fn as_bool_vec(&self) -> Result<Vec<bool>> {
        match self {
            Column { data: ColumnData::Bool(v, _) } => Ok(v.to_vec()),
            other => Err(VolasError::DType(format!(
                "expected a bool column, got {}",
                other.dtype()
            ))),
        }
    }

    /// The column as `i64` values: an `I64` column directly (exact), otherwise via
    /// lossless narrowing (errors if any value is non-integral / out of range).
    /// Used by the int paths of `select` / `binary`, where the caller has already
    /// established the values fit.
    fn as_i64_vec(&self) -> Result<Vec<i64>> {
        match self {
            Column { data: ColumnData::I64(v, _) } => Ok(v.to_vec()),
            _ => self
                .to_f64_vec()
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if self.is_valid(i) {
                        i64::try_from_f64(x).ok_or_else(|| {
                            VolasError::DType(format!("value {x} does not fit int64"))
                        })
                    } else {
                        Ok(0) // placeholder for a missing value (masked by the result validity)
                    }
                })
                .collect(),
        }
    }

    /// The column's `String` values (a `Str` column directly); errors otherwise.
    /// Used by `select` so a str `where` / `mask` keeps its values dtype-preserving.
    fn as_str_vec(&self) -> Result<Vec<String>> {
        match self {
            Column { data: ColumnData::Str(v, _) } => Ok(v.to_vec()),
            other => Err(VolasError::DType(format!(
                "cannot select a {} column as str",
                other.dtype()
            ))),
        }
    }

    /// The column's epoch-ns values (a `Datetime` column directly); errors otherwise.
    fn as_datetime_vec(&self) -> Result<Vec<i64>> {
        match self {
            Column { data: ColumnData::Datetime(v) } => Ok(v.to_vec()),
            other => Err(VolasError::DType(format!(
                "cannot select a {} column as datetime",
                other.dtype()
            ))),
        }
    }

    /// The column as `i32` values: an `I32` column directly, a `Bool` as 0/1,
    /// otherwise via lossless narrowing (errors if out of range / non-integral).
    fn as_i32_vec(&self) -> Result<Vec<i32>> {
        match self {
            Column { data: ColumnData::I32(v, _) } => Ok(v.to_vec()),
            Column { data: ColumnData::Bool(v, _) } => Ok(v.iter().map(|&b| b as i32).collect()),
            _ => self
                .to_f64_vec()
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if self.is_valid(i) {
                        i32::try_from_f64(x).ok_or_else(|| {
                            VolasError::DType(format!("value {x} does not fit int32"))
                        })
                    } else {
                        Ok(0) // placeholder for a missing value (masked by the result validity)
                    }
                })
                .collect(),
        }
    }

    /// The column as `f32` values (an `F32` column directly; else converted).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Column { data: ColumnData::F32(v) } => v.to_vec(),
            _ => self.to_f64_vec().iter().map(|&x| x as f32).collect(),
        }
    }

    /// Render each value as a `String` (for `astype(str)`).
    fn to_string_vec(&self) -> Vec<String> {
        match self {
            Column { data: ColumnData::Str(v, _) } => v.to_vec(),
            Column { data: ColumnData::F64(v) } => v.iter().map(|x| x.to_string()).collect(),
            Column { data: ColumnData::F32(v) } => v.iter().map(|x| x.to_string()).collect(),
            Column { data: ColumnData::I64(v, _) } => v.iter().map(|x| x.to_string()).collect(),
            Column { data: ColumnData::I32(v, _) } => v.iter().map(|x| x.to_string()).collect(),
            Column { data: ColumnData::Bool(v, _) } => v
                .iter()
                .map(|&b| if b { "True" } else { "False" }.to_string())
                .collect(),
            Column { data: ColumnData::Datetime(v) } => v.iter().map(|&ns| datetime::format_ns(ns)).collect(),
        }
    }
}

/// Collect a value iterator to `f64`, mapping missing positions to `NaN` so the
/// f64-funnel reductions (mean / std / median …) skip them. Dense ⇒ a plain
/// collect with no per-element validity check, so the indicator feed path stays
/// unchanged.
fn mask_f64(vals: impl Iterator<Item = f64>, validity: &Validity) -> Vec<f64> {
    if validity.has_nulls() {
        vals.enumerate()
            .map(|(i, x)| if validity.is_valid(i) { x } else { f64::NAN })
            .collect()
    } else {
        vals.collect()
    }
}

/// A bool / i32 column widened to `i64` (for the i64-accumulator reductions).
fn widen_i64<T: Copy + Into<i64>>(v: &[T]) -> Vec<i64> {
    v.iter().map(|&x| x.into()).collect()
}

/// Shift `v` by `n` (pandas `shift`), filling vacated cells with `fill`; the kept
/// region moves as a single `memcpy`.
fn shift_fill<T: Copy>(v: &[T], n: isize, fill: T) -> Vec<T> {
    let len = v.len();
    let mut out = vec![fill; len];
    if n >= 0 {
        let n = (n as usize).min(len);
        out[n..].copy_from_slice(&v[..len - n]);
    } else {
        let n = ((-n) as usize).min(len);
        out[..len - n].copy_from_slice(&v[n..]);
    }
    out
}

/// `x[i] - x[i-n]` (pandas `diff`) in one pass; the shift-gap cells are `missing`.
fn diff_kernel<T: Copy + std::ops::Sub<Output = T>>(v: &[T], n: isize, missing: T) -> Vec<T> {
    let len = v.len();
    let mut out = vec![missing; len];
    if n >= 0 {
        let k = n as usize;
        for i in k..len {
            out[i] = v[i] - v[i - k];
        }
    } else {
        let k = (-n) as usize;
        for i in 0..len.saturating_sub(k) {
            out[i] = v[i] - v[i + k];
        }
    }
    out
}

/// Extend `av` (validity of an `alen`-long column) with `bv` (a `blen`-long
/// column's validity) so it stays aligned after an `append`. Dense + dense is a
/// no-op (the result is still fully present).
fn append_validity(av: &mut Validity, alen: usize, bv: &Validity, blen: usize) {
    if !av.has_nulls() && !bv.has_nulls() {
        return;
    }
    let flags: Vec<bool> = (0..alen)
        .map(|i| av.is_valid(i))
        .chain((0..blen).map(|i| bv.is_valid(i)))
        .collect();
    *av = Validity::from_valid_iter(alen + blen, flags);
}

/// Sum of the present values in their element type (skip `volas.NA`), checked
/// (an overflow is an error, contract #25). Dense ⇒ the plain kernel.
fn sum_valid<T: Numeric>(v: &[T], val: &Validity) -> Result<T> {
    if !val.has_nulls() {
        return stats::sum(v);
    }
    stats::fold(present(v, val), T::ZERO, T::checked_add, "sum")
}

/// Product of the present values (skip `volas.NA`), checked. Dense ⇒ the plain
/// kernel.
fn prod_valid<T: Numeric>(v: &[T], val: &Validity) -> Result<T> {
    if !val.has_nulls() {
        return stats::prod(v);
    }
    stats::fold(present(v, val), T::ONE, T::checked_mul, "prod")
}

/// The present values of `v` under `val`, in order.
fn present<'a, T: Copy>(v: &'a [T], val: &'a Validity) -> impl Iterator<Item = T> + 'a {
    v.iter().enumerate().filter(|(i, _)| val.is_valid(*i)).map(|(_, &x)| x)
}

/// A checked element-wise map over the present values; a missing position keeps
/// its placeholder (masked by the carried validity), so a payload under a null
/// never raises. Backs the integer `abs` / `round`.
fn map_present<T: Numeric>(
    v: &[T],
    val: &Validity,
    f: impl Fn(T) -> Option<T>,
    op: &str,
) -> Result<Vec<T>> {
    let dense = !val.has_nulls();
    v.iter()
        .enumerate()
        .map(|(i, &x)| {
            if dense || val.is_valid(i) {
                f(x).ok_or_else(|| overflow(T::DTYPE, op))
            } else {
                Ok(x)
            }
        })
        .collect()
}

/// Min / max of the present values (skip `volas.NA`); `None` when none present.
fn extreme_valid<T: Numeric>(v: &[T], val: &Validity, want_max: bool) -> Option<T> {
    if !val.has_nulls() {
        return stats::extreme(v, want_max);
    }
    let mut it = v
        .iter()
        .enumerate()
        .filter(|(i, _)| val.is_valid(*i))
        .map(|(_, &x)| x);
    let first = it.next()?;
    Some(if want_max {
        it.fold(first, |a, x| if x > a { x } else { a })
    } else {
        it.fold(first, |a, x| if x < a { x } else { a })
    })
}

/// Running cumulative `op` over present values; a missing position gets
/// `placeholder` (masked by the carried validity, so its value is irrelevant). A
/// `None` from `op` is a failed (overflowing) step and fails the whole fold.
fn cum_valid<T: Copy>(
    v: &[T],
    val: &Validity,
    placeholder: T,
    op: impl Fn(T, T) -> Option<T>,
) -> Option<Vec<T>> {
    let mut acc: Option<T> = None;
    v.iter()
        .enumerate()
        .map(|(i, &x)| {
            if val.is_valid(i) {
                let next = match acc {
                    None => x,
                    Some(a) => op(a, x)?,
                };
                acc = Some(next);
                Some(next)
            } else {
                Some(placeholder)
            }
        })
        .collect()
}

/// Dtype-preserving cumulative for a numeric column: the dense kernel when fully
/// present, else a skip-and-propagate fold carrying the input validity through.
/// `op` is checked: a `None` step is an overflow error named `name` (#25).
fn cum<T: Numeric>(
    v: &[T],
    val: &Validity,
    dense: fn(&[T]) -> Result<Vec<T>>,
    op: fn(T, T) -> Option<T>,
    name: &str,
) -> Result<Column> {
    let out = if val.has_nulls() {
        cum_valid(v, val, T::ZERO, op).ok_or_else(|| overflow(T::DTYPE, name))?
    } else {
        dense(v)?
    };
    Ok(T::into_column(out).with_validity(val.clone()))
}

/// Clamp each element to `[lo, hi]` (either optional); missing passes through.
/// Bounds are narrowed to `T` losslessly (the caller only stays in an int dtype
/// when the bounds fit it).
fn clip_vec<T: Numeric>(v: &[T], lo: Option<f64>, hi: Option<f64>) -> Vec<T> {
    let lo = lo.and_then(T::try_from_f64);
    let hi = hi.and_then(T::try_from_f64);
    v.iter()
        .map(|&x| {
            if x.is_missing() {
                return x;
            }
            let mut y = x;
            if let Some(l) = lo {
                if y < l {
                    y = l;
                }
            }
            if let Some(h) = hi {
                if y > h {
                    y = h;
                }
            }
            y
        })
        .collect()
}

/// Integer floor division (toward negative infinity, unlike Rust's truncating
/// `/`); `None` for the one overflowing quotient, `i64::MIN // -1` (contract #25;
/// NumPy wraps it). The divisor is non-zero (callers route a zero divisor through
/// the float path).
fn ifloordiv(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    let r = a % b; // cannot overflow: the only overflowing pair failed above
    // a remainder of the opposite sign means `q` was truncated up: q - 1 >= i64::MIN
    Some(if r != 0 && (r < 0) != (b < 0) { q - 1 } else { q })
}

/// A hashable key for a float value, or `None` for `NaN` (which groups as NA).
/// `-0.0` is canonicalized to `+0.0` so the two zeros count as one value.
fn float_key(x: f64) -> Option<u64> {
    if x.is_nan() {
        None
    } else if x == 0.0 {
        Some(0)
    } else {
        Some(x.to_bits())
    }
}

/// Group `0..len` by `key(i)` (a `None` key is the single NA group), returning
/// `(first_index, count, is_na)` per distinct value in order of first appearance.
fn group_by<K: Eq + std::hash::Hash>(
    len: usize,
    key: impl Fn(usize) -> Option<K>,
) -> Vec<(usize, usize, bool)> {
    let mut order: Vec<(usize, usize, bool)> = Vec::new();
    let mut seen: HashMap<K, usize> = HashMap::new();
    let mut na: Option<usize> = None;
    for i in 0..len {
        match key(i) {
            None => match na {
                Some(g) => order[g].1 += 1,
                None => {
                    na = Some(order.len());
                    order.push((i, 1, true));
                }
            },
            Some(k) => match seen.get(&k) {
                Some(&g) => order[g].1 += 1,
                None => {
                    seen.insert(k, order.len());
                    order.push((i, 1, false));
                }
            },
        }
    }
    order
}

/// Running OR (`or = true`, backs bool `cummax`) / running AND (`cummin`).
fn bool_running(v: &[bool], or: bool) -> Vec<bool> {
    let mut acc = !or; // OR seeds false; AND seeds true
    v.iter()
        .map(|&b| {
            acc = if or { acc || b } else { acc && b };
            acc
        })
        .collect()
}

/// Clamp a bool column (pandas `clip`): a `True` lower bound forces all true, a
/// `False` upper bound forces all false, otherwise unchanged (bool has only the
/// two values, so the bounds decide the whole column).
fn clip_bool(v: &[bool], lo: Option<f64>, hi: Option<f64>) -> Vec<bool> {
    let force_true = lo.is_some_and(|x| x != 0.0);
    let force_false = hi == Some(0.0);
    v.iter()
        .map(|&b| {
            if force_false {
                false
            } else if force_true {
                true
            } else {
                b
            }
        })
        .collect()
}

/// Mixed-dtype arithmetic retains the original checked-kernel path.
fn float_kernel<T: Numeric>(a: &[T], b: &[T], op: BinOp) -> Vec<T> {
    binary_kernel(a, b, op, &Validity::dense()).expect("float arithmetic never overflows")
}

/// Dispatch once, then write IEEE arithmetic directly into the final buffer.
fn float_slice_kernel<T>(a: &[T], b: &[T], op: BinOp) -> Vec<T>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    match op {
        BinOp::Add => float_output(a, b, |x, y| x + y),
        BinOp::Sub => float_output(a, b, |x, y| x - y),
        BinOp::Mul => float_output(a, b, |x, y| x * y),
    }
}

/// Statically dispatched arithmetic; no input copies or per-row Result machinery.
fn float_output<T: Copy, U>(a: &[T], b: &[T], f: impl Fn(T, T) -> U) -> Vec<U> {
    let mut out = Vec::with_capacity(a.len());
    out.extend(a.iter().zip(b).map(|(&x, &y)| f(x, y)));
    out
}

/// Element-wise `a ∘ b` for `Add` / `Sub` / `Mul` over the positions `valid`
/// marks present: checked, so an integer overflow at a present position is an
/// error (contract #25); a missing position holds a `0` placeholder (masked by the
/// result validity) and is never computed, so its payload cannot raise.
fn binary_kernel<T: Numeric>(a: &[T], b: &[T], op: BinOp, valid: &Validity) -> Result<Vec<T>> {
    type Op<T> = fn(T, T) -> Option<T>;
    let (f, name): (Op<T>, &str) = match op {
        BinOp::Add => (T::checked_add, "`+`"),
        BinOp::Sub => (T::checked_sub, "`-`"),
        BinOp::Mul => (T::checked_mul, "`*`"),
    };
    let dense = !valid.has_nulls();
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (&x, &y))| {
            if dense || valid.is_valid(i) {
                f(x, y).ok_or_else(|| overflow(T::DTYPE, name))
            } else {
                Ok(T::ZERO)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_na;
#[cfg(test)]
mod tests_overflow;
#[cfg(test)]
mod tests_convert;
#[cfg(test)]
mod tests_exact;
#[cfg(test)]
mod tests_fold;
#[cfg(test)]
mod tests_layout;
#[cfg(test)]
mod tests_arithmetic_bits;
