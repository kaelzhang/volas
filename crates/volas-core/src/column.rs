//! Column: a single typed, contiguous buffer.
//!
//! Each variant holds its buffer behind an `Arc`, so a `Column` — and the
//! `DataFrame` / `Series` that contain it — is cheap to clone: cloning shares the
//! buffer (an O(1) refcount bump, not an O(n) copy). Mutation (`append`) is
//! copy-on-write via `Arc::make_mut`: it grows the `Vec` in place when the buffer
//! is uniquely owned, and copies only when a view (another `Series`, a zero-copy
//! export) is still alive. `F64` columns use `NaN` for missing values (matching
//! stock-pandas / pandas semantics).

use std::sync::Arc;

use crate::datetime;
use crate::dtype::DType;
use crate::error::{Result, VolasError};
use crate::numeric::{binary_supertype, fits, Numeric};
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
            Column::F64(buf) => {
                let $slice: &[f64] = buf.as_slice();
                Ok($body)
            }
            Column::F32(buf) => {
                let $slice: &[f32] = buf.as_slice();
                Ok($body)
            }
            Column::I64(buf, _) => {
                let $slice: &[i64] = buf.as_slice();
                Ok($body)
            }
            Column::I32(buf, _) => {
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

/// A typed, contiguous column of values. The buffer is `Arc`-shared (cheap clone)
/// and mutated copy-on-write.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// 64-bit floats; `NaN` denotes missing.
    F64(Arc<Vec<f64>>),
    /// 32-bit floats (narrow storage); `NaN` denotes missing.
    F32(Arc<Vec<f32>>),
    /// Booleans (comparison / signal results); `Validity` marks missing cells.
    Bool(Arc<Vec<bool>>, Validity),
    /// 64-bit signed integers; `Validity` marks missing cells.
    I64(Arc<Vec<i64>>, Validity),
    /// 32-bit signed integers (narrow storage); `Validity` marks missing cells.
    I32(Arc<Vec<i32>>, Validity),
    /// UTF-8 strings.
    Str(Arc<Vec<String>>),
    /// Datetimes as i64 nanoseconds since the Unix epoch (UTC-naive).
    Datetime(Arc<Vec<i64>>),
}

/// A scalar to assign into a column (boolean-mask / positional assignment). Kept
/// distinct from a plain `f64` so a real `bool` can be told from a number — they
/// fit different dtypes (pandas rejects a number into a bool column, and an
/// integral number stays in an int column while a real bool does not change it).
#[derive(Clone, Copy, Debug)]
pub enum SetVal {
    /// A boolean scalar.
    Bool(bool),
    /// A numeric scalar (an integral, finite value can stay in an int column).
    Num(f64),
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

impl Column {
    /// Build an `F64` column.
    pub fn f64(v: Vec<f64>) -> Column {
        Column::F64(Arc::new(v))
    }
    /// Build an `F32` column.
    pub fn f32(v: Vec<f32>) -> Column {
        Column::F32(Arc::new(v))
    }
    /// Build an `I32` column (all values present).
    pub fn i32(v: Vec<i32>) -> Column {
        Column::I32(Arc::new(v), Validity::dense())
    }
    /// Build a `Bool` column (all values present).
    pub fn bool(v: Vec<bool>) -> Column {
        Column::Bool(Arc::new(v), Validity::dense())
    }
    /// Build an `I64` column (all values present).
    pub fn i64(v: Vec<i64>) -> Column {
        Column::I64(Arc::new(v), Validity::dense())
    }
    /// Build a `Str` column.
    pub fn str(v: Vec<String>) -> Column {
        Column::Str(Arc::new(v))
    }
    /// Build a `Datetime` column (epoch nanoseconds).
    pub fn datetime(v: Vec<i64>) -> Column {
        Column::Datetime(Arc::new(v))
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::F32(v) => v.len(),
            Column::Bool(v, _) => v.len(),
            Column::I64(v, _) => v.len(),
            Column::I32(v, _) => v.len(),
            Column::Str(v) => v.len(),
            Column::Datetime(v) => v.len(),
        }
    }

    /// Whether the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The logical dtype of the column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::F64(_) => DType::F64,
            Column::F32(_) => DType::F32,
            Column::Bool(_, _) => DType::Bool,
            Column::I64(_, _) => DType::I64,
            Column::I32(_, _) => DType::I32,
            Column::Str(_) => DType::Utf8,
            Column::Datetime(_) => DType::Datetime,
        }
    }

    /// Borrow the underlying `f64` slice, if this is an `F64` column.
    pub fn as_f64(&self) -> Option<&[f64]> {
        if let Column::F64(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `bool` slice, if this is a `Bool` column.
    pub fn as_bool(&self) -> Option<&[bool]> {
        if let Column::Bool(v, _) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `i64` slice, if this is an `I64` column.
    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Column::I64(v, _) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `String` slice, if this is a `Str` column.
    pub fn as_str(&self) -> Option<&[String]> {
        if let Column::Str(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying epoch-ns slice, if this is a `Datetime` column.
    pub fn as_datetime(&self) -> Option<&[i64]> {
        if let Column::Datetime(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Materialize the values as `f64` (`bool` -> 0.0/1.0, `i64` / `datetime` ->
    /// as f64, `str` -> NaN). Used to feed indicator kernels, which operate on
    /// `f64`.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Column::F64(v) => v.to_vec(),
            Column::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Bool(v, _) => v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
            Column::I64(v, _) => v.iter().map(|&i| i as f64).collect(),
            Column::I32(v, _) => v.iter().map(|&i| i as f64).collect(),
            Column::Str(v) => vec![f64::NAN; v.len()],
            Column::Datetime(v) => v.iter().map(|&i| i as f64).collect(),
        }
    }

    /// Value at position `i` coerced to `f64` (for NumPy 2-D export).
    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            Column::F64(v) => v[i],
            Column::F32(v) => v[i] as f64,
            Column::Bool(v, _) => {
                if v[i] {
                    1.0
                } else {
                    0.0
                }
            }
            Column::I64(v, _) => v[i] as f64,
            Column::I32(v, _) => v[i] as f64,
            Column::Str(_) => f64::NAN,
            Column::Datetime(v) => v[i] as f64,
        }
    }

    /// A contiguous `[start, end)` slice (a fresh buffer).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::f64(v[start..end].to_vec()),
            Column::F32(v) => Column::f32(v[start..end].to_vec()),
            Column::Bool(v, _) => Column::bool(v[start..end].to_vec()),
            Column::I64(v, _) => Column::i64(v[start..end].to_vec()),
            Column::I32(v, _) => Column::i32(v[start..end].to_vec()),
            Column::Str(v) => Column::str(v[start..end].to_vec()),
            Column::Datetime(v) => Column::datetime(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::f64(idx.iter().map(|&i| v[i]).collect()),
            Column::F32(v) => Column::f32(idx.iter().map(|&i| v[i]).collect()),
            Column::Bool(v, _) => Column::bool(idx.iter().map(|&i| v[i]).collect()),
            Column::I64(v, _) => Column::i64(idx.iter().map(|&i| v[i]).collect()),
            Column::I32(v, _) => Column::i32(idx.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::str(idx.iter().map(|&i| v[i].clone()).collect()),
            Column::Datetime(v) => Column::datetime(idx.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Append another column of the same dtype, copy-on-write (grows in place
    /// when the buffer is uniquely owned).
    pub fn append(&mut self, other: &Column) -> Result<()> {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::F32(a), Column::F32(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Bool(a, _), Column::Bool(b, _)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::I64(a, _), Column::I64(b, _)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::I32(a, _), Column::I32(b, _)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Str(a), Column::Str(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Datetime(a), Column::Datetime(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (s, o) => Err(VolasError::DType(format!(
                "cannot append a {} column onto a {} column",
                o.dtype(),
                s.dtype()
            ))),
        }
    }

    /// Extend a stale computed column with placeholder missing values, avoiding a
    /// temporary one-row [`Column`] allocation on the live append path.
    pub fn append_missing(&mut self, len: usize) -> Result<()> {
        match self {
            Column::F64(v) => {
                Arc::make_mut(v).extend(std::iter::repeat(f64::NAN).take(len));
                Ok(())
            }
            Column::F32(v) => {
                Arc::make_mut(v).extend(std::iter::repeat(f32::NAN).take(len));
                Ok(())
            }
            Column::Bool(v, _) => {
                Arc::make_mut(v).extend(std::iter::repeat(false).take(len));
                Ok(())
            }
            other => Err(VolasError::DType(format!(
                "column type {} has no missing-value placeholder",
                other.dtype()
            ))),
        }
    }

    /// Parse this column into a [`Column::Datetime`] (epoch ns). `Str` cells are
    /// parsed via [`datetime::parse_ns`]; an already-`Datetime` column is shared
    /// back (cheap). Errors on an unparseable cell or an unsupported dtype.
    pub fn to_datetime(&self) -> Result<Column> {
        self.to_datetime_tz(crate::tz::Tz::Utc)
    }

    /// Parse a string column to a UTC `Datetime` column, interpreting **naive**
    /// strings in `tz` (offset-aware strings are absolute; an existing `Datetime`
    /// column is already UTC and returned as-is). The tz is then attached to the
    /// *index* (storage stays UTC).
    pub fn to_datetime_tz(&self, tz: crate::tz::Tz) -> Result<Column> {
        match self {
            Column::Datetime(_) => Ok(self.clone()),
            Column::Str(v) => {
                let mut out = Vec::with_capacity(v.len());
                for s in v.iter() {
                    let ns = datetime::parse_ns_in_tz(s, tz).ok_or_else(|| {
                        VolasError::Value(format!("could not parse datetime {s:?}"))
                    })?;
                    out.push(ns);
                }
                Ok(Column::datetime(out))
            }
            other => Err(VolasError::DType(format!(
                "cannot parse a {} column as datetime",
                other.dtype()
            ))),
        }
    }

    /// Interpret a numeric (epoch) column as a UTC `Datetime` column, scaling by
    /// `unit` (`"s"` / `"ms"` / `"us"` / `"ns"`). Float epochs are **truncated** to
    /// the whole `unit` (matching a NumPy / pandas `astype('datetime64[unit]')`
    /// cast). The robust ingestion path for exchange APIs that return numeric
    /// timestamps.
    pub fn epoch_to_datetime(&self, unit: &str) -> Result<Column> {
        self.epoch_to_datetime_with(unit, |x| datetime::epoch_to_ns(x as i64, unit))
    }

    /// Like [`epoch_to_datetime`](Self::epoch_to_datetime) but **rounds** float
    /// epochs to the nearest nanosecond, preserving sub-`unit` fractions (matching
    /// `pandas.to_datetime(..., unit=...)`). Identical for integer columns.
    pub fn epoch_to_datetime_rounded(&self, unit: &str) -> Result<Column> {
        self.epoch_to_datetime_with(unit, |x| datetime::epoch_to_ns_f64(x, unit))
    }

    /// Shared epoch → `Datetime` conversion; `f64_to_ns` chooses how float epochs
    /// map to nanoseconds (truncate vs round). Integers always scale exactly.
    fn epoch_to_datetime_with(
        &self,
        unit: &str,
        f64_to_ns: impl Fn(f64) -> Option<i64>,
    ) -> Result<Column> {
        match self {
            Column::I64(v, _) => v
                .iter()
                .map(|&x| {
                    datetime::epoch_to_ns(x, unit).ok_or_else(|| {
                        VolasError::Value(format!("invalid epoch unit {unit:?} or overflow"))
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::datetime),
            Column::F64(v) => v
                .iter()
                .map(|&x| {
                    f64_to_ns(x).ok_or_else(|| {
                        VolasError::Value(format!("invalid epoch unit {unit:?} or overflow"))
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::datetime),
            other => Err(VolasError::DType(format!(
                "cannot read a {} column as an epoch timestamp (need int64 / float64)",
                other.dtype()
            ))),
        }
    }

    /// Cast to another dtype (best-effort, pandas `astype`-like). A no-op when
    /// already the target dtype.
    pub fn cast(&self, to: DType) -> Result<Column> {
        if self.dtype() == to {
            return Ok(self.clone());
        }
        match to {
            DType::F64 => Ok(Column::f64(self.to_f64_vec())),
            DType::I64 => match self {
                Column::F64(v) => {
                    // pandas raises (IntCastingNaNError) rather than silently
                    // turning NaN -> 0 / inf -> i64::MAX, which corrupts data.
                    if let Some(x) = v.iter().copied().find(|x| !x.is_finite()) {
                        return Err(VolasError::Value(format!(
                            "cannot convert non-finite value ({x}) to int64 (NaN / inf); \
                             fill or drop it first"
                        )));
                    }
                    Ok(Column::i64(v.iter().map(|&x| x as i64).collect()))
                }
                Column::Bool(v, _) => Ok(Column::i64(v.iter().map(|&b| b as i64).collect())),
                Column::Datetime(v) => Ok(Column::i64(v.to_vec())),
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to int64",
                    other.dtype()
                ))),
            },
            DType::Bool => match self {
                Column::F64(v) => Ok(Column::bool(v.iter().map(|&x| x != 0.0).collect())),
                Column::I64(v, _) => Ok(Column::bool(v.iter().map(|&x| x != 0).collect())),
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to bool",
                    other.dtype()
                ))),
            },
            DType::F32 => Ok(Column::f32(self.to_f64_vec().iter().map(|&x| x as f32).collect())),
            DType::I32 => self
                .to_f64_vec()
                .iter()
                .map(|&x| {
                    i32::try_from_f64(x).ok_or_else(|| {
                        VolasError::Value(format!(
                            "cannot convert {x} to int32 (non-finite / non-integral / out of range)"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::i32),
            DType::Utf8 => Ok(Column::str(self.to_string_vec())),
            DType::Datetime => self.to_datetime(),
        }
    }

    /// Assign a scalar at the given positions, following pandas 3.0's in-place
    /// dtype rules: keep the column dtype when the value fits losslessly; upcast
    /// an int column to float for `NaN`; reject a lossy write (a non-integral
    /// number into an int column, or a number into a bool column) with a `DType`
    /// error — surfaces as `TypeError`, like pandas' `LossySetitemError`.
    /// `positions` are assumed in bounds (callers validate the mask / index).
    pub fn set_scalar_at(&self, positions: &[usize], value: SetVal) -> Result<Column> {
        match self {
            // A float column absorbs any value (with rounding for f32).
            Column::F64(v) => Ok(set_float_at(v, positions, value)),
            Column::F32(v) => Ok(set_float_at(v, positions, value)),
            // An int column keeps the value if it fits, upcasts to float for NaN,
            // and rejects a lossy (non-integral / out-of-range) write.
            Column::I64(v, _) => set_int_at(v, positions, value, "int64"),
            Column::I32(v, _) => set_int_at(v, positions, value, "int32"),
            Column::Bool(v, _) => match value {
                SetVal::Bool(b) => {
                    let mut nv = v.to_vec();
                    for &i in positions {
                        nv[i] = b;
                    }
                    Ok(Column::bool(nv))
                }
                SetVal::Num(x) => Err(VolasError::DType(format!(
                    "Invalid value '{x}' for dtype 'bool'"
                ))),
            },
            other => Err(VolasError::DType(format!(
                "cannot assign a scalar into a {} column",
                other.dtype()
            ))),
        }
    }

    // --- dtype-preserving numeric transforms (pandas 3.0) ---------------------
    // Each dispatches the kernel over the column's element type so an int column
    // stays int and computes natively (no f64 round-trip). A non-numeric column
    // is a `DType` error.

    /// Cumulative sum (pandas `cumsum`, skipna), dtype-preserving. A `bool` column
    /// sums to `int64` (counts trues), matching pandas.
    pub fn cumsum(&self) -> Result<Column> {
        if let Column::Bool(v, _) = self {
            return Ok(Column::i64(stats::cumsum(&bool_as_i64(v))));
        }
        numeric_dispatch!(self, v => Numeric::into_column(stats::cumsum(v)))
    }
    /// Cumulative maximum (pandas `cummax`), dtype-preserving. A `bool` column
    /// stays `bool` (running OR).
    pub fn cummax(&self) -> Result<Column> {
        if let Column::Bool(v, _) = self {
            return Ok(Column::bool(bool_running(v, true)));
        }
        numeric_dispatch!(self, v => Numeric::into_column(stats::cummax(v)))
    }
    /// Cumulative minimum (pandas `cummin`), dtype-preserving. A `bool` column
    /// stays `bool` (running AND).
    pub fn cummin(&self) -> Result<Column> {
        if let Column::Bool(v, _) = self {
            return Ok(Column::bool(bool_running(v, false)));
        }
        numeric_dispatch!(self, v => Numeric::into_column(stats::cummin(v)))
    }
    /// Cumulative product (pandas `cumprod`), dtype-preserving. A `bool` column
    /// products to `int64`, matching pandas.
    pub fn cumprod(&self) -> Result<Column> {
        if let Column::Bool(v, _) = self {
            return Ok(Column::i64(stats::cumprod(&bool_as_i64(v))));
        }
        numeric_dispatch!(self, v => Numeric::into_column(stats::cumprod(v)))
    }

    /// Element-wise absolute value (pandas `abs`), dtype-preserving. `abs` of a
    /// missing (`NaN`) stays missing; `abs(i64::MIN)` wraps to `i64::MIN` (pandas);
    /// a `bool` column is unchanged (`abs(bool) == bool`).
    pub fn abs(&self) -> Result<Column> {
        if matches!(self, Column::Bool(_, _)) {
            return Ok(self.clone());
        }
        numeric_dispatch!(self, v => Numeric::into_column(stats::abs(v)))
    }

    /// Round to `decimals` places (pandas `round`), dtype-preserving: banker's
    /// (half-to-even) for floats, and for ints an identity at `decimals >= 0` or a
    /// banker's round to the nearest power-of-ten multiple at negative `decimals`.
    pub fn round(&self, decimals: i32) -> Result<Column> {
        match self {
            Column::F64(v) => Ok(Column::f64(v.iter().map(|&x| round_f64(x, decimals)).collect())),
            Column::F32(v) => {
                Ok(Column::f32(v.iter().map(|&x| round_f64(x as f64, decimals) as f32).collect()))
            }
            Column::I64(v, _) => Ok(Column::i64(v.iter().map(|&x| round_i64(x, decimals)).collect())),
            Column::I32(v, _) => Ok(Column::i32(
                v.iter().map(|&x| round_i64(x as i64, decimals) as i32).collect(),
            )),
            Column::Bool(_, _) => Ok(self.clone()), // round(bool) == bool (pandas no-op)
            other => Err(VolasError::DType(format!("cannot round a {} column", other.dtype()))),
        }
    }

    /// Clamp to `[lower, upper]` (either bound optional), pandas `clip`. Stays in
    /// the column dtype when every present bound fits it losslessly; otherwise (an
    /// int column with a non-integral bound) promotes to float, matching pandas.
    pub fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> Result<Column> {
        match self {
            // bool stays bool (pandas): a True lower bound forces all true, a
            // False upper bound forces all false, otherwise unchanged.
            Column::Bool(v, _) => return Ok(Column::bool(clip_bool(v, lower, upper))),
            Column::F64(_) | Column::F32(_) | Column::I64(_, _) | Column::I32(_, _) => {}
            other => return Err(VolasError::DType(format!("cannot clip a {} column", other.dtype()))),
        }
        let bound_fits = |b: Option<f64>| b.map_or(true, |x| fits(self.dtype(), x));
        // Stay in dtype when every present bound fits it losslessly (a float dtype
        // always does); an int column with a non-integral bound promotes to float.
        let stay = self.dtype().is_float() || (bound_fits(lower) && bound_fits(upper));
        if stay {
            numeric_dispatch!(self, v => Numeric::into_column(clip_vec(v, lower, upper)))
        } else {
            Ok(Column::f64(clip_vec(&self.to_f64_vec(), lower, upper)))
        }
    }

    /// `where` / `mask` core: pick `self` where `cond` is true, else `other`,
    /// producing `target` dtype (the caller resolves keep-vs-promote so the fill's
    /// value/type is accounted for). Picks i64 natively when `target` is `I64`
    /// (no f64 round-trip, so large ints stay exact). Equal lengths assumed.
    pub fn select(&self, cond: &[bool], other: &Column, target: DType) -> Result<Column> {
        match target {
            DType::I64 => Ok(Column::i64(stats::select(cond, &self.as_i64_vec()?, &other.as_i64_vec()?))),
            DType::I32 => Ok(Column::i32(stats::select(cond, &self.as_i32_vec()?, &other.as_i32_vec()?))),
            DType::F32 => Ok(Column::f32(stats::select(cond, &self.to_f32_vec(), &other.to_f32_vec()))),
            // bool ∘ bool stays bool (pandas keeps a bool result when the fill is
            // also bool); `bool` isn't `Numeric`, so the pick is inlined.
            DType::Bool => {
                let (a, b) = (self.as_bool_vec()?, other.as_bool_vec()?);
                Ok(Column::bool((0..cond.len()).map(|i| if cond[i] { a[i] } else { b[i] }).collect()))
            }
            _ => Ok(Column::f64(stats::select(cond, &self.to_f64_vec(), &other.to_f64_vec()))),
        }
    }

    /// The column as `bool` values (a `Bool` column directly; else an error).
    fn as_bool_vec(&self) -> Result<Vec<bool>> {
        match self {
            Column::Bool(v, _) => Ok(v.to_vec()),
            other => Err(VolasError::DType(format!("expected a bool column, got {}", other.dtype()))),
        }
    }

    /// Binary `+ - *` against `other`, dtype-preserving via [`binary_supertype`]
    /// (`int ∘ int → i64`, else f64). `bool ∘ bool` is logical, matching pandas:
    /// `+` is OR, `*` is AND, `-` is an error (numpy disallows bool subtraction);
    /// `bool ∘ number` promotes (bool acts as 0/1). Wrapping int ops match pandas
    /// overflow. Equal lengths assumed.
    pub fn binary(&self, other: &Column, op: BinOp) -> Result<Column> {
        if let (Column::Bool(a, _), Column::Bool(b, _)) = (self, other) {
            return match op {
                BinOp::Add => Ok(Column::bool(a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect())),
                BinOp::Mul => Ok(Column::bool(a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect())),
                BinOp::Sub => Err(VolasError::DType(
                    "the `-` operator is not supported for bool columns (use `^`)".into(),
                )),
            };
        }
        match binary_supertype(self.dtype(), other.dtype()) {
            DType::I64 => Ok(Column::i64(binary_kernel(&self.as_i64_vec()?, &other.as_i64_vec()?, op))),
            DType::I32 => Ok(Column::i32(binary_kernel(&self.as_i32_vec()?, &other.as_i32_vec()?, op))),
            DType::F32 => Ok(Column::f32(binary_kernel(&self.to_f32_vec(), &other.to_f32_vec(), op))),
            _ => Ok(Column::f64(binary_kernel(&self.to_f64_vec(), &other.to_f64_vec(), op))),
        }
    }

    /// True division `self / other` (pandas `/`): always float. `bool / bool` is
    /// an error, matching pandas (division is not defined on bool).
    pub fn div(&self, other: &Column) -> Result<Column> {
        if matches!((self, other), (Column::Bool(_, _), Column::Bool(_, _))) {
            return Err(VolasError::DType(
                "division is not supported between two bool columns".into(),
            ));
        }
        let (a, b) = (self.to_f64_vec(), other.to_f64_vec());
        Ok(Column::f64(a.iter().zip(&b).map(|(&x, &y)| x / y).collect()))
    }

    /// Sum, dtype-preserving (pandas): a float column -> float, int/bool -> i64
    /// (bool counts trues). Computed natively (i64 in i64, exact past 2^53).
    pub fn sum(&self) -> Scalar {
        match self {
            Column::F64(v) => Scalar::F64(stats::sum(v.as_slice())),
            Column::F32(v) => Scalar::F32(stats::sum(v.as_slice())),
            Column::I64(v, _) => Scalar::I64(stats::sum(v.as_slice())),
            // int32 sum promotes to int64 (pandas / numpy accumulator)
            Column::I32(v, _) => Scalar::I64(stats::sum(&v.iter().map(|&x| x as i64).collect::<Vec<_>>())),
            Column::Bool(v, _) => Scalar::I64(stats::sum(&bool_as_i64(v))),
            other => Scalar::F64(stats::sum(&other.to_f64_vec())),
        }
    }

    /// Product, dtype-preserving (float -> float, int/bool -> i64).
    pub fn prod(&self) -> Scalar {
        match self {
            Column::F64(v) => Scalar::F64(stats::prod(v.as_slice())),
            Column::F32(v) => Scalar::F32(stats::prod(v.as_slice())),
            Column::I64(v, _) => Scalar::I64(stats::prod(v.as_slice())),
            Column::I32(v, _) => Scalar::I64(stats::prod(&v.iter().map(|&x| x as i64).collect::<Vec<_>>())),
            Column::Bool(v, _) => Scalar::I64(stats::prod(&bool_as_i64(v))),
            other => Scalar::F64(stats::prod(&other.to_f64_vec())),
        }
    }

    /// Minimum (`want_max = false`) / maximum, dtype-preserving (float -> float,
    /// int -> i64, bool -> bool). Empty / all-missing -> `F64(NaN)`.
    pub fn extreme(&self, want_max: bool) -> Scalar {
        match self {
            Column::I64(v, _) => match stats::extreme(v.as_slice(), want_max) {
                Some(x) => Scalar::I64(x),
                None => Scalar::F64(f64::NAN),
            },
            Column::I32(v, _) => match stats::extreme(v.as_slice(), want_max) {
                Some(x) => Scalar::I32(x),
                None => Scalar::F64(f64::NAN),
            },
            Column::F32(v) => {
                Scalar::F32(stats::extreme(v.as_slice(), want_max).unwrap_or(f32::NAN))
            }
            Column::Bool(v, _) => {
                // min = all (AND), max = any (OR); empty -> NaN
                if v.is_empty() {
                    Scalar::F64(f64::NAN)
                } else if want_max {
                    Scalar::Bool(v.iter().any(|&b| b))
                } else {
                    Scalar::Bool(v.iter().all(|&b| b))
                }
            }
            Column::F64(v) => Scalar::F64(stats::extreme(v.as_slice(), want_max).unwrap_or(f64::NAN)),
            other => {
                Scalar::F64(stats::extreme(&other.to_f64_vec(), want_max).unwrap_or(f64::NAN))
            }
        }
    }

    /// The column as `i64` values: an `I64` column directly (exact), otherwise via
    /// lossless narrowing (errors if any value is non-integral / out of range).
    /// Used by the int paths of `select` / `binary`, where the caller has already
    /// established the values fit.
    fn as_i64_vec(&self) -> Result<Vec<i64>> {
        match self {
            Column::I64(v, _) => Ok(v.to_vec()),
            _ => self
                .to_f64_vec()
                .iter()
                .map(|&x| {
                    i64::try_from_f64(x)
                        .ok_or_else(|| VolasError::DType(format!("value {x} does not fit int64")))
                })
                .collect(),
        }
    }

    /// The column as `i32` values: an `I32` column directly, a `Bool` as 0/1,
    /// otherwise via lossless narrowing (errors if out of range / non-integral).
    fn as_i32_vec(&self) -> Result<Vec<i32>> {
        match self {
            Column::I32(v, _) => Ok(v.to_vec()),
            Column::Bool(v, _) => Ok(v.iter().map(|&b| b as i32).collect()),
            _ => self
                .to_f64_vec()
                .iter()
                .map(|&x| {
                    i32::try_from_f64(x)
                        .ok_or_else(|| VolasError::DType(format!("value {x} does not fit int32")))
                })
                .collect(),
        }
    }

    /// The column as `f32` values (an `F32` column directly; else converted).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Column::F32(v) => v.to_vec(),
            _ => self.to_f64_vec().iter().map(|&x| x as f32).collect(),
        }
    }

    /// Render each value as a `String` (for `astype(str)`).
    fn to_string_vec(&self) -> Vec<String> {
        match self {
            Column::Str(v) => v.to_vec(),
            Column::F64(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::F32(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::I64(v, _) => v.iter().map(|x| x.to_string()).collect(),
            Column::I32(v, _) => v.iter().map(|x| x.to_string()).collect(),
            Column::Bool(v, _) => v
                .iter()
                .map(|&b| if b { "True" } else { "False" }.to_string())
                .collect(),
            Column::Datetime(v) => v.iter().map(|&ns| datetime::format_ns(ns)).collect(),
        }
    }

    /// Value equality where `NaN == NaN` (pandas `equals` semantics), unlike the
    /// derived `PartialEq` (which uses IEEE `NaN != NaN`).
    pub fn equals(&self, other: &Column) -> bool {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            (Column::F32(a), Column::F32(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            _ => self == other,
        }
    }
}

/// Banker's (half-to-even) round of `x` to `decimals` places; `NaN` stays `NaN`.
fn round_f64(x: f64, decimals: i32) -> f64 {
    if x.is_nan() {
        return x;
    }
    let f = 10f64.powi(decimals);
    (x * f).round_ties_even() / f
}

/// Round an int to `decimals` places: identity for `decimals >= 0`; for negative
/// `decimals`, banker's round to the nearest `10^|decimals|` multiple (integer
/// arithmetic, exact for all i64 — no f64 round-trip).
fn round_i64(x: i64, decimals: i32) -> i64 {
    if decimals >= 0 {
        return x;
    }
    let factor = match 10i64.checked_pow(decimals.unsigned_abs()) {
        Some(f) => f,
        None => return 0, // 10^k beyond i64 -> everything rounds to 0
    };
    let q = x.div_euclid(factor);
    let r = x.rem_euclid(factor); // 0..factor
    let half = factor / 2; // factor = 10^k is even, so this is exact
    let up = r > half || (r == half && q.rem_euclid(2) != 0); // tie -> even multiple
    if up { q + 1 } else { q }.wrapping_mul(factor)
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

/// Write a scalar into a float column at `positions` (any value fits; f32 rounds).
fn set_float_at<T: Numeric>(v: &[T], positions: &[usize], value: SetVal) -> Column {
    let x = match value {
        SetVal::Num(x) => T::try_from_f64(x).unwrap_or(T::ZERO), // float try_from is always Some
        SetVal::Bool(b) => {
            if b {
                T::ONE
            } else {
                T::ZERO
            }
        }
    };
    let mut nv = v.to_vec();
    for &i in positions {
        nv[i] = x;
    }
    T::into_column(nv)
}

/// Write a scalar into an int column: keep the dtype if it fits, upcast to float
/// for `NaN`, error on a lossy (non-integral / out-of-range) write.
fn set_int_at<T: Numeric>(
    v: &[T],
    positions: &[usize],
    value: SetVal,
    dtype: &str,
) -> Result<Column> {
    match value {
        SetVal::Num(x) if x.is_nan() => {
            let mut nv: Vec<f64> = v.iter().map(|&n| n.to_f64()).collect();
            for &i in positions {
                nv[i] = x;
            }
            Ok(Column::f64(nv))
        }
        SetVal::Num(x) => match T::try_from_f64(x) {
            Some(iv) => {
                let mut nv = v.to_vec();
                for &i in positions {
                    nv[i] = iv;
                }
                Ok(T::into_column(nv))
            }
            None => Err(VolasError::DType(format!("Invalid value '{x}' for dtype '{dtype}'"))),
        },
        SetVal::Bool(b) => {
            let iv = if b { T::ONE } else { T::ZERO };
            let mut nv = v.to_vec();
            for &i in positions {
                nv[i] = iv;
            }
            Ok(T::into_column(nv))
        }
    }
}

/// A bool column as `i64` (0/1), for `cumsum` / `cumprod` (pandas -> int64).
fn bool_as_i64(v: &[bool]) -> Vec<i64> {
    v.iter().map(|&b| b as i64).collect()
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
    let force_true = lo.map_or(false, |x| x != 0.0);
    let force_false = hi.map_or(false, |x| x == 0.0);
    v.iter()
        .map(|&b| if force_false { false } else if force_true { true } else { b })
        .collect()
}

/// Element-wise `a ∘ b` for `Add` / `Sub` / `Mul` (wrapping, pandas overflow).
fn binary_kernel<T: Numeric>(a: &[T], b: &[T], op: BinOp) -> Vec<T> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
        })
        .collect()
}

#[cfg(test)]
mod tests {
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

        // -> F64 (incl. the Str -> NaN arm of to_f64_vec)
        assert_eq!(
            Column::i64(vec![3]).cast(DType::F64).unwrap(),
            Column::f64(vec![3.0])
        );
        assert_eq!(
            Column::bool(vec![true, false]).cast(DType::F64).unwrap(),
            Column::f64(vec![1.0, 0.0])
        );
        let from_str = Column::str(vec!["a".into(), "b".into()])
            .cast(DType::F64)
            .unwrap();
        assert_eq!(from_str.dtype(), DType::F64);
        assert!(from_str.to_f64_vec().iter().all(|x| x.is_nan()));

        // -> I64 (F64 / Bool / Datetime; Str errors)
        assert_eq!(
            Column::f64(vec![2.9]).cast(DType::I64).unwrap(),
            Column::i64(vec![2])
        );
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
    fn set_scalar_at_follows_pandas_dtype_rules() {
        use SetVal::{Bool, Num};
        // F64 stays F64 for a number or a bool.
        let f = Column::f64(vec![1.0, 2.0, 3.0]);
        assert_eq!(f.set_scalar_at(&[1], Num(9.0)).unwrap(), Column::f64(vec![1.0, 9.0, 3.0]));
        assert_eq!(f.set_scalar_at(&[0], Bool(false)).unwrap(), Column::f64(vec![0.0, 2.0, 3.0]));
        // I64 keeps int for an integral number or a bool.
        let i = Column::i64(vec![1, 2, 3]);
        assert_eq!(i.set_scalar_at(&[2], Num(0.0)).unwrap(), Column::i64(vec![1, 2, 0]));
        assert_eq!(i.set_scalar_at(&[0], Bool(false)).unwrap(), Column::i64(vec![0, 2, 3]));
        // I64 + NaN upcasts the whole column to float.
        let up = i.set_scalar_at(&[1], Num(f64::NAN)).unwrap();
        assert_eq!(up.dtype(), DType::F64);
        assert!(matches!(&up, Column::F64(v) if v[0] == 1.0 && v[1].is_nan() && v[2] == 3.0));
        // I64 + a non-integral number is lossy -> error.
        assert!(i.set_scalar_at(&[0], Num(2.5)).is_err());
        // Bool keeps bool for a bool; a number into bool is lossy -> error.
        let b = Column::bool(vec![true, false]);
        assert_eq!(b.set_scalar_at(&[1], Bool(true)).unwrap(), Column::bool(vec![true, true]));
        assert!(b.set_scalar_at(&[0], Num(0.0)).is_err());
        // A scalar into a str column is unsupported -> error.
        assert!(Column::str(vec!["a".into()]).set_scalar_at(&[0], Num(1.0)).is_err());
    }

    #[test]
    fn cumulatives_preserve_dtype() {
        // i64 stays i64, computed natively
        assert_eq!(Column::i64(vec![1, 2, 3, 4]).cumsum().unwrap(), Column::i64(vec![1, 3, 6, 10]));
        assert_eq!(Column::i64(vec![3, 1, 4, 1]).cummax().unwrap(), Column::i64(vec![3, 3, 4, 4]));
        assert_eq!(Column::i64(vec![3, 1, 4, 1]).cummin().unwrap(), Column::i64(vec![3, 1, 1, 1]));
        assert_eq!(Column::i64(vec![1, 2, 3]).cumprod().unwrap(), Column::i64(vec![1, 2, 6]));
        // f64 keeps NaN in place (compare with equals: NaN == NaN)
        assert!(Column::f64(vec![1.0, f64::NAN, 2.0, 4.0]).cumsum().unwrap()
            .equals(&Column::f64(vec![1.0, f64::NAN, 3.0, 7.0])));
        assert!(Column::f64(vec![1.0, f64::NAN, 4.0, 2.0]).cummax().unwrap()
            .equals(&Column::f64(vec![1.0, f64::NAN, 4.0, 4.0])));
        assert!(Column::f64(vec![3.0, f64::NAN, 1.0]).cummin().unwrap()
            .equals(&Column::f64(vec![3.0, f64::NAN, 1.0])));
        assert!(Column::f64(vec![2.0, f64::NAN, 3.0]).cumprod().unwrap()
            .equals(&Column::f64(vec![2.0, f64::NAN, 6.0])));
        // bool is treated as i64 (pandas bool.cumsum -> int64); str -> error
        assert_eq!(Column::bool(vec![true, false, true]).cumsum().unwrap(), Column::i64(vec![1, 1, 2]));
        assert!(Column::str(vec!["a".into()]).cumsum().is_err());
    }

    #[test]
    fn abs_preserves_dtype_and_wraps() {
        assert!(Column::f64(vec![-1.0, f64::NAN, 2.0]).abs().unwrap()
            .equals(&Column::f64(vec![1.0, f64::NAN, 2.0])));
        // abs(i64::MIN) wraps to i64::MIN (pandas / numpy)
        assert_eq!(Column::i64(vec![-3, 4, i64::MIN]).abs().unwrap(), Column::i64(vec![3, 4, i64::MIN]));
    }

    #[test]
    fn round_preserves_dtype() {
        // f64 banker's, NaN passthrough
        assert!(Column::f64(vec![0.5, 1.5, 2.5, f64::NAN]).round(0).unwrap()
            .equals(&Column::f64(vec![0.0, 2.0, 2.0, f64::NAN])));
        // i64 identity at decimals>=0; banker's-to-multiple at negative decimals
        assert_eq!(Column::i64(vec![7, 8]).round(0).unwrap(), Column::i64(vec![7, 8]));
        assert_eq!(Column::i64(vec![15, 25, 35, 45, 5]).round(-1).unwrap(), Column::i64(vec![20, 20, 40, 40, 0]));
        assert_eq!(Column::i64(vec![16, 13]).round(-1).unwrap(), Column::i64(vec![20, 10])); // r>half / r<half
        assert_eq!(Column::i64(vec![-15, -25]).round(-1).unwrap(), Column::i64(vec![-20, -20])); // negative
        assert_eq!(Column::i64(vec![123]).round(-25).unwrap(), Column::i64(vec![0])); // 10^25 overflows -> 0
        assert_eq!(Column::bool(vec![true, false]).round(0).unwrap(), Column::bool(vec![true, false])); // bool no-op
        assert!(Column::str(vec!["a".into()]).round(0).is_err());
    }

    #[test]
    fn clip_preserves_dtype_or_promotes() {
        use DType::{F64, I64};
        // f64: both bounds, lo-only, hi-only, no bounds, NaN passthrough
        assert!(Column::f64(vec![-1.0, 1.0, 3.0, f64::NAN]).clip(Some(0.0), Some(2.0)).unwrap()
            .equals(&Column::f64(vec![0.0, 1.0, 2.0, f64::NAN])));
        assert_eq!(Column::f64(vec![-1.0, 5.0]).clip(Some(0.0), None).unwrap(), Column::f64(vec![0.0, 5.0]));
        assert_eq!(Column::f64(vec![-1.0, 5.0]).clip(None, Some(2.0)).unwrap(), Column::f64(vec![-1.0, 2.0]));
        assert_eq!(Column::f64(vec![1.0, 5.0]).clip(None, None).unwrap(), Column::f64(vec![1.0, 5.0]));
        // i64 with integral bounds stays int
        assert_eq!(Column::i64(vec![1, 5, 9]).clip(Some(2.0), Some(8.0)).unwrap(), Column::i64(vec![2, 5, 8]));
        // i64 with a non-integral bound promotes to float (pandas)
        let p = Column::i64(vec![1, 5, 9]).clip(Some(2.5), None).unwrap();
        assert_eq!(p.dtype(), F64);
        assert_eq!(p, Column::f64(vec![2.5, 5.0, 9.0]));
        let _ = I64;
        // bool stays bool: clip(F,T) no-op, clip(T,T) forces true, clip(F,F) forces false
        assert_eq!(Column::bool(vec![true, false]).clip(Some(0.0), Some(1.0)).unwrap(),
                   Column::bool(vec![true, false]));
        assert_eq!(Column::bool(vec![true, false]).clip(Some(1.0), Some(1.0)).unwrap(),
                   Column::bool(vec![true, true]));
        assert_eq!(Column::bool(vec![true, false]).clip(Some(0.0), Some(0.0)).unwrap(),
                   Column::bool(vec![false, false]));
        assert_eq!(Column::bool(vec![true, false]).clip(None, None).unwrap(),
                   Column::bool(vec![true, false]));
        assert!(Column::str(vec!["a".into()]).clip(None, None).is_err());
    }

    #[test]
    fn select_picks_in_target_dtype() {
        let cond = [true, false, true];
        let a = Column::i64(vec![1, 2, 3]);
        // target I64: other is i64 (direct) and f64-integral (lossless narrow)
        assert_eq!(a.select(&cond, &Column::i64(vec![10, 20, 30]), DType::I64).unwrap(),
                   Column::i64(vec![1, 20, 3]));
        assert_eq!(a.select(&cond, &Column::f64(vec![10.0, 20.0, 30.0]), DType::I64).unwrap(),
                   Column::i64(vec![1, 20, 3]));
        // target F64
        assert_eq!(Column::f64(vec![1.0, 2.0, 3.0])
            .select(&cond, &Column::f64(vec![10.0, 20.0, 30.0]), DType::F64).unwrap(),
            Column::f64(vec![1.0, 20.0, 3.0]));
        // as_i64_vec error: target I64 but a value is non-integral
        assert!(a.select(&cond, &Column::f64(vec![1.5, 2.0, 3.0]), DType::I64).is_err());
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
        assert_eq!(Column::i64(vec![i64::MAX]).binary(&Column::i64(vec![1]), BinOp::Add).unwrap(),
                   Column::i64(vec![i64::MIN]));
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
        assert_eq!(Column::bool(vec![true, false, true]).extreme(false), SB(false)); // AND
        assert_eq!(Column::bool(vec![true, false, true]).extreme(true), SB(true)); // OR
        assert_eq!(Column::f64(vec![3.0, 1.0]).extreme(false), F64(1.0));
        assert!(matches!(Column::str(vec!["a".into()]).extreme(true), F64(_)));
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
        assert_eq!(Column::f64(vec![1.5]).cast(DType::F32).unwrap(), Column::f32(vec![1.5]));
        assert_eq!(Column::f64(vec![3.0]).cast(DType::I32).unwrap(), Column::i32(vec![3]));
        assert!(Column::f64(vec![2.5]).cast(DType::I32).is_err()); // non-integral
        assert!(Column::f64(vec![3e9]).cast(DType::I32).is_err()); // out of range
        assert_eq!(f.cast(DType::F64).unwrap(), Column::f64(vec![1.5, 2.5, 3.5]));
        // reductions: f32 -> F32; i32 sum -> I64 (promotes), min -> I32
        assert_eq!(f.sum(), F32(7.5));
        assert_eq!(f.extreme(false), F32(1.5));
        assert_eq!(i.sum(), I64(8));
        assert_eq!(i.prod(), I64(12));
        assert_eq!(i.extreme(true), I32(4));
        // round / clip preserve dtype
        assert_eq!(Column::f32(vec![1.4, 2.6]).round(0).unwrap(), Column::f32(vec![1.0, 3.0]));
        assert_eq!(i.round(-1).unwrap().dtype(), DType::I32);
        assert_eq!(f.clip(Some(2.0), Some(3.0)).unwrap(), Column::f32(vec![2.0, 2.5, 3.0]));
        assert_eq!(i.clip(Some(2.0), Some(3.0)).unwrap().dtype(), DType::I32);
        // binary: same-dtype preserves
        assert_eq!(f.binary(&f, BinOp::Add).unwrap(), Column::f32(vec![3.0, 5.0, 7.0]));
        assert_eq!(i.binary(&i, BinOp::Add).unwrap(), Column::i32(vec![6, 2, 8]));
        // select (where/mask) in f32 / i32 target
        let cond = [true, false, true];
        assert_eq!(f.select(&cond, &Column::f32(vec![0.0, 0.0, 0.0]), DType::F32).unwrap(),
                   Column::f32(vec![1.5, 0.0, 3.5]));
        assert_eq!(i.select(&cond, &Column::i32(vec![0, 0, 0]), DType::I32).unwrap(),
                   Column::i32(vec![3, 0, 4]));
        // assignment: f32 writes, i32 keeps / upcasts NaN / rejects lossy
        assert_eq!(f.set_scalar_at(&[1], SetVal::Num(9.0)).unwrap(), Column::f32(vec![1.5, 9.0, 3.5]));
        assert_eq!(i.set_scalar_at(&[1], SetVal::Bool(true)).unwrap(), Column::i32(vec![3, 1, 4]));
        assert_eq!(i.set_scalar_at(&[1], SetVal::Num(9.0)).unwrap(), Column::i32(vec![3, 9, 4]));
        assert_eq!(i.set_scalar_at(&[0], SetVal::Num(f64::NAN)).unwrap().dtype(), DType::F64);
        assert!(i.set_scalar_at(&[0], SetVal::Num(2.5)).is_err());
        assert_eq!(f.set_scalar_at(&[0], SetVal::Bool(false)).unwrap(), Column::f32(vec![0.0, 2.5, 3.5]));
        // remaining f32/i32 arms (both directions of slice/take, the other reductions,
        // sub/mul kernels through the trait, bool->i32, append/append_missing)
        assert_eq!(f.get_f64(0), 1.5);
        assert_eq!(i.slice(1, 3), Column::i32(vec![1, 4]));
        assert_eq!(f.take(&[2, 0]), Column::f32(vec![3.5, 1.5]));
        assert_eq!(f.to_string_vec(), vec!["1.5", "2.5", "3.5"]);
        assert_eq!(f.prod(), F32(13.125));
        assert!(matches!(Column::i32(vec![]).extreme(false), Scalar::F64(x) if x.is_nan()));
        assert_eq!(f.binary(&f, BinOp::Sub).unwrap(), Column::f32(vec![0.0, 0.0, 0.0]));
        assert_eq!(f.binary(&f, BinOp::Mul).unwrap(), Column::f32(vec![2.25, 6.25, 12.25]));
        assert_eq!(i.binary(&i, BinOp::Sub).unwrap(), Column::i32(vec![0, 0, 0]));
        assert_eq!(i.binary(&i, BinOp::Mul).unwrap(), Column::i32(vec![9, 1, 16]));
        assert_eq!(Column::bool(vec![true, false, true]).binary(&i, BinOp::Add).unwrap(),
                   Column::i32(vec![4, 1, 5]));
        // as_i32_vec f64 fallback (lossless narrow + lossy error)
        assert_eq!(i.select(&cond, &Column::f64(vec![0.0, 0.0, 0.0]), DType::I32).unwrap(),
                   Column::i32(vec![3, 0, 4]));
        assert!(i.select(&cond, &Column::f64(vec![2.5, 0.0, 0.0]), DType::I32).is_err());
        assert_eq!(f.set_scalar_at(&[0], SetVal::Bool(true)).unwrap(), Column::f32(vec![1.0, 2.5, 3.5]));
        let mut ii = Column::i32(vec![1]);
        ii.append(&Column::i32(vec![2])).unwrap();
        assert_eq!(ii, Column::i32(vec![1, 2]));
        assert!(Column::f32(vec![1.0]).append(&Column::i32(vec![1])).is_err());
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
        assert_eq!(b().cummin().unwrap(), Column::bool(vec![true, false, false]));
        // abs -> bool (identity)
        assert_eq!(b().abs().unwrap(), b());
        // + is OR, * is AND, - is an error
        assert_eq!(b().binary(&c, BinOp::Add).unwrap(), Column::bool(vec![true, true, true]));
        assert_eq!(b().binary(&c, BinOp::Mul).unwrap(), Column::bool(vec![true, false, false]));
        assert!(b().binary(&c, BinOp::Sub).is_err());
        // bool / bool -> error; bool ∘ number promotes (bool acts as 0/1)
        assert!(b().div(&c).is_err());
        assert_eq!(b().binary(&Column::i64(vec![1, 1, 1]), BinOp::Add).unwrap(), Column::i64(vec![2, 1, 2]));
        let f = b().binary(&Column::f64(vec![1.0, 1.0, 1.0]), BinOp::Add).unwrap();
        assert_eq!(f.dtype(), DType::F64);
        // where/mask with a bool fill stays bool (Column::select Bool target)
        let cond = [true, false, true];
        assert_eq!(
            b().select(&cond, &Column::bool(vec![false, false, false]), DType::Bool).unwrap(),
            Column::bool(vec![true, false, true])
        );
        assert!(Column::i64(vec![1]).as_bool_vec().is_err()); // non-bool -> error
    }
}
