//! `Column` element-wise transforms — cumulatives, abs/round/clip,
//! shift/diff, fillna/ffill, and the conditional `select`.

use super::*;

impl Column {
    /// Cumulative sum (pandas `cumsum`, skipna), dtype-preserving (`bool` -> int64
    /// counting trues). Missing values are skipped and stay missing (NA in → NA out).
    pub fn cumsum(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let iv = widen_i64(v);
                let out = if val.has_nulls() {
                    cum_valid(&iv, val, 0, i64::wrapping_add)
                } else {
                    stats::cumsum(&iv)
                };
                Ok(Column::i64_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cumsum, i64::wrapping_add)),
            Column::I32(v, val) => Ok(cum(v, val, stats::cumsum, i32::wrapping_add)),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cumsum(v))),
        }
    }

    /// Cumulative maximum (pandas `cummax`), dtype-preserving (`bool` running OR).
    /// Missing values are skipped and stay missing.
    pub fn cummax(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let out = if val.has_nulls() {
                    cum_valid(v, val, false, |a, b| a || b)
                } else {
                    bool_running(v, true)
                };
                Ok(Column::bool_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cummax, |a, x| if x > a { x } else { a })),
            Column::I32(v, val) => Ok(cum(v, val, stats::cummax, |a, x| if x > a { x } else { a })),
            // running extreme is order-based, so str (lexical) / datetime (instant)
            // are valid, like min/max — not a numeric-only op.
            Column::Str(..) | Column::Datetime(..) => self.cum_extreme(true),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cummax(v))),
        }
    }

    /// Cumulative minimum (pandas `cummin`), dtype-preserving (`bool` running AND).
    /// Missing values are skipped and stay missing.
    pub fn cummin(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let out = if val.has_nulls() {
                    cum_valid(v, val, true, |a, b| a && b)
                } else {
                    bool_running(v, false)
                };
                Ok(Column::bool_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cummin, |a, x| if x < a { x } else { a })),
            Column::I32(v, val) => Ok(cum(v, val, stats::cummin, |a, x| if x < a { x } else { a })),
            Column::Str(..) | Column::Datetime(..) => self.cum_extreme(false),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cummin(v))),
        }
    }

    /// Cumulative product (pandas `cumprod`), dtype-preserving (`bool` -> int64).
    /// Missing values are skipped and stay missing.
    pub fn cumprod(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let iv = widen_i64(v);
                let out = if val.has_nulls() {
                    cum_valid(&iv, val, 1, i64::wrapping_mul)
                } else {
                    stats::cumprod(&iv)
                };
                Ok(Column::i64_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cumprod, i64::wrapping_mul)),
            Column::I32(v, val) => Ok(cum(v, val, stats::cumprod, i32::wrapping_mul)),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cumprod(v))),
        }
    }

    /// Element-wise absolute value (pandas `abs`), dtype-preserving. `abs` of a
    /// missing (`NaN`) stays missing; `abs(i64::MIN)` wraps to `i64::MIN` (pandas);
    /// a `bool` column is unchanged (`abs(bool) == bool`).
    pub fn abs(&self) -> Result<Column> {
        if matches!(self, Column::Bool(_, _)) {
            return Ok(self.clone());
        }
        let validity = self.validity().cloned().unwrap_or_default();
        let out = numeric_dispatch!(self, v => Numeric::into_column(stats::abs(v)))?;
        Ok(out.with_validity(validity))
    }

    /// Round to `decimals` places (pandas `round`), dtype-preserving: banker's
    /// (half-to-even) for floats, and for ints an identity at `decimals >= 0` or a
    /// banker's round to the nearest power-of-ten multiple at negative `decimals`.
    pub fn round(&self, decimals: i32) -> Result<Column> {
        match self {
            Column::F64(v) => Ok(Column::f64(
                v.iter().map(|&x| round_f64(x, decimals)).collect(),
            )),
            Column::F32(v) => Ok(Column::f32(
                v.iter()
                    .map(|&x| round_f64(x as f64, decimals) as f32)
                    .collect(),
            )),
            Column::I64(v, val) => Ok(Column::i64_with(
                v.iter().map(|&x| round_i64(x, decimals)).collect(),
                val.clone(),
            )),
            Column::I32(v, val) => Ok(Column::i32_with(
                v.iter()
                    .map(|&x| round_i64(x as i64, decimals) as i32)
                    .collect(),
                val.clone(),
            )),
            Column::Bool(_, _) => Ok(self.clone()), // round(bool) == bool (pandas no-op)
            other => Err(VolasError::DType(format!(
                "cannot round a {} column",
                other.dtype()
            ))),
        }
    }

    /// Clamp to `[lower, upper]` (either bound optional), pandas `clip`. Stays in
    /// the column dtype when every present bound fits it losslessly; otherwise (an
    /// int column with a non-integral bound) promotes to float, matching pandas.
    pub fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> Result<Column> {
        match self {
            // bool stays bool (pandas): a True lower bound forces all true, a
            // False upper bound forces all false, otherwise unchanged.
            Column::Bool(v, val) => {
                return Ok(Column::bool_with(clip_bool(v, lower, upper), val.clone()))
            }
            Column::F64(_) | Column::F32(_) | Column::I64(_, _) | Column::I32(_, _) => {}
            other => {
                return Err(VolasError::DType(format!(
                    "cannot clip a {} column",
                    other.dtype()
                )))
            }
        }
        let bound_fits = |b: Option<f64>| b.map_or(true, |x| fits(self.dtype(), x));
        // Stay in dtype when every present bound fits it losslessly (a float dtype
        // always does); an int column with a non-integral bound promotes to float.
        let stay = self.dtype().is_float() || (bound_fits(lower) && bound_fits(upper));
        if stay {
            let validity = self.validity().cloned().unwrap_or_default();
            numeric_dispatch!(self, v => Numeric::into_column(clip_vec(v, lower, upper)))
                .map(|c| c.with_validity(validity))
        } else {
            // int -> f64 promotion: to_f64_vec already maps a missing value to NaN
            Ok(Column::f64(clip_vec(&self.to_f64_vec(), lower, upper)))
        }
    }

    /// Shift values by `n` periods (pandas `shift`), dtype-preserving for every
    /// dtype (`str` included). Vacated cells become missing (`NaN` for float,
    /// `volas.NA` for int/bool/str, `NaT` for datetime); a shifted-in value keeps
    /// its own missingness.
    pub fn shift(&self, n: isize) -> Column {
        let len = self.len();
        // Validity of the result: a cell is present when its source cell exists and
        // was itself present. Only the nullable (int/bool) variants build it; the
        // value buffers shift with a single `memcpy` via `shift_fill`.
        let nulls = || {
            Validity::from_valid_iter(
                len,
                (0..len).map(|i| {
                    let s = i as isize - n;
                    s >= 0 && (s as usize) < len && self.is_valid(s as usize)
                }),
            )
        };
        match self {
            Column::F64(v) => Column::f64(shift_fill(v, n, f64::NAN)),
            Column::F32(v) => Column::f32(shift_fill(v, n, f32::NAN)),
            Column::I64(v, _) => Column::i64_with(shift_fill(v, n, 0), nulls()),
            Column::I32(v, _) => Column::i32_with(shift_fill(v, n, 0), nulls()),
            Column::Bool(v, _) => Column::bool_with(shift_fill(v, n, false), nulls()),
            Column::Datetime(v) => Column::datetime(shift_fill(v, n, i64::MIN)),
            Column::Str(v, _) => {
                let shifted = (0..len)
                    .map(|i| {
                        let s = i as isize - n;
                        if s >= 0 && (s as usize) < len {
                            v.get(s as usize)
                        } else {
                            "" // placeholder, masked by the gap validity
                        }
                    })
                    .collect();
                Column::Str(shifted, nulls())
            }
        }
    }

    /// Discrete difference `x[i] - x[i-n]` (pandas `diff`), dtype-preserving: the
    /// first `n` cells (the shift gap) are missing. `diff` **is** subtraction, so it
    /// inherits [`binary`](Self::binary)'s rules: int stays int (NA gap), bool-bool
    /// subtraction is unsupported, and `str` / `datetime` raise — a datetime
    /// difference is a `timedelta64`, the open `O3` decision, not a float64-of-ns.
    pub fn diff(&self, n: isize) -> Result<Column> {
        match self {
            Column::F64(v) => Ok(Column::f64(diff_kernel(v, n, f64::NAN))),
            Column::F32(v) => Ok(Column::f32(diff_kernel(v, n, f32::NAN))),
            // int -> int diff; bool -> "bool subtraction unsupported"; str/datetime
            // -> require_numeric error (all enforced by `binary`).
            _ => self.binary(&self.shift(n), BinOp::Sub),
        }
    }

    /// Replace missing cells with the constant `value` (pandas `fillna`),
    /// dtype-preserving when `value` fits the dtype, else promoting an int column
    /// to float (a non-integral fill).
    pub fn fillna(&self, value: f64) -> Result<Column> {
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let len = self.len();
        Ok(match self {
            Column::F64(v) => Column::f64(
                v.iter()
                    .map(|&x| if x.is_nan() { value } else { x })
                    .collect(),
            ),
            Column::F32(v) => Column::f32(
                v.iter()
                    .map(|&x| if x.is_nan() { value as f32 } else { x })
                    .collect(),
            ),
            Column::I64(v, val) => match i64::try_from_f64(value) {
                Some(iv) => Column::i64(
                    (0..len)
                        .map(|i| if val.is_valid(i) { v[i] } else { iv })
                        .collect(),
                ),
                None => Column::f64(
                    (0..len)
                        .map(|i| if val.is_valid(i) { v[i] as f64 } else { value })
                        .collect(),
                ),
            },
            Column::I32(v, val) => match i32::try_from_f64(value) {
                Some(iv) => Column::i32(
                    (0..len)
                        .map(|i| if val.is_valid(i) { v[i] } else { iv })
                        .collect(),
                ),
                None => Column::f64(
                    (0..len)
                        .map(|i| if val.is_valid(i) { v[i] as f64 } else { value })
                        .collect(),
                ),
            },
            // a 0/1 fill keeps bool; a non-0/1 fill promotes the (numeric-family)
            // bool column to float.
            Column::Bool(v, val) if value == 0.0 || value == 1.0 => Column::bool(
                (0..len)
                    .map(|i| if val.is_valid(i) { v[i] } else { value != 0.0 })
                    .collect(),
            ),
            Column::Bool(..) => Column::f64(
                self.to_f64_vec()
                    .iter()
                    .map(|&x| if x.is_nan() { value } else { x })
                    .collect(),
            ),
            // A numeric fill cannot apply to a non-numeric column: volas has no
            // `object` dtype to hold a mixed string/number or datetime/number
            // column, and the old f64 funnel silently turned valid strings into the
            // fill and lost the datetime dtype. Reject it.
            Column::Str(..) | Column::Datetime(..) => {
                return Err(VolasError::DType(format!(
                    "cannot fill a {} column with the numeric value {value} (volas has \
                     no object dtype); drop or select the missing rows instead",
                    self.dtype(),
                )))
            }
        })
    }

    /// Forward-fill (`forward = true`, pandas `ffill`) or backward-fill (`bfill`)
    /// missing cells from the nearest present value in that direction,
    /// dtype-preserving; leading / trailing cells with no source stay missing.
    pub fn fill_dir(&self, forward: bool) -> Column {
        if self.null_count() == 0 {
            return self.clone();
        }
        let len = self.len();
        // For each position, the source index of the value to carry in, or `None`.
        let mut src = vec![None; len];
        let mut last: Option<usize> = None;
        for k in 0..len {
            let i = if forward { k } else { len - 1 - k };
            if self.is_valid(i) {
                last = Some(i);
            }
            src[i] = last;
        }
        let validity = Validity::from_valid_iter(len, src.iter().map(|s| s.is_some()));
        match self {
            Column::F64(v) => {
                Column::f64(src.iter().map(|s| s.map_or(f64::NAN, |j| v[j])).collect())
            }
            Column::F32(v) => {
                Column::f32(src.iter().map(|s| s.map_or(f32::NAN, |j| v[j])).collect())
            }
            Column::I64(v, _) => Column::i64_with(
                src.iter().map(|s| s.map_or(0, |j| v[j])).collect(),
                validity,
            ),
            Column::I32(v, _) => Column::i32_with(
                src.iter().map(|s| s.map_or(0, |j| v[j])).collect(),
                validity,
            ),
            Column::Bool(v, _) => Column::bool_with(
                src.iter().map(|s| s.map_or(false, |j| v[j])).collect(),
                validity,
            ),
            Column::Datetime(v) => {
                Column::datetime(src.iter().map(|s| s.map_or(i64::MIN, |j| v[j])).collect())
            }
            // str carries missing too: gather the carried value (empty placeholder
            // for an unfilled cell, which `validity` then marks NA), like int/bool.
            Column::Str(v, _) => Column::Str(
                src.iter().map(|s| s.map_or("", |j| v.get(j))).collect(),
                validity,
            ),
        }
    }

    /// `where` / `mask` core: pick `self` where `cond` is true, else `other`,
    /// producing `target` dtype (the caller resolves keep-vs-promote so the fill's
    /// value/type is accounted for). Picks i64 natively when `target` is `I64`
    /// (no f64 round-trip, so large ints stay exact). Equal lengths assumed.
    pub fn select(&self, cond: &[bool], other: &Column, target: DType) -> Result<Column> {
        match target {
            DType::I64 => Ok(Column::i64_with(
                stats::select(cond, &self.as_i64_vec()?, &other.as_i64_vec()?),
                self.select_nulls(cond, other),
            )),
            DType::I32 => Ok(Column::i32_with(
                stats::select(cond, &self.as_i32_vec()?, &other.as_i32_vec()?),
                self.select_nulls(cond, other),
            )),
            DType::F32 => Ok(Column::f32(stats::select(
                cond,
                &self.to_f32_vec(),
                &other.to_f32_vec(),
            ))),
            // bool ∘ bool stays bool (pandas keeps a bool result when the fill is
            // also bool); `bool` isn't `Numeric`, so the pick is inlined.
            DType::Bool => {
                let (a, b) = (self.as_bool_vec()?, other.as_bool_vec()?);
                Ok(Column::bool_with(
                    (0..cond.len())
                        .map(|i| if cond[i] { a[i] } else { b[i] })
                        .collect(),
                    self.select_nulls(cond, other),
                ))
            }
            // str / datetime stay in their dtype (a default `other` of NA keeps the
            // kept values and marks the rest missing) instead of an f64 funnel that
            // turned every kept string into NaN.
            DType::Utf8 => {
                let (a, b) = (self.as_str_vec()?, other.as_str_vec()?);
                Ok(Column::str_with(
                    (0..cond.len())
                        .map(|i| if cond[i] { a[i].clone() } else { b[i].clone() })
                        .collect(),
                    self.select_nulls(cond, other),
                ))
            }
            DType::Datetime => {
                let (a, b) = (self.as_datetime_vec()?, other.as_datetime_vec()?);
                Ok(Column::datetime(
                    (0..cond.len())
                        .map(|i| if cond[i] { a[i] } else { b[i] })
                        .collect(),
                ))
            }
            _ => Ok(Column::f64(stats::select(
                cond,
                &self.to_f64_vec(),
                &other.to_f64_vec(),
            ))),
        }
    }

    /// Picked validity for `select`: position `i` is present when the *chosen*
    /// side (`self` where `cond`, else `other`) is present there.
    fn select_nulls(&self, cond: &[bool], other: &Column) -> Validity {
        Validity::from_valid_iter(
            cond.len(),
            (0..cond.len()).map(|i| {
                if cond[i] {
                    self.is_valid(i)
                } else {
                    other.is_valid(i)
                }
            }),
        )
    }
}
