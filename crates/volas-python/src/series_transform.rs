//! `Series` element-wise transforms + missing-value handling (cumulatives, shift,
//! fillna/ffill, clip/round/rank, the math/trig functions, rolling/ewm, where/mask).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{
    stats, Column, Series,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {

    /// Element-wise membership in `values` -> bool Series (pandas `isin`).
    /// A missing cell is False (NA is not "in" anything).
    pub(crate) fn isin(&self, values: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        let probe = pyany_to_column(values)?;
        let keys: std::collections::HashSet<String> =
            (0..probe.len()).filter_map(|i| cell_key(&probe, i)).collect();
        let out: Vec<bool> = (0..self.inner.len())
            .map(|i| cell_key(&self.inner.data, i).is_some_and(|k| keys.contains(&k)))
            .collect();
        Ok(bool_series(&self.inner, out))
    }

    /// Range membership -> bool Series (pandas `between`); `inclusive` selects
    /// which bounds are closed (`'both'` | `'left'` | `'right'` | `'neither'`).
    /// A missing cell is False.
    #[pyo3(signature = (left, right, inclusive = "both"))]
    pub(crate) fn between(&self, left: f64, right: f64, inclusive: &str) -> PyResult<PySeries> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        type Cmp = fn(f64, f64) -> bool;
        let (lo_ok, hi_ok): (Cmp, Cmp) = match inclusive {
            "both" => (|x, l| x >= l, |x, r| x <= r),
            "left" => (|x, l| x >= l, |x, r| x < r),
            "right" => (|x, l| x > l, |x, r| x <= r),
            "neither" => (|x, l| x > l, |x, r| x < r),
            other => {
                return Err(PyValueError::new_err(format!(
                    "between: inclusive must be 'both', 'left', 'right' or 'neither', got {other:?}"
                )))
            }
        };
        let v = self.inner.data.to_f64_vec();
        let out: Vec<bool> = (0..v.len())
            .map(|i| self.inner.data.is_valid(i) && lo_ok(v[i], left) && hi_ok(v[i], right))
            .collect();
        Ok(bool_series(&self.inner, out))
    }

    /// Replace every cell equal to `to_replace` with `value`, dtype-preserving
    /// (pandas scalar `replace`). Types that cannot match the column are an error.
    pub(crate) fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PySeries> {
        let n = self.inner.len();
        let old = cmp_scalar_col(to_replace, self.inner.data.dtype(), 1)?;
        let newc = scalar_to_column(value, self.inner.data.dtype())?;
        let old_key = cell_key(&old, 0);
        let positions: Vec<usize> = (0..n)
            .filter(|&i| cell_key(&self.inner.data, i) == old_key)
            .collect();
        if positions.is_empty() {
            return Ok(col_to_series(&self.inner, self.inner.data.clone()));
        }
        let data = self.inner.data.scatter(&positions, &newc).map_err(pyerr)?;
        Ok(col_to_series(&self.inner, data))
    }

    /// Linear interpolation across interior missing values (pandas
    /// `interpolate(method='linear')`); leading/trailing gaps stay missing.
    pub(crate) fn interpolate(&self) -> PyResult<PySeries> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let v = self.inner.data.to_f64_vec();
        let n = v.len();
        let valid: Vec<bool> = (0..n).map(|i| self.inner.data.is_valid(i)).collect();
        let mut out = v.clone();
        let mut last: Option<usize> = None;
        for i in 0..n {
            if valid[i] {
                if let Some(l) = last {
                    if i > l + 1 {
                        let step = (v[i] - v[l]) / (i - l) as f64;
                        for (k, slot) in out.iter_mut().enumerate().take(i).skip(l + 1) {
                            *slot = v[l] + step * (k - l) as f64;
                        }
                    }
                }
                last = Some(i);
            }
        }
        Ok(col_to_series(&self.inner, Column::f64(out)))
    }

    /// A fixed-window rolling aggregator (pandas `rolling(window, min_periods,
    /// center)`): `s.rolling(20).mean()` etc. `min_periods` defaults to the
    /// window; `center=True` labels each window at its center (reads FUTURE
    /// rows relative to the label — fine for labeling, look-ahead in live use).
    ///
    /// This surface is pandas COMPATIBILITY: the result is a plain Series that
    /// `append` does not refresh. Prefer the directive forms in live systems.
    #[pyo3(signature = (window, min_periods = None, center = false))]
    pub(crate) fn rolling(
        &self,
        window: i64,
        min_periods: Option<i64>,
        center: bool,
    ) -> PyResult<crate::window::PyRolling> {
        // i64 params + explicit guards: a negative value must be a clean
        // ValueError, not pyo3's unsigned-conversion OverflowError leak (R-1).
        if window < 1 {
            return Err(PyValueError::new_err("rolling window must be >= 1"));
        }
        if min_periods.is_some_and(|m| m < 0) {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        if min_periods.is_some_and(|m| m > window) {
            return Err(PyValueError::new_err(
                "min_periods must be <= window",
            ));
        }
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(crate::window::PyRolling {
            series: self.inner.clone(),
            spec: crate::window::WinSpec {
                window: window as usize,
                min_periods: min_periods.unwrap_or(window) as usize,
                center,
            },
        })
    }

    /// An expanding (cumulative) window aggregator (pandas `expanding()`).
    /// Compatibility surface — see `rolling`.
    #[pyo3(signature = (min_periods = 1))]
    pub(crate) fn expanding(&self, min_periods: i64) -> PyResult<crate::window::PyExpanding> {
        if min_periods < 0 {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(crate::window::PyExpanding {
            series: self.inner.clone(),
            spec: crate::window::WinSpec {
                window: usize::MAX,
                min_periods: min_periods as usize,
                center: false,
            },
        })
    }

    /// An exponentially-weighted aggregator (pandas `ewm`): exactly one of
    /// `com` / `span` / `halflife` / `alpha`, with `adjust` and `ignore_na`.
    /// Compatibility surface — see `rolling`.
    #[pyo3(signature = (com = None, span = None, halflife = None, alpha = None, min_periods = 0, adjust = true, ignore_na = false))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ewm(
        &self,
        com: Option<f64>,
        span: Option<f64>,
        halflife: Option<f64>,
        alpha: Option<f64>,
        min_periods: i64,
        adjust: bool,
        ignore_na: bool,
    ) -> PyResult<crate::window::PyEwm> {
        if min_periods < 0 {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(crate::window::PyEwm {
            series: self.inner.clone(),
            alpha: crate::window::resolve_alpha(com, span, halflife, alpha)?,
            adjust,
            ignore_na,
            min_periods: min_periods as usize,
        })
    }

    /// Shift the values by ``n`` rows, padding vacated cells with NaN.
    ///
    /// Args:
    ///     n (int): rows to shift; positive shifts down (default 1), negative up.
    ///
    /// Usage::
    ///
    ///     s.shift(1)    # lag by one bar
    ///     s.shift(-1)   # lead by one bar
    ///
    /// Returns:
    ///     Series: a new series of the same length.
    #[pyo3(signature = (n = 1))]
    pub(crate) fn shift(&self, n: isize) -> PySeries {
        col_to_series(&self.inner, self.inner.data.shift(n))
    }

    /// Discrete difference ``x[i] - x[i-n]`` (equivalent to ``s - s.shift(n)``).
    ///
    /// Args:
    ///     n (int): periods to difference; the first ``n`` rows are NaN
    ///         (default 1). Negative ``n`` differences against later rows.
    ///
    /// Returns:
    ///     Series: a new series of the same length.
    #[pyo3(signature = (n = 1))]
    pub(crate) fn diff(&self, n: isize) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.diff(n).map_err(pyerr)?))
    }

    /// Replace a missing cell with `value` (pandas `fillna`), typed to the
    /// column's dtype like `where` / `mask`: a numeric column fills with a number
    /// (promoting only when the fill needs it — an integral fill keeps int, a
    /// fractional fill promotes to float), a `str` column with a string, a
    /// `datetime` column with a parsed timestamp. An incompatible fill (a number
    /// into a `str` column) raises a `TypeError`. For directional fill use
    /// `ffill` / `bfill` (pandas 3.0 removed `fillna(method=)`).
    #[pyo3(signature = (value, limit = None))]
    pub(crate) fn fillna(&self, value: &Bound<'_, PyAny>, limit: Option<i64>) -> PyResult<PySeries> {
        if limit.is_some_and(|l| l < 0) {
            return Err(PyValueError::new_err("limit must be >= 0"));
        }
        let limit = limit.map(|l| l as usize);
        // fillna keeps present cells and fills missing ones, so it is `where` over
        // the validity mask — reusing the same typed-fill resolution. `limit`
        // caps how many leading missing cells are filled (pandas).
        let mut budget = limit.unwrap_or(usize::MAX);
        let keep: Vec<bool> = (0..self.inner.len())
            .map(|i| {
                if self.inner.data.is_valid(i) {
                    true
                } else if budget > 0 {
                    budget -= 1;
                    false                       // fill this one
                } else {
                    true                        // budget spent: keep it missing
                }
            })
            .collect();
        self.select_with(&keep, Some(value))
    }

    /// Forward-fill NaN cells from the last valid value (pandas `ffill`).
    pub(crate) fn ffill(&self) -> PySeries {
        self.fill_dir(true)
    }

    /// Backward-fill NaN cells from the next valid value (pandas `bfill`).
    pub(crate) fn bfill(&self) -> PySeries {
        self.fill_dir(false)
    }

    /// Boolean mask of missing (`volas.NA`) values, across every dtype (a float
    /// `NaN`, an int/bool validity hole, a datetime `NaT`).
    pub(crate) fn isna(&self) -> PySeries {
        let c = &self.inner.data;
        bool_series(&self.inner, (0..c.len()).map(|i| !c.is_valid(i)).collect())
    }

    /// Boolean mask of present (non-missing) values.
    pub(crate) fn notna(&self) -> PySeries {
        let c = &self.inner.data;
        bool_series(&self.inner, (0..c.len()).map(|i| c.is_valid(i)).collect())
    }

    /// Drop missing (NaN) elements (carries their index labels with them).
    pub(crate) fn dropna(&self) -> PySeries {
        let c = &self.inner.data;
        let keep: Vec<usize> = (0..c.len()).filter(|&i| c.is_valid(i)).collect();
        let data = self.inner.data.take(&keep);
        let index = Arc::new(self.inner.index.take(&keep));
        PySeries {
            inner: Series::new(self.inner.name.clone(), data, index),
        }
    }

    /// Cast to a dtype (`'float64'` / `'int64'` / `'bool'` / `'str'` /
    /// `'datetime64[ns]'` / ...), pandas `astype`.
    pub(crate) fn astype(&self, dtype: &str) -> PyResult<PySeries> {
        let col = if let Some(unit) = datetime_unit_of(dtype) {
            match &self.inner.data {
                Column::Datetime(_) | Column::Str(_, _) => {
                    self.inner.data.to_datetime().map_err(pyerr)?
                }
                _ => self.inner.data.epoch_to_datetime(unit).map_err(pyerr)?,
            }
        } else {
            self.inner
                .data
                .cast(parse_dtype(dtype)?)
                .map_err(pyerr)?
        };
        Ok(PySeries {
            inner: Series::new(self.inner.name.clone(), col, Arc::clone(&self.inner.index)),
        })
    }

    /// Cumulative sum (pandas `cumsum`, skipna=True), dtype-preserving.
    pub(crate) fn cumsum(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cumsum().map_err(pyerr)?))
    }

    /// Cumulative maximum (pandas `cummax`, skipna=True), dtype-preserving.
    pub(crate) fn cummax(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cummax().map_err(pyerr)?))
    }

    /// Cumulative minimum (pandas `cummin`, skipna=True), dtype-preserving.
    pub(crate) fn cummin(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cummin().map_err(pyerr)?))
    }

    /// Cumulative product (pandas `cumprod`, skipna=True), dtype-preserving.
    pub(crate) fn cumprod(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cumprod().map_err(pyerr)?))
    }

    /// Round each value to `decimals` places (pandas `round`), dtype-preserving:
    /// banker's (half-to-even) for floats, integer-exact for ints; NaN stays NaN.
    #[pyo3(signature = (decimals = 0))]
    pub(crate) fn round(&self, decimals: i32) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.round(decimals).map_err(pyerr)?))
    }

    /// Numerical rank (pandas `rank`, 1-based, NaN kept as NaN). Ties resolve by
    /// `method` (`'average'` | `'min'` | `'max'` | `'first'` | `'dense'`); `pct`
    /// returns ranks scaled to (0, 1].
    #[pyo3(signature = (method = "average", ascending = true, pct = false, na_option = "keep"))]
    pub(crate) fn rank(&self, method: &str, ascending: bool, pct: bool, na_option: &str) -> PyResult<PySeries> {
        if !matches!(na_option, "keep" | "top" | "bottom") {
            return Err(PyValueError::new_err(format!(
                "rank: na_option must be 'keep', 'top' or 'bottom', got {na_option:?}"
            )));
        }
        // rank is order-based, so it serves any ordered dtype (str lexically,
        // datetime by raw i64) — not the f64 funnel, which loses sub-256ns
        // datetime order. The rank VALUES are always float64 (pandas).
        let m = match method {
            "average" => stats::RankMethod::Average,
            "min" => stats::RankMethod::Min,
            "max" => stats::RankMethod::Max,
            "first" => stats::RankMethod::First,
            "dense" => stats::RankMethod::Dense,
            other => return Err(PyValueError::new_err(format!("rank: unknown method '{other}'"))),
        };
        let mut ranks = self.inner.data.rank(m, ascending, pct);
        if na_option != "keep" {
            // 'top'/'bottom': NA cells receive the extreme ranks instead of NaN;
            // present ranks shift to make room (pandas semantics).
            let n = self.inner.len();
            let na_count = (0..n).filter(|&i| !self.inner.data.is_valid(i)).count();
            if na_count > 0 && !pct {
                let mut na_rank = if na_option == "top" { 1.0 } else { (n - na_count + 1) as f64 };
                for (i, r) in ranks.iter_mut().enumerate() {
                    if !self.inner.data.is_valid(i) {
                        *r = na_rank;
                        na_rank += 1.0;
                    } else if na_option == "top" {
                        *r += na_count as f64;
                    }
                }
            }
        }
        Ok(f64_series(&self.inner, ranks))
    }

    /// Element-wise absolute value (pandas `abs`), dtype-preserving.
    pub(crate) fn abs(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.abs().map_err(pyerr)?))
    }

    /// Clip values into `[lower, upper]` (either bound optional), dtype-preserving;
    /// NaN stays NaN. An int column with a non-integral bound promotes to float
    /// (pandas `clip`).
    #[pyo3(signature = (lower = None, upper = None))]
    pub(crate) fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> PyResult<PySeries> {
        // F19: an inverted interval (lower > upper) is a coding error -> fail-loud
        // (C5), not a silent collapse-to-upper nor pandas's quiet interval swap.
        if let (Some(lo), Some(hi)) = (lower, upper) {
            if lo > hi {
                return Err(PyValueError::new_err(format!(
                    "clip lower ({lo}) must be <= upper ({hi})"
                )));
            }
        }
        Ok(col_to_series(&self.inner, self.inner.data.clip(lower, upper).map_err(pyerr)?))
    }

    /// pandas `Series.where`: keep self where `cond` is True, else `other`
    /// (default NaN). `cond` is a boolean Series; `other` is a scalar or a
    /// (same-index) Series.
    #[pyo3(name = "where", signature = (cond, other = None))]
    pub(crate) fn where_(&self, cond: &PySeries, other: Option<&Bound<'_, PyAny>>) -> PyResult<PySeries> {
        self.select_where(cond, other, false)
    }

    /// pandas `Series.mask`: the inverse of `where` — replace with `other` where
    /// `cond` is True, keep self elsewhere.
    #[pyo3(signature = (cond, other = None))]
    pub(crate) fn mask(&self, cond: &PySeries, other: Option<&Bound<'_, PyAny>>) -> PyResult<PySeries> {
        self.select_where(cond, other, true)
    }

    // --- TA-Lib "Math Transform" group: element-wise, NaN-preserving (a NaN or an
    // out-of-domain input — e.g. sqrt of a negative, asin outside [-1, 1] — yields
    // NaN, matching TA-Lib). Implemented as Series methods, not directives.
    /// Element-wise arc cosine (TA-Lib ACOS).
    pub(crate) fn acos(&self) -> PyResult<PySeries> {
        self.map_f64(f64::acos)
    }
    /// Element-wise arc sine (TA-Lib ASIN).
    pub(crate) fn asin(&self) -> PyResult<PySeries> {
        self.map_f64(f64::asin)
    }
    /// Element-wise arc tangent (TA-Lib ATAN).
    pub(crate) fn atan(&self) -> PyResult<PySeries> {
        self.map_f64(f64::atan)
    }
    /// Element-wise ceiling (TA-Lib CEIL).
    pub(crate) fn ceil(&self) -> PyResult<PySeries> {
        self.map_f64(f64::ceil)
    }
    /// Element-wise cosine (TA-Lib COS).
    pub(crate) fn cos(&self) -> PyResult<PySeries> {
        self.map_f64(f64::cos)
    }
    /// Element-wise hyperbolic cosine (TA-Lib COSH).
    pub(crate) fn cosh(&self) -> PyResult<PySeries> {
        self.map_f64(f64::cosh)
    }
    /// Element-wise base-e exponential (TA-Lib EXP).
    pub(crate) fn exp(&self) -> PyResult<PySeries> {
        self.map_f64(f64::exp)
    }
    /// Element-wise floor (TA-Lib FLOOR).
    pub(crate) fn floor(&self) -> PyResult<PySeries> {
        self.map_f64(f64::floor)
    }
    /// Element-wise natural logarithm (TA-Lib LN).
    pub(crate) fn ln(&self) -> PyResult<PySeries> {
        self.map_f64(f64::ln)
    }
    /// Element-wise base-10 logarithm (TA-Lib LOG10).
    pub(crate) fn log10(&self) -> PyResult<PySeries> {
        self.map_f64(f64::log10)
    }
    /// Element-wise sine (TA-Lib SIN).
    pub(crate) fn sin(&self) -> PyResult<PySeries> {
        self.map_f64(f64::sin)
    }
    /// Element-wise hyperbolic sine (TA-Lib SINH).
    pub(crate) fn sinh(&self) -> PyResult<PySeries> {
        self.map_f64(f64::sinh)
    }
    /// Element-wise square root (TA-Lib SQRT).
    pub(crate) fn sqrt(&self) -> PyResult<PySeries> {
        self.map_f64(f64::sqrt)
    }
    /// Element-wise tangent (TA-Lib TAN).
    pub(crate) fn tan(&self) -> PyResult<PySeries> {
        self.map_f64(f64::tan)
    }
    /// Element-wise hyperbolic tangent (TA-Lib TANH).
    pub(crate) fn tanh(&self) -> PyResult<PySeries> {
        self.map_f64(f64::tanh)
    }
}
