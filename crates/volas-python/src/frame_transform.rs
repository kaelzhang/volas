//! `DataFrame` column-wise transforms and missing-value handling
//! (cumulatives, diff/shift/rank, clip/round, fillna/ffill/where/mask, rolling/ewm).


use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{
    binary_supertype, Column,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {

    /// Element-wise membership in `values` -> a boolean frame (pandas `isin`).
    pub(crate) fn isin(&self, values: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.isin(values))
    }
    /// Replace cells equal to `to_replace` with `value`, per column,
    /// dtype-preserving; columns whose dtype cannot hold the scalars are kept
    /// unchanged (pandas `replace`).
    pub(crate) fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        self.map_cols(|s| match s.replace(to_replace, value) {
            Ok(r) => Ok(r),
            Err(_) => Ok(PySeries { inner: s.inner.clone() }),   // dtype can't match -> untouched
        })
    }
    /// Linear interpolation per numeric column (pandas `interpolate`).
    pub(crate) fn interpolate(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.interpolate())
    }
    /// A fixed-window rolling aggregator over every numeric column (pandas
    /// `df.rolling(window, min_periods, center)`). Compatibility surface —
    /// prefer directives in live systems (see `Series.rolling`).
    #[pyo3(signature = (window, min_periods = None, center = false))]
    pub(crate) fn rolling(
        &self,
        window: i64,
        min_periods: Option<i64>,
        center: bool,
    ) -> PyResult<PyRollingFrame> {
        if window < 1 {
            return Err(PyValueError::new_err("rolling window must be >= 1"));
        }
        if min_periods.is_some_and(|m| m < 0) {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        if min_periods.is_some_and(|m| m > window) {
            return Err(PyValueError::new_err("min_periods must be <= window"));
        }
        Ok(PyRollingFrame {
            frame: self.logical().into_owned(),
            window: window as usize,
            min_periods: min_periods.unwrap_or(window) as usize,
            center,
        })
    }
    /// An expanding (cumulative) aggregator over every numeric column.
    #[pyo3(signature = (min_periods = 1))]
    pub(crate) fn expanding(&self, min_periods: i64) -> PyResult<PyExpandingFrame> {
        if min_periods < 0 {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        Ok(PyExpandingFrame {
            frame: self.logical().into_owned(),
            min_periods: min_periods as usize,
        })
    }
    /// An exponentially-weighted aggregator over every numeric column (pandas
    /// `ewm`: exactly one of com / span / halflife / alpha).
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
    ) -> PyResult<PyEwmFrame> {
        if min_periods < 0 {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        Ok(PyEwmFrame {
            frame: self.logical().into_owned(),
            alpha: crate::window::resolve_alpha(com, span, halflife, alpha)?,
            adjust,
            ignore_na,
            min_periods: min_periods as usize,
        })
    }

    /// Drop rows containing missing values, across every dtype (via the column
    /// validity). `how='any'` (default) drops a row if any column is missing
    /// there; `how='all'` only if every column is missing. An invalid `how`
    /// raises `ValueError`.
    #[pyo3(signature = (how = "any"))]
    pub(crate) fn dropna(&self, how: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        if how != "any" && how != "all" {
            return Err(PyValueError::new_err(format!(
                "dropna: invalid `how` {how:?} (expected 'any' or 'all')"
            )));
        }
        let view = self.logical();
        let df = view.as_ref();
        let cols = df.columns();
        let total = cols.len();
        let keep: Vec<usize> = (0..df.height())
            .filter(|&i| {
                let nan = cols.iter().filter(|c| !c.is_valid(i)).count();
                match how {
                    "all" => nan < total.max(1),
                    _ => nan == 0,
                }
            })
            .collect();
        Ok(PyDataFrame::plain(take_frame(df, &keep)))
    }

    /// Replace missing values with `value` in every column (pandas `fillna`),
    /// dtype-preserving like the Series version: `value` is a typed scalar — a
    /// string fills a str column, a Timestamp / datetime string fills a datetime
    /// column, a number / bool fills the numeric family, and `volas.NA` is a no-op.
    /// A dense (no-hole) column is untouched (so a numeric `fillna(0)` over a mixed
    /// frame skips its holeless str / datetime columns); a column whose hole the
    /// fill can't take raises a `TypeError` (volas has no `object` dtype to mix
    /// types — C4). For directional fill use `ffill` / `bfill` (pandas 3.0 removed
    /// `fillna(method=)`).
    #[pyo3(signature = (value, limit = None))]
    pub(crate) fn fillna(&self, value: &Bound<'_, PyAny>, limit: Option<i64>) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        // Per column, fill the missing cells with the typed scalar (str into str,
        // Timestamp / datetime string into datetime, number / bool into a numeric-
        // family column), mirroring the Series surface. A `volas.NA` fill is a
        // dtype-preserving no-op. A dense (no-hole) column is untouched, so a numeric
        // `fillna(0)` over a mixed frame skips its holeless str / datetime columns
        // and only rejects a column that actually has a hole the fill can't take
        // (C4 — no silent numeric -> non-numeric coercion). Atomic: every column is
        // resolved before any frame is built, so a rejected column mutates nothing.
        if limit.is_some_and(|l| l < 0) {
            return Err(PyValueError::new_err("limit must be >= 0"));
        }
        let na_like = is_na_like_py(value);
        let view = self.logical();
        let cols = view
            .as_ref()
            .columns()
            .iter()
            .map(|c| -> PyResult<Column> {
                if c.null_count() == 0 {
                    return Ok(c.clone());
                }
                let n = c.len();
                let kd = c.dtype();
                // `limit` caps how many leading holes are filled, PER COLUMN
                // (pandas semantics); a hole beyond the budget stays missing.
                let mut budget = limit.map(|l| l as usize).unwrap_or(usize::MAX);
                let keep: Vec<bool> = (0..n)
                    .map(|i| {
                        if c.is_valid(i) {
                            true
                        } else if budget > 0 {
                            budget -= 1;
                            false
                        } else {
                            true
                        }
                    })
                    .collect();
                let (other_col, target) = if na_like {
                    (Column::na_of(kd, n), kd)
                } else {
                    let (oc, odt) = scalar_fill_col(value, kd, n)?;
                    // a float column absorbs any fill; a same-dtype fill keeps that
                    // dtype; a mixed numeric fill promotes by the supertype.
                    let target = if kd.is_float() || kd == odt {
                        kd
                    } else {
                        binary_supertype(kd, odt)
                    };
                    (oc, target)
                };
                c.select(&keep, &other_col, target).map_err(pyerr)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.with_columns(cols)
    }

    /// Forward-fill missing cells in every column (pandas `ffill`), dtype-aware.
    pub(crate) fn ffill(&self) -> PyResult<PyDataFrame> {
        self.fill_dir(true)
    }

    /// Backward-fill missing cells in every column (pandas `bfill`), dtype-aware.
    pub(crate) fn bfill(&self) -> PyResult<PyDataFrame> {
        self.fill_dir(false)
    }

    /// Round each float column to `decimals` places (pandas `round`), banker's
    /// rounding; non-float columns are unchanged.
    #[pyo3(signature = (decimals = 0))]
    pub(crate) fn round(&self, decimals: i32) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        // Round numeric columns dtype-preservingly (banker's f64, integer-exact
        // i64); leave bool / str / datetime untouched, like pandas df.round.
        self.map_columns(|c| {
            if c.dtype().is_numeric() {
                c.round(decimals).map_err(pyerr)
            } else {
                Ok(c.clone())
            }
        })
    }

    // --- column-wise numeric transforms (-> a new frame, dtype-preserving per
    // column, pandas df.cumsum() etc.). cumulatives / abs / clip keep dtype;
    // diff / shift / rank are always float. -------------------------------------

    /// Column-wise cumulative sum (pandas `cumsum`), dtype-preserving.
    pub(crate) fn cumsum(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cumsum())
    }
    /// Column-wise cumulative maximum (pandas `cummax`).
    pub(crate) fn cummax(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cummax())
    }
    /// Column-wise cumulative minimum (pandas `cummin`).
    pub(crate) fn cummin(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cummin())
    }
    /// Column-wise cumulative product (pandas `cumprod`).
    pub(crate) fn cumprod(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cumprod())
    }
    /// Column-wise absolute value (pandas `abs`), dtype-preserving.
    pub(crate) fn abs(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.abs())
    }
    /// Column-wise clip into `[lower, upper]` (pandas `clip`), dtype-preserving.
    #[pyo3(signature = (lower = None, upper = None))]
    pub(crate) fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> PyResult<PyDataFrame> {
        // F19: inverted interval (lower > upper) -> fail-loud (C5), not silent.
        if let (Some(lo), Some(hi)) = (lower, upper) {
            if lo > hi {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clip lower ({lo}) must be <= upper ({hi})"
                )));
            }
        }
        self.map_cols(|s| s.clip(lower, upper))
    }
    /// Column-wise discrete difference (pandas `diff`), dtype-preserving; the gap
    /// is missing (`volas.NA` for int/bool).
    #[pyo3(signature = (n = 1))]
    pub(crate) fn diff(&self, n: isize) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.diff(n))
    }
    /// Column-wise shift by `n` rows (pandas `shift`), dtype-preserving; the
    /// vacated cells are missing (`volas.NA` for int/bool).
    #[pyo3(signature = (n = 1))]
    pub(crate) fn shift(&self, n: isize) -> PyResult<PyDataFrame> {
        self.map_cols(|s| Ok(s.shift(n)))
    }
    /// Column-wise rank (pandas `rank`); always float.
    #[pyo3(signature = (method = "average", ascending = true, pct = false, na_option = "keep"))]
    pub(crate) fn rank(&self, method: &str, ascending: bool, pct: bool, na_option: &str) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.rank(method, ascending, pct, na_option))
    }

    /// pandas `DataFrame.where`: keep each cell where `cond` is True, else `other`
    /// (a typed scalar, resolved per column like the Series surface; default /
    /// `volas.NA` is a dtype-preserving NA fill). `cond` is a same-shape boolean
    /// frame (e.g. from `isna`). The inverse is `mask`.
    #[pyo3(name = "where", signature = (cond, other = None))]
    pub(crate) fn where_(
        &self,
        cond: &PyDataFrame,
        other: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, true)
    }

    /// pandas `DataFrame.mask`: replace each cell with `other` where `cond` is
    /// True, keep it elsewhere — the inverse of `where`.
    #[pyo3(signature = (cond, other = None))]
    pub(crate) fn mask(
        &self,
        cond: &PyDataFrame,
        other: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, false)
    }

    /// Boolean mask of missing cells (every dtype via the column validity) -> a
    /// bool DataFrame (pandas `isna`).
    pub(crate) fn isna(&self) -> PyResult<PyDataFrame> {
        self.mask_na(true)
    }

    /// Boolean mask of present (non-NaN) cells -> a bool DataFrame (pandas `notna`).
    pub(crate) fn notna(&self) -> PyResult<PyDataFrame> {
        self.mask_na(false)
    }
}
