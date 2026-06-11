//! The `volas.Series` pyclass and its positional `.iloc` accessor.

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};
use volas_core::{
    binary_supertype, stats, BinOp, BoolOp, CmpOp, Column, DType, Index, Series, Tz,
};

use crate::format::{
    render_series, NA_REPR,
};
#[allow(unused_imports)]
use crate::*;

/// ``volas.Series`` — a single named, indexed column (usually obtained from
/// ``df['col']`` or a directive like ``df['ma:5']``).
///
/// Supports NaN-skipping reductions (``mean`` / ``sum`` / ``min`` / ``max`` /
/// ``std`` / ``var`` / ``median``), element-wise arithmetic / comparison /
/// boolean operators (``+ - * /``, ``< <= == != >= >``, ``& | ^ ~``) against a
/// scalar or another equal-length Series, the TA-Lib math transforms
/// (``sin`` / ``sqrt`` / ``ln`` / …), and ``shift`` / ``diff`` / ``fillna`` /
/// ``isna`` / ``notna`` / ``dropna``. Index by position via ``s.iloc[...]`` or
/// label via ``s.loc[...]``; export with ``to_numpy`` / ``to_list``.
///
/// Usage::
///
///     close = df['close']
///     close.mean()            # NaN-skipping mean
///     (close - df['open'])    # element-wise difference
///     close.shift(1)          # lag by one bar
///     close.iloc[-1]          # last value
#[pyclass(name = "Series")]
pub struct PySeries {
    pub(crate) inner: Series,
}

#[pymethods]
impl PySeries {
    /// The series name — the column it was drawn from, or ``None``.
    ///
    /// Returns:
    ///     str | None
    #[getter]
    pub(crate) fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// The dtype name (``'float64'``, ``'float32'``, ``'int64'``, ``'int32'``,
    /// ``'bool'``, ``'str'``, or ``'datetime64[ns]'`` — never ``'object'``, which
    /// volas has no dtype for).
    ///
    /// Returns:
    ///     str
    #[getter]
    pub(crate) fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// The row index shared with the parent frame, as a NumPy array (a
    /// ``datetime64[ns]`` array for a DatetimeIndex, an object array for a string
    /// index).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    pub(crate) fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, &self.inner.index)
    }

    /// The DatetimeIndex timezone name, or `None` for a tz-naive / non-datetime
    /// index (mirrors `df.tz`).
    #[getter]
    pub(crate) fn tz(&self) -> Option<String> {
        match self.inner.index.tz() {
            Tz::Utc => None,
            other => Some(other.name()),
        }
    }

    /// Positional (integer-location) accessor: ``s.iloc[i]`` returns the i-th
    /// value (negative indices count from the end); ``s.iloc[a:b]`` returns a
    /// sub-series. Read-only.
    ///
    /// Usage::
    ///
    ///     s.iloc[0]      # first value
    ///     s.iloc[-1]     # last value
    ///     s.iloc[1:4]    # a sub-series
    #[getter]
    pub(crate) fn iloc(&self) -> SeriesILoc {
        SeriesILoc {
            inner: self.inner.clone(),
        }
    }

    /// Label-based accessor: ``s.loc[label]`` returns the value at an index
    /// label; ``s.loc[a:b]`` returns the (stop-inclusive) label slice. Read-only.
    ///
    /// Usage::
    ///
    ///     s.loc[20210104]              # by integer label
    ///     s.loc['2021-01-04':'2021-02-01']  # inclusive datetime slice
    #[getter]
    pub(crate) fn loc(&self) -> SeriesLoc {
        SeriesLoc {
            inner: self.inner.clone(),
        }
    }

    /// The shape as a 1-tuple `(len,)` (pandas `Series.shape`).
    #[getter]
    pub(crate) fn shape(&self) -> (usize,) {
        (self.inner.len(),)
    }

    pub(crate) fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Guard the ambiguous `if series:` footgun: a Series has a single truth value
    /// only when it holds exactly one element (pandas-style).
    pub(crate) fn __bool__(&self) -> PyResult<bool> {
        match self.inner.len() {
            1 => Ok(to_bool_vec(&self.inner.data)[0]),
            _ => Err(PyValueError::new_err(
                "The truth value of a Series is ambiguous — use s.any() or s.all()",
            )),
        }
    }

    /// The values as a typed NumPy array; `dtype` casts (e.g. `'float32'`).
    #[pyo3(signature = (dtype = None))]
    pub(crate) fn to_numpy<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => Ok(arr.call_method1("astype", (dt,))?),
            None => Ok(arr),
        }
    }

    /// NumPy array protocol, so `np.isnan(series)` etc. work directly. Honors a
    /// requested `dtype` (casts).
    #[pyo3(signature = (dtype = None, copy = None))]
    pub(crate) fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => Ok(arr.call_method1("astype", (dt,))?),
            None => Ok(arr),
        }
    }

    // Reductions return numpy scalars (pandas' boundary representation). The
    // dtype-preserving ones (sum/prod/min/max) carry the column's result dtype
    // (np.int64 for an int column, etc.); the always-float statistics box np.float64.

    // Each numeric reduction first asserts the column is numeric — a str/datetime
    // reduction used to funnel through to_f64_vec and silently return 0.0 / NaN,
    // which the API contract (C4) forbids (V3).

    /// NaN-skipping mean (pandas `mean`) -> `np.float64`.
    pub(crate) fn mean(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.mean_f64()))
    }
    /// Sum (pandas `sum`), dtype-preserving: float -> `np.float64`, int / bool ->
    /// `np.int64` (computed natively).
    pub(crate) fn sum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.sum()))
    }
    /// Product (pandas `prod`), dtype-preserving.
    pub(crate) fn prod(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.prod()))
    }
    /// Minimum (pandas `min`). Order-based, so it serves any ordered dtype:
    /// numeric/bool reduce to a numpy scalar (int -> `np.int64`, exact even past
    /// 2^53), str to a Python str, datetime to `np.datetime64` — not the f64
    /// funnel. See [`extreme_value`].
    pub(crate) fn min(&self, py: Python<'_>) -> Py<PyAny> {
        extreme_value(py, &self.inner.data, false)
    }
    /// Maximum (pandas `max`), order-based and dtype-typed. See [`extreme_value`].
    pub(crate) fn max(&self, py: Python<'_>) -> Py<PyAny> {
        extreme_value(py, &self.inner.data, true)
    }
    /// Sample variance (`ddof=1`, pandas `var`) -> `np.float64`.
    pub(crate) fn var(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.var_f64()))
    }
    /// Sample standard deviation (`ddof=1`, pandas `std`) -> `np.float64`.
    pub(crate) fn std(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.var_f64().sqrt()))
    }
    /// Median (pandas `median`) -> `np.float64`.
    pub(crate) fn median(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.median_f64()))
    }
    /// Standard error of the mean (`ddof=1`, pandas `sem`) -> `np.float64`.
    pub(crate) fn sem(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::sem(&self.inner.data.to_f64_vec())))
    }
    /// Adjusted Fisher-Pearson skewness (pandas `skew`) -> `np.float64`.
    pub(crate) fn skew(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::skew(&self.inner.data.to_f64_vec())))
    }
    /// Excess kurtosis, Fisher's definition (pandas `kurt`) -> `np.float64`.
    pub(crate) fn kurt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::kurt(&self.inner.data.to_f64_vec())))
    }

    /// Pairwise Pearson correlation with `other` (pandas `corr`); positional
    /// alignment (volas does not reindex), dropping NaN pairs.
    pub(crate) fn corr(&self, other: &PySeries) -> PyResult<f64> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        other.inner.data.require_numeric().map_err(pyerr)?;
        Ok(stats::corr(&self.inner.data.to_f64_vec(), &other.inner.data.to_f64_vec()))
    }

    /// Pairwise sample covariance with `other`, ddof=1 (pandas `cov`); positional
    /// alignment, dropping NaN pairs.
    pub(crate) fn cov(&self, other: &PySeries) -> PyResult<f64> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        other.inner.data.require_numeric().map_err(pyerr)?;
        Ok(stats::cov(&self.inner.data.to_f64_vec(), &other.inner.data.to_f64_vec()))
    }

    /// Summary statistics (pandas `describe`): a Series indexed by
    /// `count / mean / std / min / 25% / 50% / 75% / max`.
    pub(crate) fn describe(&self) -> PyResult<PySeries> {
        // describe is a numeric summary (mean/std/quantiles); a str/datetime column
        // would funnel through to_f64_vec to nonsense, so it raises (C4) until a
        // dtype-aware categorical/datetime describe is designed.
        self.inner.data.require_numeric().map_err(pyerr)?;
        let v = self.inner.data.to_f64_vec();
        let count = non_nan(&self.inner.data).len() as f64;
        let vals = vec![
            count,
            self.mean_f64(),
            self.var_f64().sqrt(),
            stats::extreme(&v, false).unwrap_or(f64::NAN),
            self.quantile_f64(0.25)?,
            self.quantile_f64(0.5)?,
            self.quantile_f64(0.75)?,
            stats::extreme(&v, true).unwrap_or(f64::NAN),
        ];
        let labels = describe_labels();
        Ok(PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                Column::f64(vals),
                Arc::new(Index::str(labels)),
            ),
        })
    }

    /// Number of non-missing values (pandas `count`) -> `int`.
    pub(crate) fn count(&self) -> usize {
        self.inner.data.count()
    }

    /// Number of distinct non-missing values (pandas `nunique`) -> `int`.
    pub(crate) fn nunique(&self) -> usize {
        self.inner.data.nunique()
    }

    /// The distinct values in order of first appearance (pandas `unique`), as a
    /// **`Series`** that preserves the dtype and `volas.NA` (API contract C1: a
    /// variable-length column result stays a `Series`, not a numpy array that would
    /// collapse a nullable int/bool to float64 + NaN). One missing slot is kept if
    /// the series has any NA; the result carries a fresh `RangeIndex` (the distinct
    /// values have no row correspondence to the original).
    pub(crate) fn unique(&self) -> PySeries {
        let idx = self.inner.data.unique_indices();
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                self.inner.data.take(&idx),
                Arc::new(Index::range(idx.len())),
            ),
        }
    }

    /// Sort by value (pandas `sort_values`), stable, with missing values last; the
    /// index follows the permutation.
    #[pyo3(signature = (ascending = true))]
    pub(crate) fn sort_values(&self, ascending: bool) -> PySeries {
        self.reindexed(&self.inner.data.argsort(ascending))
    }

    /// First `n` rows (pandas `head` = `iloc[:n]`, so a negative `n` drops the
    /// last `-n` rows — Python slicing semantics).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn head(&self, n: isize) -> PySeries {
        let (a, b) = head_tail_window(n, self.inner.len(), true);
        self.sliced(a, b)
    }

    /// Last `n` rows (pandas `tail` = `iloc[-n:]`, so a negative `n` drops the
    /// first `-n` rows — Python slicing semantics).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn tail(&self, n: isize) -> PySeries {
        let (a, b) = head_tail_window(n, self.inner.len(), false);
        self.sliced(a, b)
    }

    /// True if any element is truthy (NaN skipped) — pandas `any` -> `np.bool_`.
    /// A bool/numeric truthiness reduction, so str/datetime raise rather than
    /// funnel to a silent (and dtype-dependent) answer (C4).
    pub(crate) fn any(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let r = match &self.inner.data {
            // skipna: a NA bool is its `false` placeholder in the buffer, so read the
            // validity — only a *present* true counts (matching pandas nullable any).
            Column::Bool(v, val) => v.iter().enumerate().any(|(i, &b)| val.is_valid(i) && b),
            other => other.to_f64_vec().iter().any(|&x| !x.is_nan() && x != 0.0),
        };
        Ok(np_bool(py, r))
    }

    /// True if every non-missing element is truthy (empty / all-NA -> True) — pandas
    /// `all` -> `np.bool_`, default `skipna=True`. Bool/numeric only (see `any`).
    pub(crate) fn all(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let r = match &self.inner.data {
            // skipna: a NA is ignored (vacuously satisfies), only a present false fails.
            Column::Bool(v, val) => v.iter().enumerate().all(|(i, &b)| !val.is_valid(i) || b),
            other => other.to_f64_vec().iter().all(|&x| x.is_nan() || x != 0.0),
        };
        Ok(np_bool(py, r))
    }

    /// The values as a Python list of typed scalars (pandas `to_list`).
    pub(crate) fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = (0..self.inner.len())
            .map(|i| scalar_to_py(py, &self.inner.data, i))
            .collect();
        PyList::new(py, items)
    }

    /// Alias of [`to_list`](Self::to_list) (the numpy / pandas spelling).
    pub(crate) fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.to_list(py)
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
    pub(crate) fn fillna(&self, value: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        // fillna keeps present cells and fills missing ones, so it is `where` over
        // the validity mask — reusing the same typed-fill resolution.
        let keep: Vec<bool> = (0..self.inner.len()).map(|i| self.inner.data.is_valid(i)).collect();
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

    /// pandas-style equality: **same dtype** and value-equal (NaN equals NaN).
    pub(crate) fn equals(&self, other: &PySeries) -> bool {
        self.inner.data.dtype() == other.inner.data.dtype()
            && self.inner.data.equals(&other.inner.data)
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
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    pub(crate) fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PySeries> {
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
        Ok(f64_series(&self.inner, self.inner.data.rank(m, ascending, pct)))
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

    /// The `q`-quantile in `[0, 1]` (linear interpolation, NaN-skipping) — pandas
    /// `quantile` -> `np.float64`.
    #[pyo3(signature = (q = 0.5))]
    pub(crate) fn quantile(&self, py: Python<'_>, q: f64) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.quantile_f64(q)?))
    }

    /// The index **label** of the maximum value (NaN-skipping); raises on an
    /// all-NA series (pandas `idxmax`).
    pub(crate) fn idxmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(label_to_py(
            py,
            &self.inner.index,
            argext(&self.inner.data, true)?,
        ))
    }

    /// The index **label** of the minimum value (NaN-skipping); raises on an
    /// all-NA series (pandas `idxmin`).
    pub(crate) fn idxmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(label_to_py(
            py,
            &self.inner.index,
            argext(&self.inner.data, false)?,
        ))
    }

    pub(crate) fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Add, false)
    }
    pub(crate) fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Sub, false)
    }
    pub(crate) fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Mul, false)
    }
    pub(crate) fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_div(&self.inner, other, false)
    }
    pub(crate) fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Add, true)
    }
    pub(crate) fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Sub, true)
    }
    pub(crate) fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Mul, true)
    }
    pub(crate) fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_div(&self.inner, other, true)
    }
    pub(crate) fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_floordiv(&self.inner, other, false)
    }
    pub(crate) fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_floordiv(&self.inner, other, true)
    }

    // Element-wise comparisons -> bool Series (pandas-style), dtype-aware.
    pub(crate) fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Lt)
    }
    pub(crate) fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Le)
    }
    pub(crate) fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Eq)
    }
    pub(crate) fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Ne)
    }
    pub(crate) fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Ge)
    }
    pub(crate) fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Gt)
    }

    // Element-wise boolean logic -> bool Series (operands coerced to bool).
    pub(crate) fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::And)
    }
    pub(crate) fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Or)
    }
    pub(crate) fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Xor)
    }
    pub(crate) fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::And)
    }
    pub(crate) fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Or)
    }
    pub(crate) fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Xor)
    }
    pub(crate) fn __invert__(&self) -> PySeries {
        col_to_series(&self.inner, self.inner.data.not())
    }

    /// `series[key]`: an integer position, a datetime label, or a slice.
    pub(crate) fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // boolean mask -> the True rows, as a new Series (pandas `s[bool_mask]`)
        if let Some(mask) = bool_mask_key(key)? {
            let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
            return Ok(Py::new(py, PySeries { inner: sub })?.into_any());
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return Ok(np_scalar_to_py(py, &self.inner.data, i));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        // label lookup
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(np_scalar_to_py(py, &self.inner.data, pos))
    }

    /// In-place assignment by boolean mask (`s[mask] = v`) or integer position
    /// (`s[i] = v`). Follows pandas 3.0 dtype rules: the column dtype is kept when
    /// the value fits (an integral number stays in an int series), `NaN` upcasts
    /// an int series to float, and a lossy write (e.g. `2.5` into an int series)
    /// raises `TypeError`.
    pub(crate) fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.len();
        let positions: Vec<usize> = if let Some(mask) = bool_mask_key(key)? {
            if mask.len() != n {
                return Err(PyValueError::new_err(format!(
                    "boolean mask length {} != series length {n}",
                    mask.len()
                )));
            }
            mask.iter()
                .enumerate()
                .filter_map(|(i, &m)| m.then_some(i))
                .collect()
        } else if let Ok(i) = key.extract::<isize>() {
            vec![norm_idx(i, n)?]
        } else {
            return Err(PyTypeError::new_err(
                "Series assignment takes a boolean mask or an integer position",
            ));
        };
        // One assignment path for every value kind (number, bool, string, datetime
        // string, None/NaN): convert to a typed single-cell column for this dtype
        // and scatter it — identical rules to the DataFrame indexers and mask
        // assignment (keep dtype, update validity, lossy values error).
        self.inner.data = scatter_scalar(&self.inner.data, &positions, value)?;
        Ok(())
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

    /// pandas-style vertical repr (`label   value` rows + a
    /// `Name: <name>, dtype: <dtype>` footer), truncating to 5 head + 5 tail rows
    /// past 60 (`display.max_rows` / `min_rows`). `str` and `repr` are identical.
    pub(crate) fn __repr__(&self) -> String {
        let truncate = if self.inner.len() > 60 { Some(5) } else { None };
        render_series(&self.inner, NA_REPR, None, truncate, true)
    }

    pub(crate) fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Render the whole series as text (pandas `Series.to_string`): no truncation
    /// by default and no `Name/dtype` footer; `max_rows` truncates.
    #[pyo3(signature = (na_rep = NA_REPR, float_format = None, max_rows = None))]
    pub(crate) fn to_string(
        &self,
        na_rep: &str,
        float_format: Option<&str>,
        max_rows: Option<usize>,
    ) -> PyResult<String> {
        let ff = parse_ff(float_format)?;
        let truncate = match max_rows {
            Some(m) if self.inner.len() > m => Some((m / 2).max(1)),
            _ => None,
        };
        Ok(render_series(&self.inner, na_rep, ff, truncate, false))
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

impl PySeries {
    /// A new Series whose data and index are gathered by `idx` (backs
    /// `sort_values` and any fancy-index reorder).
    pub(crate) fn reindexed(&self, idx: &[usize]) -> PySeries {
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                self.inner.data.take(idx),
                Arc::new(self.inner.index.take(idx)),
            ),
        }
    }

    /// A new Series with the `[start, end)` row slice of both data and index
    /// (backs `head` / `tail`).
    pub(crate) fn sliced(&self, start: usize, end: usize) -> PySeries {
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                self.inner.data.slice(start, end),
                Arc::new(self.inner.index.slice(start, end)),
            ),
        }
    }

    /// Box a float statistic as the column's float dtype: `np.float32` for an f32
    /// column, else `np.float64` (pandas: `f32.mean() -> np.float32`).
    pub(crate) fn box_float(&self, py: Python<'_>, value: f64) -> Py<PyAny> {
        if self.inner.data.dtype() == DType::F32 {
            np_f32(py, value as f32)
        } else {
            np_f64(py, value)
        }
    }

    // Raw f64 reduction values (the public methods box these as numpy scalars;
    // `describe` reuses them, so they stay unboxed here).
    pub(crate) fn mean_f64(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    pub(crate) fn var_f64(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        let n = v.len();
        if n < 2 {
            return f64::NAN;
        }
        let mean = v.iter().sum::<f64>() / n as f64;
        v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    }
    pub(crate) fn median_f64(&self) -> f64 {
        let mut v = non_nan(&self.inner.data);
        let n = v.len();
        if n == 0 {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    }
    /// The `q`-quantile as a raw f64 (linear interpolation, NaN-skipping).
    pub(crate) fn quantile_f64(&self, q: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("quantile: q must be in [0, 1]"));
        }
        let mut v = non_nan(&self.inner.data);
        if v.is_empty() {
            return Ok(f64::NAN);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = q * (v.len() - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        Ok(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
    }

    /// The boolean mask for a `where` / `mask` condition, validating it matches
    /// this series' length (pandas requires equal-shape conditionals).
    pub(crate) fn cond_mask(&self, cond: &PySeries) -> PyResult<Vec<bool>> {
        if !matches!(cond.inner.data, Column::Bool(..)) {
            return Err(PyTypeError::new_err(format!(
                "where/mask: `cond` must be a boolean Series, got {}",
                cond.inner.data.dtype()
            )));
        }
        let c = bool_mask_vec(&cond.inner.data)?;
        if c.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "Array conditional must be same shape as self ({} != {})",
                c.len(),
                self.inner.len()
            )));
        }
        Ok(c)
    }

    /// `where` (`invert = false`) / `mask` (`invert = true`) shared core: pick
    /// `self` where the (possibly inverted) condition holds, else `other`, in the
    /// promoted dtype. `mask` is `where(!cond)`.
    pub(crate) fn select_where(
        &self,
        cond: &PySeries,
        other: Option<&Bound<'_, PyAny>>,
        invert: bool,
    ) -> PyResult<PySeries> {
        let mut c = self.cond_mask(cond)?;
        if invert {
            c.iter_mut().for_each(|b| *b = !*b);
        }
        self.select_with(&c, other)
    }

    /// Keep `self` where `keep[i]` is true, else take the resolved `other` (the
    /// typed-scalar / Series / default-NA fill). The shared core of `where` /
    /// `mask` / `fillna`, so all three resolve a fill the same dtype-typed way
    /// (str scalar, parsed timestamp, NA-preserving) instead of funnelling to f64.
    pub(crate) fn select_with(&self, keep: &[bool], other: Option<&Bound<'_, PyAny>>) -> PyResult<PySeries> {
        // Lazy: if every cell is kept, no fill is applied, so the fill's type /
        // alignment is irrelevant — return self unchanged (dtype intact). This keeps
        // an all-keep where/mask (and a dense-column fillna) from type-checking a
        // fill it never uses, matching the per-column DataFrame surface.
        if keep.iter().all(|&b| b) {
            return Ok(PySeries { inner: self.inner.clone() });
        }
        let (other_col, other_dt) = where_other_resolve(other, &self.inner)?;
        // A float column keeps its float dtype (it absorbs any fill); a same-dtype
        // `other` (incl. the default NA, and bool/str/datetime) keeps that dtype so
        // it is not funneled to f64; a mixed int/float promotes by the supertype.
        let self_dt = self.inner.data.dtype();
        let target = if self_dt.is_float() {
            self_dt
        } else if self_dt == other_dt {
            self_dt
        } else {
            binary_supertype(self_dt, other_dt)
        };
        Ok(col_to_series(
            &self.inner,
            self.inner.data.select(keep, &other_col, target).map_err(pyerr)?,
        ))
    }

    /// Apply an element-wise `f64 -> f64` map, preserving name and index. The single
    /// guard for the whole Math Transform family: a `str` / `datetime` column would
    /// funnel through `to_f64_vec` to silent `NaN` (str) or `sin(epoch-as-f64)`
    /// (datetime), which the contract (C4) forbids, so it raises here.
    pub(crate) fn map_f64(&self, f: impl Fn(f64) -> f64) -> PyResult<PySeries> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let data = Column::f64(self.inner.data.to_f64_vec().iter().map(|&x| f(x)).collect());
        Ok(PySeries {
            inner: Series::new(self.inner.name.clone(), data, Arc::clone(&self.inner.index)),
        })
    }

    /// Directional fill (`forward` = ffill, else bfill), dtype-aware over every
    /// dtype (int / bool / str / datetime / float). Shared by `ffill` / `bfill`.
    pub(crate) fn fill_dir(&self, forward: bool) -> PySeries {
        col_to_series(&self.inner, self.inner.data.fill_dir(forward))
    }
}

/// `series.iloc[...]` positional indexer.
#[pyclass]
pub struct SeriesILoc {
    inner: Series,
}

#[pymethods]
impl SeriesILoc {
    pub(crate) fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return Ok(np_scalar_to_py(py, &self.inner.data, i));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        Err(PyIndexError::new_err(
            "iloc key must be an integer or slice",
        ))
    }
}
