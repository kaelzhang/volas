//! `DataFrame` structural / row-reshaping ops (head/tail, sort, drop, rename,
//! set_index, astype, dedup, nlargest, resample).

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use volas_core::{
    Column, DataFrame, Index, Label, Series,
};
use volas_time::Cumulator;

use crate::timeframe::{build_agg_spec_for, resolve_time_frame};
#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {

    /// First `n` rows (pandas `head` = `iloc[:n]`, so a negative `n` drops the
    /// last `-n` rows — Python slicing semantics).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn head(&self, n: isize) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let (start, len) = self.logical_range();
        let (a, b) = head_tail_window(n, len, true);
        Ok(PyDataFrame::plain(self.inner.slice(start + a, start + b)))
    }

    /// Last `n` rows (pandas `tail` = `iloc[-n:]`, so a negative `n` drops the
    /// first `-n` rows — Python slicing semantics).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn tail(&self, n: isize) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let (start, len) = self.logical_range();
        let (a, b) = head_tail_window(n, len, false);
        Ok(PyDataFrame::plain(self.inner.slice(start + a, start + b)))
    }
    /// The `n` rows with the largest values in `column` (pandas `nlargest`).
    pub(crate) fn nlargest(&self, n: i64, column: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        if n < 0 {
            return Err(PyValueError::new_err("n must be >= 0"));
        }
        self.extreme_rows(n as usize, column, false)
    }
    /// The `n` rows with the smallest values in `column` (pandas `nsmallest`).
    pub(crate) fn nsmallest(&self, n: i64, column: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        if n < 0 {
            return Err(PyValueError::new_err("n must be >= 0"));
        }
        self.extreme_rows(n as usize, column, true)
    }
    /// Drop later duplicate ROWS, keeping the first (pandas
    /// `drop_duplicates(keep='first')`, over all columns).
    #[pyo3(signature = (keep = "first"))]
    pub(crate) fn drop_duplicates(&self, keep: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let dup = self.row_duplicated(keep)?;
        let view = self.logical();
        let df = view.as_ref();
        let positions: Vec<usize> = (0..df.height()).filter(|&i| !dup[i]).collect();
        Ok(PyDataFrame::plain(take_frame(df, &positions)))
    }
    /// True per row for a later duplicate of an earlier row (pandas `duplicated`).
    #[pyo3(signature = (keep = "first"))]
    pub(crate) fn duplicated(&self, keep: &str) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let dup = self.row_duplicated(keep)?;
        Ok(PySeries {
            inner: Series::new(None, Column::bool(dup), Arc::clone(self.logical().index())),
        })
    }
    /// The first (smallest-position) mode of each column, as a 1-row frame.
    /// (pandas pads multi-modal columns into extra rows; volas keeps the single
    /// deterministic first mode per column — documented divergence.)
    pub(crate) fn mode(&self) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let idx = df.index();
        let mut cols = Vec::with_capacity(df.width());
        for (name, col) in df.names().iter().zip(df.columns()) {
            let s = PySeries {
                inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)),
            };
            let m = s.mode();
            let take: Vec<usize> = if m.inner.is_empty() { vec![] } else { vec![0] };
            cols.push(m.inner.data.take(&take));
        }
        Ok(PyDataFrame::plain(
            DataFrame::new(df.names().to_vec(), cols, None).map_err(pyerr)?,
        ))
    }
    /// Counts of unique values (pandas `df.value_counts()`); volas has no
    /// MultiIndex, so only a single-column frame is supported — call it on the
    /// column (`df[col].value_counts()`) otherwise.
    pub(crate) fn value_counts(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        if df.width() != 1 {
            return Err(PyTypeError::new_err(
                "DataFrame.value_counts needs a single column (volas has no MultiIndex); \
                 use df[col].value_counts()",
            ));
        }
        let name = df.names()[0].clone();
        let col = df.columns()[0].clone();
        let s = PySeries {
            inner: Series::new(Some(name), col, Arc::clone(df.index())),
        };
        s.value_counts(false, true, false, true)
    }

    /// Sort rows by index label (pandas `sort_index`).
    #[pyo3(signature = (ascending = true))]
    pub(crate) fn sort_index(&self, ascending: bool) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let perm = df.index().argsort(ascending);
        Ok(PyDataFrame::plain(take_frame(df, &perm)))
    }

    /// Move the row index into an `'index'` column and restore a RangeIndex
    /// (pandas `reset_index`); `drop=True` discards the old index.
    #[pyo3(signature = (drop = false))]
    pub(crate) fn reset_index(&self, drop: bool) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let h = df.height();
        let (names, columns): (Vec<String>, Vec<Column>) = if drop {
            (df.names().to_vec(), df.columns().to_vec())
        } else {
            // Restore the index's name as the new column label (pandas parity);
            // an unnamed index falls back to "index".
            let label = df.index().name().unwrap_or("index").to_string();
            // F39: the restored index label must not collide with an existing
            // column — a duplicate column name violates the unique-name contract.
            if df.names().iter().any(|n| n == &label) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "reset_index: column name {label:?} already exists (would duplicate)"
                )));
            }
            let mut names = vec![label];
            names.extend(df.names().iter().cloned());
            let mut cols = vec![df.index().to_column()];
            cols.extend(df.columns().iter().cloned());
            (names, cols)
        };
        Ok(PyDataFrame::plain(
            DataFrame::new(names, columns, Some(Index::range(h))).map_err(pyerr)?,
        ))
    }

    /// Drop rows by index label (`axis=0`) or columns by name (`axis=1`) —
    /// returns a new DataFrame. Row labels are parsed against the index kind.
    #[pyo3(signature = (labels, axis = 0, errors = "raise"))]
    pub(crate) fn drop(&self, py: Python<'_>, labels: Vec<Py<PyAny>>, axis: i64, errors: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let ignore_missing = match errors {
            "raise" => false,
            "ignore" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "drop: errors must be 'raise' or 'ignore', got {other:?}"
                )))
            }
        };
        if axis == 1 {
            let drop_names: Vec<String> = labels
                .iter()
                .map(|l| l.bind(py).extract::<String>())
                .collect::<PyResult<_>>()?;
            // F37: a missing label is an error (pandas KeyError), not a silent
            // no-op — unless explicitly opted out with errors='ignore' (F44).
            let names = df.names();
            if !ignore_missing {
                for n in &drop_names {
                    if !names.iter().any(|m| m == n) {
                        return Err(PyKeyError::new_err(format!("[{n:?}] not found in axis")));
                    }
                }
            }
            let keep: Vec<String> = names
                .iter()
                .filter(|n| !drop_names.contains(n))
                .cloned()
                .collect();
            return Ok(PyDataFrame::plain(df.select(&keep).map_err(pyerr)?));
        }
        let index = df.index();
        let targets: Vec<Label> = labels
            .iter()
            .map(|l| parse_label(l.bind(py), index))
            .collect::<PyResult<_>>()?;
        // F37 (row axis): every label must exist in the index, else KeyError —
        // unless errors='ignore' (F44).
        if !ignore_missing {
            let present: Vec<Label> = (0..df.height()).map(|i| index.label_at(i)).collect();
            for t in &targets {
                if !present.contains(t) {
                    return Err(PyKeyError::new_err("label not found in axis"));
                }
            }
        }
        let positions: Vec<usize> = (0..df.height())
            .filter(|&i| !targets.contains(&index.label_at(i)))
            .collect();
        Ok(PyDataFrame::plain(take_frame(df, &positions)))
    }

    /// Resample to a coarser timeframe (OHLCV cumulation / down-sampling),
    /// returning a **tf-aware** DataFrame you can keep ``append``-ing finer bars
    /// into (the forming period is the live last row).
    ///
    /// Requires a DatetimeIndex. Each column is aggregated with a sensible
    /// default (open=first, high=max, low=min, close=last, volume=sum); override
    /// per column via ``cumulators``. If the source already has a ``time_frame``,
    /// the target must be a whole multiple of it (e.g. 5m→15m, not 5m→7m, and not
    /// a week/3-day into a month); cumulating to the *same* frame is a ``copy()``.
    ///
    /// Args:
    ///     time_frame (str | TimeFrame): the target bucket, e.g. ``'1d'``,
    ///         ``'15m'``, ``'1w'``.
    ///     cumulators (dict[str, str], optional): per-column aggregator
    ///         overrides, e.g. ``{'volume': 'sum', 'close': 'last'}``.
    ///
    /// Usage::
    ///
    ///     daily = df.cumulate('1d')          # a tf-aware 1d frame
    ///     daily.append(intraday_bar)         # folds into the forming day
    ///
    /// Returns:
    ///     DataFrame: the resampled, tf-aware frame.
    #[pyo3(signature = (time_frame, cumulators = None))]
    pub(crate) fn cumulate(
        &self,
        time_frame: &Bound<'_, PyAny>,
        cumulators: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let target = resolve_time_frame(time_frame)?;
        if let Some(tfs) = &self.tf {
            // Same frame is a no-op resample == copy() (keeps the cursor & state).
            if target == tfs.time_frame {
                return Ok(self.copy());
            }
            if !tfs.time_frame.can_coarsen(target) {
                return Err(PyValueError::new_err(format!(
                    "cannot cumulate {} -> {}: the target is not a whole multiple of the source frame",
                    tfs.time_frame.label(),
                    target.label()
                )));
            }
        }
        let view = self.logical();
        let base = view.as_ref();
        let spec = build_agg_spec_for(cumulators, Some(base.names()))?;
        let mut cum = Cumulator::new(target, spec.clone());
        cum.append(base).map_err(pyerr)?;
        let frame = cum.frame().map_err(pyerr)?;
        // The result is a fresh frame (no cached directive columns -> cursor 0)
        // that carries the open period's fine bars so further appends fold in.
        Ok(PyDataFrame {
            inner: frame,
            tf: Some(TfState {
                time_frame: target,
                cumulators: spec,
                open: cum.open_clone(),
                fold_plan: None,
            }),
            window: None,
        })
    }

    /// Rename columns (pandas `rename(columns={old: new})`), returning a new
    /// frame.
    #[pyo3(signature = (columns))]
    pub(crate) fn rename(&self, columns: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let mut mapping = HashMap::new();
        for (k, v) in columns.iter() {
            mapping.insert(k.extract::<String>()?, v.extract::<String>()?);
        }
        // F39: a rename must not collide two columns onto one name (duplicate
        // column names violate the unique-name contract) — fail-loud (C4).
        let result: Vec<String> = df
            .names()
            .iter()
            .map(|n| mapping.get(n).cloned().unwrap_or_else(|| n.clone()))
            .collect();
        let mut seen = std::collections::HashSet::new();
        for n in &result {
            if !seen.insert(n) {
                return Err(PyValueError::new_err(format!(
                    "rename would produce duplicate column {n:?}"
                )));
            }
        }
        Ok(PyDataFrame::plain(df.rename(&mapping).map_err(pyerr)?))
    }

    /// Move a column into the row index (pandas `set_index(col)`), returning a
    /// new frame. A datetime / int / string column becomes the matching index.
    #[pyo3(signature = (keys))]
    pub(crate) fn set_index(&self, keys: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        Ok(PyDataFrame::plain(
            self.logical().set_index(keys).map_err(pyerr)?,
        ))
    }

    /// Cast columns to new dtypes (pandas `astype({col: dtype})`), returning a
    /// new frame.
    pub(crate) fn astype(&self, dtypes: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let mut df = self.logical().into_owned();
        let mut mapping = HashMap::new();
        for (k, v) in dtypes.iter() {
            let name = k.extract::<String>()?;
            let dt = v.extract::<String>()?;
            if let Some(unit) = datetime_unit_of(&dt) {
                // datetime target: parse a string column, or scale a numeric epoch
                // column by the dtype's unit (truncating, like a NumPy
                // `datetime64[unit]` cast).
                let col = df.column(&name).map_err(pyerr)?.clone();
                let converted = match &col {
                    Column::Datetime(_) | Column::Str(_, _) => col.to_datetime().map_err(pyerr)?,
                    _ => col.epoch_to_datetime(unit).map_err(pyerr)?,
                };
                df.set_column(&name, converted).map_err(pyerr)?;
            } else {
                mapping.insert(name, parse_dtype(&dt)?);
            }
        }
        if !mapping.is_empty() {
            df = df.astype(&mapping).map_err(pyerr)?;
        }
        Ok(PyDataFrame::plain(df))
    }
}

impl PyDataFrame {

    /// The `n` extreme rows by `column` (ascending for nsmallest).
    fn extreme_rows(&self, n: usize, column: &str, ascending: bool) -> PyResult<PyDataFrame> {
        let view = self.logical();
        let df = view.as_ref();
        let col = df.column(column).map_err(pyerr)?;
        col.require_numeric().map_err(pyerr)?;
        let v = col.to_f64_vec();
        let mut order: Vec<usize> = (0..v.len()).filter(|&i| col.is_valid(i) && !v[i].is_nan()).collect();
        order.sort_by(|&a, &b| {
            let o = v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal);
            if ascending { o } else { o.reverse() }
        });
        order.truncate(n);
        Ok(PyDataFrame::plain(take_frame(df, &order)))
    }

    /// Row-level duplicate mask over all columns, honoring `keep` ('first'|'last').
    fn row_duplicated(&self, keep: &str) -> PyResult<Vec<bool>> {
        let view = self.logical();
        let df = view.as_ref();
        let h = df.height();
        let key_of = |i: usize| -> Vec<Option<String>> {
            df.columns()
                .iter()
                .map(|c| crate::series::cell_key(c, i))
                .collect()
        };
        let mut seen = std::collections::HashSet::with_capacity(h);
        match keep {
            "first" => Ok((0..h).map(|i| !seen.insert(key_of(i))).collect()),
            "last" => {
                let mut out = vec![false; h];
                for i in (0..h).rev() {
                    out[i] = !seen.insert(key_of(i));
                }
                Ok(out)
            }
            other => Err(PyValueError::new_err(format!(
                "keep must be 'first' or 'last', got {other:?}"
            ))),
        }
    }
}
