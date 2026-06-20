//! `Series` structural ops (sort, head/tail, unique, value_counts, dedup,
//! reset_index, rename, to_frame, monotonicity checks).

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use volas_core::{
    Column, Index, Series,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {

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

    /// Sort by value (pandas `sort_values`), stable; `na_position` places the
    /// missing values `'last'` (default) or `'first'`; the index follows.
    #[pyo3(signature = (ascending = true, na_position = "last"))]
    pub(crate) fn sort_values(&self, ascending: bool, na_position: &str) -> PyResult<PySeries> {
        let perm = self.inner.data.argsort(ascending);
        let perm = match na_position {
            "last" => perm,
            "first" => {
                // argsort sinks NA last; rotate the NA block to the front, both
                // halves keeping their stable order.
                let (mut nas, present): (Vec<usize>, Vec<usize>) =
                    perm.into_iter().partition(|&i| !self.inner.data.is_valid(i));
                nas.extend(present);
                nas
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "sort_values: na_position must be 'first' or 'last', got {other:?}"
                )))
            }
        };
        Ok(self.reindexed(&perm))
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

    /// The values as a Python list of typed scalars (pandas `to_list`).
    /// Counts of unique values, most frequent first, indexed by the value
    /// (pandas `value_counts`). Discrete dtypes only: volas has no float index,
    /// so a float series must be rounded / astype'd first (C4 fail-loud).
    #[pyo3(signature = (normalize = false, sort = true, ascending = false, dropna = true))]
    pub(crate) fn value_counts(
        &self,
        normalize: bool,
        sort: bool,
        ascending: bool,
        dropna: bool,
    ) -> PyResult<PySeries> {
        if !dropna {
            // an NA bucket would need an NA index label, which int/str indexes
            // forbid by design (the NA-label guard) — fail loud, not silently drop.
            return Err(PyValueError::new_err(
                "value_counts(dropna=False) is unsupported: a volas index has no                  missing-label slot; count NA separately via isna().sum()",
            ));
        }
        let n = self.inner.len();
        let mut order: Vec<String> = Vec::new();
        let mut counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut sample: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for i in 0..n {
            let Some(key) = cell_key(&self.inner.data, i) else { continue };
            match counts.get_mut(&key) {
                Some(c) => *c += 1,
                None => {
                    counts.insert(key.clone(), 1);
                    sample.insert(key.clone(), i);
                    order.push(key);
                }
            }
        }
        // most frequent first (or ascending); ties keep first-appearance order.
        if sort {
            if ascending {
                order.sort_by_key(|k| counts[k]);
            } else {
                order.sort_by_key(|k| std::cmp::Reverse(counts[k]));
            }
        }
        let positions: Vec<usize> = order.iter().map(|k| sample[k]).collect();
        let labels = self.inner.data.take(&positions);
        let index = match &labels {
            Column::I64(v, _) => Index::int64(v.to_vec()),
            Column::I32(v, _) => Index::int64(v.iter().map(|&x| x as i64).collect()),
            Column::Bool(v, _) => Index::int64(v.iter().map(|&b| b as i64).collect()),
            Column::Str(v, _) => Index::str(v.to_vec()),
            Column::Datetime(v) => Index::datetime(v.to_vec(), self.inner.index.tz()),
            _ => {
                return Err(PyTypeError::new_err(
                    "value_counts needs discrete labels; volas has no float index — \
                     round or astype the series first",
                ))
            }
        };
        let data = if normalize {
            let total: i64 = counts.values().sum();
            Column::f64(order.iter().map(|k| counts[k] as f64 / total as f64).collect())
        } else {
            Column::i64(order.iter().map(|k| counts[k]).collect())
        };
        Ok(PySeries { inner: Series::new(None, data, Arc::new(index)) })
    }

    /// The most frequent value(s), ascending, on a fresh RangeIndex (pandas `mode`).
    pub(crate) fn mode(&self) -> PySeries {
        let n = self.inner.len();
        let mut counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut sample: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for i in 0..n {
            let Some(key) = cell_key(&self.inner.data, i) else { continue };
            *counts.entry(key.clone()).or_insert(0) += 1;
            sample.entry(key).or_insert(i);
        }
        let top = counts.values().copied().max().unwrap_or(0);
        let mut positions: Vec<usize> = counts
            .iter()
            .filter(|(_, &c)| c == top)
            .map(|(k, _)| sample[k])
            .collect();
        positions.sort_unstable();
        let data = self.inner.data.take(&positions);
        let h = positions.len();
        PySeries { inner: Series::new(self.inner.name.clone(), data, Arc::new(Index::range(h))) }
    }

    /// The `n` largest values, descending (pandas `nlargest`).
    pub(crate) fn nlargest(&self, n: i64) -> PyResult<PySeries> {
        if n < 0 {
            return Err(PyValueError::new_err("n must be >= 0"));
        }
        self.inner.data.require_numeric().map_err(pyerr)?;
        let sorted = self.sort_values(false, "last")?;
        Ok(slice_head(&sorted.inner, n as usize))
    }

    /// The `n` smallest values, ascending (pandas `nsmallest`).
    pub(crate) fn nsmallest(&self, n: i64) -> PyResult<PySeries> {
        if n < 0 {
            return Err(PyValueError::new_err("n must be >= 0"));
        }
        self.inner.data.require_numeric().map_err(pyerr)?;
        let sorted = self.sort_values(true, "last")?;
        Ok(slice_head(&sorted.inner, n as usize))
    }

    /// Drop duplicate values, keeping the `keep` occurrence (`'first'` |
    /// `'last'`; pandas `drop_duplicates`).
    #[pyo3(signature = (keep = "first"))]
    pub(crate) fn drop_duplicates(&self, keep: &str) -> PyResult<PySeries> {
        let dup = duplicated_mask_keep(&self.inner.data, keep)?;
        let positions: Vec<usize> = (0..self.inner.len()).filter(|&i| !dup[i]).collect();
        let data = self.inner.data.take(&positions);
        let index = Arc::new(self.inner.index.take(&positions));
        Ok(PySeries { inner: Series::new(self.inner.name.clone(), data, index) })
    }

    /// True for each duplicate occurrence other than the `keep` one (`'first'` |
    /// `'last'`; pandas `duplicated`).
    #[pyo3(signature = (keep = "first"))]
    pub(crate) fn duplicated(&self, keep: &str) -> PyResult<PySeries> {
        Ok(bool_series(&self.inner, duplicated_mask_keep(&self.inner.data, keep)?))
    }

    /// Whether the values are monotonically non-decreasing, NA-free (pandas).
    #[getter]
    pub(crate) fn is_monotonic_increasing(&self) -> bool {
        monotonic(&self.inner.data, true)
    }

    /// Whether the values are monotonically non-increasing, NA-free (pandas).
    #[getter]
    pub(crate) fn is_monotonic_decreasing(&self) -> bool {
        monotonic(&self.inner.data, false)
    }

    /// Whether every value is distinct (pandas `is_unique`); NA cells count as
    /// one shared "missing" value.
    #[getter]
    pub(crate) fn is_unique(&self) -> bool {
        !duplicated_mask_keep(&self.inner.data, "first")
            .expect("'first' is valid")
            .iter()
            .any(|&d| d)
    }

    /// Restore a RangeIndex. `drop=True` returns a Series; otherwise (pandas)
    /// the old index becomes an `'index'` column of a 2-column DataFrame.
    #[pyo3(signature = (drop = false))]
    pub(crate) fn reset_index(&self, py: Python<'_>, drop: bool) -> PyResult<Py<PyAny>> {
        let h = self.inner.len();
        if drop {
            let s = PySeries {
                inner: Series::new(
                    self.inner.name.clone(),
                    self.inner.data.clone(),
                    Arc::new(Index::range(h)),
                ),
            };
            return Ok(Py::new(py, s)?.into_any());
        }
        let label = self.inner.index.name().unwrap_or("index").to_string();
        let vname = self.inner.name.clone().unwrap_or_else(|| "0".to_string());
        let df = volas_core::DataFrame::new(
            vec![label, vname],
            vec![self.inner.index.to_column(), self.inner.data.clone()],
            Some(Index::range(h)),
        )
        .map_err(pyerr)?;
        Ok(Py::new(py, PyDataFrame::plain(df))?.into_any())
    }

    /// Sort by index labels (pandas `sort_index`).
    #[pyo3(signature = (ascending = true))]
    pub(crate) fn sort_index(&self, ascending: bool) -> PySeries {
        let perm = self.inner.index.argsort(ascending);
        let data = self.inner.data.take(&perm);
        let index = Arc::new(self.inner.index.take(&perm));
        PySeries { inner: Series::new(self.inner.name.clone(), data, index) }
    }

    /// A copy with a new name (pandas scalar `rename`).
    #[pyo3(signature = (name = None))]
    pub(crate) fn rename(&self, name: Option<String>) -> PySeries {
        PySeries {
            inner: Series::new(name, self.inner.data.clone(), Arc::clone(&self.inner.index)),
        }
    }

    /// An independent copy (pandas `copy`; columns are copy-on-write).
    pub(crate) fn copy(&self) -> PySeries {
        PySeries { inner: self.inner.clone() }
    }

    /// This series as a 1-column DataFrame (pandas `to_frame`).
    #[pyo3(signature = (name = None))]
    pub(crate) fn to_frame(&self, name: Option<String>) -> PyResult<PyDataFrame> {
        let col_name = name
            .or_else(|| self.inner.name.clone())
            .unwrap_or_else(|| "0".to_string());
        let df = volas_core::DataFrame::new(
            vec![col_name],
            vec![self.inner.data.clone()],
            Some((*self.inner.index).clone()),
        )
        .map_err(pyerr)?;
        Ok(PyDataFrame::plain(df))
    }
}
