//! The binding's projection of the one derivation owner — `volas_core::DataFrame`'s
//! derivations (`volas-core/src/dataframe/derive.rs` states what each keeps). This module
//! only maps a Python selection onto the visible rows of the physical frame, hands it the
//! directive layer's replay of a recursion's carried states ([`Replayer`]), and keeps a
//! frame's wrappers where the result is still a frame of that kind:
//!
//! - a **time-frame** frame's model (its time frame, aggregators and open period) is kept
//!   by a result that is still a time-frame frame of the same bars: a selection ending at
//!   the forming row keeps the open period; any other selection of bars in time order is
//!   the time-frame frame of its rows, its last one the forming bar (the constructor's
//!   seed rule); a projection, a rename and a time-zone change whose bars are still
//!   periods of the frame's grid carry the open period over. Rows out of time order, or
//!   labels that are not periods of the grid, give a plain frame.
//! - a **bounded window** is kept by `copy()` / `DataFrame(wf)` alone: a frame derived
//!   from a windowed frame is an unbounded frame of its visible rows (the hidden margin
//!   only serves as the lead of its cached columns).

use std::collections::HashMap;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use volas_core::DataFrame;
use volas_directive::Replayer;

#[allow(unused_imports)]
use crate::*;

/// A selection of the visible rows, by position.
pub(crate) enum RowSel {
    /// The rows `[start, end)`.
    Range(usize, usize),
    /// The rows at these positions, in this order.
    Positions(Vec<usize>),
}

impl RowSel {
    /// The rows a boolean `mask` over the visible rows selects.
    pub(crate) fn mask(mask: &[bool]) -> RowSel {
        RowSel::Positions(mask.iter().enumerate().filter_map(|(i, &b)| b.then_some(i)).collect())
    }

    /// The positions these rows are, in order.
    pub(crate) fn positions(&self) -> Vec<usize> {
        match self {
            RowSel::Range(a, b) => (*a..(*b).max(*a)).collect(),
            RowSel::Positions(p) => p.clone(),
        }
    }

    /// The contiguous range these rows are, if they are one (in order, at least one row).
    fn range(&self) -> Option<(usize, usize)> {
        match self {
            RowSel::Range(a, b) => Some((*a, (*b).max(*a))),
            RowSel::Positions(p) => match p.as_slice() {
                [first, ..] if p.windows(2).all(|w| w[1] == w[0] + 1) => Some((*first, first + p.len())),
                _ => None,
            },
        }
    }
}

impl Fresh<'_> {
    /// The visible rows `[a, b)` as the contiguous selection of the physical frame (its
    /// hidden margin, if any, serving as the lead).
    pub(crate) fn contiguous(&self, a: usize, b: usize) -> DataFrame {
        let (start, len) = self.range();
        let (a, b) = (a.min(len), b.max(a).min(len));
        self.physical().rows(start + a, start + b, &Replayer)
    }

    /// The visible rows `sel` selects, as the one derivation decides it.
    pub(crate) fn selected(&self, sel: &RowSel) -> DataFrame {
        match (sel.range(), sel) {
            (Some((a, b)), _) => self.contiguous(a, b),
            (None, RowSel::Positions(p)) => {
                let start = self.range().0;
                let physical: Vec<usize> = p.iter().map(|&i| start + i).collect();
                self.physical().take_data(&physical)
            }
            (None, RowSel::Range(..)) => unreachable!("a range is contiguous"), // LCOV_EXCL_LINE
        }
    }
}

impl PyDataFrame {
    /// The visible rows `sel` selects as a derived frame (see the module docs): a
    /// contiguous selection continues the cached columns, any other is a new series; a
    /// time-frame frame's model is kept where the result is still one of its bars.
    pub(crate) fn derive_rows(&self, view: &Fresh, sel: &RowSel) -> PyResult<PyDataFrame> {
        let inner = view.selected(sel);
        let ends_forming = sel.range().is_some_and(|(a, b)| b > a && b == view.rows());
        let tf = match &self.tf {
            Some(t) if ends_forming => Some(t.clone()),
            Some(t) => t.relabelled(&inner)?,
            None => None,
        };
        Ok(PyDataFrame { inner, tf, window: None })
    }

    /// The visible rows a boolean `mask` over them selects ([`derive_rows`](Self::derive_rows)).
    pub(crate) fn masked(&self, mask: &[bool]) -> PyResult<PyDataFrame> {
        let view = self.read()?;
        if mask.len() != view.rows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                view.rows()
            )));
        }
        self.derive_rows(&view, &RowSel::mask(mask))
    }

    /// `inner`, a column projection of this frame's visible rows: the time-frame model
    /// follows the projection.
    pub(crate) fn derive_projection(&self, inner: DataFrame) -> PyResult<PyDataFrame> {
        let tf = match &self.tf {
            Some(t) => Some(t.projected(&inner)?),
            None => None,
        };
        Ok(PyDataFrame { inner, tf, window: None })
    }

    /// `inner`, a new series of this frame's visible rows re-labelled or re-typed (a new
    /// index, a cast): the time-frame frame of its rows when they are still bars in time
    /// order, else a plain frame.
    pub(crate) fn derive_relabelled(&self, inner: DataFrame) -> PyResult<PyDataFrame> {
        let tf = match &self.tf {
            Some(t) => t.relabelled(&inner)?,
            None => None,
        };
        Ok(PyDataFrame { inner, tf, window: None })
    }

    /// `inner`, this frame's visible rows renamed by `mapping`: the time-frame model keeps
    /// the open period under the new names.
    pub(crate) fn derive_renamed(&self, inner: DataFrame, mapping: &HashMap<String, String>) -> PyResult<PyDataFrame> {
        let tf = match &self.tf {
            Some(t) => Some(t.renamed(mapping)?),
            None => None,
        };
        Ok(PyDataFrame { inner, tf, window: None })
    }

    /// `inner`, this frame's visible rows in another zone, `rezone` moving a frame of
    /// bars there too: the time-frame model is kept when every row is still a period of
    /// the grid in that zone and the open period's bars fall in the last one.
    pub(crate) fn derive_rezoned(
        &self,
        inner: DataFrame,
        rezone: impl Fn(&DataFrame) -> volas_core::Result<DataFrame>,
    ) -> PyResult<PyDataFrame> {
        let tf = match &self.tf {
            Some(t) => t.rezoned(&inner, rezone)?,
            None => None,
        };
        Ok(PyDataFrame { inner, tf, window: None })
    }
}

/// The error of a 2-D selection of rows that are not contiguous naming cached column
/// `name`: such a selection is a new series, which does not carry it.
pub(crate) fn uncarried_cached(name: &str) -> PyErr {
    PyKeyError::new_err(format!(
        "'{name}' is a cached directive column: a selection of rows that are not contiguous is \
         a new series, which does not carry it. Read its values as a Series — \
         df.loc[rows, '{name}'] — or compute it on the selected frame."
    ))
}
