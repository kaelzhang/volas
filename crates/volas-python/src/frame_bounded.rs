//! A bounded rolling-window frame (`window=` / `max_lookback=`) and its history: how
//! many rows it retains, which directives it computes exactly, and the compaction that
//! drops its head while every recursive column keeps the checkpoint of its history.
//!
//! The frame shows its last `window` rows and keeps a hidden margin of `max_lookback`
//! rows behind them (and at least `max_lookback + 2` rows in all, so a replaced last row
//! resumes from the row before it, which reads `max_lookback + 1` rows behind). Its
//! physical rows grow to twice that retention, then the head is dropped back to it —
//! amortized O(1) per row, memory bounded.
//!
//! A value that depends only on a trailing window of `lookback` rows (no recursion
//! anywhere — the sliding sums are exact, so a running accumulator is window-local too)
//! is computed exactly from the retained rows whenever `lookback <= max_lookback`. A
//! recursion depends on every row before: the frame computes it exactly only when it is
//! **declared** in `max_lookback` — computed at construction over the whole seed, then
//! carried across every drop by a checkpoint (`volas_core::Checkpoint`) from which a
//! recompute replays. Anything else requested on a windowed frame cannot be exact and
//! raises, naming the two correct paths: declare it, or compute it on an unbounded frame.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::History;
use volas_directive::types::Ast;
use volas_directive::{parse, stringify};

#[allow(unused_imports)]
use crate::*;

/// The windowed-frame lookback bound — how `max_lookback=` is supplied: either an
/// explicit row count, or a list of indicator directives whose largest lookback it is
/// derived from (so the caller never hand-computes a compound indicator's warm-up).
#[derive(FromPyObject)]
pub(crate) enum LookbackBound {
    Count(usize),
    Directives(Vec<String>),
}

/// A bounded frame's retention (see the module docs). Every count is checked when it is
/// made, so retaining and dropping never overflow.
#[derive(Clone)]
pub(crate) struct WindowState {
    /// The output window M — the logical row count once warmed.
    pub(crate) window: usize,
    /// The hidden margin: the largest lookback a directive computed on the frame may
    /// read (`max_lookback`).
    pub(crate) lookback: usize,
    /// The rows retained after a drop: `window + lookback`, at least `lookback + 2`.
    pub(crate) capacity: usize,
    /// The physical row count that triggers a drop: twice the capacity.
    bound: usize,
}

impl WindowState {
    /// The retention of a `window`-row frame with a `lookback`-row margin; a count that
    /// does not fit in a machine word is a `ValueError`.
    fn new(window: usize, lookback: usize) -> PyResult<WindowState> {
        let too_large = || {
            PyValueError::new_err(format!(
                "window={window} with max_lookback={lookback} is too large: the frame could not \
                 count the rows it retains"
            ))
        };
        let capacity = window.max(2).checked_add(lookback).ok_or_else(too_large)?;
        let bound = capacity.checked_mul(2).ok_or_else(too_large)?;
        Ok(WindowState { window, lookback, capacity, bound })
    }

    /// The window and its margin, `window + max_lookback` rows: what a warmed frame
    /// holds (`ready`) and the rows its labels stay unique over.
    pub(crate) fn window_and_margin(&self) -> usize {
        self.window + self.lookback // fits: at most `capacity`
    }
}

/// A windowed frame's construction arguments, validated: its retention and the declared
/// directives (`max_lookback=[...]`), parsed.
pub(crate) struct Bounds {
    state: WindowState,
    declared: Vec<String>,
}

/// Validate the windowing arguments — before any data is touched. `max_lookback` only
/// applies with `window`; with `window` it is required, an int or a non-empty list of
/// indicator directives (whose largest lookback is the margin).
pub(crate) fn bounds(window: Option<usize>, max_lookback: Option<LookbackBound>) -> PyResult<Option<Bounds>> {
    let Some(w) = window else {
        if max_lookback.is_some() {
            return Err(PyValueError::new_err(
                "max_lookback only applies to a windowed frame — pass window=",
            ));
        }
        return Ok(None);
    };
    if w == 0 {
        return Err(PyValueError::new_err("window must be a positive integer"));
    }
    let (lookback, declared) = match max_lookback {
        None => {
            return Err(PyValueError::new_err(
                "a windowed frame needs its lookback bound — pass max_lookback= \
                 (an int, or a list of indicator directives to derive it from)",
            ))
        }
        Some(LookbackBound::Count(l)) => (l, Vec::new()),
        Some(LookbackBound::Directives(inds)) => {
            if inds.is_empty() {
                return Err(PyValueError::new_err(
                    "max_lookback=[] declares no directive — pass the directives the frame \
                     computes, or an int (max_lookback=0 for none)",
                ));
            }
            // Each entry must name an INDICATOR: `indicator_lookback` rejects a typo'd
            // / bare-column directive (e.g. `ma5` for `ma:5`) instead of deriving a
            // silent 0-row margin.
            let mut max = 0usize;
            for d in &inds {
                let node = parse(d).map_err(pyerr)?;
                max = max.max(volas_directive::lookback::indicator_lookback(&node).map_err(pyerr)?);
            }
            (max, inds)
        }
    };
    Ok(Some(Bounds { state: WindowState::new(w, lookback)?, declared }))
}

impl PyDataFrame {
    /// This freshly built frame, bounded by `bounds` when given (see [`bound`](Self::bound)).
    pub(crate) fn bounded(mut self, bounds: Option<Bounds>) -> PyResult<Self> {
        if let Some(b) = bounds {
            self.bound(b)?;
        }
        Ok(self)
    }

    /// Make this freshly built frame bounded: every cached column it carries (a copied
    /// source's) and every declared directive must be one the window keeps exactly; the
    /// declared ones are computed over all the given rows — the whole history — and
    /// cached; then the head beyond the retention is dropped, each recursive column
    /// keeping the checkpoint of its history.
    fn bound(&mut self, Bounds { state, declared }: Bounds) -> PyResult<()> {
        self.window = Some(state);
        let cached: Vec<String> =
            self.inner.names().iter().filter(|n| self.inner.is_computed(n)).cloned().collect();
        for name in &cached {
            self.check_kept(&*parse(name).map_err(directive_err)?, name)?;
        }
        let mut nodes = Vec::with_capacity(declared.len());
        for text in &declared {
            let node = parse(text).map_err(directive_err)?;
            self.check_kept(&node, text)?;
            nodes.push(node);
        }
        for (text, node) in declared.iter().zip(&nodes) {
            if !self.inner.is_computed(&stringify(node)) {
                self.cache_new(text, node)?;
            }
        }
        self.maybe_compact();
        Ok(())
    }

    /// The error of a directive no bounded frame keeps exactly (a recursion no carried
    /// state covers), or `None`.
    fn unkept(&self, node: &Ast, text: &str) -> Option<PyErr> {
        (volas_directive::lookback::history(node) == History::Unkept).then(|| {
            PyValueError::new_err(format!(
                "cannot keep '{text}' exact on a windowed frame: it depends on every row \
                 before through a recursion no carried state covers (one inside an operand or \
                 combined with other values, or a command that recomputes it), so a frame that \
                 drops its oldest rows cannot recompute it. Declare the recursive directives it \
                 combines and combine their columns, or compute it on an unbounded frame."
            ))
        })
    }

    /// `node` (spelled `text`) is kept exactly by this bounded frame once declared: it is
    /// not [`unkept`](Self::unkept), and it reads no more rows than the margin holds.
    fn check_kept(&self, node: &Ast, text: &str) -> PyResult<()> {
        let w = self.window.as_ref().expect("a bounded frame");
        if let Some(err) = self.unkept(node, text) {
            return Err(err);
        }
        let lookback = volas_directive::lookback::lookback(node);
        if lookback > w.lookback {
            return Err(PyValueError::new_err(format!(
                "cannot keep '{text}' on a windowed frame with max_lookback={}: it reads {lookback} \
                 rows before each row. Declare it in max_lookback, or drop the column first.",
                w.lookback
            )));
        }
        Ok(())
    }

    /// Whether a directive requested after construction (`df[d]`, `df[[...]]`,
    /// `df.exec(d)`) can be computed exactly on this frame — always on an unbounded one;
    /// on a bounded one only when it depends on a trailing window its margin holds. A
    /// recursion must be declared at construction. The error names the two correct paths.
    pub(crate) fn admit_new(&self, node: &Ast, text: &str) -> PyResult<()> {
        let Some(w) = &self.window else { return Ok(()) };
        if let Some(err) = self.unkept(node, text) {
            return Err(err);
        }
        let reach = volas_directive::lookback::reach(node);
        if reach.history.recursive() {
            return Err(PyValueError::new_err(format!(
                "cannot compute '{text}' on this windowed frame: each of its values depends on \
                 every row before it, and the frame keeps only its last {} rows. Declare it when \
                 constructing the frame — DataFrame(data, window={}, max_lookback=[..., '{text}']) \
                 — and read its column df['{text}'], or compute it on an unbounded frame.",
                w.capacity, w.window
            )));
        }
        if reach.lookback > w.lookback {
            return Err(PyValueError::new_err(format!(
                "cannot compute '{text}' on this windowed frame: it reads {} rows before each \
                 row, more than the {} it keeps (max_lookback). Declare it when constructing the \
                 frame — DataFrame(data, window={}, max_lookback=[..., '{text}']) — and read its \
                 column df['{text}'], or compute it on an unbounded frame.",
                reach.lookback, w.lookback, w.window
            )));
        }
        Ok(())
    }

    /// After rows were added: once the physical rows reach the drop bound, drop the head
    /// down to the retention (amortized O(1) per row). No-op when unbounded.
    pub(crate) fn maybe_compact(&mut self) {
        let Some(w) = &self.window else { return };
        if self.inner.height() >= w.bound {
            self.compact(w.capacity);
        }
    }

    /// Keep the last `keep` rows. Every recursive column whose history starts in the
    /// dropped rows first moves its checkpoint to the new head — replaying it over the
    /// dropped rows, or capturing it from the history the frame holds — so a later
    /// recompute replays from there. Nothing is refreshed: a stale column stays stale (its
    /// cursor and carried states move with the rows, `DataFrame::retain_tail`), and the
    /// checkpoint reads input rows only, so dropping never fails. A recursion whose
    /// checkpoint cannot be formed keeps none; recomputing it then raises.
    fn compact(&mut self, keep: usize) {
        let h = self.inner.height();
        if keep >= h {
            return;
        }
        let origin = self.inner.origin();
        let head = origin + (h - keep); // the absolute position of the first kept row
        let mut base: Option<volas_core::DataFrame> = None;
        for (name, meta) in self.inner.computed_columns() {
            let History::Carried { operands } = meta.reach.history else { continue };
            if meta.since >= head {
                continue; // its history starts in the kept rows
            }
            // the last row before the kept rows whose operands the kept rows recompute
            let end = head + operands - origin; // frame rows the checkpoint covers
            let Ok(node) = parse(&name) else { continue }; // LCOV_EXCL_LINE (a cached name parses)
            if base.is_none() && !directive_uses_default_series(&node) {
                base = self.refresh_frame().ok();
            }
            let frame = match &base {
                Some(f) if !directive_uses_default_series(&node) => f,
                _ => &self.inner,
            };
            let checkpoint = match self.inner.computed_checkpoint(&name) {
                Some(cp) => volas_directive::exec::replay(frame, &node, cp, end).map(|(_, cp)| cp),
                // its history starts inside the frame: capture it from there
                None if meta.since >= origin => {
                    let start = meta.since - origin;
                    match start {
                        0 => volas_directive::exec::capture(frame, &node, end),
                        _ => volas_directive::exec::capture(&frame.slice_data(start, h), &node, end - start),
                    }
                }
                None => None, // the rows its history started on are gone with no checkpoint
            };
            self.inner.set_computed_checkpoint(&name, checkpoint);
        }
        self.inner = self.inner.retain_tail(keep);
    }
}
