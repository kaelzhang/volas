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
    /// The columns the frame keeps by declaration (canonical names): those declared in
    /// `max_lookback`, and the cached columns a copied source brought.
    declared: Vec<String>,
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
        Ok(WindowState { window, lookback, capacity, bound, declared: Vec::new() })
    }

    /// The window and its margin, `window + max_lookback` rows: the rows its labels stay
    /// unique over.
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
    fn bound(&mut self, Bounds { mut state, declared }: Bounds) -> PyResult<()> {
        let cached: Vec<String> =
            self.inner.names().iter().filter(|n| self.inner.is_computed(n)).cloned().collect();
        state.declared = cached.clone();
        self.window = Some(state);
        for name in &cached {
            self.check_kept(&*parse(name).map_err(directive_err)?, name)?;
        }
        self.checkpoint_carried_states();
        let mut nodes = Vec::with_capacity(declared.len());
        for text in &declared {
            let node = parse(text).map_err(directive_err)?;
            self.check_kept(&node, text)?;
            nodes.push(node);
        }
        for (text, node) in declared.iter().zip(&nodes) {
            let canonical = stringify(node);
            if !self.inner.is_computed(&canonical) {
                self.cache_new(text, node)?;
            }
            self.window.as_mut().expect("bounded above").declared.push(canonical);
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
    /// on a bounded one only a class-1 directive (it depends on a trailing window with no
    /// running accumulator) whose window the margin holds. Classes 2 and 3 must be
    /// declared at construction. The error names the two correct paths — or, for a
    /// directive this frame already caches, its column.
    pub(crate) fn admit_new(&self, node: &Ast, text: &str) -> PyResult<()> {
        let Some(w) = &self.window else { return Ok(()) };
        let reach = volas_directive::lookback::reach(node);
        let why = match reach.history {
            History::Window if reach.lookback <= w.lookback => return Ok(()),
            History::Window => format!(
                "it reads {} rows before each row, more than the {} it keeps (max_lookback)",
                reach.lookback, w.lookback
            ),
            History::Unkept => return Err(self.unkept(node, text).expect("an unkept directive")),
            History::Accumulator => "it carries a running accumulator (a class-2 directive)".to_string(),
            History::Counted => "it counts a run of rows (a class-3 directive)".to_string(),
            History::Carried { .. } => format!(
                "each of its values depends on every row before it, and the frame keeps only its \
                 last {} rows",
                w.capacity
            ),
        };
        let canonical = stringify(node);
        if self.inner.is_computed(&canonical) {
            return Err(PyValueError::new_err(format!(
                "cannot compute '{text}' from this windowed frame's rows: {why}. It is declared — \
                 read its column df['{canonical}']."
            )));
        }
        Err(PyValueError::new_err(format!(
            "cannot compute '{text}' on this windowed frame: {why}. Declare it when constructing \
             the frame — DataFrame(data, window={}, max_lookback=[..., '{text}']) — and read its \
             column df['{text}'], or compute it on an unbounded frame.",
            w.window
        )))
    }

    /// Whether every declared directive is past its warm-up on every visible row: its
    /// first present value (where its warm-up ended, counted in the rows it reads
    /// complete) is at or before the first visible row. O(declared columns).
    pub(crate) fn warmed(&self, w: &WindowState) -> bool {
        let first_visible = self.inner.origin() + self.logical_range().0;
        w.declared
            .iter()
            .filter(|n| self.inner.is_computed(n))
            .all(|n| self.inner.computed_first_present(n).is_some_and(|f| f <= first_visible))
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
        for (name, history, since) in self.inner.computed_histories() {
            let History::Carried { operands } = history else { continue };
            if since >= head {
                continue; // its history starts in the kept rows
            }
            // the last row before the kept rows whose operands the kept rows recompute
            let end = head + operands - origin; // frame rows the checkpoint covers
            let at = origin + end - 1;
            if let Some(cp) = self.inner.take_computed_pending(&name, at) {
                self.inner.set_computed_checkpoint(&name, Ok(cp)); // the refresh passed it
                continue;
            }
            let Ok(node) = parse(&name) else { continue }; // LCOV_EXCL_LINE (a cached name parses)
            let frame = self.history_frame(&node, &mut base);
            let checkpoint = match self.inner.computed_checkpoint(&name) {
                // a checkpoint at or past the new head (formed from a carried state) stays
                Some(cp) if cp.at >= at => Some(cp.clone()),
                Some(cp) => volas_directive::exec::replay(frame, &node, cp, end).map(|(_, cp)| cp),
                // its history starts inside the frame: capture it from there
                None if since >= origin => {
                    let start = since - origin;
                    match start {
                        0 => volas_directive::exec::capture(frame, &node, end),
                        _ => volas_directive::exec::capture(&frame.slice_data(start, h), &node, end - start),
                    }
                }
                None => None, // the rows its history started on are gone with no checkpoint
            };
            // none formed: an input the computation cannot read is why, if one is (a reason
            // found at an earlier drop stays — its row may be gone now)
            let checkpoint = checkpoint.ok_or_else(|| {
                self.inner
                    .computed_history_lost(&name)
                    .cloned()
                    .or_else(|| volas_directive::exec::execute_refresh(frame, &node).err())
            });
            self.inner.set_computed_checkpoint(&name, checkpoint);
        }
        self.inner = self.inner.retain_tail(keep, &Replayer);
    }

    /// The frame a cached `node` reads its inputs from: the live frame for a directive on
    /// its default series, else the refresh frame (`base`, made once).
    fn history_frame<'a>(&'a self, node: &Ast, base: &'a mut Option<volas_core::DataFrame>) -> &'a volas_core::DataFrame {
        if volas_directive::uses_default_series(node) {
            return &self.inner;
        }
        if base.is_none() {
            *base = self.refresh_frame().ok();
        }
        base.as_ref().unwrap_or(&self.inner)
    }

    /// A recursion the frame carries in a state whose history started before the frame's
    /// row 0 (a source frame's tail): its checkpoint is that state, as of its last valid
    /// row, with the inputs a resume from it reads — when the frame holds them.
    fn checkpoint_carried_states(&mut self) {
        let origin = self.inner.origin();
        let mut base: Option<volas_core::DataFrame> = None;
        for (name, meta) in self.inner.computed_columns() {
            if !matches!(meta.reach.history, History::Carried { .. })
                || meta.since >= origin
                || self.inner.computed_checkpoint(&name).is_some()
            {
                continue;
            }
            let Ok(node) = parse(&name) else { continue }; // LCOV_EXCL_LINE (a cached name parses)
            let state = self.inner.computed_resume_state(&name).map(|(s, _)| s.to_vec());
            let frame = self.history_frame(&node, &mut base);
            let cp = state.and_then(|s| volas_directive::exec::checkpoint(frame, &node, Some(s), meta.valid_rows, false));
            self.inner.set_computed_checkpoint(&name, cp.ok_or(None));
        }
    }
}
