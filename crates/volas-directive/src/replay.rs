//! The directive layer's half of a derived frame: what the core derivation owner
//! (`volas_core::DataFrame::rows`) cannot compute itself — a recursive cached column's
//! carried states as of a row its source frame does not carry them for ([`Replayer`]) —
//! and the frame a cached directive is computed over ([`refresh_frame`]).

use std::borrow::Cow;

use volas_core::{DataFrame, Replay};

use crate::types::{Ast, Node};

/// Whether `node` reads its frame's default series only — a bare command (`macd`) or a
/// command whose every operand is the default series — so it is computed over the live
/// frame rather than its [`refresh_frame`].
pub fn uses_default_series(node: &Ast) -> bool {
    match node {
        Node::Name(_) => true,
        Node::Command(cmd) => cmd.series.iter().all(|series| matches!(series, Node::Name(name) if name.is_empty())),
        _ => false,
    }
}

/// The frame a CACHED directive of `df` is computed over: its data columns, less those
/// named like a built-in command usable bare. In a cached directive such a bare name is
/// always the command: a data column of that name present at bind time made the name
/// ambiguous (an error), and one added later is refused while a cached directive reads it
/// — and the canonical form renders `change:` as `change`, so the computation must not
/// read it as the column. Borrowed when it drops nothing.
pub fn refresh_frame(df: &DataFrame) -> Cow<'_, DataFrame> {
    let drop = |n: &str| df.is_computed(n) || crate::is_bare_command(n);
    if !df.names().iter().any(|n| drop(n)) {
        return Cow::Borrowed(df);
    }
    let kept: Vec<String> = df.names().iter().filter(|n| !drop(n)).cloned().collect();
    Cow::Owned(df.select(&kept).expect("existing columns"))
}

/// What a cached directive carries after a computation over its whole history
/// ([`resume_states`]).
#[derive(Debug, Default)]
pub struct ResumeStates {
    /// Its resume state as of the last row.
    pub state: Option<Vec<f64>>,
    /// Its resume state as of the row before the last.
    pub before: Option<Vec<f64>>,
    /// The checkpoints of its history at its anchor rows
    /// ([`Anchors`](volas_core::Anchors)) — for a recursion carried by its state.
    pub anchors: Vec<volas_core::Checkpoint>,
}

/// What a cached directive `node` computed over `frame` carries ([`ResumeStates`]): its
/// resume states as of `frame`'s last row and the row before, read in place (none for a
/// directive with no resume kernel) — and, for a recursion carried by its state whose
/// history starts at `frame`'s row 0, the checkpoints of its history at the anchor rows
/// it passes: the state is carried there on the way to the last rows, so a later
/// selection of the frame finds its history's nearest point with no replay.
pub fn resume_states(frame: &DataFrame, node: &Ast) -> ResumeStates {
    let end = frame.height();
    let carried = matches!(crate::lookback::reach(node).history, volas_core::History::Carried { .. });
    let spacing = volas_core::dataframe::ANCHOR_SPACING;
    // the row counts at which an anchor row ends (never the last row, which a re-sent
    // row replaces)
    let first = spacing - frame.origin() % spacing;
    let marks: Vec<usize> = match carried {
        true => (first..end).step_by(spacing).collect(),
        false => Vec::new(),
    };
    let since = frame.origin();
    let mut at: Option<(usize, Vec<f64>)> = None; // the state as of row `.0 - 1`
    let mut points = Vec::with_capacity(marks.len());
    for k in marks {
        let state = at
            .as_ref()
            .and_then(|(done, s)| crate::exec::execute_resume(frame, node, s, *done, k, since))
            .map(|(_, s)| s)
            .or_else(|| crate::exec::initial_state(frame, node, k));
        at = state.map(|s| (k, s));
        points.extend(at.clone());
    }
    let (state, before) = states_upto(frame, node, end, at);
    let anchors = match points.is_empty() {
        true => Vec::new(),
        false => crate::exec::checkpoints(frame, node, points),
    };
    ResumeStates { state, before, anchors }
}

/// The states of `node` over `df`'s rows `[0, end)` as of rows `end - 1` and `end - 2`:
/// the state before the last resumed from `at` (the state as of an earlier row), or
/// from a full computation up to it, then the last row resumed from it — one pass, the
/// rows read in place.
fn states_upto(df: &DataFrame, node: &Ast, end: usize, at: Option<(usize, Vec<f64>)>) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let since = df.origin();
    let resume = |(done, s): (usize, Vec<f64>)| match done == end - 1 {
        true => Some(s),
        false => crate::exec::execute_resume(df, node, &s, done, end - 1, since).map(|(_, s)| s),
    };
    let before = (end > 1)
        .then(|| at.filter(|(done, _)| *done < end).and_then(resume).or_else(|| crate::exec::initial_state(df, node, end - 1)))
        .flatten();
    let resumed = before
        .as_deref()
        .and_then(|b| crate::exec::execute_resume(df, node, b, end - 1, end, since))
        .map(|(_, after)| after);
    match resumed {
        Some(after) => (Some(after), before),
        None => (crate::exec::initial_state(df, node, end), before.filter(|_| end > 1)),
    }
}

/// The directive layer's replay of a recursive cached column's history at a contiguous
/// selection's boundary ([`Replay`]). A checkpoint as of a row is replayed from the nearest
/// of the column's [`Anchors`](volas_core::Anchors) before it — or from the history's
/// start the frame holds, or the checkpoint it keeps of a history that started before its
/// row 0 — forming the anchors it passes, so any edge costs at most the anchor spacing
/// once the history has been passed.
pub struct Replayer;

impl Replay for Replayer {
    fn boundary(&self, frame: &DataFrame, name: &str, start: usize) -> Option<volas_core::Checkpoint> {
        checkpoint_at(frame, name, (frame.origin() + start).checked_sub(1)?)
    }
}

/// The carried states of recursive cached column `name` as of `frame`'s rows `end - 1` and
/// `end - 2` (the latter when there is one), replayed from the nearest point of its
/// history before them the frame keeps ([`checkpoint_at`]) — what a selection ending
/// before its source's last row continues the column from. `None` when they cannot be
/// replayed, or that point is more than `ANCHOR_SPACING` rows before them (the rows a
/// write left without anchors: the column is recomputed in one pass instead).
pub fn states_at(frame: &DataFrame, name: &str, end: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
    let last = (frame.origin() + end).checked_sub(1)?;
    replayed(frame, name, |history| {
        let from = history.nearest(last).map_or(history.history.since, |cp| cp.at + 1);
        if last - from.min(last) > volas_core::dataframe::ANCHOR_SPACING {
            return None;
        }
        let before = (end > 1).then(|| history.at(last - 1)).flatten();
        let at = match &before {
            Some(b) => history.step(b, last)?,
            None => history.at(last)?,
        };
        Some((at.state?, before.and_then(|b| b.state)))
    })
}

/// The checkpoint of recursive cached column `name`'s history as of absolute row `at` of
/// `frame` (a row it holds, or one of its lead's), replayed from the nearest point of it
/// before `at` the frame keeps — an anchor, its checkpoint, or the history's start in its
/// rows or its lead's. `None` when the frame keeps none of them before `at`, or the
/// replay cannot run.
pub fn checkpoint_at(frame: &DataFrame, name: &str, at: usize) -> Option<volas_core::Checkpoint> {
    replayed(frame, name, |history| history.at(at))
}

/// `f` over the history of recursive cached column `name` as `frame` holds it, ready to
/// replay ([`Replayed`]); `None` when the frame holds neither its start nor a checkpoint.
fn replayed<R>(frame: &DataFrame, name: &str, f: impl FnOnce(&Replayed<'_>) -> Option<R>) -> Option<R> {
    let node = crate::parse(name).ok()?;
    let base;
    let df: &DataFrame = if uses_default_series(&node) {
        frame
    } else {
        base = refresh_frame(frame);
        &base
    };
    let since = frame.computed_since(name)?;
    let anchors = frame.computed_anchors(name)?;
    f(&History { df, node: &node, name, frame, since, anchors }.over_lead()?)
}

/// A recursive cached column's history as a frame holds it, replayed to any row.
struct History<'a> {
    /// The rows it is computed over (the frame's, or its refresh frame's).
    df: &'a DataFrame,
    node: &'a Ast,
    /// The cached column (its canonical directive).
    name: &'a str,
    frame: &'a DataFrame,
    /// The absolute row the history starts at.
    since: usize,
    anchors: volas_core::Anchors,
}

/// The history as [`History::over_lead`] gives it: the rows a replay from a checkpoint
/// runs over — the frame's lead too, when it keeps one (an operand of the recursion reads
/// the rows before row 0).
struct Replayed<'a> {
    history: History<'a>,
    rows: std::borrow::Cow<'a, DataFrame>,
    /// The checkpoint the frame keeps of a history that started before its row 0.
    kept: Option<volas_core::Checkpoint>,
}

impl<'a> History<'a> {
    /// This history ready to replay; `None` when the frame holds neither its start (in
    /// its rows or its lead's) nor a checkpoint of it.
    fn over_lead(self) -> Option<Replayed<'a>> {
        let kept = self.frame.computed_checkpoint(self.name).cloned();
        let rows = match self.frame.lead_rows() {
            0 => std::borrow::Cow::Borrowed(self.df),
            lead => std::borrow::Cow::Owned(self.frame.led(self.df, lead)),
        };
        if kept.is_none() && self.since < rows.origin() {
            return None;
        }
        Some(Replayed { history: self, rows, kept })
    }
}

impl Replayed<'_> {
    /// The checkpoint as of absolute row `at`, replayed from the nearest anchor before it
    /// (or the history's start, or the frame's checkpoint), forming the anchors it passes —
    /// never at the frame's last row, which a re-sent row replaces.
    fn at(&self, at: usize) -> Option<volas_core::Checkpoint> {
        let h = &self.history;
        let spacing = volas_core::dataframe::ANCHOR_SPACING;
        let mut from = self.nearest(at);
        let last_row = h.frame.origin() + h.frame.height() - 1;
        loop {
            let done = from.as_ref().map_or(h.since, |cp| cp.at + 1); // the next row to replay
            let stop = ((done / spacing + 1) * spacing - 1).min(at); // the next anchor row, or `at`
            let next = match &from {
                // from the history's start, which the frame's rows (or its lead's) hold
                None => {
                    let k = stop + 1 - h.since;
                    let df = match h.since.checked_sub(self.rows.origin())? {
                        _ if h.df.origin() == h.since => std::borrow::Cow::Borrowed(h.df),
                        skip => std::borrow::Cow::Owned(self.rows.slice_data(skip, skip + k)),
                    };
                    crate::exec::checkpoint(&df, h.node, crate::exec::initial_state(&df, h.node, k), k, true)?
                }
                Some(cp) => self.step(cp, stop)?,
            };
            let next = std::sync::Arc::new(next);
            if (next.at + 1) % spacing == 0 && next.at < last_row {
                h.anchors.keep(std::sync::Arc::clone(&next));
            }
            if next.at == at {
                return Some(std::sync::Arc::unwrap_or_clone(next));
            }
            from = Some(next);
        }
    }

    /// The nearest checkpoint at or before absolute row `at` a replay can start from over
    /// the rows it holds — an anchor, or the frame's checkpoint; `None` for the history's
    /// start.
    fn nearest(&self, at: usize) -> Option<std::sync::Arc<volas_core::Checkpoint>> {
        let usable = |cp: &volas_core::Checkpoint| cp.at <= at && cp.at + 1 >= self.rows.origin();
        let kept = self.kept.clone().map(std::sync::Arc::new);
        self.history.anchors.nearest(at).filter(|cp| usable(cp)).or_else(|| kept.filter(|cp| usable(cp)))
    }

    /// The checkpoint as of absolute row `at`, replayed from `from` (a checkpoint before it).
    fn step(&self, from: &volas_core::Checkpoint, at: usize) -> Option<volas_core::Checkpoint> {
        crate::exec::replay(&self.rows, self.history.node, from, at + 1 - self.rows.origin()).map(|(_, cp)| cp)
    }
}
