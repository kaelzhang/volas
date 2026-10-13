//! The directive layer's half of a derived frame: what the core derivation owner
//! (`volas_core::DataFrame::rows`) cannot compute itself — a recursive cached column's
//! carried states as of a row its source frame does not carry them for ([`Replayer`]) —
//! and the frame a cached directive is computed over ([`refresh_frame`]).

use std::borrow::Cow;

use volas_core::{Continuation, DataFrame, Replay};

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

/// The carried resume states of a cached directive `node` computed over `frame`: the
/// state as of `frame`'s last row and as of the row before it, read in place. `(None,
/// None)` for a directive with no resume kernel.
pub fn resume_states(frame: &DataFrame, node: &Ast) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    states_upto(frame, node, frame.height())
}

/// The states of `node` over `df`'s rows `[0, end)` as of rows `end - 1` and `end - 2`:
/// the state before the last from a full computation up to it, then the last row
/// resumed from it — one pass, the rows read in place.
fn states_upto(df: &DataFrame, node: &Ast, end: usize) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let before = (end > 1).then(|| crate::exec::initial_state(df, node, end - 1)).flatten();
    let since = df.origin();
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
/// selection's edges ([`Replay`]). A checkpoint as of a row is replayed from the nearest
/// of the column's [`Anchors`](volas_core::Anchors) before it — or from the history's
/// start the frame holds, or the checkpoint it keeps of a history that started before its
/// row 0 — forming the anchors it passes, so any edge costs at most the anchor spacing
/// once the history has been passed.
pub struct Replayer;

impl Replay for Replayer {
    fn continuation(&self, frame: &DataFrame, name: &str, start: usize, end: usize, boundary: bool, ends: bool) -> Continuation {
        let Ok(node) = crate::parse(name) else { return Continuation::default() };
        let base;
        let df: &DataFrame = if uses_default_series(&node) {
            frame
        } else {
            base = refresh_frame(frame);
            &base
        };
        let (Some(since), Some(anchors)) = (frame.computed_since(name), frame.computed_anchors(name)) else {
            return Continuation::default();
        };
        let history = History { df, node: &node, name, frame, since, anchors };
        let Some(history) = history.over_lead() else { return Continuation::default() };
        let boundary = (boundary && start > 0).then(|| history.at(frame.origin() + start - 1)).flatten();
        let ends = ends
            .then(|| {
                let last = frame.origin() + end - 1;
                let before = (end > 1).then(|| history.at(last - 1)).flatten();
                let at = match &before {
                    Some(b) => history.step(b, last)?,
                    None => history.at(last)?,
                };
                Some((at.state?, before.and_then(|b| b.state)))
            })
            .flatten();
        Continuation { boundary, ends }
    }
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
    /// This history ready to replay; `None` when the frame holds neither its start nor a
    /// checkpoint of it.
    fn over_lead(self) -> Option<Replayed<'a>> {
        let kept = match self.since == self.frame.origin() {
            true => None,
            false => Some(self.frame.computed_checkpoint(self.name)?.clone()),
        };
        let rows = match self.frame.lead_rows() {
            0 => std::borrow::Cow::Borrowed(self.df),
            lead => std::borrow::Cow::Owned(self.frame.led(self.df, lead)),
        };
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
        let usable = |cp: &volas_core::Checkpoint| cp.at <= at && cp.at + 1 >= self.rows.origin();
        let mut from = h.anchors.nearest(at).filter(usable).or_else(|| self.kept.clone().filter(usable));
        let last_row = h.frame.origin() + h.frame.height() - 1;
        loop {
            let done = from.as_ref().map_or(h.since, |cp| cp.at + 1); // the next row to replay
            let stop = ((done / spacing + 1) * spacing - 1).min(at); // the next anchor row, or `at`
            let next = match &from {
                // from the history's start, which the frame's rows hold from its row 0
                None => {
                    let k = stop + 1 - h.df.origin();
                    crate::exec::checkpoint(h.df, h.node, crate::exec::initial_state(h.df, h.node, k), k, true)?
                }
                Some(cp) => self.step(cp, stop)?,
            };
            if (next.at + 1) % spacing == 0 && next.at < last_row {
                h.anchors.keep(next.clone());
            }
            if next.at == at {
                return Some(next);
            }
            from = Some(next);
        }
    }

    /// The checkpoint as of absolute row `at`, replayed from `from` (a checkpoint before it).
    fn step(&self, from: &volas_core::Checkpoint, at: usize) -> Option<volas_core::Checkpoint> {
        crate::exec::replay(&self.rows, self.history.node, from, at + 1 - self.rows.origin()).map(|(_, cp)| cp)
    }
}
