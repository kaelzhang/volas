//! The directive layer's half of core row derivation. Replay uses the cached bound
//! directive and the continuation state's single owner, including retained input tails.
//! Anchor snapshots are published by the same pass that computed the cached values.

use std::borrow::Cow;

use volas_core::{DataFrame, Replay};

use crate::types::{Ast, Node};

/// Whether a bound node reads only its frame's default series.
pub fn uses_default_series(node: &Ast) -> bool {
    match node.node() {
        Node::Name(_) => true,
        Node::Command(cmd) => cmd.series.iter().all(|series| matches!(series, Node::Name(name) if name.is_empty())),
        _ => false,
    }
}

/// A data projection excluding cached columns and names usable as bare commands.
/// Retained for callers that explicitly request such a projection; replay itself reads
/// its stored bound directive over the frame, without re-parsing or re-binding names.
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
    crate::exec::execute_for_cache(frame, node).map(|(_, state, before, anchors)| {
        ResumeStates { state, before, anchors }
    }).unwrap_or_default()
}

/// The directive layer's replay of a recursive cached column's history at a contiguous
/// selection's boundary ([`Replay`]). A checkpoint as of a row is replayed from the nearest
/// of the column's [`Anchors`](volas_core::Anchors) before it — or from the history's
/// start the frame holds, or the checkpoint it keeps of a history that started before its
/// row 0 — forming the anchors it passes, so any edge costs at most the anchor spacing
/// once the history has been passed.
pub struct Replayer;

impl Replay for Replayer {
    fn continuation(&self, frame: &DataFrame, name: &str, end: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
        states_at(frame, name, end)
    }

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
    if end == 0 || end > frame.height() { return None; }
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
        Some((at.state?.to_vec(), before.and_then(|b| b.state).map(|state| state.to_vec())))
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
    let node = frame.computed_directive(name)?.downcast_ref::<Ast>()?;
    let df = frame;
    let since = frame.computed_since(name)?;
    let anchors = frame.computed_anchors(name)?;
    f(&History { df, node, name, frame, since, anchors }.over_lead()?)
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
            let next = volas_core::CheckpointRef::from(next);
            if (next.at + 1) % spacing == 0 && next.at < last_row {
                h.anchors.keep(next.clone());
            }
            if next.at == at {
                return Some(next.into_owned());
            }
            from = Some(next);
        }
    }

    /// The nearest checkpoint at or before absolute row `at` a replay can start from over
    /// the rows it holds — an anchor, or the frame's checkpoint; `None` for the history's
    /// start.
    fn nearest(&self, at: usize) -> Option<volas_core::CheckpointRef> {
        let usable = |cp: &volas_core::Checkpoint| cp.at <= at && cp.at + 1 >= self.rows.origin();
        let kept = self.kept.clone().map(volas_core::CheckpointRef::from);
        let anchor = self.history.anchors.nearest(at).filter(|cp| usable(cp));
        match (anchor, kept.filter(|cp| usable(cp))) {
            (Some(anchor), Some(kept)) => Some(if anchor.at >= kept.at { anchor } else { kept }),
            (anchor, kept) => anchor.or(kept),
        }
    }

    /// The checkpoint as of absolute row `at`, replayed from `from` (a checkpoint before it).
    fn step(&self, from: &volas_core::Checkpoint, at: usize) -> Option<volas_core::Checkpoint> {
        crate::exec::replay(&self.rows, self.history.node, from, at + 1 - self.rows.origin()).map(|(_, cp)| cp)
    }
}

#[cfg(test)]
mod cache_policy_tests {
    use super::*;
    use volas_core::Column;

    fn closes(n: usize, origin: usize) -> DataFrame {
        DataFrame::new(vec!["close".into()], vec![Column::f64((0..n).map(|i| 100.0 + (i as f64 * 0.7).sin()).collect::<Vec<_>>())], None).unwrap().with_origin(origin)
    }
    fn bits(values: &[f64]) -> Vec<u64> { values.iter().map(|v| v.to_bits()).collect() }

    #[test]
    fn finite_cache_does_not_request_anchors_or_lose_terminal_states() {
        let source = closes(1600, 0);
        for text in ["ma:20", "close>ma:20", "ma:20@(roc:2)"] {
            let node = crate::parse(text).unwrap();
            let cached = resume_states(&source, &node);
            assert!(cached.anchors.is_empty(), "finite cache: {text}");
            let (_, state, before, anchors) = crate::exec::execute_with_states_and_anchors(&source, &node).unwrap();
            assert_eq!(anchors.len(), 3, "explicit anchor request: {text}");
            assert_eq!(cached.state.as_deref().map(bits), state.as_deref().map(bits));
            assert_eq!(cached.before.as_deref().map(bits), before.as_deref().map(bits));
        }
    }

    #[test]
    fn finite_cached_tail_resumes_without_periodic_anchors() {
        for origin in [0, 17] {
            let source = closes(1700, origin);
            for text in ["ma:20", "close>ma:20", "ma:20@(roc:2)"] {
                let node = crate::parse(text).unwrap();
                let mut cached = source.slice_data(0, 1600);
                let (column, state, before, anchors) = crate::exec::execute_for_cache(&cached, &node).unwrap();
                assert!(anchors.is_empty());
                cached.set_column(text, column).unwrap();
                cached.set_computed(text, std::sync::Arc::new((*node).clone()), crate::lookback::reach(&node));
                cached.set_computed_states(text, state, before);
                let mut part = cached.rows(600, 1600, &Replayer);
                let carried = part.computed_resume_state(text).unwrap().0.to_vec();
                let rows = DataFrame::new(vec!["close".into()], vec![source.column("close").unwrap().slice(1600, 1700)], None).unwrap();
                part.append(&rows).unwrap();
                let (tail, next) = crate::exec::execute_resume(&part, &node, &carried, 1000, part.height(), part.origin()).unwrap();
                let expected = crate::execute(&source, &node).unwrap().to_f64_vec();
                assert_eq!(bits(&tail.to_f64_vec()), bits(&expected[1600..]), "{text}, origin{origin}");
                assert_eq!(bits(&next), bits(&crate::exec::initial_state(&source, &node, 1700).unwrap()), "state {text}, origin{origin}");
            }
        }
    }

    #[test]
    fn recursive_cache_keeps_finite_child_snapshots_at_every_requested_row() {
        for origin in [0, 17] {
            for missing in [false, true] {
                let close: Vec<_> = (0..1600).map(|i| if missing && (i % 19 == 0 || (490..530).contains(&i)) {
                    f64::NAN
                } else { 100.0 + (i as f64 * 0.7).sin() }).collect();
                let source = DataFrame::new(vec!["close".into()], vec![Column::f64(close)], None).unwrap().with_origin(origin);
                for text in ["ema:5@(ma:20)", "ema:5@(close>ma:20)"] {
                    let node = crate::parse(text).unwrap();
                    let (column, state, before, anchors) = crate::exec::execute_for_cache(&source, &node).unwrap();
                    let ends: Vec<_> = (512 - origin % 512..1600).step_by(512).collect();
                    assert_eq!(anchors.iter().map(|cp| cp.at + 1 - origin).collect::<Vec<_>>(), ends, "{text}");
                    for cp in &anchors {
                        let prefix = crate::exec::initial_state(&source, &node, cp.at + 1 - origin).unwrap();
                        assert_eq!(bits(cp.state.as_ref().unwrap()), bits(&prefix), "{text}, origin{origin}, missing{missing}, at{}", cp.at);
                    }
                    let explicit = crate::exec::execute_with_states_and_anchors(&source, &node).unwrap();
                    assert_eq!(bits(&column.to_f64_vec()), bits(&explicit.0.to_f64_vec()));
                    assert_eq!(state.as_deref().map(bits), explicit.1.as_deref().map(bits));
                    assert_eq!(before.as_deref().map(bits), explicit.2.as_deref().map(bits));
                    assert_eq!(resume_states(&source, &node).anchors.len(), ends.len());
                }
            }
        }
    }

}
