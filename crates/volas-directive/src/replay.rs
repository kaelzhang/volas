//! The directive layer's half of a derived frame: what the core derivation owner
//! (`volas_core::DataFrame::rows`) cannot compute itself — a recursive cached column's
//! carried states as of a row its source frame does not carry them for ([`Replayer`]) —
//! and the frame a cached directive is computed over ([`refresh_frame`]).

use std::borrow::Cow;

use volas_core::{Column, DataFrame, Replay};

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

/// The carried resume states of a cached directive `node` freshly computed over `frame`
/// (`col` is that computation): the state as of `frame`'s last row and as of the row
/// before it. `(None, None)` for a directive with no resume kernel.
pub fn resume_states(frame: &DataFrame, node: &Ast, col: &Column) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let after = crate::exec::initial_state(frame, node, col);
    let h = frame.height();
    if after.is_none() || h == 0 {
        return (after, None);
    }
    let before = crate::exec::initial_state(&frame.slice_data(0, h - 1), node, &col.slice(0, h - 1));
    (after, before)
}

/// The directive layer's replay of a recursive cached column's carried states, for a
/// contiguous selection that ends before the column's last valid row: from the rows of
/// the column's history the source frame holds (computed up to the selection's end), or
/// from the checkpoint it keeps of a history that started before its row 0 (replayed up
/// to it).
pub struct Replayer;

impl Replay for Replayer {
    fn states(&self, frame: &DataFrame, name: &str, end: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
        let node = crate::parse(name).ok()?;
        let base;
        let df: &DataFrame = if uses_default_series(&node) {
            frame
        } else {
            base = refresh_frame(frame);
            &base
        };
        let origin = frame.origin();
        let since = frame.computed_since(name)?;
        if since == origin {
            // the frame holds the history from its row 0 (a history never starts after
            // it): the state before the last row from a full computation up to it, then
            // the last row resumed from it — one pass (a resume reads rows `[0, end)` only)
            let (rows, n) = (df, end);
            let col = frame.column(name).ok()?.slice(0, end);
            let before = (n > 1)
                .then(|| crate::exec::initial_state(&rows.slice_data(0, n - 1), &node, &col.slice(0, n - 1)))
                .flatten();
            let resumed = before
                .as_deref()
                .and_then(|b| crate::exec::execute_resume(rows, &node, b, n - 1, n, since))
                .map(|(_, after)| after);
            return match resumed {
                Some(after) => Some((after, before)),
                None => {
                    let (after, before) = resume_states(&rows.slice_data(0, n), &node, &col);
                    Some((after?, before))
                }
            };
        }
        let cp = frame.computed_checkpoint(name)?;
        let (_, at) = crate::exec::replay(df, &node, cp, end)?;
        let before = crate::exec::replay(df, &node, cp, end - 1).and_then(|(_, b)| b.state);
        Some((at.state?, before))
    }
}
