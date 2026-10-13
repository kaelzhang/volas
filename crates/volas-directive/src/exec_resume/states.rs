//! A full computation together with the states a cached column carries — the state as
//! of the last row and the one before — in one pass over the rows.
//!
//! Sliding accumulators publish both terminal states from their batch kernel's own
//! pass. The continuation owner frames those words and the bounded input tails;
//! it never reconstructs the accumulator by stepping through the history again.
//!
//! Recursive compositions publish their terminal states from their batch kernels too.
//! Recurrences with direct vector resumes run once through
//! the command's own paths: the full computation over a short head (its warm-up and some
//! rows past it), the state as of the head's last row, then the resume from it over the
//! rest — which yields those rows' values and the state before the last row — and the
//! last row's step. Each path is bit-identical to the full computation (the
//! resume-versus-full matrices), so the values are too.

use volas_core::{Column, DataFrame, Result};

use crate::complete::into_f64;
use crate::lookback::lookback;
use crate::types::Ast;

/// A column's values and its carried states `(state, state_before_last)`.
pub type WithStates = (Column, Option<Vec<f64>>, Option<Vec<f64>>);
/// Values, terminal states, and same-pass framed continuation snapshots.
pub(super) type WithAnchors = (Column, Option<Vec<f64>>, Option<Vec<f64>>, Vec<(usize, volas_core::StateWords)>);
/// Public result with checkpoints owned by the continuation representation.
pub type WithCheckpoints = (Column, Option<Vec<f64>>, Option<Vec<f64>>, Vec<volas_core::Checkpoint>);

pub(super) fn marks(df: &DataFrame) -> Vec<usize> {
    let spacing = volas_core::dataframe::ANCHOR_SPACING;
    (spacing - df.origin() % spacing..df.height()).step_by(spacing).collect()
}

/// The full computation of `node` over `df` and its carried states: the state as of the
/// last row and the one before (`None` for a directive with no resume).
pub fn execute_with_states(df: &DataFrame, node: &Ast) -> Result<WithStates> {
    let (column, state, before, _) = execute_anchored(df, node)?;
    Ok((column, state, before))
}

/// Compute values and capture anchor checkpoints in the producing pass.
pub fn execute_with_states_and_anchors(df: &DataFrame, node: &Ast) -> Result<WithCheckpoints> {
    let (column, state, before, points) = execute_anchored(df, node)?;
    Ok((column, state, before, super::history::checkpoints(df, node, points)))
}

/// Cache execution requests anchors only for carried recursions. Every node in a
/// chain shares that root request and the same continuation encoder.
pub fn execute_for_cache(df: &DataFrame, node: &Ast) -> Result<WithCheckpoints> {
    let requested = if matches!(crate::lookback::reach(node).history, volas_core::History::Carried { .. }) { marks(df) } else { Vec::new() };
    let (column, state, before, points) = execute_marked(df, node, &requested)?;
    Ok((column, state, before, super::history::checkpoints(df, node, points)))
}

pub(super) fn execute_anchored(df: &DataFrame, node: &Ast) -> Result<WithAnchors> {
    execute_marked(df, node, &marks(df))
}

pub(super) fn execute_marked(df: &DataFrame, node: &Ast, requested: &[usize]) -> Result<WithAnchors> {
    if super::chain::chained(node) {
        return super::chain::with_marked(df, node, requested);
    }
    if let Some(done) = one_pass(df, node, requested) {
        return Ok(done);
    }
    let col = crate::exec::execute(df, node)?;
    let (state, before) = capture(df, node);
    Ok((col, state, before, Vec::new()))
}

/// The carried states of `node` over `df`'s rows: the state as of the row before the
/// last (one capture pass, read in place), then the last row's continuation from it.
pub fn capture(df: &DataFrame, node: &Ast) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let h = df.height();
    if h == 0 {
        return (None, None);
    }
    let before = super::initial_state(df, node, h - 1);
    let after = before
        .as_deref()
        .and_then(|b| super::execute_resume(df, node, b, h - 1, h, df.origin()).map(|(_, state)| state))
        .or_else(|| super::initial_state(df, node, h));
    match after {
        Some(after) => (Some(after), before),
        None => (None, None),
    }
}

/// The one-pass computation of a carried line: `None` when it has no batch/resume path, the
/// frame is too short for a head past the warm-up, or a path declines.
fn one_pass(df: &DataFrame, node: &Ast, marks: &[usize]) -> Option<WithAnchors> {
    if !matches!(node.continuation(), crate::types::Continuation::Own(_)) {
        return None;
    }
    let crate::types::Continuation::Own(cmd) = node.continuation() else { return None; };
    let h = df.height();
    let head = 2 * (lookback(node) + 1);
    if h < head + 2 { return None; }
    let bc = super::bound(cmd);
    if bc.spec.history(bc.args) == crate::spec::History::Recursive {
        if let Some(pass) = super::batch::recursive_pass(df, &bc, marks) { return Some(pass); }
    } else {
        let ops = crate::operands::Operands::new(df, bc.series);
        if let Some(pass) = super::batch::compute(&ops, &bc, marks) {
            let encoder = super::state::FrameEncoder::new(&ops, &bc, false, None).with_kernel(&pass.after);
            let after = encoder.capture(h, &pass.after)?;
            let before = encoder.capture(h - 1, &pass.before)?;
            let mut points = super::state::FramedPoints::with_capacity(marks.len(), after.len());
            for (end, kernel) in pass.points.iter() { points.capture(&encoder, end, kernel); }
            if marks.contains(&(h - 1)) { points.push(h - 1, &before); }
            let points = points.finish();
            return Some((Column::f64(pass.values), Some(after), Some(before), points));
        }
    }
    let head = marks.first().copied().filter(|&mark| mark < head).unwrap_or(head);
    let mut state = super::initial_state(df, node, head)?;
    let since = df.origin();
    // the head's values in front of the resumed rows, the last row after them
    let head_values = crate::exec::execute(&df.slice_data(0, head), node).ok()?;
    let mut values = Vec::with_capacity(h);
    values.extend(into_f64(head_values));
    let mut from = head;
    let mut points = super::state::FramedPoints::with_capacity(marks.len(), state.len());
    if marks.contains(&head) { points.push(head, &state); }
    for end in marks.iter().copied().filter(|&end| end > head && end < h - 1) {
        state = super::resume::resume_into(df, node, &state, from, end, since, &mut values)?;
        points.push(end, &state);
        from = end;
    }
    let before = super::resume::resume_into(df, node, &state, from, h - 1, since, &mut values)?;
    if marks.contains(&(h - 1)) { points.push(h - 1, &before); }
    let after = super::resume::resume_into(df, node, &before, h - 1, h, since, &mut values)?;
    Some((Column::f64(values), Some(after), Some(before), points.finish()))
}
