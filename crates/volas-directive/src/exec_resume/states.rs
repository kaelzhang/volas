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

/// The full computation of `node` over `df` and its carried states: the state as of the
/// last row and the one before (`None` for a directive with no resume).
pub fn execute_with_states(df: &DataFrame, node: &Ast) -> Result<WithStates> {
    if super::chain::chained(node) {
        return super::chain::with_states(df, node);
    }
    if let Some(done) = one_pass(df, node) {
        return Ok(done);
    }
    let col = crate::exec::execute(df, node)?;
    let (state, before) = capture(df, node);
    Ok((col, state, before))
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
fn one_pass(df: &DataFrame, node: &Ast) -> Option<WithStates> {
    if !matches!(node.continuation(), crate::types::Continuation::Own(_)) {
        return None;
    }
    let crate::types::Continuation::Own(cmd) = node.continuation() else { return None; };
    let h = df.height();
    let head = 2 * (lookback(node) + 1);
    if h < head + 2 { return None; }
    let bc = super::bound(cmd);
    if bc.spec.history(bc.args) == crate::spec::History::Recursive {
        if let Some(pass) = super::batch::recursive_pass(df, &bc) { return Some(pass); }
    } else {
        let ops = crate::operands::Operands::new(df, bc.series);
        if let Some(pass) = super::batch::compute(&ops, &bc) {
            let after = super::state::capture_read(&ops, &bc, h, pass.after, false)?;
            let before = super::state::capture_read(&ops, &bc, h - 1, pass.before, false)?;
            return Some((Column::f64(pass.values), Some(after), Some(before)));
        }
    }
    let state = super::initial_state(df, node, head)?;
    let since = df.origin();
    // the head's values in front of the resumed rows, the last row after them
    let head_values = crate::exec::execute(&df.slice_data(0, head), node).ok()?;
    let mut values = Vec::with_capacity(h);
    values.extend(into_f64(head_values));
    let before = super::resume::resume_into(df, node, &state, head, h - 1, since, &mut values)?;
    let after = super::resume::resume_into(df, node, &before, h - 1, h, since, &mut values)?;
    Some((Column::f64(values), Some(after), Some(before)))
}
