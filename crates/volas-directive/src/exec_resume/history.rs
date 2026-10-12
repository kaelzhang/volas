//! The checkpoint of a recursive cached column's history ([`volas_core::Checkpoint`]):
//! capturing it from a full computation, and replaying the column forward from it.
//!
//! A recursion's value at a row depends on every row before it. A frame that drops the
//! rows its history started on (a bounded window) keeps, per recursive column, the state
//! as of a row before the rows it keeps and the inputs a resume reads before its first
//! row — the last complete rows of each input the command reads, as many as its resume
//! reaches. While the history is too short to define a state (inside the warm-up) it
//! keeps every complete row of it instead.
//!
//! A replay runs the command's own paths — the resume from the state, or the full
//! computation of a warm-up history — over a frame of just its inputs: the checkpoint's
//! complete rows, then the frame's rows after it. A recursion runs on its complete rows
//! only, so that frame's complete rows are exactly those the whole history's are, and
//! the replayed values are bit-identical to a computation over the whole history.

use volas_compute::missing::ResumeRows;
use volas_core::{Checkpoint, Column, DataFrame};

use crate::complete::BoundCommand;
use crate::operands::Operands;
use crate::types::{ArgValue, Ast, Command, Node};

use super::as_command;

/// Each input slot the recursive command `cmd` reads over `df`, with its values over the
/// whole frame. The command's state arm is probed — it records every input it reads and
/// computes nothing — so the slots are exactly those its paths read. `None` when an
/// input cannot be read (an absent column, an operand that fails).
fn inputs(df: &DataFrame, cmd: &BoundCommand) -> Option<Vec<(usize, Vec<f64>)>> {
    let ops = Operands::new(df, cmd.series);
    ops.probe();
    let _ = super::initial_state::state_of(&ops, cmd.name, cmd.sub, cmd.args);
    if ops.failed() {
        return None;
    }
    Some(ops.reads().map(|(slot, read)| (slot, read.view().into_owned())).collect())
}

/// The checkpoint as of absolute row `at`, the last of `inputs`' first `n` rows, from
/// `state` (the state as of it, if defined). With at least `reach` complete rows behind
/// it — all a resume reads — it keeps the state and those rows; with fewer (the history
/// is inside its resume's warm-up) it keeps every complete row and no state, and a
/// replay computes the whole history from them.
fn keep(at: usize, inputs: &[(usize, Vec<f64>)], n: usize, reach: usize, state: Option<Vec<f64>>) -> Checkpoint {
    let refs: Vec<&[f64]> = inputs.iter().map(|(_, v)| &v[..n]).collect();
    let last = ResumeRows::of(&refs, n, reach);
    let state = state.filter(|_| last.from >= reach);
    let rows = if state.is_some() { last.rows } else { ResumeRows::of(&refs, n, n).rows };
    Checkpoint {
        at,
        state,
        inputs: inputs.iter().map(|(slot, v)| (*slot, rows.iter().map(|&i| v[i]).collect())).collect(),
    }
}

/// The command `cmd` reading each input slot from a column of its own, and the frame of
/// those columns (`inputs`, by slot): a slot the command does not read is all missing.
fn over_inputs(cmd: &BoundCommand, inputs: &[(usize, Vec<f64>)]) -> Option<(DataFrame, Ast)> {
    let rows = inputs.first().map_or(0, |(_, v)| v.len());
    let slots = inputs.iter().map(|(slot, _)| slot + 1).max().unwrap_or(0);
    let column = |slot: usize| format!("\u{0}{slot}");
    let mut values: Vec<Vec<f64>> = vec![vec![f64::NAN; rows]; slots];
    for (slot, v) in inputs {
        values[*slot] = v.clone();
    }
    let names: Vec<String> = (0..slots).map(column).collect();
    let frame = DataFrame::new(names.clone(), values.into_iter().map(Column::f64).collect(), None).ok()?;
    let node = Node::Command(Command {
        name: cmd.name.to_string(),
        sub: cmd.sub.map(str::to_string),
        args: cmd.args.iter().cloned().collect::<Box<[ArgValue]>>(),
        series: names.into_iter().map(Node::Name).collect(),
    });
    Some((frame, node))
}

/// The checkpoint of the recursive `node`'s history over `df`'s rows `[0, end)` — a
/// history that starts at `df`'s row 0 — as of row `end - 1`: the state a full
/// computation over those rows ends with (when defined) and the inputs a replay from it
/// reads. `None` when `node` is not a recursion, `end` is not a row count of `df`, or an
/// input cannot be read.
pub fn capture(df: &DataFrame, node: &Ast, end: usize) -> Option<Checkpoint> {
    let (name, sub, args, series) = as_command(node)?;
    if !crate::spec::is_recursive(&name, &args) || end == 0 || end > df.height() {
        return None;
    }
    let cmd = BoundCommand { name: name.as_ref(), sub: sub.as_deref(), args: &args, series };
    let head = df.slice_data(0, end);
    let state = super::initial_state(&head, node, &Column::f64(Vec::new()));
    let inputs = inputs(&head, &cmd)?;
    Some(keep(df.origin() + end - 1, &inputs, end, cmd.reach(), state))
}

/// Replay the recursive `node` from `checkpoint` over `df`'s rows after it up to `end`
/// (`df` holds the rows from the one after the checkpoint on, or more): those rows'
/// values and the checkpoint as of row `end - 1`. The values are bit-identical to a full
/// computation over the whole history. `None` when the replay cannot run — `node` is not
/// a recursion, `df` does not hold the row after the checkpoint, an input cannot be read
/// or is not one the checkpoint recorded, or the resume declines.
pub fn replay(df: &DataFrame, node: &Ast, checkpoint: &Checkpoint, end: usize) -> Option<(Column, Checkpoint)> {
    let (name, sub, args, series) = as_command(node)?;
    if !crate::spec::is_recursive(&name, &args) {
        return None;
    }
    let cmd = BoundCommand { name: name.as_ref(), sub: sub.as_deref(), args: &args, series };
    let first = (checkpoint.at + 1).checked_sub(df.origin())?;
    if first > end || end > df.height() {
        return None;
    }
    if first == end {
        return Some((Column::f64(Vec::new()), checkpoint.clone())); // nothing after it yet
    }
    let current = inputs(df, &cmd)?;
    let same_slots = current.len() == checkpoint.inputs.len()
        && current.iter().zip(&checkpoint.inputs).all(|((a, _), (b, _))| a == b);
    if !same_slots {
        return None;
    }
    // Each input: the checkpoint's complete rows, then the frame's rows after it.
    let from = checkpoint.inputs.first().map_or(0, |(_, v)| v.len());
    let joined: Vec<(usize, Vec<f64>)> = current
        .iter()
        .zip(&checkpoint.inputs)
        .map(|((slot, now), (_, then))| (*slot, then.iter().chain(&now[first..end]).copied().collect()))
        .collect();
    let n = from + (end - first);
    let (frame, over) = over_inputs(&cmd, &joined)?;
    let (values, state) = match &checkpoint.state {
        Some(state) => {
            let (values, state) = super::execute_resume(&frame, &over, state, from, n, 0)?;
            (values, Some(state))
        }
        None => {
            let values = crate::exec::execute(&frame, &over).ok()?;
            let state = super::initial_state(&frame, &over, &values);
            (values.slice(from, n), state)
        }
    };
    Some((values, keep(df.origin() + end - 1, &joined, n, cmd.reach(), state)))
}
