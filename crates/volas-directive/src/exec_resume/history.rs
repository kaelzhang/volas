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

/// The recursive command `node` binds to, as a [`BoundCommand`] passed to `f`; `None`
/// for anything else.
fn recursion<R>(node: &Ast, f: impl FnOnce(&BoundCommand) -> Option<R>) -> Option<R> {
    let (name, sub, args, series) = as_command(node)?;
    if !crate::spec::is_recursive(&name, &args) {
        return None;
    }
    f(&BoundCommand { name: name.as_ref(), sub: sub.as_deref(), args: &args, series })
}

/// `f` of each input slot the recursive command `cmd` reads over `df`, with its values
/// over the whole frame (a frame column borrowed). The command's state arm is probed — it
/// records every input it reads and computes nothing — so the slots are exactly those its
/// paths read. `None` when an input cannot be read (an absent column, an operand that
/// fails).
fn with_inputs<R>(df: &DataFrame, cmd: &BoundCommand, f: impl FnOnce(&[(usize, &[f64])]) -> R) -> Option<R> {
    let ops = Operands::new(df, cmd.series);
    ops.probe();
    let _ = super::initial_state::state_of(&ops, cmd.name, cmd.sub, cmd.args);
    if ops.failed() {
        return None;
    }
    let views: Vec<(usize, std::borrow::Cow<[f64]>)> = ops.reads().map(|(slot, read)| (slot, read.view())).collect();
    let refs: Vec<(usize, &[f64])> = views.iter().map(|(slot, v)| (*slot, v.as_ref())).collect();
    Some(f(&refs))
}

/// The checkpoint as of absolute row `at`, the last of `inputs`' first `n` rows, from
/// `state` (the state as of it, if defined). With a state and at least `reach` complete
/// rows behind it — all a resume reads — it keeps the state and those rows. Otherwise, when
/// `whole` (the rows hold the whole history), it keeps every complete row and no state —
/// the history is inside its resume's warm-up, and a replay computes all of it from them;
/// when not, the rows cannot vouch for the history: `None`.
fn keep(at: usize, inputs: &[(usize, &[f64])], n: usize, reach: usize, state: Option<Vec<f64>>, whole: bool) -> Option<Checkpoint> {
    let refs: Vec<&[f64]> = inputs.iter().map(|(_, v)| &v[..n]).collect();
    let last = ResumeRows::of(&refs, n, reach);
    let (state, rows) = match state {
        Some(state) if last.from >= reach => (Some(state), last.rows),
        _ if whole => (None, ResumeRows::of(&refs, n, n).rows),
        _ => return None,
    };
    Some(Checkpoint {
        at,
        state,
        inputs: inputs.iter().map(|(slot, v)| (*slot, rows.iter().map(|&i| v[i]).collect())).collect(),
    })
}

/// The command `cmd` reading each input slot from a column of its own, and the frame of
/// those columns (`inputs`, by slot, taken over): a slot the command does not read is all
/// missing.
fn over_inputs(cmd: &BoundCommand, inputs: Vec<(usize, Vec<f64>)>) -> Option<(DataFrame, Ast)> {
    let rows = inputs.first().map_or(0, |(_, v)| v.len());
    let slots = inputs.iter().map(|(slot, _)| slot + 1).max().unwrap_or(0);
    let mut values: Vec<Option<Vec<f64>>> = vec![None; slots];
    for (slot, v) in inputs {
        values[slot] = Some(v);
    }
    let names: Vec<String> = (0..slots).map(|slot| format!("\u{0}{slot}")).collect();
    let columns = values.into_iter().map(|v| Column::f64(v.unwrap_or_else(|| vec![f64::NAN; rows]))).collect();
    let frame = DataFrame::new(names.clone(), columns, None).ok()?;
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
    if end == 0 || end > df.height() {
        return None;
    }
    recursion(node, |cmd| {
        let state = super::initial_state(df, node, end);
        with_inputs(df, cmd, |inputs| keep(df.origin() + end - 1, inputs, end, cmd.reach(), state, true))?
    })
}

/// The checkpoint of the recursive `node` as of `df`'s row `end - 1` from `state`, its
/// carried state as of that row: the state and the inputs before `end` a resume from it
/// reads. `whole` says `df`'s rows hold the whole history (it starts at row 0); when they
/// do not, a frame with fewer than the resume's reach of complete rows before `end`
/// cannot form one. `None` then, and when `node` is not a recursion, `end` is not a row
/// count of `df`, or an input cannot be read.
pub fn checkpoint(df: &DataFrame, node: &Ast, state: Option<Vec<f64>>, end: usize, whole: bool) -> Option<Checkpoint> {
    if end == 0 || end > df.height() {
        return None;
    }
    recursion(node, |cmd| with_inputs(df, cmd, |inputs| keep(df.origin() + end - 1, inputs, end, cmd.reach(), state, whole))?)
}

/// The checkpoints of the recursive `node` over `df`, whose rows hold the whole history
/// (it starts at row 0), as of each `(end, state)` of `states`: as of row `end - 1`, from
/// its carried state there — the inputs read once for all of them. Those that cannot be
/// formed are left out.
pub fn checkpoints(df: &DataFrame, node: &Ast, states: Vec<(usize, Vec<f64>)>) -> Vec<Checkpoint> {
    recursion(node, |cmd| {
        with_inputs(df, cmd, |inputs| {
            states
                .into_iter()
                .filter(|(end, _)| (1..=df.height()).contains(end))
                .filter_map(|(end, state)| keep(df.origin() + end - 1, inputs, end, cmd.reach(), Some(state), true))
                .collect()
        })
    })
    .unwrap_or_default()
}

/// Replay the recursive `node` from `checkpoint` over `df`'s rows after it up to `end`
/// (`df` holds the rows from the one after the checkpoint on, or more): those rows'
/// values and the checkpoint as of row `end - 1`. The values are bit-identical to a full
/// computation over the whole history. `None` when the replay cannot run — `node` is not
/// a recursion, `df` does not hold the row after the checkpoint, an input cannot be read
/// or is not one the checkpoint recorded, or the resume declines.
pub fn replay(df: &DataFrame, node: &Ast, checkpoint: &Checkpoint, end: usize) -> Option<(Column, Checkpoint)> {
    let first = (checkpoint.at + 1).checked_sub(df.origin())?;
    if first > end || end > df.height() {
        return None;
    }
    if first == end {
        return Some((Column::f64(Vec::new()), checkpoint.clone())); // nothing after it yet
    }
    recursion(node, |cmd| {
        // Each input: the checkpoint's complete rows, then the frame's rows after it.
        let joined = with_inputs(df, cmd, |current| {
            let same = current.len() == checkpoint.inputs.len()
                && current.iter().zip(&checkpoint.inputs).all(|((a, _), (b, _))| a == b);
            same.then(|| {
                current
                    .iter()
                    .zip(&checkpoint.inputs)
                    .map(|((slot, now), (_, then))| (*slot, then.iter().chain(&now[first..end]).copied().collect()))
                    .collect::<Vec<(usize, Vec<f64>)>>()
            })
        })??;
        let from = checkpoint.inputs.first().map_or(0, |(_, v)| v.len());
        let n = from + (end - first);
        let slots: Vec<usize> = joined.iter().map(|(slot, _)| *slot).collect();
        let (frame, over) = over_inputs(cmd, joined)?;
        let (values, state) = match &checkpoint.state {
            Some(state) => {
                let (values, state) = super::execute_resume(&frame, &over, state, from, n, 0)?;
                (values, Some(state))
            }
            None => {
                let values = crate::exec::execute(&frame, &over).ok()?;
                let state = super::initial_state(&frame, &over, frame.height());
                (values.slice(from, n), state)
            }
        };
        let inputs: Vec<(usize, &[f64])> = slots
            .iter()
            .map(|&slot| (slot, frame.columns()[slot].as_f64().expect("an f64 input column")))
            .collect();
        let next = keep(df.origin() + end - 1, &inputs, n, cmd.reach(), state, true)?;
        Some((values, next))
    })
}
