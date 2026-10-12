//! Incremental directive resume support for cached computed columns.
//!
//! A recursive indicator's whole history compresses into a small fixed-size state (a
//! `Vec<f64>`); a finite-window one carrying a running accumulator keeps its running
//! sums; a formula of its own row keeps nothing. `initial_state` captures that state
//! after a full compute; `execute_resume` continues over only the new tail rows and
//! `execute_resume_step` one row at a time, bit-identical to a fresh full recompute.
//! Which of them a command line has is its spec's fact ([`Resume`]); a line without one
//! (or a composite expression) returns `None`, and the caller recomputes.
//!
//! A continuation reads a trailing window of the frame — its lookback twice over, see
//! [`window_start`] — so its cost is the window's, not the frame's height.

use std::borrow::Cow;

use crate::bind::bind_bare;
use crate::complete::BoundCommand;
use crate::lookback::{evaluates_on_a_window, lookback};
use crate::spec::{command_spec, History, Resume};
use crate::types::{Ast, Bound, Command, Node};

mod history;
mod initial_state;
mod resume;
mod scalar;
mod window;

pub use history::{capture, checkpoint, replay};
pub use initial_state::initial_state;
pub use resume::execute_resume;
pub use scalar::execute_resume_step;

/// The command a continuation of `node` runs: a plain `Node::Command` (borrowed), or a
/// bare `Node::Name` no-argument command bound on the fly (a tree not bound against a
/// frame) — under the command it runs as (`ma` at a recursive MA type is that average's
/// own command, see [`crate::spec::CommandSpec::runs_as_command`]). `None` for anything
/// else.
fn as_command(node: &Ast) -> Option<Cow<'_, Command<Bound>>> {
    let cmd = match node {
        Node::Command(cmd) => Cow::Borrowed(cmd),
        Node::Name(name) if !name.is_empty() => Cow::Owned(bind_bare(name).ok()?),
        _ => return None,
    };
    let Some((name, values)) = cmd.args.spec.runs_as_command(&cmd.args) else {
        return Some(cmd);
    };
    let spec = command_spec(name, None)?;
    Some(Cow::Owned(Command {
        name: name.to_string(),
        sub: None,
        args: Bound { values: values.into_boxed_slice(), spec, bare: None },
        series: cmd.series.clone(),
    }))
}

/// The bound command `cmd` as the missing-row rule and the arms read it.
fn bound(cmd: &Command<Bound>) -> BoundCommand<'_> {
    BoundCommand {
        name: &cmd.name,
        sub: cmd.sub.as_deref(),
        args: &cmd.args.values,
        series: &cmd.series,
        spec: &cmd.args.spec,
    }
}

/// The resume `cmd` has under its arguments.
fn resume_of(cmd: &Command<Bound>) -> Resume {
    cmd.args.spec.resume(&cmd.args)
}

/// Whether `cmd` is a recursion: it runs on the complete rows of its inputs.
fn recursive(cmd: &Command<Bound>) -> bool {
    cmd.args.spec.history(&cmd.args) == History::Recursive
}

/// The first frame row a continuation of `node` from `row` reads: twice its lookback
/// plus one before `row` — the rows its own window and its `@` operands' windows need,
/// with margin — when its operands can be evaluated over a trailing window
/// ([`evaluates_on_a_window`]); row 0 otherwise (a recursive operand depends on every
/// row before).
fn window_start(node: &Ast, cmd: &Command<Bound>, row: usize) -> usize {
    if !cmd.series.iter().all(evaluates_on_a_window) {
        return 0;
    }
    row.saturating_sub(2 * (lookback(node) + 1))
}
