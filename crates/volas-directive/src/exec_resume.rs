//! Incremental directive resume support for cached computed columns.

use std::borrow::Cow;

use crate::bind::bind_command;
use crate::types::{ArgValue, Ast, Node};

mod initial_state;
mod resume;

pub use initial_state::initial_state;
pub use resume::{execute_resume, execute_resume_default_series, execute_resume_default_series_one};

// --- state-carry resume (additive; fallback path stays correct) -------------
//
// A recursive indicator's whole history compresses into a small fixed-size state
// (a `Vec<f64>`). `initial_state` captures that state after a full compute;
// `execute_resume` continues the recursion over only the new tail rows, producing
// values bit-identical to a fresh full recompute. Both return `None` for any
// directive without a resume kernel, so the caller transparently falls back to the
// correct full-recompute path. Only the canonical no-operand forms (the directives
// volas auto-caches) are handled; an unusual `@`-operand override returns `None`
// and stays on the fallback.

/// Resolve a command node to `(name, sub, args, series)` when it is a plain
/// `Node::Command` (canonical name / sub and resolved arguments, borrowed) or a bare
/// `Node::Name` no-argument command (e.g. `obv`/`ad`, or a default-only canonical form
/// like `efi`). For a bare name the command is bound on the fly — the same single
/// `bind` `exec` uses — so its defaults are resolved and the resume kernels read a
/// complete argument list rather than an empty slice. `None` for anything else.
fn as_command(node: &Ast) -> Option<(String, Option<String>, Cow<'_, [ArgValue]>, &[Ast])> {
    match node {
        Node::Command(cmd) => Some((
            cmd.name.clone(),
            cmd.sub.clone(),
            Cow::Borrowed(&cmd.args[..]),
            &cmd.series,
        )),
        Node::Name(name) if !name.is_empty() => {
            let cmd = bind_command(name, None, &[], &[]).ok()?;
            Some((cmd.name, cmd.sub, Cow::Owned(cmd.args.into_vec()), &[]))
        }
        _ => None,
    }
}
