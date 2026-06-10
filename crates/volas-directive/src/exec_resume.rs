//! Incremental directive resume support for cached computed columns.

use crate::types::Node;

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

/// Resolve a command node to `(name_lc, sub, args, series)` when it is a plain
/// `Node::Command` (or a bare `Node::Name` no-arg command, e.g. `obv`/`ad`); `None`
/// otherwise. The name is lower-cased and `cdl`→`style` aliased, matching
/// [`exec_command`]. A `Node::Name` carries no sub / args / series — the same way
/// [`execute`] dispatches it via `exec_command(df, name, None, &[], &[])`.
fn as_command(node: &Node) -> Option<(String, Option<String>, &[Option<String>], &[Node])> {
    let lc = |name: &str| {
        let name = name.to_ascii_lowercase();
        if name == "cdl" {
            "style".to_string()
        } else {
            name
        }
    };
    match node {
        Node::Command(cmd) => Some((lc(&cmd.name), cmd.sub.clone(), &cmd.args, &cmd.series)),
        Node::Name(name) if !name.is_empty() => Some((lc(name), None, &[], &[])),
        _ => None,
    }
}
