//! The state chain of a directive whose recursions are not all at its top: a recursion
//! inside an `@` operand (`ema:5@(ema:3)`), or combined with other values
//! (`close - ma:5 + obv`).
//!
//! Every node of the tree that keeps a state of its own — a recursive command, or one
//! whose output is a row position — carries that state, and its values on the last `L`
//! rows (`L` twice the directive's lookback, plus two: every row a consumer of it reads
//! before a new row). A continuation over new rows runs the nodes children first: each
//! resumes from its state over the new rows of a trailing window of the frame — its
//! `@` operands that keep a state read as columns of their own values, the remembered
//! ones before the new rows — and then the whole tree is evaluated over that window, as
//! the parts that keep no state are window functions. So one new row costs the window,
//! whatever the frame's height, and every value is the full computation's.
//!
//! The chain is the column's carried state, flat: per node, children first, its state's
//! length, its state, and its last `L` values (missing before the frame's first row).

use volas_core::{Column, DataFrame};

use super::{as_command, resume_of};
use crate::complete::into_f64;
use crate::lookback::lookback_node;
use crate::spec::{History, Resume};
use crate::types::{Ast, BoundNode, ChainPlan, Command, Continuation, Node};

/// Whether `node` keeps a state of its own: a recursive command, or one whose output is
/// a row position.
fn keeps_state(node: &BoundNode) -> bool {
    if !matches!(node, Node::Command(_) | Node::Name(_)) {
        return false;
    }
    as_command(node).is_some_and(|c| {
        c.args.spec.history(&c.args) == History::Recursive || c.args.spec.emits_position()
    })
}

/// The nodes of `node` that keep a state, children first.
fn stateful<'a>(node: &'a BoundNode, out: &mut Vec<&'a BoundNode>) {
    match node {
        Node::Command(c) => c.series.iter().for_each(|s| stateful(s, out)),
        Node::Unary { operand, .. } => stateful(operand, out),
        Node::Binary { left, right, .. } => {
            stateful(left, out);
            stateful(right, out);
        }
        Node::Name(_) | Node::Scalar(_) => {}
    }
    if keeps_state(node) {
        out.push(node);
    }
}

/// Whether `node` continues through a state chain: some node below its top keeps a
/// state (a command whose operands keep none continues by its own resume), and every node
/// that keeps one resumes.
pub(crate) fn plan(node: &BoundNode) -> Continuation {
    let mut nodes = Vec::new();
    stateful(node, &mut nodes);
    let own_only = nodes.len() == 1 && std::ptr::eq(nodes[0], node);
    if !nodes.is_empty() && !own_only
        && nodes.iter().all(|n| as_command(n).is_some_and(|c| resume_of(&c) != Resume::Recompute))
    {
        return Continuation::Chain(ChainPlan {
            nodes: (0..nodes.len()).map(|k| crate::bind::finish(own_node(&nodes, k))).collect(),
            top: Box::new(substitute(node, &nodes)),
            names: (0..nodes.len()).map(hidden).collect(),
            span: 2 * (lookback_node(node) + 1),
        });
    }
    if as_command(node).is_some_and(|c| resume_of(&c) != Resume::Recompute) {
        Continuation::Own
    } else {
        Continuation::None
    }
}

/// The decision is stored at binding time; this hot-path check never walks the tree.
pub(super) fn chained(node: &Ast) -> bool {
    matches!(node.continuation(), Continuation::Chain(_))
}

fn chain_plan(node: &Ast) -> &ChainPlan {
    let Continuation::Chain(plan) = node.continuation() else { unreachable!("a chained directive") };
    plan
}

/// The name of the column holding the values of the `k`-th stateful node.
fn hidden(k: usize) -> String {
    format!("\u{1}chain{k}")
}

/// `node` with each of its subtrees among `done` (by identity) read from its column.
fn substitute(node: &BoundNode, done: &[&BoundNode]) -> BoundNode {
    if let Some(k) = done.iter().position(|d| std::ptr::eq(*d, node)) {
        return Node::Name(hidden(k));
    }
    match node {
        Node::Command(c) => Node::Command(Command {
            name: c.name.clone(),
            sub: c.sub.clone(),
            args: c.args.clone(),
            series: c.series.iter().map(|s| substitute(s, done)).collect(),
        }),
        Node::Unary { op, operand } => Node::Unary { op: *op, operand: Box::new(substitute(operand, done)) },
        Node::Binary { left, op, right } => Node::Binary {
            left: Box::new(substitute(left, done)),
            op: *op,
            right: Box::new(substitute(right, done)),
        },
        Node::Name(_) | Node::Scalar(_) => node.clone(),
    }
}

/// The `k`-th stateful node with its children read from their columns.
fn own_node(nodes: &[&BoundNode], k: usize) -> BoundNode {
    match nodes[k] {
        Node::Command(c) => Node::Command(Command {
            name: c.name.clone(),
            sub: c.sub.clone(),
            args: c.args.clone(),
            series: c.series.iter().map(|s| substitute(s, &nodes[..k])).collect(),
        }),
        other => other.clone(), // a bare name: no operands
    }
}

/// `values` (of rows `[0, len)`) on the `l` rows before `end`, missing before row 0.
fn tail(values: &[f64], end: usize, l: usize) -> impl Iterator<Item = f64> + '_ {
    (end as isize - l as isize..end as isize).map(move |i| if i < 0 { f64::NAN } else { values[i as usize] })
}

/// Append one node's state and its last `l` values (of rows before `end`) to `chain`.
fn keep(chain: &mut Vec<f64>, state: Vec<f64>, values: &[f64], end: usize, l: usize) {
    chain.push(state.len() as f64);
    chain.extend(state);
    chain.extend(tail(values, end, l));
}

/// The full computation of the chained `node` over `df`, with its chain as of the last
/// row and the one before (see the module docs): each `None` while a node has no state
/// as of that row (still in its warm-up), independently of the other.
pub(super) fn with_states(df: &DataFrame, node: &Ast) -> volas_core::Result<super::states::WithStates> {
    let plan = chain_plan(node);
    let (h, l) = (df.height(), plan.span);
    let mut frame = df.clone();
    let (mut after, mut before) = (Some(Vec::new()), Some(Vec::new()));
    for (k, own) in plan.nodes.iter().enumerate() {
        let (col, a, b) = super::states::execute_with_states(&frame, own)?;
        let values = into_f64(col);
        after = after.zip(a).map(|(mut chain, a)| {
            keep(&mut chain, a, &values, h, l);
            chain
        });
        before = before.zip(b).map(|(mut chain, b)| {
            keep(&mut chain, b, &values, h.saturating_sub(1), l);
            chain
        });
        frame.set_column(&plan.names[k], Column::f64(values))?;
    }
    let top = crate::exec::execute_node(&frame, &plan.top)?;
    Ok((top, after, before))
}

/// The chain of `node` as of `df`'s row `end - 1`.
pub(super) fn capture(df: &DataFrame, node: &Ast, end: usize) -> Option<Vec<f64>> {
    with_states(&df.slice_data(0, end.min(df.height())), node).ok()?.1
}

/// Continue the chained `node` from `chain` (as of row `from - 1`) over rows `[from,
/// end)`: those rows' values and the chain as of row `end - 1`. `None` when the chain
/// does not fit the directive or a node's resume declines (the caller recomputes).
pub(super) fn resume(
    df: &DataFrame,
    node: &Ast,
    chain: &[f64],
    from: usize,
    end: usize,
    since: usize,
) -> Option<(Column, Vec<f64>)> {
    let plan = chain_plan(node);
    let l = plan.span;
    let end = end.min(df.height());
    if from > end {
        return None;
    }
    let start = from.saturating_sub(l);
    let mut win = df.slice_data(start, end);
    let (new, held) = (end - from, from - start);
    let mut next = Vec::with_capacity(chain.len());
    let mut at = 0;
    for (k, own) in plan.nodes.iter().enumerate() {
        let len = *chain.get(at)? as usize;
        let state = chain.get(at + 1..at + 1 + len)?;
        let ring = chain.get(at + 1 + len..at + 1 + len + l)?;
        at += 1 + len + l;
        let (col, state) = super::execute_resume(&win, own, state, held, held + new, since)?;
        // the node's values over the window: those remembered, then the new rows'
        let mut values = Vec::with_capacity(held + new);
        values.extend_from_slice(&ring[l - held..]);
        values.extend(into_f64(col));
        next.push(state.len() as f64);
        next.extend(state);
        next.extend(ring.iter().copied().chain(values[held..].iter().copied()).skip(new));
        win.set_column(&plan.names[k], Column::f64(values)).ok()?;
    }
    if at != chain.len() {
        return None;
    }
    let top = crate::exec::execute_node(&win, &plan.top).ok()?;
    Some((top.slice(held, held + new), next))
}
