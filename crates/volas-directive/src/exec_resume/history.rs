//! Checkpoints store exactly the continuation state. Replay is ordinary resume.
//! During warm-up the raw input columns retain the history, including missing rows,
//! integer precision and absolute row positions, for both Own and Chain plans.

use volas_core::{Checkpoint, Column, DataFrame};
use crate::types::{Ast, BoundNode, Node};

fn supported(node: &Ast) -> bool {
    !matches!(node.continuation(), crate::types::Continuation::None)
}

fn columns(node: &BoundNode, names: &mut Vec<String>) {
    match node {
        Node::Name(n) if super::as_command(node).is_none() => {
            if !names.contains(n) { names.push(n.clone()); }
        }
        Node::Command(_) | Node::Name(_) => {
            if let Some(cmd) = super::as_command(node) {
                for (i, default) in cmd.args.spec.series.iter().enumerate() {
                    if let Some(s) = crate::exec::operand(&cmd.series, i) { columns(s, names); }
                    else if !names.iter().any(|n| n == default) { names.push((*default).into()); }
                }
            }
        }
        Node::Unary { operand, .. } => columns(operand, names),
        Node::Binary { left, right, .. } => { columns(left, names); columns(right, names); }
        Node::Scalar(_) => {}
    }
}

pub fn capture(df: &DataFrame, node: &Ast, end: usize) -> Option<Checkpoint> {
    if !supported(node) || end == 0 || end > df.height() { return None; }
    // Validate every input even while no node has seeded yet.
    crate::exec::execute(&df.slice_data(0, end), node).ok()?;
    checkpoint(df, node, super::initial_state(df, node, end), end, true)
}

pub fn checkpoint(df: &DataFrame, node: &Ast, state: Option<Vec<f64>>, end: usize, whole: bool) -> Option<Checkpoint> {
    if !supported(node) || df.origin() + end == 0 || end > df.height() { return None; }
    let raw = if state.is_none() {
        if !whole { return None; }
        let mut names = Vec::new();
        columns(node.node(), &mut names);
        names.into_iter().map(|n| Some((n.clone(), df.column(&n).ok()?.slice(0, end)))).collect::<Option<Vec<_>>>()?
    } else { Vec::new() };
    Some(Checkpoint { at: df.origin() + end - 1, state, raw })
}

pub fn replay(df: &DataFrame, node: &Ast, cp: &Checkpoint, end: usize) -> Option<(Column, Checkpoint)> {
    if !supported(node) { return None; }
    let first = (cp.at + 1).checked_sub(df.origin())?;
    if first > end || end > df.height() { return None; }
    if first == end { return Some((Column::f64(Vec::new()), cp.clone())); }
    if let Some(state) = &cp.state {
        let (values, state) = super::execute_resume(df, node, state, first, end, 0)?;
        return Some((values, Checkpoint { at: df.origin() + end - 1, state: Some(state), raw: Vec::new() }));
    }
    let held = cp.raw.first()?.1.len();
    let names: Vec<String> = cp.raw.iter().map(|(n, _)| n.clone()).collect();
    let cols = cp.raw.iter().map(|(n, c)| {
        let mut c = c.clone();
        c.append(&df.column(n).ok()?.slice(first, end)).ok()?;
        Some(c)
    }).collect::<Option<Vec<_>>>()?;
    let joined = DataFrame::new(names, cols, None).ok()?.with_origin(cp.at + 1 - held);
    let (values, state, _) = super::execute_with_states(&joined, node).ok()?;
    let next = checkpoint(&joined, node, state, joined.height(), true)?;
    Some((values.slice(held, joined.height()), next))
}
