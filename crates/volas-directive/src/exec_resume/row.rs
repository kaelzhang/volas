//! Scalar evaluation of a plan's stateless remainder. Stateful names read the freshly
//! stepped node rings; numeric comparisons preserve integer precision.
use volas_core::DataFrame;
use crate::element::{self, Value};
use crate::types::{BoundNode, Node, Number, Op};

pub(super) fn column(df: &DataFrame, name: &str, row: usize) -> Option<Value> {
    element::at(df.column(name).ok()?, row)
}

pub(super) fn evaluate(
    df: &DataFrame, node: &BoundNode, row: usize,
    hidden: &impl Fn(&str, usize) -> Option<f64>,
) -> Option<Value> {
    let eval = |n, r| evaluate(df, n, r, hidden);
    Some(match node {
        Node::Scalar(Number::Int(i)) => Value::Int(*i),
        Node::Scalar(Number::Float(f)) => Value::Float(*f),
        Node::Name(n) if n.starts_with('\u{1}') => Value::Float(hidden(n, row)?),
        Node::Name(n) => column(df, n, row)?,
        Node::Unary { op, operand } => {
            let v = eval(operand, row)?;
            element::unary(*op, v)?
        }
        Node::Binary { left, op, right } => {
            let (a, b) = (eval(left, row)?, eval(right, row)?);
            let previous = if matches!(op, Op::Cross | Op::CrossUp | Op::CrossDown) && row > 0 {
                Some((eval(left, row - 1)?, eval(right, row - 1)?))
            } else { None };
            element::binary(*op, a, b, previous)?
        }
        // Pure window commands without a continuation still use their published full
        // evaluator on a bounded window. Their stateful operands are handled by the plan.
        Node::Command(_) => return None,
    })
}

/// The shared allocation-free operand reader for Own and Chain plans.
pub(super) fn operand(
    df: &DataFrame, cmd: &crate::types::Command<crate::types::Bound>,
    slot: usize, default: &str, row: usize, hidden: &impl Fn(&str, usize) -> Option<f64>,
) -> Option<f64> {
    match crate::exec::operand(&cmd.series, slot) {
        Some(n) => evaluate(df, n, row, hidden)?.float(),
        None => column(df, default, row)?.float(),
    }
}
