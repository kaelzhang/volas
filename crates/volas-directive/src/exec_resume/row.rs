//! Scalar evaluation of a plan's stateless remainder. Stateful names read the freshly
//! stepped node rings; numeric comparisons preserve integer precision.
use std::cmp::Ordering;
use volas_core::{Column, DataFrame};
use crate::types::{BoundNode, Node, Number, Op, UnaryOp};

#[derive(Clone, Copy)]
pub(super) enum Value { Int(i64), Float(f64), Bool(Option<bool>) }
impl Value {
    pub fn float(self) -> Option<f64> {
        match self {
            Self::Int(i) if i.unsigned_abs() > (1u64 << 53) => None,
            Self::Int(i) => Some(i as f64),
            Self::Float(f) => Some(f),
            Self::Bool(b) => Some(b.map_or(f64::NAN, |v| f64::from(u8::from(v)))),
        }
    }
    fn truth(self) -> Option<bool> {
        match self { Self::Int(i) => Some(i != 0), Self::Float(f) => (!f.is_nan()).then_some(f != 0.0), Self::Bool(b) => b }
    }
    fn order(self, other: Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Some(a.cmp(&b)),
            (Self::Int(a), Self::Float(b)) => volas_core::numeric::cmp_i64_f64(a, b),
            (Self::Float(a), Self::Int(b)) => volas_core::numeric::cmp_i64_f64(b, a).map(Ordering::reverse),
            (a, b) => a.float()?.partial_cmp(&b.float()?),
        }
    }
}
fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a,b) { (Some(false), _) | (_, Some(false)) => Some(false), (Some(true),Some(true)) => Some(true), _ => None }
}
fn or(a: Option<bool>, b: Option<bool>) -> Option<bool> { and(a.map(|v| !v), b.map(|v| !v)).map(|v| !v) }

pub(super) fn column(df: &DataFrame, name: &str, row: usize) -> Option<Value> {
    let col = df.column(name).ok()?;
    if row >= col.len() { return None; }
    if !col.is_valid(row) { return Some(Value::Float(f64::NAN)); }
    Some(match col {
        Column::I64(v, _) => Value::Int(v[row]),
        Column::I32(v, _) => Value::Int(v[row] as i64),
        Column::Bool(v, _) => Value::Bool(Some(v[row])),
        Column::F64(_) | Column::F32(_) => Value::Float(col.get_f64(row)),
        _ => return None,
    })
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
            match op { UnaryOp::Neg => Value::Float(-v.float()?), UnaryOp::Not => Value::Bool(v.truth().map(|v| !v)) }
        }
        Node::Binary { left, op, right } => {
            let (a, b) = (eval(left, row)?, eval(right, row)?);
            match op {
                Op::Add => Value::Float(a.float()? + b.float()?),
                Op::Sub => Value::Float(a.float()? - b.float()?),
                Op::Mul => Value::Float(a.float()? * b.float()?),
                Op::Div => Value::Float(a.float()? / b.float()?),
                Op::And => Value::Bool(and(a.truth(), b.truth())),
                Op::Or => Value::Bool(or(a.truth(), b.truth())),
                Op::Xor => Value::Bool(a.truth().zip(b.truth()).map(|(a,b)| a ^ b)),
                Op::Lt => Value::Bool(a.order(b).map(|o| o.is_lt())),
                Op::Le => Value::Bool(a.order(b).map(|o| o.is_le())),
                Op::Eq => Value::Bool(a.order(b).map(|o| o.is_eq())),
                Op::Ne => Value::Bool(a.order(b).map(|o| !o.is_eq())),
                Op::Ge => Value::Bool(a.order(b).map(|o| o.is_ge())),
                Op::Gt => Value::Bool(a.order(b).map(|o| o.is_gt())),
                Op::Cross | Op::CrossUp | Op::CrossDown => {
                    let prev = row.checked_sub(1).and_then(|p| eval(left,p)?.order(eval(right,p)?));
                    let now = a.order(b);
                    let up = and(prev.map(|o| o.is_le()), now.map(|o| o.is_gt()));
                    let down = and(prev.map(|o| o.is_ge()), now.map(|o| o.is_lt()));
                    Value::Bool(match op { Op::CrossUp => up, Op::CrossDown => down, _ => or(up, down) })
                }
            }
        }
        // Pure window commands without a continuation still use their published full
        // evaluator on a bounded window. Their stateful operands are handled by the plan.
        Node::Command(_) => return None,
    })
}
