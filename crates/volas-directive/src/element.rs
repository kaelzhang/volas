//! Element operations shared by batch evaluation and single-row continuation.
//! `None` from an operation means it cannot be evaluated; a missing truth is
//! represented independently as `Value::Bool(None)`.
use std::cmp::Ordering;
use volas_core::{Column, Result, VolasError};
use crate::types::{Op, UnaryOp};

#[derive(Clone, Copy)]
pub(crate) enum Value { Int(i64), Float(f64), Bool(Option<bool>) }
impl Value {
    pub fn float(self) -> Option<f64> {
        match self {
            Self::Int(i) if !volas_core::numeric::in_f64_domain(i) => None,
            Self::Int(i) => Some(i as f64),
            Self::Float(f) => Some(f),
            Self::Bool(b) => Some(b.map_or(f64::NAN, |v| f64::from(u8::from(v)))),
        }
    }
    pub fn truth(self) -> Option<bool> {
        match self { Self::Int(i) => Some(i != 0), Self::Float(f) => (!f.is_nan()).then_some(f != 0.0), Self::Bool(b) => b }
    }
    pub fn order(self, other: Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Some(a.cmp(&b)),
            (Self::Int(a), Self::Float(b)) => volas_core::numeric::cmp_i64_f64(a, b),
            (Self::Float(a), Self::Int(b)) => volas_core::numeric::cmp_i64_f64(b, a).map(Ordering::reverse),
            (Self::Bool(a), b) => Self::Int(i64::from(a?)).order(b),
            (a, Self::Bool(b)) => a.order(Self::Int(i64::from(b?))),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(&b),
        }
    }
}
pub(crate) fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a,b) { (Some(false), _) | (_, Some(false)) => Some(false), (Some(true),Some(true)) => Some(true), _ => None }
}
pub(crate) fn or(a: Option<bool>, b: Option<bool>) -> Option<bool> { and(a.map(|v| !v), b.map(|v| !v)).map(|v| !v) }


pub(crate) fn unary(op: UnaryOp, v: Value) -> Option<Value> {
    Some(match op {
        UnaryOp::Not => Value::Bool(v.truth().map(|b| !b)),
        UnaryOp::Neg => match v {
            Value::Int(i) => Value::Int(i.checked_neg()?),
            Value::Bool(Some(b)) => Value::Int(-i64::from(b)),
            Value::Bool(None) => Value::Bool(None),
            Value::Float(f) => Value::Float(-f),
        },
    })
}

pub(crate) fn binary(op: Op, a: Value, b: Value, previous: Option<(Value, Value)>) -> Option<Value> {
    Some(match op {
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
            let prev = previous.and_then(|(a, b)| a.order(b));
            let now = a.order(b);
            let up = and(prev.map(|o| o.is_le()), now.map(|o| o.is_gt()));
            let down = and(prev.map(|o| o.is_ge()), now.map(|o| o.is_lt()));
            Value::Bool(match op { Op::CrossUp => up, Op::CrossDown => down, _ => or(up, down) })
        }
    })
}

pub(crate) fn at(col: &Column, row: usize) -> Option<Value> {
    if row >= col.len() { return None; }
    if !col.is_valid(row) { return Some(Value::Bool(None)); }
    Some(match col {
        Column::I64(v, _) => Value::Int(v[row]),
        Column::I32(v, _) => Value::Int(v[row] as i64),
        Column::Bool(v, _) => Value::Bool(Some(v[row])),
        Column::F64(_) | Column::F32(_) => Value::Float(col.get_f64(row)),
        _ => return None,
    })
}

pub(crate) fn unary_column(op: UnaryOp, c: &Column) -> Result<Column> {
    let values = (0..c.len()).map(|i| unary(op, at(c, i).expect("numeric operand"))
        .ok_or_else(|| volas_core::numeric::overflow(volas_core::DType::I64, "unary -")));
    if op == UnaryOp::Not {
        Ok(Column::bool_from_options(values.map(|v| v.map(Value::truth)).collect::<Result<Vec<_>>>()?))
    } else if c.dtype().is_float() {
        Ok(Column::f64(values.map(|v| v.map(|v| v.float().expect("float"))).collect::<Result<_>>()?))
    } else {
        let values = values.map(|v| v.map(|v| match v { Value::Int(i) => Some(i), _ => None })).collect::<Result<Vec<_>>>()?;
        Ok(Column::i64_from_options(values))
    }
}

pub(crate) fn binary_column(op: Op, l: &Column, r: &Column) -> Result<Column> {
    let arithmetic = matches!(op, Op::Add | Op::Sub | Op::Mul | Op::Div);
    // Validate the entire input domain, including a value paired with NA.
    if arithmetic {
        for c in [l, r] {
            for i in 0..c.len() {
                if let Some(Value::Int(v)) = at(c, i) {
                    if !volas_core::numeric::in_f64_domain(v) {
                        return Err(volas_core::numeric::outside_f64_domain(v, "directive arithmetic"));
                    }
                }
            }
        }
    }
    let values = (0..l.len()).map(|i| {
        let a = at(l, i).expect("numeric operand");
        let b = at(r, i).expect("numeric operand");
        let previous = i.checked_sub(1).and_then(|p| Some((at(l,p)?, at(r,p)?)));
        binary(op, a, b, previous).ok_or_else(|| VolasError::Value("invalid numeric operation".into()))
    });
    if arithmetic {
        Ok(Column::f64(values.map(|v| v.map(|v| v.float().expect("float"))).collect::<Result<_>>()?))
    } else {
        Ok(Column::bool_from_options(values.map(|v| v.map(Value::truth)).collect::<Result<Vec<_>>>()?))
    }
}
