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
    let crossing = matches!(op, Op::Cross | Op::CrossUp | Op::CrossDown);
    let values = (0..l.len()).map(|i| {
        let a = at(l, i).expect("numeric operand");
        let b = at(r, i).expect("numeric operand");
        let previous = if crossing { i.checked_sub(1).and_then(|p| Some((at(l,p)?, at(r,p)?))) } else { None };
        binary(op, a, b, previous).ok_or_else(|| VolasError::Value("invalid numeric operation".into()))
    });
    if arithmetic {
        Ok(Column::f64(values.map(|v| v.map(|v| v.float().expect("float"))).collect::<Result<_>>()?))
    } else {
        Ok(Column::bool_from_options(values.map(|v| v.map(Value::truth)).collect::<Result<Vec<_>>>()?))
    }
}

#[cfg(test)]
mod traversal_tests {
    use super::*;

    #[test]
    fn batch_binary_matches_scalar_owner_for_every_op_and_numeric_type() {
        let columns = [
            Column::f64(vec![f64::NAN, -0.0, 0.0, -1.0, 2.0, f64::INFINITY]),
            Column::f32(vec![f32::NAN, 0.0, 1.0, f32::NEG_INFINITY, 2.0, -0.0]),
            Column::i64_from_options(vec![Some(i64::MAX), None, Some(9_007_199_254_740_993), Some(-9_007_199_254_740_993), Some(i64::MIN), Some(0)]),
            Column::i64(vec![0, -1, 2, 1, 9_007_199_254_740_992, -9_007_199_254_740_992]),
            Column::i32(vec![0, -1, 2, i32::MAX, i32::MIN, 1]),
            Column::bool_from_options(vec![Some(true), None, Some(false), Some(true), Some(false), Some(true)]),
        ];
        for left in &columns {
            for right in &columns {
                for op in [Op::Lt, Op::Le, Op::Eq, Op::Ne, Op::Ge, Op::Gt, Op::CrossUp, Op::CrossDown,
                    Op::Cross, Op::Add, Op::Sub, Op::Mul, Op::Div, Op::And, Op::Or, Op::Xor] {
                    for (start, end) in [(0, 0), (0, 1), (0, 6), (2, 6)] {
                        let l = left.slice(start, end);
                        let r = right.slice(start, end);
                        let expected: Vec<_> = (0..l.len()).map(|i| {
                            let previous = i.checked_sub(1).and_then(|p| Some((at(&l, p)?, at(&r, p)?)));
                            binary(op, at(&l, i).unwrap(), at(&r, i).unwrap(), previous)
                        }).collect();
                        let actual = binary_column(op, &l, &r);
                        if expected.iter().any(Option::is_none) { assert!(actual.is_err()); continue; }
                        let actual = actual.unwrap();
                        for (i, expected) in expected.into_iter().enumerate() {
                            let expected = expected.unwrap();
                            if matches!(op, Op::Add | Op::Sub | Op::Mul | Op::Div) {
                                assert_eq!(actual.get_f64(i).to_bits(), expected.float().unwrap().to_bits(), "{op:?}, row{i}, range{start}..{end}");
                            } else {
                                let truth = expected.truth();
                                assert_eq!(actual.is_valid(i), truth.is_some());
                                if let Some(truth) = truth { assert_eq!(actual.get_f64(i), f64::from(u8::from(truth))); }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn crossings_keep_the_prior_row_and_exact_integer_boundary() {
        let l = Column::i64(vec![0, 9_007_199_254_740_993, 0]);
        let r = Column::f64(vec![9_007_199_254_740_992.0; 3]);
        for (op, middle, last) in [(Op::Cross, true, true), (Op::CrossUp, true, false), (Op::CrossDown, false, true)] {
            let result = binary_column(op, &l, &r).unwrap();
            assert_eq!(result.is_valid(0), op == Op::CrossUp);
            if op == Op::CrossUp { assert_eq!(result.get_f64(0), 0.0); }
            assert_eq!(result.get_f64(1), f64::from(u8::from(middle)));
            assert_eq!(result.get_f64(2), f64::from(u8::from(last)));
            let slice = binary_column(op, &l.slice(1, 3), &r.slice(1, 3)).unwrap();
            // A slice starts a new traversal and must not read outside its boundary.
            if op != Op::CrossDown { assert!(!slice.is_valid(0)); }
        }
        let missing = Column::f64(vec![f64::NAN; 3]);
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            assert!(binary_column(op, &l, &missing).is_err());
        }
    }
}
