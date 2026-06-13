//! Render a directive AST back to its canonical string: whitespace removed,
//! minimal parenthesization by operator priority, default arguments and default
//! series dropped (so `boll`, `boll:20`, `boll:20@close` all stringify to
//! `boll`). The canonical form is volas-native — numeric arguments are not
//! coerced to a typed display (e.g. `100` stays `100`, not `100.0`).

use crate::spec::command_spec;
use crate::types::{ArgValue, Ast, Command, Node, UnaryOp};

/// Canonical string form of a directive AST.
pub fn stringify(node: &Ast) -> String {
    match node {
        Node::Name(s) => s.clone(),
        Node::Scalar(v) => format_num(*v),
        Node::Command(cmd) => stringify_command(cmd),
        Node::Unary { op, operand } => {
            let op_s = match op {
                UnaryOp::Not => "~",
                UnaryOp::Neg => "-",
            };
            let inner = match operand.as_ref() {
                Node::Binary { .. } => format!("({})", stringify(operand)),
                _ => stringify(operand),
            };
            format!("{op_s}{inner}")
        }
        Node::Binary { left, op, right } => {
            let p = op.priority();
            // The left child only needs parens if it binds *looser* than us.
            let l = match left.as_ref() {
                Node::Binary { op: lop, .. } if lop.priority() < p => {
                    format!("({})", stringify(left))
                }
                _ => stringify(left),
            };
            // The right child needs parens at equal priority too (left-assoc).
            let r = match right.as_ref() {
                Node::Binary { op: rop, .. } if rop.priority() <= p => {
                    format!("({})", stringify(right))
                }
                _ => stringify(right),
            };
            format!("{l}{}{r}", op.token())
        }
    }
}

fn format_num(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn stringify_command(cmd: &Command<Box<[ArgValue]>>) -> String {
    // The name and sub are already canonical (bound), so they render directly.
    let mut out = cmd.name.clone();
    if let Some(s) = &cmd.sub {
        out.push('.');
        out.push_str(s);
    }
    let spec = command_spec(&cmd.name, cmd.sub.as_deref());

    // Arguments: keep one only if it differs from its default. Every argument is
    // present (resolved), so an argument equal to its default renders empty and
    // trailing defaults fall away (`boll:20@close` -> `boll`).
    let arg_tokens: Vec<String> = cmd
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let is_default = spec
                .as_ref()
                .and_then(|s| s.args.get(i))
                .and_then(|spec_arg| spec_arg.default.value())
                .is_some_and(|default| default == *a);
            if is_default {
                String::new()
            } else {
                a.to_token()
            }
        })
        .collect();
    out.push_str(&join_with(':', arg_tokens));

    // Series: keep one only if it differs from the default column.
    let series_tokens: Vec<String> = cmd
        .series
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let default = spec.as_ref().and_then(|s| s.series.get(i)).copied();
            match node {
                Node::Name(s) if s.is_empty() || Some(s.as_str()) == default => String::new(),
                other => stringify_series(other),
            }
        })
        .collect();
    out.push_str(&join_with('@', series_tokens));

    out
}

fn stringify_series(node: &Ast) -> String {
    match node {
        Node::Name(s) => s.clone(),
        Node::Scalar(v) => format_num(*v),
        Node::Command(c) => stringify_command(c),
        // An expression as a series operand stays grouped so it re-parses.
        _ => format!("({})", stringify(node)),
    }
}

/// Join `tokens` with `,` under `prefix` (`:` or `@`), dropping trailing empties.
/// Returns empty if nothing remains.
fn join_with(prefix: char, mut tokens: Vec<String>) -> String {
    while tokens.last().is_some_and(|t| t.is_empty()) {
        tokens.pop();
    }
    if tokens.is_empty() {
        String::new()
    } else {
        format!("{prefix}{}", tokens.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn s(input: &str) -> String {
        stringify(&parse(input).unwrap())
    }

    #[test]
    fn drops_default_args_and_series() {
        assert_eq!(s("boll"), "boll");
        assert_eq!(s("boll:20@close"), "boll");
        assert_eq!(s("boll:30@close"), "boll:30");
        assert_eq!(s("macd:12,26"), "macd");
        assert_eq!(s("ma:5@close"), "ma:5");
    }

    #[test]
    fn operator_priority_parens() {
        assert_eq!(s("close + open * high"), "close+open*high");
        assert_eq!(s("3 * (high - low)"), "3*(high-low)");
        assert_eq!(s("(kdj.j > 100) | (kdj.j <= 100)"), "kdj.j>100|kdj.j<=100");
        assert_eq!(s("~ ( kdj.j < 0 )"), "~(kdj.j<0)");
    }

    #[test]
    fn sub_canonicalization() {
        assert_eq!(s("macd.dif"), "macd");
        assert_eq!(s("macd.s"), "macd.signal");
        assert_eq!(s("boll.u:20"), "boll.upper");
    }

    #[test]
    fn default_argument_slots() {
        assert_eq!(s("kdj.j:,4"), "kdj.j:,4");
        assert_eq!(s("kdj.j:@,high"), "kdj.j@,high");
    }

    #[test]
    fn left_parens_fractional_scalar_and_expression_series() {
        // left child binds looser than the parent -> it gets parenthesised
        assert_eq!(s("(close + 1) * 2.5"), "(close+1)*2.5");
        // a fractional scalar renders with its decimals (the non-integer branch)
        assert_eq!(s("close > 2.5"), "close>2.5");
        // an expression as a series operand stays grouped so it re-parses
        assert_eq!(s("ma:5@(close + 1)"), "ma:5@(close+1)");
    }

    #[test]
    fn scalar_series_operand() {
        // a bare scalar in a series slot exercises stringify_series's Scalar arm
        use crate::types::{ArgValue, Ast, Command, Node};
        let node: Ast = Node::Command(Command {
            name: "correl".into(),
            sub: None,
            args: vec![ArgValue::Usize(5)].into_boxed_slice(),
            series: vec![Node::Name("close".into()), Node::Scalar(3.0)],
        });
        assert!(stringify(&node).contains('3'));
    }
}
