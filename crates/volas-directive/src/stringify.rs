//! Render a directive AST back to its canonical string: whitespace removed,
//! minimal parenthesization by operator priority, default arguments and default
//! series dropped (so `boll`, `boll:20`, `boll:20@close` all stringify to
//! `boll`). The canonical form is volas-native — numeric arguments are not
//! coerced to a typed display (e.g. `100` stays `100`, not `100.0`).

use crate::parser::{is_ident_char, is_ident_start};
use crate::spec::{canon_sub, command_spec, normalize};
use crate::types::{ArgTokens, Command, Node, UnaryOp};

/// Canonical string form of a directive tree. Generic over the argument payload so
/// it canonicalizes both a bound [`Ast`](crate::types::Ast) (the frame-cache key)
/// and a raw [`Cst`](crate::types::Cst) (the form-level `directive_stringify`,
/// where a required argument may be absent — the form name is still canonical).
pub fn stringify<A: ArgTokens>(node: &Node<A>) -> String {
    match node {
        Node::Name(s) => s.clone(),
        Node::Scalar(v) => v.token(),
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

fn stringify_command<A: ArgTokens>(cmd: &Command<A>) -> String {
    // Canonicalize the name / sub here (case-folding, `cdl` -> `style`, alias subs);
    // this is idempotent for an already-bound `Ast` and the real work for a raw `Cst`.
    let name = normalize(&cmd.name);
    let sub = canon_sub(&name, cmd.sub.as_deref());
    let mut out = name.clone().into_owned();
    if let Some(s) = &sub {
        out.push('.');
        out.push_str(s);
    }
    let spec = command_spec(&name, sub.as_deref());

    // Arguments: keep a slot only if it is present and differs from its default. An
    // absent slot — or one equal to its default — renders empty, and trailing empties
    // fall away (`boll:20@close` -> `boll`; the absent period of `donchian.upper`
    // simply drops).
    let arg_tokens: Vec<String> = (0..cmd.args.arg_len())
        .map(|i| {
            let default = spec
                .as_ref()
                .and_then(|s| s.args.get(i))
                .and_then(|spec_arg| spec_arg.default.value())
                .map(|d| d.to_token());
            match cmd.args.arg_token(i) {
                Some(v) if Some(&v) != default.as_ref() => v,
                _ => String::new(),
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

/// A series operand. The parser reads a bare identifier in a series slot and nothing
/// else bare, so any other operand — a command with a sub / arguments / series, a
/// scalar, an expression — is grouped; one that renders as a bare identifier (a
/// command whose arguments are all defaults) is not, so the form is a fixed point.
fn stringify_series<A: ArgTokens>(node: &Node<A>) -> String {
    let text = stringify(node);
    if is_identifier(&text) {
        text
    } else {
        format!("({text})")
    }
}

fn is_identifier(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes.next().is_some_and(is_ident_start) && bytes.all(is_ident_char)
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

    /// A series slot reads a bare identifier only, so every other operand — a command
    /// with a sub, arguments or series of its own, a scalar — keeps its parentheses
    /// and the canonical name re-parses to the same tree.
    #[test]
    fn a_non_name_series_operand_stays_grouped() {
        for (input, canonical) in [
            ("ema:5@(rsi:5)", "ema:5@(rsi:5)"),
            ("ema:5@(macd.signal)", "ema:5@(macd.signal)"),
            ("ema:5@(boll.upper:30)", "ema:5@(boll.upper:30)"),
            ("ema:5@(ema:3@(rsi:4))", "ema:5@(ema:3@(rsi:4))"),
            ("correl:5@,(3)", "correl:5@,(3)"),
        ] {
            assert_eq!(s(input), canonical, "{input}");
            assert_eq!(parse(canonical).unwrap(), parse(input).unwrap(), "{input}");
        }
    }

    /// A canonical name drops the spaces around an operator, so an argument must end
    /// at every binary operator for `ma:5 OP ma:3` to re-parse as written.
    #[test]
    fn an_argument_ends_at_every_binary_operator() {
        for op in [
            "+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "&", "|", "^", "//", "\\\\",
            "><",
        ] {
            let input = format!("ma:5 {op} ma:3");
            let canonical = s(&input);
            assert_eq!(canonical, format!("ma:5{op}ma:3"));
            assert_eq!(parse(&canonical).unwrap(), parse(&input).unwrap(), "{input}");
        }
    }

    /// A whole number past `i64`'s range renders as itself, not saturated.
    #[test]
    fn a_large_whole_number_keeps_its_value() {
        assert_eq!(s("close > 100000000000000000000"), "close>100000000000000000000");
        assert_eq!(s("boll.upper:20,100000000000000000000"), "boll.upper:,100000000000000000000");
        assert_eq!(s("close > 2"), "close>2");
        assert_eq!(s("close > 0.000001"), "close>0.000001");
    }

    #[test]
    fn scalar_series_operand() {
        // a bare scalar in a series slot exercises stringify_series's Scalar arm
        use crate::types::{ArgValue, Ast, Command, Node};
        let node: Ast = Node::Command(Command {
            name: "correl".into(),
            sub: None,
            args: vec![ArgValue::Usize(5)].into_boxed_slice(),
            series: vec![Node::Name("close".into()), Node::Scalar(crate::types::Number::Int(3))],
        });
        assert!(stringify(&node).contains('3'));
    }
}
