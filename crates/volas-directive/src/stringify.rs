//! The one renderer of a directive's canonical name — the name volas caches it under,
//! and what `directive_stringify` returns: whitespace removed, minimal
//! parenthesization by operator priority, every argument rendered from its bound value
//! (`ma:05` is `ma:5`, `hv:10,1W` is `hv:10`), default arguments and default series
//! dropped. The name is independent of the spelling: `macd`, `macd:`, `MACD` and
//! `macd:12,26` all name `macd`. A cached column refreshes from the bound directive it
//! keeps, never from a re-read of its name, so a name a data column could also hold
//! needs no mark of its own.

use crate::bind::is_bare_command;
use crate::parser::{is_ident_char, is_ident_start};
use crate::spec::normalize;
use crate::types::{Command, Node, Payload, UnaryOp};

/// Canonical string form of a directive tree: a bound [`Ast`](crate::types::Ast) (the
/// cache key) or a directive [`Form`](crate::types::Form) (`directive_stringify`, where
/// a required argument may be absent), rendered alike.
pub(crate) fn stringify_node<A: Payload>(node: &Node<A>) -> String {
    match node {
        // a bare name that is a command usable bare names that command (a frame-bound
        // tree holds none: there it is a command node, or a resolved Data reference)
        Node::Name(s) if is_bare_command(s) => normalize(s).into_owned(),
        Node::Data(s) | Node::Name(s) => s.clone(),
        Node::Scalar(v) => v.token(),
        Node::Command(cmd) => stringify_command(cmd),
        Node::Unary { op, operand } => {
            let op_s = match op {
                UnaryOp::Not => "~",
                UnaryOp::Neg => "-",
            };
            // A negated number keeps its parentheses: `-5` is an exact literal, `-(5)`
            // the negation of one, so each form re-parses as itself.
            let inner = match operand.as_ref() {
                Node::Binary { .. } => format!("({})", stringify_node(operand)),
                Node::Scalar(_) if *op == UnaryOp::Neg => format!("({})", stringify_node(operand)),
                _ => stringify_node(operand),
            };
            format!("{op_s}{inner}")
        }
        Node::Binary { left, op, right } => {
            let p = op.priority();
            // The left child only needs parens if it binds *looser* than us.
            let l = match left.as_ref() {
                Node::Binary { op: lop, .. } if lop.priority() < p => {
                    format!("({})", stringify_node(left))
                }
                _ => stringify_node(left),
            };
            // The right child needs parens at equal priority too (left-assoc).
            let r = match right.as_ref() {
                Node::Binary { op: rop, .. } if rop.priority() <= p => {
                    format!("({})", stringify_node(right))
                }
                _ => stringify_node(right),
            };
            format!("{l}{}{r}", op.token())
        }
    }
}

fn stringify_command<A: Payload>(cmd: &Command<A>) -> String {
    let spec = cmd.args.spec();
    let mut out = cmd.name.clone();
    if let Some(s) = &cmd.sub {
        out.push('.');
        out.push_str(s);
    }

    // Arguments: keep a slot only if it is present and differs from its default (as a
    // token, so a `-0` multiplier is not its default `0`). An absent slot — or one equal
    // to its default — renders empty, and trailing empties fall away.
    let arg_tokens: Vec<String> = (0..cmd.args.arg_len())
        .map(|i| {
            let default = spec.args.get(i).and_then(|a| a.default.value()).map(|d| d.to_token());
            match cmd.args.arg(i).map(|v| v.to_token()) {
                Some(v) if Some(&v) != default.as_ref() => v,
                _ => String::new(),
            }
        })
        .collect();
    let args = join_with(':', arg_tokens);

    // Series: keep one only if it differs from the default column.
    let series_tokens: Vec<String> = cmd
        .series
        .iter()
        .enumerate()
        .map(|(i, node)| match node {
            Node::Data(s) | Node::Name(s) if s.is_empty() || spec.series.get(i) == Some(&s.as_str()) => String::new(),
            other => stringify_series(other),
        })
        .collect();
    let series = join_with('@', series_tokens);

    out.push_str(&args);
    out.push_str(&series);
    out
}

/// A series operand. The parser reads a bare identifier in a series slot and nothing
/// else bare, so any other operand — a command with a sub / arguments / series, a
/// scalar, an expression — is grouped; one that renders as a bare identifier (a column,
/// or a command written bare) is not, so the form is a fixed point.
fn stringify_series<A: Payload>(node: &Node<A>) -> String {
    let text = stringify_node(node);
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

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::types::Ast {}
    impl Sealed for crate::types::Form {}
}

/// The two published directive representations accepted by the canonical renderer.
pub trait Render: sealed::Sealed {
    /// Render with the shared payload-generic node renderer.
    fn render(&self) -> String;
}
impl Render for crate::types::Ast {
    fn render(&self) -> String { stringify_node(self.node()) }
}
impl Render for crate::types::Form {
    fn render(&self) -> String { stringify_node(self) }
}

/// Canonical spelling of a validated directive or a lenient directive form.
pub fn stringify(tree: &(impl Render + ?Sized)) -> String { tree.render() }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, parse_cst};

    fn s(input: &str) -> String {
        stringify(&*parse(input).unwrap())
    }

    /// The form a frame-blind `directive_stringify` names: bound leniently.
    fn form(input: &str) -> String {
        stringify(&crate::bind::bind_form(&parse_cst(input).unwrap()).unwrap())
    }

    /// A command written bare names its command, in its canonical case; any other bare
    /// name (a column) keeps its spelling.
    #[test]
    fn a_bare_command_names_its_command() {
        for d in ["OBV", "obv", "Obv"] {
            assert_eq!(s(d), "obv");
            assert_eq!(form(d), "obv");
        }
        assert_eq!(s("close - OBV"), "close-obv");
        assert_eq!(s("Close + RSI"), "Close+RSI"); // `rsi` needs a period: a column
        let columns = |n: &str| n == "Close";
        let bound = crate::bind::bind_frame(&parse("Close - OBV").unwrap(), &columns, "").unwrap();
        assert_eq!(stringify(&bound), "Close-obv");
        assert!(crate::bind::uses_bare_command(&bound, "obv") && !crate::bind::uses_bare_command(&bound, "OBV"));
    }

    #[test]
    fn drops_default_args_and_series() {
        assert_eq!(s("boll"), "boll");
        assert_eq!(s("boll:30@close"), "boll:30");
        assert_eq!(s("ma:5@close"), "ma:5");
    }

    /// The name is independent of the spelling: an explicit command whose arguments and
    /// series are all defaults names the same column as its bare name.
    #[test]
    fn every_spelling_of_a_command_has_one_name() {
        for d in ["macd", "macd:", "MACD", "Macd:12,26", "macd:012,26@close", "macd.dif"] {
            assert_eq!(s(d), "macd", "{d}");
        }
        assert_eq!(s("boll:20@close"), "boll");
        assert_eq!(s("change: * 2"), "change*2");
        assert_eq!(s("ma:5@(obv:)"), "ma:5@obv");
        assert_eq!(s("macd.signal:12"), "macd.signal");
        assert_eq!(form("rsi:"), "rsi");
    }

    #[test]
    fn operator_priority_parens() {
        assert_eq!(s("close + open * high"), "close+open*high");
        assert_eq!(s("3 * (high - low)"), "3*(high-low)");
        assert_eq!(s("(kdj.j > 100) | (kdj.j <= 100)"), "kdj.j>100|kdj.j<=100");
        assert_eq!(s("~ ( kdj.j < 0 )"), "~(kdj.j<0)");
        assert_eq!(s("-(5) + -5 - -(close)"), "-(5)+-5--close");
    }

    #[test]
    fn sub_canonicalization() {
        assert_eq!(s("macd.dif"), "macd");
        assert_eq!(s("macd.s"), "macd.signal");
        assert_eq!(s("boll.u:20"), "boll.upper");
    }

    /// Arguments render from their bound values, whatever their spelling.
    #[test]
    fn arguments_render_from_their_values() {
        assert_eq!(s("ma:05"), "ma:5");
        assert_eq!(s("ma:+2"), "ma:2");
        assert_eq!(s("boll.upper:20,2.0"), "boll.upper");
        assert_eq!(s("MACD:012,26"), "macd");
        assert_eq!(s("hv:10,1W"), "hv:10,1w");
        assert_eq!(s("stddev:2,1.50"), "stddev:2,1.5");
        assert_eq!(s("boll.upper:20,-0.0"), "boll.upper:,-0");
    }

    /// A form names exactly what its directive caches under; an absent required
    /// argument or operand simply drops.
    #[test]
    fn a_form_renders_like_its_directive() {
        for d in ["ma:05", "kdj.j:9,3,2,100@high,close,close", "MACD:12,26", "boll:20@close", "obv", "close > 2"] {
            assert_eq!(form(d), s(d), "{d}");
        }
        assert_eq!(form("donchian.upper"), "donchian.upper");
        assert_eq!(form("correl:5"), "correl:5");
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
}
