//! Directive AST.
//!
//! Two shapes share one tree definition via the [`Node`] type parameter, which
//! is only ever the per-command argument payload:
//!
//! * [`Cst`] — the parser's output, arguments as raw token strings.
//! * [`Ast`] — the *bound* tree: arguments parsed, defaulted, and type-checked
//!   against the command spec exactly once (see `bind`), so execution / lookback
//!   / stringify read ready typed values with no re-parsing, re-defaulting, or
//!   re-validation.

use volas_time::TimeFrame;

use crate::spec::CommandSpec;

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `>=`
    Ge,
    /// `>`
    Gt,
    /// `//` — left crosses up through right.
    CrossUp,
    /// `\\` — left crosses down through right.
    CrossDown,
    /// `><` — left crosses right in either direction.
    Cross,
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `&`
    And,
    /// `|`
    Or,
    /// `^`
    Xor,
}

impl Op {
    /// Binding priority (higher binds tighter), mirroring the parser's
    /// precedence levels — used by `stringify` to decide parenthesization.
    pub fn priority(&self) -> u8 {
        match self {
            Op::And | Op::Or | Op::Xor => 1,
            Op::Lt
            | Op::Le
            | Op::Eq
            | Op::Ne
            | Op::Ge
            | Op::Gt
            | Op::CrossUp
            | Op::CrossDown
            | Op::Cross => 2,
            Op::Add | Op::Sub => 3,
            Op::Mul | Op::Div => 4,
        }
    }

    /// The canonical source token.
    pub fn token(&self) -> &'static str {
        match self {
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Eq => "==",
            Op::Ne => "!=",
            Op::Ge => ">=",
            Op::Gt => ">",
            Op::CrossUp => "//",
            Op::CrossDown => "\\\\",
            Op::Cross => "><",
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::And => "&",
            Op::Or => "|",
            Op::Xor => "^",
        }
    }
}

/// A unary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `~` — logical NOT.
    Not,
    /// `-` — arithmetic negation.
    Neg,
}

/// A bound argument value: the result of parsing a raw token once against its
/// spec bound (or taking the spec default). The variant is fixed per argument
/// position by the command spec, so the typed accessors are infallible at the
/// dispatch / lookback boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum ArgValue {
    /// A window / period / count / matype selector (`ArgBound::IntMin` / `IntRange`).
    Usize(usize),
    /// A signed integer (`ArgBound::OneOfI64` / `I64Min`): a direction, trading-day count.
    I64(i64),
    /// A finite float (`ArgBound::Finite` / `FloatMin` / `FloatGt` / `FloatRange`).
    F64(f64),
    /// A bar time frame (`ArgBound::TimeFrame`), parsed by [`TimeFrame::from_label`].
    TimeFrame(TimeFrame),
}

impl ArgValue {
    /// Extract a `usize` argument. The spec bound fixed this position's variant,
    /// so a mismatch is a spec/dispatch disagreement, never user input.
    pub fn as_usize(&self) -> usize {
        match self {
            ArgValue::Usize(v) => *v,
            _ => unreachable!("argument is not a usize: {self:?}"), // LCOV_EXCL_LINE
        }
    }

    /// Extract an `i64` argument.
    pub fn as_i64(&self) -> i64 {
        match self {
            ArgValue::I64(v) => *v,
            _ => unreachable!("argument is not an i64: {self:?}"), // LCOV_EXCL_LINE
        }
    }

    /// Extract an `f64` argument.
    pub fn as_f64(&self) -> f64 {
        match self {
            ArgValue::F64(v) => *v,
            _ => unreachable!("argument is not an f64: {self:?}"), // LCOV_EXCL_LINE
        }
    }

    /// Extract a time-frame argument.
    pub fn as_time_frame(&self) -> TimeFrame {
        match self {
            ArgValue::TimeFrame(v) => *v,
            _ => unreachable!("argument is not a time frame: {self:?}"), // LCOV_EXCL_LINE
        }
    }

    /// The canonical source token for this value, used by `stringify`. Integers
    /// render without a decimal point; floats keep their natural form.
    pub fn to_token(&self) -> String {
        match self {
            ArgValue::Usize(v) => v.to_string(),
            ArgValue::I64(v) => v.to_string(),
            ArgValue::F64(v) => number_token(*v),
            ArgValue::TimeFrame(v) => v.label().to_string(),
        }
    }
}

/// A bound command's payload: its arguments — parsed, defaulted and type-checked once
/// against the spec of its line — and that spec, so every consumer reads the line's
/// facts from the command itself. `bare` keeps the spelling of a command written as a
/// bare name (`obv`) and resolved to the command against a frame: it renders as
/// written, where an explicit all-default command renders `obv:`.
#[derive(Clone, Debug)]
pub struct Bound {
    /// The bound arguments, one per spec slot.
    pub values: Box<[ArgValue]>,
    /// The spec of the command's line.
    pub spec: CommandSpec,
    /// The spelling of a command written as a bare name, if it was.
    pub bare: Option<Box<str>>,
}

impl std::ops::Deref for Bound {
    type Target = [ArgValue];
    fn deref(&self) -> &[ArgValue] {
        &self.values
    }
}

/// Two bound commands are equal when their arguments and spelling are: the spec is a
/// function of the command's name and sub-command.
impl PartialEq for Bound {
    fn eq(&self, other: &Bound) -> bool {
        self.values == other.values && self.bare == other.bare
    }
}

/// The payload of a command in a directive *form* (`directive_stringify`'s input): the
/// arguments supplied, each bound against its slot, and `None` where a slot is absent —
/// a required one included, since a form such as `donchian.upper` names a real line
/// whose period is supplied only at use.
#[derive(Clone, Debug)]
pub struct FormArgs {
    /// The supplied arguments, bound; `None` for an absent slot.
    pub values: Vec<Option<ArgValue>>,
    /// The spec of the command's line.
    pub spec: CommandSpec,
}

/// What the renderer reads from a command's payload: its spec, its bound arguments (an
/// absent slot is `None`) and the spelling of a bare-written command. Implemented by the
/// bound [`Ast`] and by a [`Form`], so one renderer names both.
pub trait Payload {
    /// The spec of the command's line.
    fn spec(&self) -> &CommandSpec;
    /// Number of argument slots.
    fn arg_len(&self) -> usize;
    /// The bound argument at slot `i`, or `None` if absent.
    fn arg(&self, i: usize) -> Option<&ArgValue>;
    /// The spelling of a command written as a bare name.
    fn bare(&self) -> Option<&str>;
}

impl Payload for Bound {
    fn spec(&self) -> &CommandSpec {
        &self.spec
    }
    fn arg_len(&self) -> usize {
        self.values.len()
    }
    fn arg(&self, i: usize) -> Option<&ArgValue> {
        self.values.get(i)
    }
    fn bare(&self) -> Option<&str> {
        self.bare.as_deref()
    }
}

impl Payload for FormArgs {
    fn spec(&self) -> &CommandSpec {
        &self.spec
    }
    fn arg_len(&self) -> usize {
        self.values.len()
    }
    fn arg(&self, i: usize) -> Option<&ArgValue> {
        self.values.get(i).and_then(Option::as_ref)
    }
    fn bare(&self) -> Option<&str> {
        None
    }
}

/// A parsed indicator command, e.g. `macd.signal:12,26,9@close`. The payload `A` is the
/// raw argument strings in a [`Cst`], the bound arguments and line spec ([`Bound`]) in an
/// [`Ast`].
#[derive(Clone, Debug, PartialEq)]
pub struct Command<A> {
    /// Command name (canonical in an [`Ast`]: lower-cased, `cdl` folded to `style`).
    pub name: String,
    /// Optional sub-command (canonical in an [`Ast`]).
    pub sub: Option<String>,
    /// Positional arguments.
    pub args: A,
    /// `@` series operands.
    pub series: Vec<Node<A>>,
}

/// A node in the directive AST, generic over its command argument payload `A`.
#[derive(Clone, Debug, PartialEq)]
pub enum Node<A> {
    /// A bare identifier: a column name, or a no-argument command.
    Name(String),
    /// A numeric literal.
    Scalar(Number),
    /// An indicator command.
    Command(Command<A>),
    /// A unary operation.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<Node<A>>,
    },
    /// A binary operation `left op right`.
    Binary {
        /// Left operand.
        left: Box<Node<A>>,
        /// Operator.
        op: Op,
        /// Right operand.
        right: Box<Node<A>>,
    },
}

/// The parser's concrete syntax tree: arguments as raw token strings, `None` for
/// an omitted slot. Lowered to an [`Ast`] by `bind`.
pub type Cst = Node<Vec<Option<String>>>;

/// The bound AST: arguments parsed, defaulted, and type-checked once, each command
/// carrying its line's spec. Executed, measured (lookback), and rendered (stringify)
/// without per-call re-parsing.
pub type Ast = Node<Bound>;

/// A directive form bound leniently (`directive_stringify`): every supplied argument
/// bound, an absent one left absent.
pub type Form = Node<FormArgs>;

/// A numeric literal, held exactly: an integral value inside the int64 range is an
/// `Int` — so `volume > 9007199254740993` compares against that integer, never its
/// float64 neighbour (contract #26) — and any other value is a float64. The form is
/// canonical (`2.0` is `Int(2)`), so a literal re-parses from its token unchanged. An
/// integer zero has no sign (`-0` is `Int(0)`); a float zero written with its decimal
/// point keeps it (`-0.0` is `Float(-0.0)`, rendered `-0.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    /// An integer literal.
    Int(i64),
    /// A float literal (non-integral, or past the int64 range).
    Float(f64),
}

impl Number {
    /// The literal written as `text` (an optional `-`, digits, an optional `.`). An
    /// integral value (`5`, `5.0`, `9007199254740993.000`) is read exactly, never
    /// through a float64: inside int64 it is an `Int`; past it, a float64 that holds
    /// it exactly, else [`LiteralError::Inexact`] — a written integer is never
    /// rounded silently. Any other value is a float64 in canonical form.
    pub(crate) fn parse(text: &str) -> Result<Number, LiteralError> {
        let (whole, frac) = text.split_once('.').unwrap_or((text, ""));
        let integral = frac.bytes().all(|b| b == b'0');
        if integral {
            if let Ok(i) = whole.parse::<i64>() {
                // An integer has no signed zero (`-0` is `0`); a float literal written
                // with a decimal point keeps the sign of its zero (`-0.0`).
                if i == 0 && text.starts_with('-') && text.contains('.') {
                    return Ok(Number::Float(-0.0));
                }
                return Ok(Number::Int(i));
            }
        }
        let v = text.parse::<f64>().map_err(|_| LiteralError::Invalid)?;
        if !v.is_finite() {
            return Err(LiteralError::OutOfRange);
        }
        if integral {
            // past int64: only a float64 that is exactly the written integer
            let (sign, digits) = whole.strip_prefix('-').map_or(("", whole), |d| ("-", d));
            let digits = digits.trim_start_matches('0');
            return if format!("{v:.0}") == format!("{sign}{digits}") {
                Ok(Number::Float(v))
            } else {
                Err(LiteralError::Inexact)
            };
        }
        Ok(match <i64 as volas_core::numeric::Numeric>::try_from_f64(v) {
            Some(i) => Number::Int(i),
            None => Number::Float(v),
        })
    }

    /// The literal's canonical source token.
    pub(crate) fn token(self) -> String {
        match self {
            Number::Int(i) => i.to_string(),
            // `-0` would re-parse as the integer zero, which has no sign
            Number::Float(v) if v == 0.0 && v.is_sign_negative() => "-0.0".to_string(),
            Number::Float(v) => number_token(v),
        }
    }
}

/// Why a number token is not a literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LiteralError {
    /// Not a number at all (e.g. `1.2.3`).
    Invalid,
    /// Rounds to `±inf`, which has no literal form.
    OutOfRange,
    /// An integral value past int64 that no float64 holds exactly.
    Inexact,
}

/// The canonical source token of a number that reads back as the same `f64`: a
/// whole number is its exact integer digits without a decimal point (`2`,
/// `100000000000000000000`, never a saturated integer cast nor a shortened form the
/// exact integer reading would reject), any other value its shortest decimal form.
pub(crate) fn number_token(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_OPS: [Op; 16] = [
        Op::Lt,
        Op::Le,
        Op::Eq,
        Op::Ne,
        Op::Ge,
        Op::Gt,
        Op::CrossUp,
        Op::CrossDown,
        Op::Cross,
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::And,
        Op::Or,
        Op::Xor,
    ];

    #[test]
    fn op_token_and_priority_for_every_variant() {
        for op in ALL_OPS {
            assert!(!op.token().is_empty());
            assert!((1..=4).contains(&op.priority()));
        }
        assert_eq!(Op::And.priority(), 1);
        assert_eq!(Op::Lt.priority(), 2);
        assert_eq!(Op::Add.priority(), 3);
        assert_eq!(Op::Mul.priority(), 4);
        assert_eq!(Op::CrossUp.token(), "//");
        assert_eq!(Op::CrossDown.token(), "\\\\");
        assert_eq!(Op::Cross.token(), "><");
        assert_eq!(Op::Ne.token(), "!=");
    }

    #[test]
    fn nodes_construct_and_debug() {
        let cmd = Command {
            name: "ma".into(),
            sub: Some("x".into()),
            args: vec![Some("5".into()), None],
            series: vec![Node::Name("close".into())],
        };
        assert_eq!(cmd.clone().name, "ma");
        let nodes: [Cst; 6] = [
            Node::Name("a".into()),
            Node::Scalar(Number::Float(1.5)),
            Node::Command(cmd),
            Node::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Node::Name("b".into())),
            },
            Node::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Node::Scalar(Number::Int(2))),
            },
            Node::Binary {
                left: Box::new(Node::Name("c".into())),
                op: Op::Gt,
                right: Box::new(Node::Scalar(Number::Int(0))),
            },
        ];
        for n in &nodes {
            assert!(!format!("{n:?}").is_empty());
        }
    }

    #[test]
    fn arg_value_accessors_and_tokens() {
        assert_eq!(ArgValue::Usize(14).as_usize(), 14);
        assert_eq!(ArgValue::I64(-1).as_i64(), -1);
        assert_eq!(ArgValue::F64(2.5).as_f64(), 2.5);
        assert_eq!(ArgValue::TimeFrame(TimeFrame::Day1).as_time_frame(), TimeFrame::Day1);
        assert_eq!(ArgValue::Usize(20).to_token(), "20");
        assert_eq!(ArgValue::I64(252).to_token(), "252");
        assert_eq!(ArgValue::F64(2.0).to_token(), "2");
        assert_eq!(ArgValue::F64(0.7).to_token(), "0.7");
        // A time frame renders as its canonical label, so `1W` and `1w` share one key.
        assert_eq!(ArgValue::TimeFrame(TimeFrame::Week1).to_token(), "1w");
    }
}
