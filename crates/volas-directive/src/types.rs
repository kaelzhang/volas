//! Directive AST.

/// A binary operator between two directive operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `==`
    Eq,
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
}

impl Op {
    /// The canonical source token for the operator.
    pub fn token(&self) -> &'static str {
        match self {
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Eq => "==",
            Op::Ge => ">=",
            Op::Gt => ">",
            Op::CrossUp => "//",
            Op::CrossDown => "\\\\",
            Op::Cross => "><",
        }
    }
}

/// A parsed indicator command, e.g. `macd.signal:12,26,9@close`.
#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    /// Command name, e.g. `"ma"`, `"macd"`.
    pub name: String,
    /// Optional sub-command, e.g. `"signal"`, `"upper"`, `"j"`.
    pub sub: Option<String>,
    /// Positional args as raw strings; `None` means "use the default".
    pub args: Vec<Option<String>>,
    /// `@` series operands (column names or nested directives).
    pub series: Vec<Node>,
}

/// A node in the directive AST.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// A bare identifier: a column name, or a no-argument command (e.g. `tr`).
    Name(String),
    /// A numeric literal operand (e.g. the `0` in `kdj.j < 0`).
    Scalar(f64),
    /// An indicator command.
    Command(Command),
    /// A binary operation `left op right`.
    Binary {
        /// Left operand.
        left: Box<Node>,
        /// Operator.
        op: Op,
        /// Right operand.
        right: Box<Node>,
    },
}
