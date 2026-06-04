//! Directive AST.

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
            Op::Lt | Op::Le | Op::Eq | Op::Ne | Op::Ge | Op::Gt | Op::CrossUp | Op::CrossDown
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

/// A parsed indicator command, e.g. `macd.signal:12,26,9@close`.
#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    /// Command name.
    pub name: String,
    /// Optional sub-command.
    pub sub: Option<String>,
    /// Positional args as raw strings; `None` means "use the default".
    pub args: Vec<Option<String>>,
    /// `@` series operands.
    pub series: Vec<Node>,
}

/// A node in the directive AST.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// A bare identifier: a column name, or a no-argument command.
    Name(String),
    /// A numeric literal.
    Scalar(f64),
    /// An indicator command.
    Command(Command),
    /// A unary operation.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<Node>,
    },
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
