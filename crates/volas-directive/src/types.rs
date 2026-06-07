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
        let nodes = [
            Node::Name("a".into()),
            Node::Scalar(1.5),
            Node::Command(cmd),
            Node::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Node::Name("b".into())),
            },
            Node::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Node::Scalar(2.0)),
            },
            Node::Binary {
                left: Box::new(Node::Name("c".into())),
                op: Op::Gt,
                right: Box::new(Node::Scalar(0.0)),
            },
        ];
        for n in &nodes {
            assert!(!format!("{n:?}").is_empty());
        }
    }
}
