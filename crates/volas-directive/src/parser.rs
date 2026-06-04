//! A whitespace-tolerant precedence-climbing parser for directive strings.
//!
//! Precedence, low to high: logical (`& | ^`) < comparison / cross
//! (`< <= == != >= > // \\ ><`) < additive (`+ -`) < multiplicative (`* /`) <
//! unary (`~ -`) < primary (command, scalar, parenthesised expression).

use super::types::{Command, Node, Op, UnaryOp};
use volas_core::{Result, VolasError};

/// Parse a directive string into an AST [`Node`].
pub fn parse(input: &str) -> Result<Node> {
    let mut p = Parser::new(input);
    p.skip_ws();
    if p.eof() {
        return Err(p.err("empty directive"));
    }
    let node = p.parse_expr()?;
    p.skip_ws();
    if !p.eof() {
        return Err(p.err("expected end of directive"));
    }
    Ok(node)
}

fn bin(left: Node, op: Op, right: Node) -> Node {
    Node::Binary {
        left: Box::new(left),
        op,
        right: Box::new(right),
    }
}

fn is_cmp(op: Op) -> bool {
    matches!(
        op,
        Op::Lt | Op::Le | Op::Eq | Op::Ne | Op::Ge | Op::Gt | Op::CrossUp | Op::CrossDown | Op::Cross
    )
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            s: s.as_bytes(),
            i: 0,
        }
    }

    fn eof(&self) -> bool {
        self.i >= self.s.len()
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn peek2(&self) -> Option<u8> {
        self.s.get(self.i + 1).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    /// A syntax error annotated with the current 1-based line / column.
    fn err(&self, message: impl Into<String>) -> VolasError {
        let consumed = &self.s[..self.i.min(self.s.len())];
        let line = 1 + consumed.iter().filter(|&&b| b == b'\n').count();
        let line_start = consumed
            .iter()
            .rposition(|&b| b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let column = self.i - line_start + 1;
        VolasError::Value(format!("{} (line {line}, column {column})", message.into()))
    }

    // --- precedence levels ---

    fn parse_expr(&mut self) -> Result<Node> {
        self.parse_logical()
    }

    fn parse_logical(&mut self) -> Result<Node> {
        let mut left = self.parse_cmp()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op @ (Op::And | Op::Or | Op::Xor), len)) => {
                    self.i += len;
                    self.skip_ws();
                    left = bin(left, op, self.parse_cmp()?);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Node> {
        let mut left = self.parse_add()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op, len)) if is_cmp(op) => {
                    self.i += len;
                    self.skip_ws();
                    left = bin(left, op, self.parse_add()?);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Node> {
        let mut left = self.parse_mul()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op @ (Op::Add | Op::Sub), len)) => {
                    self.i += len;
                    self.skip_ws();
                    left = bin(left, op, self.parse_mul()?);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Node> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op @ (Op::Mul | Op::Div), len)) => {
                    self.i += len;
                    self.skip_ws();
                    left = bin(left, op, self.parse_unary()?);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Node> {
        self.skip_ws();
        match self.peek() {
            Some(b'~') => {
                self.bump();
                let operand = self.parse_unary()?;
                Ok(Node::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                })
            }
            Some(b'-') if !is_number_start(b'-', self.peek2()) => {
                self.bump();
                let operand = self.parse_unary()?;
                Ok(Node::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Node> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.bump();
                let e = self.parse_expr()?;
                self.skip_ws();
                self.expect(b')')?;
                Ok(e)
            }
            Some(c) if is_number_start(c, self.peek2()) => self.parse_number(),
            Some(c) if is_ident_start(c) => self.parse_command(),
            Some(c) => Err(self.err(format!("unexpected token '{}'", c as char))),
            None => Err(self.err("unexpected end of directive")),
        }
    }

    /// Peek the next binary operator without consuming it.
    fn peek_binop(&self) -> Option<(Op, usize)> {
        let (a, b) = (self.peek(), self.peek2());
        let pair = match (a, b) {
            (Some(b'<'), Some(b'=')) => (Op::Le, 2),
            (Some(b'>'), Some(b'=')) => (Op::Ge, 2),
            (Some(b'='), Some(b'=')) => (Op::Eq, 2),
            (Some(b'!'), Some(b'=')) => (Op::Ne, 2),
            (Some(b'/'), Some(b'/')) => (Op::CrossUp, 2),
            (Some(b'\\'), Some(b'\\')) => (Op::CrossDown, 2),
            (Some(b'>'), Some(b'<')) => (Op::Cross, 2),
            (Some(b'\\'), _) => (Op::CrossDown, 1),
            (Some(b'<'), _) => (Op::Lt, 1),
            (Some(b'>'), _) => (Op::Gt, 1),
            (Some(b'+'), _) => (Op::Add, 1),
            (Some(b'-'), _) => (Op::Sub, 1),
            (Some(b'*'), _) => (Op::Mul, 1),
            (Some(b'/'), _) => (Op::Div, 1),
            (Some(b'&'), _) => (Op::And, 1),
            (Some(b'|'), _) => (Op::Or, 1),
            (Some(b'^'), _) => (Op::Xor, 1),
            _ => return None,
        };
        Some(pair)
    }

    // --- leaves ---

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.peek() == Some(c) {
            self.bump();
            Ok(())
        } else {
            Err(self.err(format!("expected '{}'", c as char)))
        }
    }

    fn parse_number(&mut self) -> Result<Node> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.') {
            self.bump();
        }
        let text = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
        match text.parse::<f64>() {
            Ok(v) => Ok(Node::Scalar(v)),
            Err(_) => Err(self.err(format!("invalid number '{text}'"))),
        }
    }

    fn read_ident(&mut self) -> String {
        let start = self.i;
        while matches!(self.peek(), Some(c) if is_ident_char(c)) {
            self.bump();
        }
        String::from_utf8_lossy(&self.s[start..self.i]).into_owned()
    }

    fn parse_command(&mut self) -> Result<Node> {
        let name = self.read_ident();
        let mut sub = None;
        self.skip_ws();
        if self.peek() == Some(b'.') {
            self.bump();
            self.skip_ws();
            sub = Some(self.read_ident());
        }
        self.skip_ws();
        let mut args = Vec::new();
        let mut has_args = false;
        if self.peek() == Some(b':') {
            self.bump();
            has_args = true;
            args = self.read_args();
        }
        self.skip_ws();
        let mut series = Vec::new();
        let mut has_series = false;
        if self.peek() == Some(b'@') {
            self.bump();
            has_series = true;
            series = self.read_series()?;
        }
        if sub.is_none() && !has_args && !has_series {
            Ok(Node::Name(name))
        } else {
            Ok(Node::Command(Command {
                name,
                sub,
                args,
                series,
            }))
        }
    }

    fn read_args(&mut self) -> Vec<Option<String>> {
        let mut args = Vec::new();
        loop {
            self.skip_ws();
            let start = self.i;
            while let Some(c) = self.peek() {
                if matches!(c, b',' | b'@' | b')' | b' ' | b'\t' | b'\n' | b'\r') {
                    break;
                }
                self.bump();
            }
            let text = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
            args.push(if text.is_empty() { None } else { Some(text) });
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.bump();
            } else {
                break;
            }
        }
        args
    }

    fn read_series(&mut self) -> Result<Vec<Node>> {
        let mut series = Vec::new();
        loop {
            self.skip_ws();
            let node = if self.peek() == Some(b'(') {
                self.bump();
                let e = self.parse_expr()?;
                self.skip_ws();
                self.expect(b')')?;
                e
            } else if matches!(self.peek(), Some(c) if is_ident_start(c)) {
                Node::Name(self.read_ident())
            } else {
                Node::Name(String::new())
            };
            series.push(node);
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.bump();
            } else {
                break;
            }
        }
        Ok(series)
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}
fn is_number_start(c: u8, next: Option<u8>) -> bool {
    c.is_ascii_digit()
        || (c == b'-' && matches!(next, Some(n) if n.is_ascii_digit() || n == b'.'))
        || (c == b'.' && matches!(next, Some(n) if n.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_command() {
        match parse("ma:5").unwrap() {
            Node::Command(c) => {
                assert_eq!(c.name, "ma");
                assert_eq!(c.args, vec![Some("5".to_string())]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_operator_and_scalar() {
        match parse("kdj.j < 0").unwrap() {
            Node::Binary { op, right, .. } => {
                assert_eq!(op, Op::Lt);
                assert_eq!(*right, Node::Scalar(0.0));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_cross_and_nested() {
        assert!(matches!(
            parse("macd // macd.signal").unwrap(),
            Node::Binary { op: Op::CrossUp, .. }
        ));
        if let Node::Command(c) = parse("increase:3@(ma:20@close)").unwrap() {
            assert!(matches!(c.series[0], Node::Command(_)));
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_precedence() {
        // (kdj.j + 1) != kdj.j
        match parse("kdj.j + 1 != kdj.j").unwrap() {
            Node::Binary { op: Op::Ne, left, .. } => {
                assert!(matches!(*left, Node::Binary { op: Op::Add, .. }));
            }
            _ => panic!(),
        }
        // (a > 1) | (a <= 1)
        assert!(matches!(
            parse("(kdj.j > 1) | (kdj.j <= 1)").unwrap(),
            Node::Binary { op: Op::Or, .. }
        ));
        // unary not / neg
        assert!(matches!(
            parse("~(kdj.j <= 0)").unwrap(),
            Node::Unary { op: UnaryOp::Not, .. }
        ));
        assert!(matches!(parse("kdj.j * 2").unwrap(), Node::Binary { op: Op::Mul, .. }));
    }

    #[test]
    fn parse_empty_arg_slots() {
        if let Node::Command(c) = parse("macd.signal:,,10").unwrap() {
            assert_eq!(c.args, vec![None, None, Some("10".into())]);
        } else {
            panic!();
        }
    }
}
