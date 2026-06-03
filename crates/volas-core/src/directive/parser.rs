//! A whitespace-tolerant recursive-descent parser for directive strings.

use super::types::{Command, Node, Op};
use crate::error::{Result, VolasError};

/// Parse a directive string into an AST [`Node`].
pub fn parse(input: &str) -> Result<Node> {
    let mut p = Parser::new(input);
    p.skip_ws();
    if p.eof() {
        return Err(VolasError::Value("empty directive".into()));
    }
    let node = p.parse_expr()?;
    p.skip_ws();
    if !p.eof() {
        return Err(VolasError::Value(format!(
            "unexpected trailing input at position {}",
            p.i
        )));
    }
    Ok(node)
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

    fn parse_expr(&mut self) -> Result<Node> {
        let left = self.parse_term()?;
        self.skip_ws();
        if let Some(op) = self.try_op() {
            self.skip_ws();
            let right = self.parse_term()?;
            return Ok(Node::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Node> {
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
            other => Err(VolasError::Value(format!(
                "unexpected token {:?} at position {}",
                other.map(|c| c as char),
                self.i
            ))),
        }
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.peek() == Some(c) {
            self.bump();
            Ok(())
        } else {
            Err(VolasError::Value(format!(
                "expected '{}' at position {}",
                c as char, self.i
            )))
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
        let text = String::from_utf8_lossy(&self.s[start..self.i]);
        text.parse::<f64>()
            .map(Node::Scalar)
            .map_err(|_| VolasError::Value(format!("invalid number '{text}'")))
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
                // empty slot -> use the command's default series for this position
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

    fn try_op(&mut self) -> Option<Op> {
        let (a, b) = (self.peek(), self.peek2());
        let (op, len) = match (a, b) {
            (Some(b'<'), Some(b'=')) => (Op::Le, 2),
            (Some(b'>'), Some(b'=')) => (Op::Ge, 2),
            (Some(b'='), Some(b'=')) => (Op::Eq, 2),
            (Some(b'/'), Some(b'/')) => (Op::CrossUp, 2),
            (Some(b'\\'), Some(b'\\')) => (Op::CrossDown, 2),
            (Some(b'>'), Some(b'<')) => (Op::Cross, 2),
            (Some(b'\\'), _) => (Op::CrossDown, 1),
            (Some(b'<'), _) => (Op::Lt, 1),
            (Some(b'>'), _) => (Op::Gt, 1),
            _ => return None,
        };
        self.i += len;
        Some(op)
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
        let n = parse("ma:5").unwrap();
        match n {
            Node::Command(c) => {
                assert_eq!(c.name, "ma");
                assert_eq!(c.args, vec![Some("5".to_string())]);
                assert!(c.series.is_empty());
            }
            _ => panic!("expected command"),
        }
    }

    #[test]
    fn parse_sub_and_series() {
        let n = parse("macd.signal:12,26,9@close").unwrap();
        match n {
            Node::Command(c) => {
                assert_eq!(c.name, "macd");
                assert_eq!(c.sub.as_deref(), Some("signal"));
                assert_eq!(c.args.len(), 3);
                assert_eq!(c.series, vec![Node::Name("close".into())]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_operator_and_scalar() {
        let n = parse("kdj.j < 0").unwrap();
        match n {
            Node::Binary { op, right, .. } => {
                assert_eq!(op, Op::Lt);
                assert_eq!(*right, Node::Scalar(0.0));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_nested_and_cross() {
        assert!(matches!(parse("macd // macd.signal").unwrap(), Node::Binary { op: Op::CrossUp, .. }));
        let n = parse("increase:3@(ma:20@close)").unwrap();
        if let Node::Command(c) = n {
            assert_eq!(c.name, "increase");
            assert!(matches!(c.series[0], Node::Command(_)));
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_empty_arg_slots() {
        let n = parse("macd.signal:,,10").unwrap();
        if let Node::Command(c) = n {
            assert_eq!(c.args, vec![None, None, Some("10".into())]);
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_multiline() {
        let n = parse("repeat\n  : 5\n  @ (\n    close > boll.upper\n  )").unwrap();
        if let Node::Command(c) = n {
            assert_eq!(c.name, "repeat");
            assert_eq!(c.args, vec![Some("5".into())]);
            assert!(matches!(c.series[0], Node::Binary { op: Op::Gt, .. }));
        } else {
            panic!();
        }
    }
}
