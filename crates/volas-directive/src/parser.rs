//! A whitespace-tolerant precedence-climbing parser for directive strings.
//!
//! Precedence, low to high: logical (`& | ^`) < comparison / cross
//! (`< <= == != >= > // \\ ><`) < additive (`+ -`) < multiplicative (`* /`) <
//! unary (`~ -`) < primary (command, scalar, parenthesised expression).

use std::cell::RefCell;
use std::rc::Rc;

use super::bind::bind_node;
use super::types::{Ast, Command, Cst, Node, Op, UnaryOp};
use volas_core::fxhash::FxHashMap;
use volas_core::{Result, VolasError};

/// The deepest a directive may nest. Every parenthesised group, `@(...)` series
/// group, unary operator and binary operator adds a level to the parse tree, and the
/// bound holds both for the tree's height and for the parser's own recursion. A
/// realistic directive nests fewer than ten levels; the bound only stops input that
/// would otherwise overflow the stack of the parser or of a later walk over the tree
/// (bind, stringify, lookback, exec, drop), which recurse once per level. Measured in
/// a release build, the costliest shape at this bound (nested `@(...)` series groups,
/// executed) needs under 0.6 MiB of stack, so it fits a 2 MiB thread.
pub const MAX_NESTING: usize = 128;

/// The most nodes (names, numbers, commands, operators, series slots) one directive
/// may hold. A realistic directive holds a few dozen; the bound caps the work and
/// memory any walk over the tree can spend on a single directive.
pub const MAX_NODES: usize = 4096;

/// The most directives one thread's parse memo holds.
pub const MEMO_MAX_ENTRIES: usize = 512;

/// The longest directive the parse memo keeps, in bytes. A realistic directive is
/// well under 200 bytes; a longer input is parsed on every call instead of pinning
/// its text and tree in memory.
pub const MEMO_MAX_INPUT_BYTES: usize = 4096;

/// The most bytes one thread's parse memo holds, estimated per entry as
/// [`memo_cost`]. Realistic directives fill [`MEMO_MAX_ENTRIES`] first.
pub const MEMO_MAX_BYTES: usize = 1 << 20;

/// Estimated resident bytes of one memo entry: the key, the tree's strings (no
/// longer than the input) and one tree node per parsed node.
fn memo_cost(input: &str, nodes: usize) -> usize {
    2 * input.len() + nodes * std::mem::size_of::<Ast>()
}

// Any single memoizable entry fits the byte budget on its own, so eviction always
// makes room.
const _: () =
    assert!(2 * MEMO_MAX_INPUT_BYTES + MAX_NODES * std::mem::size_of::<Ast>() <= MEMO_MAX_BYTES);

/// One thread's parse memo, bounded by [`MEMO_MAX_ENTRIES`] and [`MEMO_MAX_BYTES`].
/// When full it evicts uniformly random entries, one at a time, until the new entry
/// fits. Random replacement degrades smoothly when the working set outgrows the
/// memo; LRU, FIFO and clock all miss on every call once a cyclic access pattern
/// (a refresh walking every cached column, each bar) holds one entry too many.
#[derive(Default)]
struct Memo {
    entries: Vec<MemoEntry>,
    /// Slot of each key in `entries`.
    slots: FxHashMap<Rc<str>, usize>,
    bytes: usize,
    /// xorshift64 state picking the next victim; deterministic per thread.
    rng: u64,
}

struct MemoEntry {
    key: Rc<str>,
    ast: Rc<Ast>,
    cost: usize,
}

impl Memo {
    fn get(&self, input: &str) -> Option<Rc<Ast>> {
        self.slots
            .get(input)
            .map(|&i| Rc::clone(&self.entries[i].ast))
    }

    fn insert(&mut self, input: &str, ast: &Rc<Ast>, cost: usize) {
        while self.entries.len() >= MEMO_MAX_ENTRIES || self.bytes + cost > MEMO_MAX_BYTES {
            self.evict_one();
        }
        let key: Rc<str> = Rc::from(input);
        self.slots.insert(Rc::clone(&key), self.entries.len());
        self.entries.push(MemoEntry {
            key,
            ast: Rc::clone(ast),
            cost,
        });
        self.bytes += cost;
    }

    fn evict_one(&mut self) {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let victim = (self.rng % self.entries.len() as u64) as usize;
        let gone = self.entries.swap_remove(victim);
        self.slots.remove(&gone.key);
        self.bytes -= gone.cost;
        if let Some(moved) = self.entries.get(victim) {
            self.slots.insert(Rc::clone(&moved.key), victim);
        }
    }
}

thread_local! {
    /// Per-thread directive memo. `parse` is a pure function of its input, so caching
    /// the bound [`Ast`] is bit-identical — and `df.exec` re-parses on every call, so a
    /// repeated directive in a hot loop skips both tokenizing **and** binding/validation
    /// (the whole per-exec validation cost) and shares one `Rc<Ast>` with no deep clone.
    /// FxHash (not SipHash): the keys are internal, non-adversarial directive strings —
    /// the same trust level as `name_to_idx` — and the per-bar refresh hits this memo.
    static PARSE_CACHE: RefCell<Memo> = RefCell::new(Memo {
        rng: 0x9E37_79B9_7F4A_7C15,
        ..Memo::default()
    });
}

/// Parse a directive string into a bound, validated [`Ast`] (memoized per thread).
/// Binding happens here, once: an invalid argument / unknown command errors at parse
/// time, and the returned tree is ready to execute with no further validation. Only
/// successful parses of inputs up to [`MEMO_MAX_INPUT_BYTES`] are memoized.
pub fn parse(input: &str) -> Result<Rc<Ast>> {
    let memoize = input.len() <= MEMO_MAX_INPUT_BYTES;
    if memoize {
        if let Some(ast) = PARSE_CACHE.with(|c| c.borrow().get(input)) {
            return Ok(ast);
        }
    }
    let (cst, nodes) = parse_tree(input)?;
    let ast = Rc::new(bind_node(&cst)?);
    if memoize {
        let cost = memo_cost(input, nodes);
        PARSE_CACHE.with(|c| c.borrow_mut().insert(input, &ast, cost));
    }
    Ok(ast)
}

pub fn parse_cst(input: &str) -> Result<Cst> {
    parse_tree(input).map(|(cst, _)| cst)
}

/// Parse a directive into its tree and the number of nodes it holds.
fn parse_tree(input: &str) -> Result<(Cst, usize)> {
    let mut p = Parser::new(input);
    p.skip_ws();
    if p.eof() {
        return Err(p.err("empty directive"));
    }
    let (node, _) = p.parse_expr()?;
    p.skip_ws();
    if !p.eof() {
        return Err(p.err("expected end of directive"));
    }
    Ok((node, p.nodes))
}

/// A parsed sub-tree and its height (a leaf is 1).
type Sub = (Cst, usize);

fn is_cmp(op: Op) -> bool {
    matches!(
        op,
        Op::Lt
            | Op::Le
            | Op::Eq
            | Op::Ne
            | Op::Ge
            | Op::Gt
            | Op::CrossUp
            | Op::CrossDown
            | Op::Cross
    )
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    /// Current recursion depth of the precedence productions.
    depth: usize,
    /// Nodes built so far.
    nodes: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            s: s.as_bytes(),
            i: 0,
            depth: 0,
            nodes: 0,
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
        VolasError::Parse(format!("{} (line {line}, column {column})", message.into()))
    }

    // --- budget (see [`MAX_NESTING`] / [`MAX_NODES`]) ---

    fn too_deep(&self) -> VolasError {
        self.err(format!("directive nests deeper than {MAX_NESTING} levels"))
    }

    /// Run a recursive production one level deeper, refusing before the recursion
    /// when the directive already nests [`MAX_NESTING`] levels.
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.depth >= MAX_NESTING {
            return Err(self.too_deep());
        }
        self.depth += 1;
        let out = f(self);
        self.depth -= 1;
        out
    }

    /// Account one built node whose children are `height - 1` deep.
    fn node(&mut self, node: Cst, height: usize) -> Result<Sub> {
        if height > MAX_NESTING {
            return Err(self.too_deep());
        }
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(self.err(format!("directive holds more than {MAX_NODES} nodes")));
        }
        Ok((node, height))
    }

    fn bin(&mut self, (left, lh): Sub, op: Op, (right, rh): Sub) -> Result<Sub> {
        let node = Node::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        };
        self.node(node, 1 + lh.max(rh))
    }

    // --- precedence levels ---

    fn parse_expr(&mut self) -> Result<Sub> {
        self.nested(Self::parse_logical)
    }

    fn parse_logical(&mut self) -> Result<Sub> {
        let mut left = self.parse_cmp()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op @ (Op::And | Op::Or | Op::Xor), len)) => {
                    self.i += len;
                    self.skip_ws();
                    let right = self.parse_cmp()?;
                    left = self.bin(left, op, right)?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Sub> {
        let mut left = self.parse_add()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op, len)) if is_cmp(op) => {
                    self.i += len;
                    self.skip_ws();
                    let right = self.parse_add()?;
                    left = self.bin(left, op, right)?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Sub> {
        let mut left = self.parse_mul()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op @ (Op::Add | Op::Sub), len)) => {
                    self.i += len;
                    self.skip_ws();
                    let right = self.parse_mul()?;
                    left = self.bin(left, op, right)?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Sub> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            match self.peek_binop() {
                Some((op @ (Op::Mul | Op::Div), len)) => {
                    self.i += len;
                    self.skip_ws();
                    let right = self.parse_unary()?;
                    left = self.bin(left, op, right)?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Sub> {
        self.skip_ws();
        let op = match self.peek() {
            Some(b'~') => UnaryOp::Not,
            Some(b'-') if !is_number_start(b'-', self.peek2()) => UnaryOp::Neg,
            _ => return self.parse_primary(),
        };
        self.bump();
        let (operand, height) = self.nested(Self::parse_unary)?;
        let node = Node::Unary {
            op,
            operand: Box::new(operand),
        };
        self.node(node, height + 1)
    }

    fn parse_primary(&mut self) -> Result<Sub> {
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

    fn parse_number(&mut self) -> Result<Sub> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.') {
            self.bump();
        }
        let text = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
        // Digits and dots never parse to NaN, but too many digits round to ±inf, which
        // has no literal form: it would render as `inf`, a name, and rebind on re-parse.
        match text.parse::<f64>() {
            Ok(v) if v.is_finite() => self.node(Node::Scalar(v), 1),
            Ok(_) => Err(self.err("number literal is out of range of a 64-bit float")),
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

    fn parse_command(&mut self) -> Result<Sub> {
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
            // `cmd:` — a colon with nothing after it names no arguments, not one empty one.
            if args == [None] {
                args.clear();
            }
        }
        self.skip_ws();
        let mut series = Vec::new();
        let mut height = 0;
        let mut has_series = false;
        if self.peek() == Some(b'@') {
            self.bump();
            has_series = true;
            (series, height) = self.read_series()?;
        }
        if sub.is_none() && !has_args && !has_series {
            // A bare identifier may be a column (case-sensitive) or a no-arg command;
            // leave its case to the executor, which resolves columns first.
            self.node(Node::Name(name), 1)
        } else {
            // A sub-command / args / series makes this unambiguously a command, so the
            // command name is case-insensitive (P6): lower-case it into the AST, which
            // also flows through to stringify.
            let command = Node::Command(Command {
                name: name.to_ascii_lowercase(),
                sub,
                args,
                series,
            });
            self.node(command, height + 1)
        }
    }

    /// Whether the `+` / `-` at the cursor is a sign inside the argument that began at
    /// `start`: it opens the argument, or follows the `e` / `E` of a numeric exponent.
    fn is_sign_in_number(&self, start: usize) -> bool {
        let i = self.i;
        matches!(self.s[i], b'+' | b'-')
            && (i == start
                || (i >= start + 2
                    && matches!(self.s[i - 1], b'e' | b'E')
                    && self.s[i - 2].is_ascii_digit()))
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
                // A binary operator ends the argument, so a canonical name — which drops
                // the spaces around operators (`rsi:14>50`) — re-parses as written. A
                // sign is kept where it belongs to the number: opening the argument
                // (`-1`) or following an exponent (`1e-5`).
                if is_operator_char(c) && !self.is_sign_in_number(start) {
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

    /// The `@` series operands and the height of the tallest one.
    fn read_series(&mut self) -> Result<(Vec<Cst>, usize)> {
        let mut series = Vec::new();
        let mut height = 0;
        loop {
            self.skip_ws();
            let (node, h) = if self.peek() == Some(b'(') {
                self.bump();
                let e = self.parse_expr()?;
                self.skip_ws();
                self.expect(b')')?;
                e
            } else if matches!(self.peek(), Some(c) if is_ident_start(c)) {
                let name = self.read_ident();
                self.node(Node::Name(name), 1)?
            } else {
                self.node(Node::Name(String::new()), 1)?
            };
            height = height.max(h);
            series.push(node);
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.bump();
            } else {
                break;
            }
        }
        Ok((series, height))
    }
}

/// A character that starts a binary operator (`+ - * / \ < > = ! & | ^`) — every first
/// byte [`Parser::peek_binop`] recognizes.
fn is_operator_char(c: u8) -> bool {
    matches!(
        c,
        b'+' | b'-' | b'*' | b'/' | b'\\' | b'<' | b'>' | b'=' | b'!' | b'&' | b'|' | b'^'
    )
}

pub(crate) fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
pub(crate) fn is_ident_char(c: u8) -> bool {
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

    // The structure tests exercise the parser directly (`parse_cst` -> `Cst`,
    // arguments as raw tokens); binding/validation is `bind`'s concern, tested there.

    #[test]
    fn parse_simple_command() {
        match parse_cst("ma:5").unwrap() {
            Node::Command(c) => {
                assert_eq!(c.name, "ma");
                assert_eq!(c.args, vec![Some("5".to_string())]);
            }
            _ => panic!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn parse_operator_and_scalar() {
        match parse_cst("kdj.j < 0").unwrap() {
            Node::Binary { op, right, .. } => {
                assert_eq!(op, Op::Lt);
                assert_eq!(*right, Node::Scalar(0.0));
            }
            _ => panic!(), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn parse_cross_and_nested() {
        assert!(matches!(
            parse_cst("macd // macd.signal").unwrap(),
            Node::Binary {
                op: Op::CrossUp,
                ..
            }
        ));
        if let Node::Command(c) = parse_cst("increase:3@(ma:20@close)").unwrap() {
            assert!(matches!(c.series[0], Node::Command(_)));
        } else {
            panic!(); // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn parse_precedence() {
        // (kdj.j + 1) != kdj.j
        match parse_cst("kdj.j + 1 != kdj.j").unwrap() {
            Node::Binary {
                op: Op::Ne, left, ..
            } => {
                assert!(matches!(*left, Node::Binary { op: Op::Add, .. }));
            }
            _ => panic!(), // LCOV_EXCL_LINE
        }
        // (a > 1) | (a <= 1)
        assert!(matches!(
            parse_cst("(kdj.j > 1) | (kdj.j <= 1)").unwrap(),
            Node::Binary { op: Op::Or, .. }
        ));
        // unary not / neg
        assert!(matches!(
            parse_cst("~(kdj.j <= 0)").unwrap(),
            Node::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
        assert!(matches!(
            parse_cst("kdj.j * 2").unwrap(),
            Node::Binary { op: Op::Mul, .. }
        ));
    }

    #[test]
    fn parse_empty_arg_slots() {
        if let Node::Command(c) = parse_cst("macd.signal:,,10").unwrap() {
            assert_eq!(c.args, vec![None, None, Some("10".into())]);
        } else {
            panic!(); // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn malformed_number_is_a_syntax_error() {
        // parse_number scans digits/'.' greedily, so "1.2.3" fails f64 parsing
        assert!(parse_cst("1.2.3").is_err());
    }

    #[test]
    fn a_literal_overflowing_f64_is_a_syntax_error() {
        // It would round to `inf`, which renders as `inf` and re-parses as a Name.
        let big = format!("1{}", "0".repeat(400));
        for d in [
            format!("close < {big}"),
            format!("-{big}"),
            format!("{big}.5"),
        ] {
            match parse_cst(&d) {
                Err(VolasError::Parse(m)) => assert!(m.contains("out of range"), "{m}"),
                other => panic!("{d}: {other:?}"), // LCOV_EXCL_LINE
            }
        }
        // The largest finite literal is still a number.
        let max = format!("{}", f64::MAX);
        assert_eq!(parse_cst(&max).unwrap(), Node::Scalar(f64::MAX));
    }

    /// A ~40 KB hostile directive must be a syntax error, never a stack overflow in
    /// the parser or in a downstream walk (bind / stringify / exec / lookback / drop).
    #[test]
    fn nesting_past_the_budget_is_a_syntax_error() {
        let paren = format!("{}close{}", "(".repeat(20_000), ")".repeat(20_000));
        let not = format!("{}close", "~".repeat(30_000));
        let neg = format!("{}close", "-".repeat(30_000));
        let chain = vec!["close"; 30_000].join("+");
        let series = format!("{}close{}", "ema:2@(".repeat(5_000), ")".repeat(5_000));
        for d in [paren, not, neg, chain, series] {
            match parse_cst(&d) {
                Err(VolasError::Parse(m)) => assert!(m.contains("nests deeper than"), "{m}"),
                other => panic!("{:?}", other.map(|_| ())), // LCOV_EXCL_LINE
            }
        }
    }

    #[test]
    fn parse_memo_hit_and_eviction() {
        // Each test runs on its own thread, so the thread-local memo starts empty. A
        // repeated parse returns the cached bound `Rc<Ast>` (the cache-hit branch), and
        // parsing past the 512-entry bound evicts entries (the eviction branch) — both
        // invisible to the rest of the suite.
        assert_eq!(parse("ma:5").unwrap(), parse("ma:5").unwrap()); // 2nd hits the cache
        for i in 0..600 {
            let _ = parse(&format!("ma:{}", i + 1)).unwrap(); // crosses 512 -> evicts
        }
        assert_eq!(parse("ma:5").unwrap(), parse("ma:5").unwrap()); // correct post-eviction
    }

    // The memo tests below observe residency without touching the memo: a bound tree
    // the memo still holds has a strong count of 2 (the memo's and the test's).
    const ENTRIES: usize = MEMO_MAX_ENTRIES;
    const INPUT: usize = MEMO_MAX_INPUT_BYTES;
    const BYTES: usize = MEMO_MAX_BYTES;

    fn resident(held: &[Rc<Ast>]) -> usize {
        held.iter().filter(|a| Rc::strong_count(a) == 2).count()
    }

    #[test]
    fn memo_skips_an_input_above_the_size_threshold() {
        let big = format!("ma:2{}", " ".repeat(INPUT));
        assert!(!Rc::ptr_eq(&parse(&big).unwrap(), &parse(&big).unwrap()));
        let at = format!("ma:2{}", " ".repeat(INPUT - 4));
        assert!(Rc::ptr_eq(&parse(&at).unwrap(), &parse(&at).unwrap()));
    }

    #[test]
    fn memo_evicts_one_entry_at_a_time() {
        let held: Vec<_> = (0..ENTRIES)
            .map(|i| parse(&format!("ma:{}", i + 1)).unwrap())
            .collect();
        assert_eq!(resident(&held), ENTRIES);
        parse("ma:100000").unwrap(); // one past the entry bound
        assert_eq!(resident(&held), ENTRIES - 1);
    }

    #[test]
    fn memo_is_bounded_by_bytes() {
        // Inputs just under the size threshold: the entry bound alone keeps them all.
        let held: Vec<_> = (0..ENTRIES)
            .map(|i| parse(&format!("ma:{}{}", i + 1, " ".repeat(INPUT - 10))).unwrap())
            .collect();
        let kept = resident(&held);
        assert!(kept > 0);
        assert!(
            kept * INPUT <= BYTES,
            "{kept} inputs of {INPUT} bytes resident"
        );
    }

    #[test]
    fn memo_bookkeeping_stays_consistent_under_churn() {
        for i in 0..3 * ENTRIES {
            let pad = " ".repeat(i * 7 % INPUT);
            parse(&format!("ma:{}{pad}", i + 1)).unwrap();
        }
        PARSE_CACHE.with(|c| {
            let m = c.borrow();
            assert!(m.entries.len() <= ENTRIES && m.bytes <= BYTES);
            assert_eq!(m.bytes, m.entries.iter().map(|e| e.cost).sum::<usize>());
            assert_eq!(m.slots.len(), m.entries.len());
            for (i, e) in m.entries.iter().enumerate() {
                assert_eq!(m.slots[&e.key], i);
            }
        });
    }
}
