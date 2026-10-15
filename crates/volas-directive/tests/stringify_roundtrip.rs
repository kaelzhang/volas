//! A canonical name re-parses to the directive it names.
//!
//! The frame caches a computed column under its bound canonical name, so for every directive `d` the grammar accepts, the canonical
//! name must parse, be its own canonical name (a fixed point), and compute the same
//! column as `d`. (`parse(stringify(parse(d)))` equals `parse(d)` as a tree exactly
//! when `d` spells out nothing that canonicalizing drops — a default argument or
//! default series, a no-argument command, a redundant sign or bracket; those name the
//! same computation, which the column comparison checks.)
//!
//! The directives are generated from the grammar: commands with sub-lines, explicit
//! and empty argument slots, `@` series operands that are columns, scalars, nested
//! commands (with their own arguments and `@` operands) and operator expressions,
//! every binary operator, unary `-` / `~`, and redundant parentheses.

use volas_core::{Column, DataFrame};
use volas_directive::{exec::execute, parse, stringify};

fn ohlcv(n: usize) -> DataFrame {
    let close: Vec<f64> = (0..n)
        .map(|i| 50.0 + i as f64 * 0.3 + (i as f64 * 0.21).sin() * 2.0)
        .collect();
    let cols = [
        ("open", close.iter().map(|v| v - 0.3).collect::<Vec<_>>()),
        ("high", close.iter().map(|v| v + 1.2).collect()),
        ("low", close.iter().map(|v| v - 1.4).collect()),
        ("volume", (0..n).map(|i| 1000.0 + (i % 7) as f64 * 10.0).collect()),
        ("close", close),
    ];
    DataFrame::new(
        cols.iter().map(|(k, _)| k.to_string()).collect(),
        cols.into_iter().map(|(_, v)| Column::f64(v)).collect(),
        None,
    )
    .unwrap()
}

/// xorshift64*: a deterministic generator, so a failure names a reproducible seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

const COLUMNS: &[&str] = &["open", "high", "low", "close", "volume"];
const SCALARS: &[&str] = &["3", "2.5", "0.25", "-1.5", "-4", "100", "0.000001", "100000000000000000000"];
const BINARY: &[&str] = &[
    "+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "&", "|", "^", "//", "\\\\", "><",
];

/// `(head, args, series slots)`: a command form, the argument lists it takes (an empty
/// string spells none; an empty slot is a default), and how many `@` operands it reads.
const COMMANDS: &[(&str, &[&str], usize)] = &[
    ("ma", &["5", "3"], 1),
    ("ema", &["5", "3", ""], 1),
    ("rsi", &["4", "14"], 1),
    ("macd", &["", "12,26", "5,10", "5,10,3"], 1),
    ("macd.signal", &["", "5,10,3", ",,4"], 1),
    ("boll.upper", &["", "20", "30", "20,3"], 1),
    ("kdj.j", &["", ",4", "9,3,3"], 3),
    ("atr", &["5", "14"], 3),
    ("obv", &[""], 2),
    ("roc", &["3"], 1),
    ("stddev", &["5", "5,2"], 1),
    ("correl", &["5"], 2),
];

fn operand(rng: &mut Rng, depth: usize) -> String {
    match rng.below(if depth == 0 { 3 } else { 6 }) {
        0 => String::new(), // the default series
        1 | 2 => rng.pick(COLUMNS).to_string(),
        3 => format!("({})", rng.pick(SCALARS)),
        4 => format!("({})", command(rng, depth - 1)),
        _ => format!("({})", expr(rng, depth - 1)),
    }
}

fn command(rng: &mut Rng, depth: usize) -> String {
    let (head, args, slots) = COMMANDS[rng.below(COMMANDS.len())];
    let mut out = head.to_string();
    let a = rng.pick(args);
    if !a.is_empty() {
        out.push(':');
        out.push_str(a);
    }
    if rng.below(2) == 0 {
        let n = 1 + rng.below(slots);
        let ops: Vec<String> = (0..n).map(|_| operand(rng, depth)).collect();
        out.push('@');
        out.push_str(&ops.join(","));
    }
    out
}

fn expr(rng: &mut Rng, depth: usize) -> String {
    let leaf = |rng: &mut Rng| match rng.below(3) {
        0 => rng.pick(COLUMNS).to_string(),
        1 => rng.pick(SCALARS).to_string(),
        _ => command(rng, depth.saturating_sub(1)),
    };
    if depth == 0 {
        return leaf(rng);
    }
    match rng.below(6) {
        0 => leaf(rng),
        1 => format!("-{}", expr(rng, depth - 1)),
        2 => format!("~({})", expr(rng, depth - 1)),
        3 => format!("({})", expr(rng, depth - 1)),
        _ => {
            let sp = if rng.below(2) == 0 { " " } else { "" };
            format!("{}{sp}{}{sp}{}", expr(rng, depth - 1), rng.pick(BINARY), expr(rng, depth - 1))
        }
    }
}

fn same_column(a: &Column, b: &Column) -> bool {
    let (a, b) = (a.to_f64_vec(), b.to_f64_vec());
    a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn a_canonical_name_reparses_to_the_same_directive() {
    let df = ohlcv(80);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0;
    for case in 0..4000 {
        let d = expr(&mut rng, 4);
        let Ok(node) = parse(&d) else { continue }; // the generator may build a bind error
        let name = stringify(&*node);
        let again = parse(&name).unwrap_or_else(|e| panic!("case {case}: {d:?} -> {name:?}: {e}"));
        assert_eq!(stringify(&*again), name, "case {case}: {d:?} is not a fixed point");
        match (execute(&df, &node), execute(&df, &again)) {
            (Ok(a), Ok(b)) => assert!(same_column(&a, &b), "case {case}: {d:?} vs {name:?}"),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("case {case}: {d:?} -> {name:?}: {:?} vs {:?}", a.is_ok(), b.is_ok()),
        }
        checked += 1;
    }
    assert!(checked > 2000, "only {checked} generated directives parsed");
}

/// Spelled with nothing canonicalizing drops, a directive's canonical name re-parses
/// to the very same tree.
#[test]
fn a_directive_without_defaults_reparses_to_the_same_tree() {
    for d in [
        "ema:5@(rsi:5)",
        "ema:5@(rsi:5@(close+open))",
        "ema:5@(macd.signal:5,10,3@(close*2))",
        "atr:5@open,(low-1),(ema:3@open)",
        "correl:5@open,(3)",
        "ma:5@(boll.upper:30)>close",
        "~(ma:3@(close+1)>ma:5@(open))",
        "-(ma:3@(-close))",
        "rsi:14>50&(macd:5,10@(high/low)<0)",
        "boll.upper:30,100000000000000000000>close",
    ] {
        let node = parse(d).unwrap();
        let name = stringify(&*node);
        assert_eq!(*parse(&name).unwrap(), *node, "{d} -> {name}");
    }
}

/// A bound data reference can itself spell an expression. Its syntax still determines
/// grouping in the shared renderer, although execution reads its physical column.
#[test]
fn bound_data_syntax_roundtrips_every_operator_unary_and_series_nesting() {
    use volas_directive::{bind_frame_resolved, canonical_data_name};
    let mut df = ohlcv(80);
    let labels = ["open+close", "open*close", "open<close", "-(open+close)",
        "ma:3@(open+close)", "ACCBANDS", "odd label"];
    for label in labels {
        df.set_column(label, Column::f64((0..80).map(|i| 1.0 + i as f64).collect())).unwrap();
    }
    let resolve = |name: &str| df.names().iter().find(|n|
        canonical_data_name(n).is_ok_and(|id| id == name)).cloned();
    for label in &labels[..labels.len() - 1] {
        for op in BINARY {
            let expr = format!("({label}){op}close");
            for text in [expr.clone(), format!("close{op}({label})"),
                format!("-({expr})"), format!("~({expr})"),
                format!("ma:3@({expr})"), format!("ma:3@(-({expr}))"),
                format!("ma:3@(ma:2@({expr}))"), format!("-({label})"),
                format!("~({label})")] {
                let node = bind_frame_resolved(&parse(&text).unwrap(), &resolve, &text).unwrap();
                let name = stringify(&node);
                assert_eq!(name, stringify(&*parse(&text).unwrap()), "{text}");
                let again = bind_frame_resolved(&parse(&name).unwrap(), &resolve, &name).unwrap();
                assert_eq!(stringify(&again), name, "{text}");
                match (execute(&df, &node), execute(&df, &again)) {
                    (Ok(a), Ok(b)) => assert!(same_column(&a, &b), "{text} -> {name}"),
                    (Err(_), Err(_)) => {},
                    (a, b) => panic!("{text}: {:?} vs {:?}", a.is_ok(), b.is_ok()),
                }
            }
        }
    }
    let opaque = bind_frame_resolved(&parse("open").unwrap(), &|_| Some("odd label".into()), "").unwrap();
    assert_eq!(stringify(&opaque), "odd label");
    let opaque_operand = bind_frame_resolved(&parse("-open").unwrap(),
        &|name| (name == "open").then(|| "odd label".into()), "").unwrap();
    assert_eq!(stringify(&opaque_operand), "-odd label");
}
