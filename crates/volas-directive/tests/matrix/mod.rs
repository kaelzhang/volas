//! Inventory and data classes for the resume-vs-full matrix.
//!
//! The output lines are derived, never hand-listed: commands from `spec::COMMANDS`,
//! sub-command tokens from every `Some("..")` match in `spec.rs` (validated by
//! `command_spec` after `canon_sub`), candlestick patterns from `CANDLE_PATTERNS`, and
//! argument values from each slot's `ArgBound` / `ArgDefault`.

use std::collections::BTreeSet;

use volas_core::{Column, DataFrame};
use volas_directive::spec::{canon_sub, command_spec, Arg, ArgBound, ArgDefault, COMMANDS};

/// The spec source, read for its sub-command tokens.
const SPEC_RS: &str = include_str!("../../src/spec.rs");

/// Every token named inside a `Some(..)` in the spec (sub-commands and aliases).
fn sub_tokens() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = SPEC_RS;
    while let Some(at) = rest.find("Some(") {
        rest = &rest[at + 5..];
        let group = &rest[..rest.find(')').unwrap_or(0)];
        for (i, part) in group.split('"').enumerate() {
            let token = part.trim();
            if i % 2 == 1 && !token.is_empty() && token.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                out.insert(token.to_string());
            }
        }
    }
    out
}

/// One output line: its head (`ma`, `macd.signal`, `style.doji`) and argument slots.
pub struct Line {
    pub head: String,
    pub args: Vec<Arg>,
    /// Extra `@series` some lines need to execute (a required second series).
    pub series: &'static str,
}

/// Every distinct output line of the language (aliases collapsed to the canonical sub).
pub fn lines() -> Vec<Line> {
    let tokens = sub_tokens();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for &cmd in COMMANDS {
        let mut subs: Vec<Option<String>> = vec![None];
        subs.extend(tokens.iter().map(|t| Some(t.clone())));
        if cmd == "style" {
            subs.extend(volas_compute::indicators::CANDLE_PATTERNS.iter().map(|p| Some(p.to_string())));
        }
        for sub in subs {
            let canon = canon_sub(cmd, sub.as_deref()).map(|c| c.into_owned());
            let Some(spec) = command_spec(cmd, canon.as_deref()) else { continue };
            let head = match &canon {
                Some(s) => format!("{cmd}.{s}"),
                None => cmd.to_string(),
            };
            if !seen.insert(head.clone()) {
                continue;
            }
            let series = match cmd {
                "correl" | "beta" => "@close,open",
                "mavp" => "@close,volume",
                "repeat" => "@close",
                _ => "",
            };
            out.push(Line { head, args: spec.args.to_vec(), series });
        }
    }
    out
}

/// An in-domain token for one slot: the default when there is one (`""`), else a
/// value at `period` clamped into the bound.
fn token(arg: &Arg, period: usize, keep_default: bool) -> String {
    if keep_default && arg.default != ArgDefault::Required {
        return String::new();
    }
    match arg.bound {
        ArgBound::IntMin(min) => period.max(min).to_string(),
        ArgBound::IntRange(lo, hi) => {
            // A matype selector keeps SMA unless a default says otherwise.
            if hi <= 8 { lo.to_string() } else { period.clamp(lo, hi).to_string() }
        }
        ArgBound::OneOfI64(allowed) => allowed[allowed.len() - 1].to_string(),
        ArgBound::I64Min(min) => (period as i64).max(min).to_string(),
        ArgBound::AnyStr => "1d".to_string(),
        _ => match arg.default {
            ArgDefault::Float(f) => format!("{f:?}"),
            _ => "1.0".to_string(),
        },
    }
}

/// The directive variants of a line: all defaults (required slots at 5), and every
/// period slot short (3) so warm-up and runs longer than the lookback both occur.
pub fn variants(line: &Line) -> Vec<String> {
    let mut out = Vec::new();
    for (period, keep) in [(5, true), (3, false)] {
        let toks: Vec<String> = line.args.iter().map(|a| token(a, period, keep)).collect();
        let last = toks.iter().rposition(|t| !t.is_empty());
        let args = match last {
            Some(i) => format!(":{}", toks[..=i].join(",")),
            None => String::new(),
        };
        let text = format!("{}{args}{}", line.head, line.series);
        if !out.contains(&text) {
            out.push(text);
        }
    }
    out
}

/// A named data class: OHLCV columns over `n` rows.
pub struct Data {
    pub name: &'static str,
    pub frame: DataFrame,
}

fn frame(o: Vec<f64>, h: Vec<f64>, l: Vec<f64>, c: Vec<f64>, v: Vec<f64>) -> DataFrame {
    DataFrame::new(
        ["open", "high", "low", "close", "volume"].iter().map(|s| s.to_string()).collect(),
        vec![Column::f64(o), Column::f64(h), Column::f64(l), Column::f64(c), Column::f64(v)],
        None,
    )
    .expect("equal-length columns")
}

/// Open, high, low, close, volume.
type Ohlcv = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// A deterministic pseudo-random walk (xorshift; no RNG dependency).
fn walk(n: usize, seed: u64) -> Ohlcv {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 10_000) as f64 / 10_000.0 - 0.5
    };
    let mut close = Vec::with_capacity(n);
    let mut price = 100.0;
    for _ in 0..n {
        price += next() * 2.0;
        close.push(price);
    }
    let open: Vec<f64> = close.iter().map(|c| c + next()).collect();
    let high: Vec<f64> = close.iter().zip(&open).map(|(c, o)| c.max(*o) + next().abs() + 0.1).collect();
    let low: Vec<f64> = close.iter().zip(&open).map(|(c, o)| c.min(*o) - next().abs() - 0.1).collect();
    let volume: Vec<f64> = (0..n).map(|_| 500.0 + next() * 400.0).collect();
    (open, high, low, close, volume)
}

/// The data classes of §5.3 relevant to resume: clean, NaN in one input, leading
/// NaN, ±inf, constant / zero-range / zero-volume runs, extreme and tiny
/// magnitudes, negative prices.
pub fn classes(n: usize) -> Vec<Data> {
    let (o, h, l, c, v) = walk(n, 0x9e37_79b9);
    let mut out = vec![Data { name: "walk", frame: frame(o.clone(), h.clone(), l.clone(), c.clone(), v.clone()) }];
    let mut cc = c.clone();
    cc[n / 2] = f64::NAN;
    out.push(Data { name: "nan-close", frame: frame(o.clone(), h.clone(), l.clone(), cc, v.clone()) });
    let (mut hh, mut vv) = (h.clone(), v.clone());
    hh[n / 3] = f64::NAN;
    vv[2 * n / 3] = f64::NAN;
    out.push(Data { name: "nan-high-volume", frame: frame(o.clone(), hh, l.clone(), c.clone(), vv) });
    let lead = |x: &Vec<f64>| -> Vec<f64> { x.iter().enumerate().map(|(i, &y)| if i < 7 { f64::NAN } else { y }).collect() };
    out.push(Data { name: "leading-nan", frame: frame(lead(&o), lead(&h), lead(&l), lead(&c), lead(&v)) });
    // One whole bar missing early on (inside the warm-up of most lines).
    let gap = |x: &Vec<f64>| -> Vec<f64> { x.iter().enumerate().map(|(i, &y)| if i == 2 { f64::NAN } else { y }).collect() };
    out.push(Data { name: "early-missing-bar", frame: frame(gap(&o), gap(&h), gap(&l), gap(&c), gap(&v)) });
    out.push(Data { name: "early-missing-close", frame: frame(o.clone(), h.clone(), l.clone(), gap(&c), v.clone()) });
    let (mut ci, mut li) = (c.clone(), l.clone());
    ci[n / 3] = f64::INFINITY;
    li[n / 2] = f64::NEG_INFINITY;
    out.push(Data { name: "inf", frame: frame(o.clone(), h.clone(), li, ci, v.clone()) });
    let flat = |x: &Vec<f64>| -> Vec<f64> {
        x.iter().enumerate().map(|(i, &y)| if (n / 4..n / 2).contains(&i) { 100.0 } else { y }).collect()
    };
    let vz: Vec<f64> = v.iter().enumerate().map(|(i, &y)| if (n / 3..n / 2).contains(&i) { 0.0 } else { y }).collect();
    out.push(Data { name: "constant", frame: frame(flat(&o), flat(&h), flat(&l), flat(&c), vz) });
    let scale = |x: &Vec<f64>, k: f64| -> Vec<f64> { x.iter().map(|y| y / 100.0 * k).collect() };
    out.push(Data { name: "extreme", frame: frame(scale(&o, 1e300), scale(&h, 1e300), scale(&l, 1e300), scale(&c, 1e300), scale(&v, 1e300)) });
    out.push(Data { name: "tiny", frame: frame(scale(&o, 1e-300), scale(&h, 1e-300), scale(&l, 1e-300), scale(&c, 1e-300), v.clone()) });
    let neg = |x: &Vec<f64>| -> Vec<f64> { x.iter().map(|y| y - 200.0).collect() };
    out.push(Data { name: "negative", frame: frame(neg(&o), neg(&h), neg(&l), neg(&c), v.clone()) });
    // Zero prices for the first third (a ratio's zero prior), then the walk.
    let zero = |x: &Vec<f64>| -> Vec<f64> {
        x.iter().enumerate().map(|(i, &y)| if i < n / 3 { 0.0 } else { y }).collect()
    };
    out.push(Data { name: "zero", frame: frame(zero(&o), zero(&h), zero(&l), zero(&c), v) });
    out
}
