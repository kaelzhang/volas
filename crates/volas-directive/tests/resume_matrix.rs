//! Resume-vs-full matrix: every output line × data class × resume point.
//!
//! For each directive with a resume kernel, the carried state captured over a head
//! `[0, from)` must continue the recursion so that
//!
//! * the resumed tail equals a full recompute of the same rows, bit for bit, at every
//!   `from` in `{1, 2, lb-1, lb, lb+1, fv-1, fv, fv+1, n/2, n-1}` (`lb` the lookback,
//!   `fv` the first non-NaN row of the full output);
//! * the state the resume returns is fed through further chunks (one row, a few rows,
//!   the rest) and each chunk again equals the full recompute, and the final state
//!   equals the state captured from scratch over the whole frame;
//! * a forming row re-folded twice from the same anchor (the tf live path) equals the
//!   full recompute of each version, through both the Vec and the scalar resume.
//!
//! Lines whose resume is known to diverge are listed in `KNOWN_DIVERGENT` (each an
//! open defect): they are reported, and the test fails if one of them stops
//! diverging (so the list cannot go stale).

mod matrix;

use std::collections::{BTreeMap, BTreeSet};

use volas_core::{Column, DataFrame};
use volas_directive::exec::{execute, execute_resume, execute_resume_step, initial_state};
use volas_directive::lookback::lookback;
use volas_directive::parse;
use volas_directive::types::Ast;

const ROWS: usize = 90;

/// `(line head, data class)` pairs whose resume is known to differ from the full
/// compute (open defects); `*` matches any class.
const KNOWN_DIVERGENT: &[(&str, &str)] = &[];

fn values(col: &Column) -> Vec<f64> {
    match col.as_bool() {
        Some(b) => b.iter().map(|&x| f64::from(u8::from(x))).collect(),
        None => col.to_f64_vec(),
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) || a == b
}

/// The first row where `got` and `want` differ.
fn first_diff(got: &[f64], want: &[f64]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("length {} != {}", got.len(), want.len()));
    }
    got.iter().zip(want).position(|(a, b)| !same(*a, *b)).map(|i| format!("row +{i}: {:?} != {:?}", got[i], want[i]))
}

fn full(df: &DataFrame, node: &Ast) -> Option<Vec<f64>> {
    execute(df, node).ok().map(|c| values(&c))
}

/// Check one directive on one data class; returns a description of each failure.
fn check(df: &DataFrame, node: &Ast) -> Vec<String> {
    let mut fails = Vec::new();
    let n = df.height();
    let Some(whole) = full(df, node) else { return fails };
    let lb = lookback(node);
    let fv = whole.iter().position(|v| !v.is_nan()).unwrap_or(n);
    let mut froms: BTreeSet<usize> = BTreeSet::new();
    for f in [1, 2, lb.saturating_sub(1), lb, lb + 1, fv.saturating_sub(1), fv, fv + 1, n / 2, n - 1] {
        if (1..n).contains(&f) {
            froms.insert(f);
        }
    }
    for from in froms {
        let head = df.slice(0, from);
        let Ok(head_col) = execute(&head, node) else { continue };
        let Some(mut state) = initial_state(&head, node, &head_col) else { continue };
        let mut at = from;
        for end in [from + 1, (from + 4).min(n), n] {
            if end <= at {
                continue;
            }
            let frame = df.slice(0, end);
            let Some(want) = full(&frame, node) else { break };
            // `None` is a legitimate decline: the caller falls back to a full compute.
            let Some((tail, next)) = execute_resume(&frame, node, &state, at, frame.height(), 0) else { break };
            if let Some(d) = first_diff(&values(&tail), &want[at..end]) {
                fails.push(format!("from {from}: chunk [{at},{end}) {d}"));
                break;
            }
            state = next;
            at = end;
        }
        if at == n {
            if let Some(fresh) = initial_state(df, node, &Column::f64(whole.clone())) {
                if first_diff(&state, &fresh).is_some() {
                    fails.push(format!("from {from}: carried state {state:?} != fresh {fresh:?}"));
                }
            }
        }
        fails.extend(forming(df, node, from));
    }
    fails
}

/// The live tf path: the forming row `from` is re-folded twice from one anchor (the
/// state over the closed rows `[0, from)`); each version equals its full recompute.
fn forming(df: &DataFrame, node: &Ast, from: usize) -> Vec<String> {
    let mut fails = Vec::new();
    let head = df.slice(0, from);
    let Ok(head_col) = execute(&head, node) else { return fails };
    let Some(anchor) = initial_state(&head, node, &head_col) else { return fails };
    for bump in [1.01, 0.97] {
        let frame = with_last_row_scaled(&df.slice(0, from + 1), bump);
        let Some(want) = full(&frame, node) else { continue };
        if let Some((tail, _)) = execute_resume(&frame, node, &anchor, from, frame.height(), 0) {
            if let Some(d) = first_diff(&values(&tail), &want[from..]) {
                fails.push(format!("forming x{bump} at {from}: vec {d}"));
            }
        }
        let mut next = vec![0.0; anchor.len()];
        if let Some(v) = execute_resume_step(&frame, node, &anchor, (from, 0), &mut next) {
            if !same(v, want[from]) {
                fails.push(format!("forming x{bump} at {from}: scalar {v:?} != {:?}", want[from]));
            }
        }
    }
    fails
}

/// A copy of `df` whose last row's high / close are moved (a later fine bar folded in).
fn with_last_row_scaled(df: &DataFrame, k: f64) -> DataFrame {
    let last = df.height() - 1;
    let cols = df
        .names()
        .iter()
        .zip(df.columns())
        .map(|(name, col)| {
            let mut v = col.to_f64_vec();
            if name == "close" || name == "high" || name == "volume" {
                v[last] *= k;
            }
            Column::f64(v)
        })
        .collect();
    DataFrame::new(df.names().to_vec(), cols, None).expect("same shape")
}

#[test]
fn resume_equals_full_recompute_for_every_line_class_and_point() {
    let classes = matrix::classes(ROWS);
    let mut divergent: BTreeMap<(String, &'static str), String> = BTreeMap::new();
    let mut resumable = 0usize;
    let lines = matrix::lines();
    assert!(lines.len() > 200, "the derived inventory shrank to {} lines", lines.len());
    for line in &lines {
        for text in matrix::variants(line) {
            let Ok(node) = parse(&text) else { panic!("{text} does not parse") };
            for data in &classes {
                let probe = data.frame.slice(0, ROWS / 2);
                if let Ok(col) = execute(&probe, &node) {
                    if initial_state(&probe, &node, &col).is_some() {
                        resumable += 1;
                    }
                }
                let fails = check(&data.frame, &node);
                if let Some(first) = fails.first() {
                    divergent.entry((line.head.clone(), data.name)).or_insert(format!("{text}: {first}"));
                }
            }
        }
    }
    eprintln!("{} lines, {resumable} resumable (directive, class) pairs", lines.len());
    assert!(resumable > 50, "only {resumable} resumable (line, class) pairs — the matrix lost its reach");
    // A class pattern is exact, `*` (any class), or a `prefix*`.
    let matches = |pattern: &str, class: &str| match pattern.strip_suffix('*') {
        Some(prefix) => class.starts_with(prefix),
        None => pattern == class,
    };
    let known = |head: &str, class: &str| KNOWN_DIVERGENT.iter().any(|&(h, c)| h == head && matches(c, class));
    let unexpected: Vec<_> = divergent.iter().filter(|((h, c), _)| !known(h, c)).collect();
    let fixed: Vec<_> = KNOWN_DIVERGENT
        .iter()
        .filter(|&&(h, c)| !divergent.keys().any(|(dh, dc)| dh == h && matches(c, dc)))
        .collect();
    for ((h, c), why) in &divergent {
        eprintln!("divergent {h} [{c}]: {why}");
    }
    assert!(unexpected.is_empty(), "{} resume divergence(s) not in KNOWN_DIVERGENT:\n{}", unexpected.len(),
        unexpected.iter().map(|((h, c), w)| format!("  {h} [{c}]: {w}")).collect::<Vec<_>>().join("\n"));
    assert!(fixed.is_empty(), "KNOWN_DIVERGENT entries no longer diverge (remove them): {fixed:?}");
}
