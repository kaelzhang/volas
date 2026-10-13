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
        if execute(&head, node).is_err() {
            continue;
        }
        let Some(mut state) = initial_state(&head, node, usize::MAX) else { continue };
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
            if let Some(fresh) = initial_state(df, node, usize::MAX) {
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
    if execute(&head, node).is_err() {
        return fails;
    }
    let Some(anchor) = initial_state(&head, node, usize::MAX) else { return fails };
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
                if execute(&probe, &node).is_ok() && initial_state(&probe, &node, usize::MAX).is_some() {
                    resumable += 1;
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

/// The computation together with its carried states (one pass for a recursion that
/// resumes) gives the full computation's values bit for bit, and the states a capture
/// after it gives — on every line, data class and a range of heights.
#[test]
fn a_computation_with_its_states_is_the_full_computation() {
    use volas_directive::exec::execute_with_states;
    let mut fails = Vec::new();
    let mut one_pass = 0usize;
    for rows in [7, 40, 160] {
        for data in &matrix::classes(rows) {
            let df = &data.frame;
            let extra = ["ma:5,1", "ma:5,3", "ma:5,4", "ma:5,6", "ma:5,7", "ma:5,8", "kcw:7,4,1.5", "rsi:5@(ma:3)",
                "ema:4@(close+open)", "obv@(wma:3),volume", "ema:3@(ema:4)", "stochrsi.d:5,3,3,0"];
            let lines = matrix::lines();
            let texts = lines.iter().flat_map(matrix::variants).chain(extra.iter().map(|t| t.to_string()));
            for text in texts {
                {
                    let node = parse(&text).unwrap();
                    let Some(whole) = full(df, &node) else { continue };
                    let (col, state, before) = execute_with_states(df, &node).unwrap();
                    if let Some(d) = first_diff(&values(&col), &whole) {
                        fails.push(format!("{text} [{} {rows}]: values {d}", data.name));
                    }
                    let want = (initial_state(df, &node, rows), initial_state(df, &node, rows - 1));
                    let same_state = |a: &Option<Vec<f64>>, b: &Option<Vec<f64>>| match (a, b) {
                        (Some(a), Some(b)) => first_diff(a, b).is_none(),
                        (None, None) => true,
                        _ => false,
                    };
                    if !same_state(&state, &want.0) || (state.is_some() && !same_state(&before, &want.1)) {
                        fails.push(format!("{text} [{} {rows}]: states {state:?} / {before:?} != {want:?}", data.name));
                    }
                    one_pass += usize::from(state.is_some());
                }
            }
        }
    }
    assert!(one_pass > 1000, "only {one_pass} computations carried states");
    assert!(fails.is_empty(), "{} failure(s):\n{}", fails.len(), fails[..fails.len().min(30)].join("\n"));
}

/// Far into a long series a continuation reads only a trailing window of the frame: for
/// every line, the `Vec` resume and the single-row step from a carried state equal the
/// full recompute — and a line whose spec declares a resume never declines there, also
/// with a missing row inside or before the window.
#[test]
fn a_continuation_far_into_the_series_reads_a_window_exactly() {
    use volas_directive::spec::Resume;
    use volas_directive::types::Node;
    const LONG: usize = 400;
    let classes = matrix::classes(LONG);
    let mut fails = Vec::new();
    let mut stepped = 0usize;
    for line in &matrix::lines() {
        for text in matrix::variants(line) {
            let node = parse(&text).unwrap();
            let declared = match node.node() {
                Node::Command(c) => c.args.spec().resume(&c.args),
                _ => Resume::Recompute,
            };
            let picked = ["walk", "nan-close", "early-missing-bar", "constant"];
            for data in classes.iter().filter(|d| picked.contains(&d.name)) {
                let df = &data.frame;
                let Some(whole) = full(df, &node) else { continue };
                for from in [LONG - 40, LONG - 1] {
                    let head = df.slice(0, from);
                    if execute(&head, &node).is_err() {
                        continue;
                    }
                    let Some(state) = initial_state(&head, &node, usize::MAX) else {
                        if declared != Resume::Recompute {
                            fails.push(format!("{text} [{}] from {from}: no state", data.name));
                        }
                        continue;
                    };
                    match execute_resume(df, &node, &state, from, usize::MAX, 0) {
                        Some((tail, _)) => {
                            if let Some(d) = first_diff(&values(&tail), &whole[from..]) {
                                fails.push(format!("{text} [{}] from {from}: vec {d}", data.name));
                            }
                        }
                        None => fails.push(format!("{text} [{}] from {from}: the vec resume declined", data.name)),
                    }
                    let mut next = vec![0.0; state.len()];
                    match execute_resume_step(df, &node, &state, (from, 0), &mut next) {
                        Some(v) => {
                            stepped += 1;
                            if !same(v, whole[from]) {
                                fails.push(format!("{text} [{}] from {from}: step {v:?} != {:?}", data.name, whole[from]));
                            }
                        }
                        None if declared == Resume::Step => {
                            fails.push(format!("{text} [{}] from {from}: the step declined", data.name));
                        }
                        None => {}
                    }
                }
            }
        }
    }
    // `@` operands evaluated over the window (a pure window or an accumulator) or over
    // the frame (a recursion, a row position), under recursive and window commands
    for text in [
        "rsi:5@(ma:3)", "ema:4@(var:3)", "ma:4@(sum:3@(close-open))", "atr:5@(ma:2@high),low,close",
        "obv@(wma:3),volume", "kama:4@(close*volume)", "sum:3@(ema:4)", "ema:3@(maxindex:4)",
        "macd.signal:5,9,3@(boll.upper:6)", "stddev:5@(ma:3,1)",
    ] {
        let node = parse(text).unwrap();
        for data in classes.iter().filter(|d| ["walk", "nan-close", "early-missing-bar"].contains(&d.name)) {
            let df = &data.frame;
            let whole = full(df, &node).unwrap();
            for from in [LONG - 40, LONG - 1] {
                let head = df.slice(0, from);
                let state = initial_state(&head, &node, usize::MAX).unwrap();
                let (tail, _) = execute_resume(df, &node, &state, from, usize::MAX, 0).unwrap();
                if let Some(d) = first_diff(&values(&tail), &whole[from..]) {
                    fails.push(format!("{text} [{}] from {from}: vec {d}", data.name));
                }
                let mut next = vec![0.0; state.len()];
                let v = execute_resume_step(df, &node, &state, (from, 0), &mut next).unwrap();
                if !same(v, whole[from]) {
                    fails.push(format!("{text} [{}] from {from}: step {v:?} != {:?}", data.name, whole[from]));
                }
            }
        }
    }
    assert!(stepped > 500, "only {stepped} steps ran");
    assert!(fails.is_empty(), "{} failure(s):\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn every_recursive_composite_declares_and_captures_a_continuation() {
    use volas_directive::{spec::Resume, types::Node};
    let data = matrix::classes(400);
    for line in matrix::lines().iter().filter(|l| ["crsi", "stoch_momentum", "apo", "ppo", "macdext", "mavp", "stoch", "stochf", "stochrsi"].contains(&l.head.split('.').next().unwrap())) {
        for text in matrix::variants(line) {
            let node = parse(&text).unwrap();
            if let Node::Command(c) = node.node() {
                if c.args.spec().history(&c.args) == volas_directive::spec::History::Recursive {
                    assert_ne!(c.args.spec().resume(&c.args), Resume::Recompute, "{text}");
                    assert!(initial_state(&data[0].frame, &node, 399).is_some(), "{text}");
                }
            }
        }
    }
}

/// Verify the language-wide invariant, including the Cartesian product of every MA
/// selector (not just equal-type or rotating mixed examples), at identity and normal
/// periods. This guards removal of the old unsupported-recursion history class.
#[test]
fn every_executable_recursive_line_and_ma_combination_has_a_resume() {
    use volas_directive::{spec::{ArgBound, History, Resume}, types::Node};
    fn check(node: &Ast, text: &str) -> bool {
        let (spec, args) = match node.node() {
            Node::Command(c) => (c.args.spec().clone(), &c.args[..]),
            Node::Name(name) => return check(&volas_directive::bind_command(name, None, &[], &[]).unwrap(), text),
            _ => panic!("inventory entry must name a command: {text}"),
        };
        if spec.history(args) != History::Recursive { return false; }
        assert_ne!(spec.resume(args), Resume::Recompute, "{text}");
        assert!(volas_directive::lookback::history(node).recursive(), "{text}");
        true
    }
    let mut checked = 0;
    for line in matrix::lines() {
        for period in [1, 5] {
            let base: Vec<String> = line.args.iter().map(|a| matrix::token(a, period, false)).collect();
            let slots: Vec<usize> = line.args.iter().enumerate()
                .filter_map(|(i, a)| (a.bound == ArgBound::IntRange(0, 8)).then_some(i)).collect();
            for mut choice in 0..9usize.pow(slots.len() as u32) {
                let mut args = base.clone();
                for &i in &slots { args[i] = (choice % 9).to_string(); choice /= 9; }
                let suffix = if args.is_empty() { String::new() } else { format!(":{}", args.join(",")) };
                let text = format!("{}{suffix}{}", line.head, line.series);
                let node = parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
                checked += usize::from(check(&node, &text));
            }
        }
        for text in matrix::variants(&line) {
            check(&parse(&text).unwrap(), &text);
        }
    }
    assert!(checked > 1500, "only {checked} recursive combinations checked");
}
