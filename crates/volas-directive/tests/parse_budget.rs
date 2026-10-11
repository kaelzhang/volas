//! The parse budget ([`MAX_NESTING`] / [`MAX_NODES`]) is the only gate a directive
//! passes, so every later walk over the tree inherits it. A directive exactly at the
//! budget must run every walk (bind, stringify, lookback, exec, drop) on an 8 MiB
//! thread (a main thread's default) even in an unoptimized build; one level or node
//! past it must be a syntax error.

use volas_core::{Column, DataFrame, VolasError};
use volas_directive::parser::{MAX_NESTING, MAX_NODES};
use volas_directive::{execute, lookback::lookback, parse, parse_cst, stringify};

const N: usize = MAX_NESTING;

fn frame() -> DataFrame {
    DataFrame::new(
        vec!["close".into()],
        vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])],
        None,
    )
    .unwrap()
}

/// Directives whose parse tree (or parser recursion) is exactly `depth` deep.
fn shapes(depth: usize) -> Vec<String> {
    vec![
        // `parse_expr` nests once per group, the outermost included.
        format!("{}close{}", "(".repeat(depth - 1), ")".repeat(depth - 1)),
        format!("{}close", "~".repeat(depth - 1)),
        format!("{}close", "-".repeat(depth - 1)),
        vec!["close"; depth].join("+"),
        format!(
            "{}close{}",
            "ema:2@(".repeat(depth - 1),
            ")".repeat(depth - 1)
        ),
    ]
}

fn on_main_sized_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn a_directive_at_the_budget_runs_every_walk() {
    on_main_sized_stack(|| {
        let df = frame();
        for d in shapes(N) {
            let ast = parse(&d).unwrap();
            assert!(!stringify(&*ast).is_empty());
            let _ = lookback(&ast);
            execute(&df, &ast).unwrap();
            drop(parse_cst(&d).unwrap());
        }
    });
}

#[test]
fn one_level_past_the_budget_is_a_syntax_error() {
    on_main_sized_stack(|| {
        for d in shapes(N + 1) {
            match parse(&d) {
                Err(VolasError::Parse(m)) => assert!(m.contains("nests deeper than"), "{m}"),
                other => panic!("{d:.40}: {:?}", other.map(|_| ())),
            }
        }
    });
}

#[test]
fn one_node_past_the_budget_is_a_syntax_error() {
    // A balanced sum keeps the tree shallow, so only the node budget can stop it.
    fn balanced(leaves: usize) -> String {
        if leaves == 1 {
            "close".into()
        } else {
            let half = leaves / 2;
            format!("({}+{})", balanced(half), balanced(leaves - half))
        }
    }
    // `leaves` leaves need `2 * leaves - 1` nodes.
    let at = balanced(MAX_NODES.div_ceil(2));
    assert!(parse_cst(&at).is_ok());
    let past = format!("{at}+close+close");
    match parse_cst(&past) {
        Err(VolasError::Parse(m)) => assert!(m.contains("more than"), "{m}"),
        other => panic!("{:?}", other.map(|_| ())),
    }
}
