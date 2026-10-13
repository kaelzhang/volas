//! Continuations preserve typed operands, warm-up history and decline foreign states.
use volas_core::{Column, DataFrame};
use volas_directive::{exec::{execute, execute_resume, execute_resume_step, initial_state}, parse};
use volas_directive::exec::{capture, replay};

fn frame() -> DataFrame {
    DataFrame::new(
        ["close", "integer", "small", "flag", "text"].map(str::to_owned).to_vec(),
        vec![
            Column::f64((0..60).map(|i| 10.0 + (i as f64 * 0.7).sin()).collect()),
            Column::i64((0..60).map(|i| (1i64 << 53) + i).collect()),
            Column::i32((0..60).map(|i| i % 3).collect()),
            Column::bool((0..60).map(|i| i % 2 == 0).collect()),
            Column::str(vec!["x".to_owned(); 60]),
        ], None,
    ).unwrap()
}

fn value(col: &Column, row: usize) -> f64 {
    if col.as_bool().is_some() { col.truth_at(row).map_or(f64::NAN, |v| f64::from(u8::from(v))) }
    else { col.get_f64(row) }
}

#[test]
fn scalar_chains_preserve_integer_comparisons_and_typed_operands() {
    let df = frame();
    for text in [
        "(integer > 9007199254740993) & (close > ema:3)",
        "(integer > 1.5) & (close > ema:3)",
        "(1.5 < integer) & (close > ema:3)",
        "(small ^ flag) | (close > ema:3)",
        "ema:3@small + ema:3@flag",
    ] {
        let node = parse(text).unwrap();
        let full = execute(&df, &node).unwrap();
        let mut state = initial_state(&df, &node, 30).unwrap();
        for row in 30..60 {
            let mut next = state.clone();
            let got = execute_resume_step(&df, &node, &state, (row, 0), &mut next).unwrap();
            assert_eq!(got.to_bits(), value(&full, row).to_bits(), "{text} at {row}");
            state = next;
        }
    }
    // A string in the scalar remainder cannot be converted into a numeric operand.
    let node = parse("ema:3 + text").unwrap();
    let state = initial_state(&df, &parse("ema:3 + close").unwrap(), 30).unwrap();
    assert!(execute(&df, &node).is_err());
    assert!(execute_resume_step(&df, &node, &state, (30, 0), &mut state.clone()).is_none());
}

#[test]
fn a_warmup_checkpoint_keeps_bounded_state_without_raw_columns() {
    let df = frame();
    let node = parse("ema:5 + 2").unwrap();
    let cp = capture(&df, &node, 2).unwrap();
    assert!(cp.state.is_some());
    assert!(cp.raw.is_empty());
    let (got, next) = replay(&df.slice_data(2, 60), &node, &cp, 58).unwrap();
    let full = execute(&df, &node).unwrap();
    assert_eq!(got.to_f64_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        full.to_f64_vec()[2..].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert!(next.state.is_some());
    assert!(next.raw.is_empty());
}

#[test]
fn a_continuation_declines_foreign_kernel_shapes_and_reversed_ranges() {
    let df = frame();
    let ma = parse("ma:5").unwrap();
    let state = initial_state(&df, &ma, 30).unwrap();
    assert!(execute_resume(&df, &ma, &state, 31, 30, 0).is_none());
    // The envelope is sound, but these commands cannot consume an SMA kernel.
    for text in ["wma:5", "apo:5,10,1"] {
        assert!(execute_resume(&df, &parse(text).unwrap(), &state, 30, 60, 0).is_none(), "{text}");
    }
    let state = initial_state(&df, &parse("ma:5 + 1").unwrap(), 30).unwrap();
    assert!(execute_resume_step(&df, &parse("wma:5 + 1").unwrap(), &state,
        (30, 0), &mut state.clone()).is_none());
}

#[test]
fn a_chain_declines_an_impossible_retained_span() {
    let df = frame();
    let node = parse("ma:5 + 1").unwrap();
    let mut state = initial_state(&df, &node, 30).unwrap();
    state[0] = f64::INFINITY;
    assert!(execute_resume(&df, &node, &state, 30, 60, 0).is_none());
}

#[test]
fn an_empty_history_has_no_terminal_states() {
    let df = frame().slice_data(0, 0);
    let (values, after, before) = volas_directive::exec::execute_with_states(&df, &parse("ema:3").unwrap()).unwrap();
    assert_eq!(values.len(), 0);
    assert!(after.is_none() && before.is_none());
}

#[test]
fn a_window_remainder_declines_scalar_and_resumes_as_a_vector() {
    let df = frame();
    let node = parse("midpoint:2@(ema:3)").unwrap();
    let state = initial_state(&df, &node, 30).unwrap();
    assert!(execute_resume_step(&df, &node, &state, (30, 0), &mut state.clone()).is_none());
    let (got, _) = execute_resume(&df, &node, &state, 30, 60, 0).unwrap();
    let full = execute(&df, &node).unwrap();
    for i in 0..got.len() { assert_eq!(got.get_f64(i).to_bits(), full.get_f64(30 + i).to_bits()); }
}

#[test]
fn finite_warmup_checkpoints_retain_expression_inputs_for_replay() {
    let df = frame();
    for text in ["-ma:5 + 2", "ma:5@(close + small)"] {
        let node = parse(text).unwrap();
        let cp = capture(&df, &node, 2).unwrap();
        assert!(cp.state.is_none());
        assert!(!cp.raw.is_empty());
        let (got, _) = replay(&df.slice_data(2, 60), &node, &cp, 58).unwrap();
        let full = execute(&df, &node).unwrap();
        for i in 0..got.len() {
            let (a, b) = (got.get_f64(i), full.get_f64(i + 2));
            assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(), "{text} at {i}");
        }
    }
}
