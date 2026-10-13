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
fn a_warmup_checkpoint_keeps_columns_but_not_literal_operands() {
    let df = frame();
    let node = parse("ema:5 + 2").unwrap();
    let cp = capture(&df, &node, 2).unwrap();
    assert!(cp.state.is_none());
    assert_eq!(cp.raw.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), ["close"]);
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
