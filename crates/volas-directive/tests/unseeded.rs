use volas_core::{Column, DataFrame};
use volas_directive::{parse, exec::{execute, initial_state, execute_resume, execute_resume_step}};

#[test]
fn missing_history_carries_a_bounded_seed_through_completion() {
    for d in ["ema:20", "crsi:3,2,5", "rsi:14", "apo:5,10,1", "ema:5@(ema:3)"] {
        let data: Vec<_> = (0..240).map(|i| if i < 160 || i % 7 == 0 { f64::NAN } else { 3.0 + (i as f64).sin() }).collect();
        let df = DataFrame::new(vec!["close".into()], vec![Column::f64(data)], None).unwrap();
        let node = parse(d).unwrap();
        let full = execute(&df, &node).unwrap().to_f64_vec();
        let mut state = initial_state(&df, &node, 100).unwrap_or_else(|| panic!("unseeded {d}"));
        let short = initial_state(&df, &node, 150).unwrap();
        assert_eq!(state.len(), short.len());
        for (row, want) in full.iter().enumerate().skip(100) {
            // The caller has discarded all but its latest raw rows.
            let window = df.slice_data(row - 2, row + 1);
            let mut next = state.clone();
            let scalar = execute_resume_step(&window, &node, &state, (2, 0), &mut next);
            let (out, after) = execute_resume(&window, &node, &state, 2, 3, 0).unwrap_or_else(|| panic!("{d}: row {row}: {state:?}"));
            let got = out.to_f64_vec()[0];
            assert!(got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan(), "{d}, {row}: {got} != {want}");
            if let Some(got) = scalar {
                assert!(got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan(), "scalar {d}: {row}");
            }
            state = after;
        }
    }
}
