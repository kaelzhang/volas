//! The directive layer's half of a derived frame: the frame a cached directive is
//! computed over, and the replay of a recursion's states a middle selection asks for.

use std::borrow::Cow;

use volas_core::{Column, DataFrame, Replay};
use volas_directive::{execute, parse, refresh_frame, Replayer};

fn closes(n: usize) -> DataFrame {
    let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.7).sin()).collect();
    DataFrame::new(vec!["close".into()], vec![Column::f64(close)], None).unwrap()
}

#[test]
fn a_frame_with_only_data_columns_is_its_own_refresh_frame() {
    let df = closes(5);
    assert!(matches!(refresh_frame(&df), Cow::Borrowed(_)));
    let mut named = df.clone();
    named.set_column("change", Column::f64(vec![0.0; 5])).unwrap(); // a bare command's name
    assert_eq!(refresh_frame(&named).names(), &["close".to_string()]);
}

#[test]
fn a_replay_gives_the_states_a_full_computation_ends_with() {
    let mut df = closes(30);
    let node = parse("ema:5").unwrap();
    df.set_column("ema:5", execute(&df, &node).unwrap()).unwrap();
    df.set_computed("ema:5", volas_directive::lookback::reach(&node));
    let (at, before) = Replayer.states(&df, "ema:5", 20).unwrap();
    let head = df.slice_data(0, 20);
    let col = execute(&head, &node).unwrap();
    let want = volas_directive::resume_states(&head, &node, &col);
    assert_eq!((Some(at), before), want);
    // a name that is not a cached column, or one that does not parse, has no states
    assert!(Replayer.states(&df, "close", 20).is_none());
    assert!(Replayer.states(&df, "ema:", 20).is_none());
}
