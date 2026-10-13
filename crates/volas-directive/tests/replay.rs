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
    let got = Replayer.continuation(&df, "ema:5", 7, 20, true, true);
    let head = df.slice_data(0, 20);
    let (at, before) = volas_directive::resume_states(&head, &node);
    assert_eq!(got.ends, Some((at.unwrap(), before)));
    let boundary = got.boundary.unwrap();
    assert_eq!((boundary.at, boundary.state), (6, volas_directive::resume_states(&df.slice_data(0, 7), &node).0));
    // a name that is not a cached column, or one that does not parse, has no history
    assert_eq!(Replayer.continuation(&df, "close", 7, 20, true, true), Default::default());
    assert_eq!(Replayer.continuation(&df, "ema:", 7, 20, true, true), Default::default());
}
