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
    let ends = volas_directive::states_at(&df, "ema:5", 20);
    let carried = volas_directive::resume_states(&df.slice_data(0, 20), &node);
    assert_eq!(ends, Some((carried.state.unwrap(), carried.before)));
    // a selection's first row: replayed over its lead, with no state before it
    let sub = df.rows(10, 30, &Replayer);
    let first = volas_directive::resume_states(&df.slice_data(0, 11), &node).state;
    assert_eq!(volas_directive::states_at(&sub, "ema:5", 1), Some((first.unwrap(), None)));
    let boundary = Replayer.boundary(&df, "ema:5", 7).unwrap();
    assert_eq!((boundary.at, boundary.state), (6, volas_directive::resume_states(&df.slice_data(0, 7), &node).state));
    assert_eq!(Replayer.boundary(&df, "ema:5", 0), None); // no row before the first
    // a name that is not a cached column, or one that does not parse, has no history
    assert_eq!(Replayer.boundary(&df, "close", 7), None);
    assert_eq!(volas_directive::states_at(&df, "ema:", 20), None);
}

/// A computation over a long history keeps the checkpoints of a recursion carried by its
/// state at the anchor rows it passes — each the one a capture there gives — and its
/// states are the full computation's; a frame whose row 0 is past position 0 keeps them
/// at the same absolute rows; a directive that is no such recursion keeps none.
#[test]
fn a_computation_over_a_long_history_keeps_its_anchors() {
    use volas_core::dataframe::ANCHOR_SPACING as S;
    use volas_directive::exec::{capture, initial_state};
    let n = 3 * S + 7;
    let df = closes(n);
    for text in ["ema:5", "ema:5@(ma:3)"] {
        let node = parse(text).unwrap();
        let carried = volas_directive::resume_states(&df, &node);
        let rows: Vec<usize> = carried.anchors.iter().map(|cp| cp.at).collect();
        assert_eq!(rows, vec![S - 1, 2 * S - 1, 3 * S - 1], "{text}");
        for cp in &carried.anchors {
            assert_eq!(Some(cp), capture(&df, &node, cp.at + 1).as_ref(), "{text}");
        }
        assert_eq!(carried.state, initial_state(&df, &node, n), "{text}");
        assert_eq!(carried.before, initial_state(&df, &node, n - 1), "{text}");
    }
    let node = parse("ema:5").unwrap();
    let later = df.slice_data(10, n);
    let rows: Vec<usize> = volas_directive::resume_states(&later, &node).anchors.iter().map(|cp| cp.at).collect();
    assert_eq!(rows, vec![S - 1, 2 * S - 1, 3 * S - 1]);
    assert!(volas_directive::resume_states(&df, &parse("ma:5").unwrap()).anchors.is_empty());
    // an anchor at the row before the last: the state before the last is the anchor's
    let short = closes(S + 1);
    let carried = volas_directive::resume_states(&short, &node);
    assert_eq!(carried.anchors.iter().map(|cp| cp.at).collect::<Vec<_>>(), vec![S - 1]);
    assert_eq!(carried.before, initial_state(&short, &node, S));
    assert_eq!(carried.state, initial_state(&short, &node, S + 1));
}
