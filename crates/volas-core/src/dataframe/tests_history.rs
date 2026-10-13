//! The checkpoint of a cached column's history: where it lives and how it moves.

use super::*;

fn frame(rows: usize) -> DataFrame {
    let mut df = DataFrame::new(
        vec!["close".into(), "ema".into()],
        vec![Column::f64((0..rows).map(|i| i as f64).collect()); 2],
        None,
    )
    .unwrap();
    let reach = Reach { lookback: 2, resume: 0, continuation: 2, history: History::Carried { operands: 0 } };
    df.set_computed("ema", std::sync::Arc::new(()), reach);
    df
}

fn checkpoint(at: usize) -> Checkpoint {
    Checkpoint { at, state: Some(vec![at as f64]), raw: vec![("close".into(), Column::f64(vec![1.0, 2.0]))] }
}

#[test]
fn a_recursion_is_any_history_but_a_window() {
    assert!(!History::Window.recursive());
    assert!(History::Carried { operands: 0 }.recursive());
    assert!(History::Unkept.recursive());
}

/// A checkpoint is kept per cached column, travels with a bounded window's own drop of
/// its head, and is not carried by a slice (a frame of its own) or kept by a column
/// that is no longer cached.
#[test]
fn a_checkpoint_moves_with_the_window_drop_only() {
    let mut df = frame(10);
    assert_eq!(df.computed_checkpoint("ema"), None);
    df.set_computed_checkpoint("ema", Ok(checkpoint(5)));
    df.set_computed_checkpoint("close", Ok(checkpoint(5))); // not cached: no-op
    assert_eq!(df.computed_checkpoint("close"), None);
    assert_eq!(df.computed_checkpoint("ema"), Some(&checkpoint(5)));
    assert_eq!(df.retain_tail(4).computed_checkpoint("ema"), Some(&checkpoint(5)));
    assert_eq!(df.slice(6, 10).computed_checkpoint("ema"), None);
    assert_eq!(df.clone().computed_checkpoint("ema"), Some(&checkpoint(5)));
    // a write stales the column but keeps its history's checkpoint
    df.invalidate_computed_on_write("close");
    assert_eq!(df.computed_checkpoint("ema"), Some(&checkpoint(5)));
    df.set_computed_checkpoint("ema", Err(None));
    assert_eq!((df.computed_checkpoint("ema"), df.computed_history_lost("ema")), (None, None));
    let why = VolasError::Value("an input past 2^53".into());
    df.set_computed_checkpoint("ema", Err(Some(why.clone())));
    assert_eq!(df.retain_tail(4).computed_history_lost("ema"), Some(&why));
    assert_eq!(df.slice(6, 10).computed_history_lost("ema"), None);
    assert_eq!(df.computed_history_lost("close"), None);
}

/// A replay continues the column: its rows from the checkpoint's next one take the
/// replayed values, it is valid through the last row with both states, and its history
/// still starts where it did.
#[test]
fn a_replay_continues_the_column() {
    let mut df = frame(10).retain_tail(6); // rows 4..10, origin 4
    df.invalidate_computed_on_write("close"); // stale from row 0, no state
    let since = df.computed_columns()[0].1.since;
    df.continue_computed("ema", 2, &Column::f64(vec![7.0, 8.0, 9.0, 10.0]), Some(vec![10.0]), Some(vec![9.0]))
        .unwrap();
    let meta = df.computed_columns().into_iter().find(|(n, _)| n == "ema").unwrap().1;
    assert_eq!((meta.valid_rows, meta.since), (6, since));
    assert_eq!((meta.state, meta.state_before_last), (Some(vec![10.0]), Some(vec![9.0])));
    assert_eq!(df.computed_resume_state("ema"), Some((&[10.0][..], 4)));
    assert_eq!(df.column("ema").unwrap().to_f64_vec(), vec![4.0, 5.0, 7.0, 8.0, 9.0, 10.0]);
    // a replay ending inside the warm-up carries no state
    df.continue_computed("ema", 5, &Column::f64(vec![1.0]), None, Some(vec![0.5])).unwrap();
    let meta = df.computed_columns().into_iter().find(|(n, _)| n == "ema").unwrap().1;
    assert_eq!((meta.state, meta.state_before_last), (None, None));
}

/// A replay must end on the last row of a cached column; anything else writes nothing.
#[test]
fn a_replay_that_does_not_reach_the_last_row_is_refused() {
    let mut df = frame(5);
    let before = df.column("ema").unwrap().clone();
    let tail = Column::f64(vec![1.0, 2.0]);
    assert!(matches!(df.continue_computed("ema", 1, &tail, None, None), Err(VolasError::Shape(_))));
    assert!(matches!(df.continue_computed("ema", usize::MAX, &tail, None, None), Err(VolasError::Shape(_))));
    assert!(matches!(df.continue_computed("close", 3, &tail, None, None), Err(VolasError::ColumnNotFound(_))));
    assert!(matches!(df.continue_computed("nope", 3, &tail, None, None), Err(VolasError::ColumnNotFound(_))));
    assert!(df.continue_computed("ema", 4, &Column::bool(vec![true]), None, None).is_err());
    assert!(df.column("ema").unwrap().equals(&before));
    // a one-row frame keeps no state before its only row
    let mut one = frame(1);
    one.continue_computed("ema", 0, &Column::f64(vec![3.0]), Some(vec![3.0]), Some(vec![2.0])).unwrap();
    let meta = one.computed_columns()[0].1.clone();
    assert_eq!((meta.state, meta.state_before_last), (Some(vec![3.0]), None));
}
