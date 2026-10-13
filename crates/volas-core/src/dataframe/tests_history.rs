//! The checkpoint of a cached column's history: where it lives and how it moves.

use super::*;
use super::derive::NoReplay;

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
    Checkpoint { at, state: Some(vec![at as f64].into()), raw: vec![("close".into(), Column::f64(vec![1.0, 2.0]))] }
}

#[test]
fn a_recursion_is_any_history_but_a_window() {
    assert!(!History::Window.recursive());
    assert!(History::Carried { operands: 0 }.recursive());
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
    df.keep_computed_anchors("close", vec![checkpoint(5)]); // likewise
    assert_eq!(df.computed_checkpoint("close"), None);
    assert_eq!(df.computed_checkpoint("ema"), Some(&checkpoint(5)));
    assert_eq!(df.retain_tail(4, &NoReplay).computed_checkpoint("ema"), Some(&checkpoint(5)));
    // any selection holding the row after it keeps it; one starting later keeps it too,
    // with the rows after it in its lead (within ANCHOR_SPACING rows)
    assert_eq!(df.rows(6, 10, &NoReplay).computed_checkpoint("ema"), Some(&checkpoint(5)));
    let later = df.rows(7, 10, &NoReplay);
    assert_eq!(later.computed_checkpoint("ema"), Some(&checkpoint(5)));
    assert_eq!(later.lead_rows(), 1); // row 6; the checkpoint owns its kernel tails
    assert_eq!(df.clone().computed_checkpoint("ema"), Some(&checkpoint(5)));
    // a write stales the column but keeps its history's checkpoint
    df.invalidate_computed_on_write("close");
    assert_eq!(df.computed_checkpoint("ema"), Some(&checkpoint(5)));
    df.set_computed_checkpoint("ema", Err(None));
    assert_eq!((df.computed_checkpoint("ema"), df.computed_history_lost("ema")), (None, None));
    let why = VolasError::Value("an input past 2^53".into());
    df.set_computed_checkpoint("ema", Err(Some(why.clone())));
    assert_eq!(df.retain_tail(4, &NoReplay).computed_history_lost("ema"), Some(&why));
    assert_eq!(df.rows(6, 10, &NoReplay).computed_history_lost("ema"), Some(&why));
    assert_eq!(df.computed_history_lost("close"), None);
}

/// A replay continues the column: its rows from the checkpoint's next one take the
/// replayed values, it is valid through the last row with both states, and its history
/// still starts where it did.
#[test]
fn a_replay_continues_the_column() {
    let mut df = frame(10).retain_tail(6, &NoReplay); // rows 4..10, origin 4
    df.invalidate_computed_on_write("close"); // stale from row 0, no state
    let since = df.computed_columns()[0].1.since;
    df.continue_computed("ema", 2, &Column::f64(vec![7.0, 8.0, 9.0, 10.0]), Some(vec![10.0]), Some(vec![9.0]))
        .unwrap();
    let meta = df.computed_columns().into_iter().find(|(n, _)| n == "ema").unwrap().1;
    assert_eq!((meta.valid_rows, meta.since), (6, since));
    assert_eq!((meta.state.as_ref().map(|words| words.to_vec()), meta.state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![10.0]), Some(vec![9.0])));
    assert_eq!(df.computed_resume_state("ema"), Some((&[10.0][..], 4)));
    assert_eq!(df.column("ema").unwrap().to_f64_vec(), vec![4.0, 5.0, 7.0, 8.0, 9.0, 10.0]);
    // a replay ending inside the warm-up carries no state
    df.continue_computed("ema", 5, &Column::f64(vec![1.0]), None, Some(vec![0.5])).unwrap();
    let meta = df.computed_columns().into_iter().find(|(n, _)| n == "ema").unwrap().1;
    assert_eq!((meta.state.as_ref().map(|words| words.to_vec()), meta.state_before_last.as_ref().map(|words| words.to_vec())), (None, None));
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
    assert_eq!((meta.state.as_ref().map(|words| words.to_vec()), meta.state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![3.0]), None));
}

/// Scalar truth writes preserve the Boolean dtype and move its validity cursor,
/// including a missing value followed by a replacement on the same row.
#[test]
fn computed_boolean_scalars_preserve_nulls_and_copy_on_write() {
    let mut df = DataFrame::new(vec!["signal".into()], vec![Column::bool(vec![false; 3])], None).unwrap();
    df.set_computed("signal", std::sync::Arc::new(()), Reach {
        lookback: 0, resume: 0, continuation: 0, history: History::Window,
    });
    df.invalidate_computed_on_write("close");
    let old = df.clone();
    for (row, value) in [0.0, 1.0, f64::NAN].into_iter().enumerate() {
        df.update_computed_value("signal", row, value).unwrap();
        assert_eq!(df.computed_columns()[0].1.valid_rows, row + 1);
    }
    let signal = df.column("signal").unwrap();
    assert_eq!(signal.dtype(), crate::DType::Bool);
    assert_eq!((signal.truth_at(0), signal.truth_at(1), signal.truth_at(2)), (Some(false), Some(true), None));
    df.update_computed_value("signal", 2, 0.0).unwrap();
    assert_eq!(df.column("signal").unwrap().truth_at(2), Some(false));
    assert_eq!(old.column("signal").unwrap().as_bool().unwrap(), &[false; 3]);
    assert_eq!(old.computed_columns()[0].1.valid_rows, 0);
}

#[test]
fn computed_scalar_rejects_an_integer_column_without_writing_or_advancing() {
    let mut df = DataFrame::new(vec!["integer".into()], vec![Column::i64(vec![42])], None).unwrap();
    df.set_computed("integer", std::sync::Arc::new(()), Reach {
        lookback: 0, resume: 0, continuation: 0, history: History::Window,
    });
    df.invalidate_computed_on_write("close");
    let err = df.update_computed_value("integer", 0, 1.5).unwrap_err();
    assert!(matches!(err, VolasError::DType(_)));
    assert!(err.to_string().contains("computed scalar dtype F64 does not match"));
    assert_eq!(df.column("integer").unwrap().get_i64(0), 42);
    assert_eq!(df.computed_columns()[0].1.valid_rows, 0);
}
