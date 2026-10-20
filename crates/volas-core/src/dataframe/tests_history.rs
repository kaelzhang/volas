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

#[test]
fn an_empty_derived_memo_is_lazy_and_shares_later_publication_with_its_copies() {
    let parent = Anchors::default();
    let selected = parent.bounded(0, 2000);
    assert_eq!(selected.len(), 0);
    assert!(selected.nearest(1000).is_none());
    // A handle taken before first publication continues to observe this frame's
    // memo, while its parent owns a different lifetime.
    let handle = selected.clone();
    selected.keep(checkpoint(511));
    assert_eq!(handle.nearest(1000).unwrap().at, 511);
    assert!(parent.nearest(1000).is_none());
    let empty = selected.bounded(1000, 1500);
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.bounded(1000, 1200).len(), 0);
    assert_eq!(empty, empty);
    assert_ne!(empty, selected.bounded(1000, 1500));
    assert_ne!(empty, handle.bounded(0, 1500));
}

#[test]
fn derived_history_shares_checkpoint_and_word_chunks() {
    let anchors = Anchors::default();
    anchors.keep_all((0..32).map(|i| checkpoint((i + 1) * 512 - 1)).collect());
    let original = anchors.nearest(8191).unwrap();
    let selected = anchors.bounded(8000, 9000);
    let derived = selected.nearest(8191).unwrap();
    assert!(std::ptr::eq(&*original, &*derived));
    assert_eq!(original.state.as_ref().unwrap().as_ptr(), derived.state.as_ref().unwrap().as_ptr());
}

#[test]
fn chunk_selections_merge_later_publications_without_pinning_the_parent() {
    for sparse in [false, true] {
        let parent = Anchors::default();
        let row = |i: usize| if sparse { i * 513 + 11 } else { (i + 1) * 512 - 1 };
        parent.keep_all((0..80).map(|i| checkpoint(row(i))).collect());
        let selected = parent.bounded(row(8), row(32));
        let original = parent.nearest(row(9)).unwrap();
        assert!(std::ptr::eq(&*original, &*selected.nearest(row(9)).unwrap()));
        selected.keep(checkpoint(row(9))); // a frozen duplicate retains its original
        selected.keep_all(vec![checkpoint(row(8)), checkpoint(row(9))]);
        selected.keep(checkpoint(row(9) + 1)); // interleave a sparse publication
        selected.keep_all(vec![checkpoint(row(10)), checkpoint(row(31) + 1)]);
        selected.keep(checkpoint(row(7) + 1));
        assert_eq!(selected.nearest(row(7) + 1).unwrap().at, row(7) + 1);
        let nested = selected.bounded(row(8), row(32));
        assert_eq!(nested.nearest(row(31) + 1).unwrap().at, row(31) + 1);
        assert_eq!(nested.nearest(row(9) + 1).unwrap().at, row(9) + 1);
        assert!(std::ptr::eq(&*original, &*nested.nearest(row(9)).unwrap()));
        // Once the parent goes, an append after a clipped bank's logical end must
        // not make its hidden suffix visible again.
        let clipped = parent.bounded(row(0), row(3));
        drop(parent);
        clipped.keep(checkpoint(row(6) + 1));
        assert_eq!(clipped.nearest(row(5)).unwrap().at, row(2));
        assert_eq!(clipped.nearest(row(6) + 1).unwrap().at, row(6) + 1);
        assert_eq!(clipped.bounded(row(0), row(7)).nearest(row(6) + 1).unwrap().at, row(6) + 1);
    }
}

#[test]
fn an_anchor_handle_keeps_empty_and_short_frame_metadata_compact() {
    let size = std::mem::size_of::<Anchors>();
    println!("anchor handle: {size} bytes");
    assert!(size <= 48, "cold chunk descriptors inflated every cached column's metadata to {size} bytes");
}

#[test]
fn selection_keeps_the_boundary_and_only_its_rows_from_one_history_snapshot() {
    for sparse in [false, true] {
        let parent = Anchors::default();
        let row = |i: usize| if sparse { i * 513 + 11 } else { (i + 1) * 512 - 1 };
        parent.keep_all((0..300).map(|i| checkpoint(row(i))).collect());
        let selected = parent.bounded(row(8), row(180));
        selected.keep(checkpoint(row(181)));
        selected.keep(checkpoint(row(7) + 1));
        for (first, upto) in [(row(9), row(12)), (row(179), row(183)), (row(182), row(184)), (row(0), row(300))] {
            let (range, boundary) = selected.selection(first, upto, Some(first - 1));
            assert_eq!(boundary, selected.nearest(first - 1));
            assert_eq!(range.len(), selected.bounded(first, upto).len());
            for at in [first - 1, first, upto - 1, upto] {
                assert_eq!(range.nearest(at), selected.bounded(first, upto).nearest(at));
            }
        }
        let (empty, boundary) = selected.selection(0, 1, Some(0));
        assert_eq!(empty.len(), 0);
        assert!(boundary.is_none());
    }
}

#[test]
fn regular_chunk_lookup_survives_clipping_and_irregular_later_publications() {
    for sparse in [false, true] {
        for growing in [false, true] {
            let parent = Anchors::default();
            let row = |i: usize| if sparse { i * 513 + 11 } else { (i + 1) * 512 - 1 };
            let point = |i: usize| {
                let mut cp = checkpoint(row(i));
                cp.state = Some(vec![i as f64; if growing && i >= 128 { 4097 } else { 1 }].into());
                cp
            };
            parent.keep_all((0..300).map(point).collect());
            // Sharing a partial tail forces a new chunk on the next publication;
            // the row index must still find both old and newly published checkpoints.
            let clipped = parent.bounded(row(64), row(300));
            parent.keep(point(300));
            parent.keep_all((301..310).map(point).collect());
            let nested = clipped.bounded(row(96), row(290));
            for i in 0..310 {
                assert_eq!(parent.nearest(row(i)).unwrap().state.as_ref().unwrap()[0], i as f64);
                let expected = (i >= 96).then_some(i.min(289));
                assert_eq!(nested.nearest(row(i)).map(|cp| cp.state.as_ref().unwrap()[0] as usize), expected);
            }
        }
    }
}

#[test]
fn a_publication_identity_differs_from_an_unopened_independent_owner() {
    let owner = Anchors::default();
    let observer = owner.clone();
    let independent = Anchors::default();
    assert_eq!(owner, observer);
    assert_ne!(owner, independent);
    assert_ne!(independent, owner);
}

#[test]
fn promoting_a_pending_boundary_preserves_its_checkpoint_owner() {
    let mut df = frame(3).rows(1, 3, &NoReplay);
    df.set_computed_pending("ema", Some(checkpoint(0)));
    let before = std::ptr::from_ref(&**df.computed.get("ema").unwrap().pending.as_ref().unwrap());
    assert!(!df.promote_computed_pending("ema", 1));
    assert!(df.has_computed_pending("ema", 0));
    assert!(df.promote_computed_pending("ema", 0));
    let after = std::ptr::from_ref(df.computed_checkpoint("ema").unwrap());
    assert_eq!(before, after, "promotion must transfer the same immutable checkpoint owner");
    assert!(!df.promote_computed_pending("ema", 0));
}

#[test]
fn stale_snapshots_vouch_for_the_exact_carried_cursor_without_owning_its_state() {
    for rows in [0, 5] {
        let mut df = frame(rows);
        df.set_computed_states("ema", Some(vec![10.0]), Some(vec![9.0]));
        let row = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![3.0])], None).unwrap();
        df.append(&row).unwrap();
        let snapshot = df.shared_stale_computed_columns(Some("ema")).next().unwrap();
        assert_eq!(snapshot.state_ready, rows > 0);
        assert_eq!(snapshot.state_ready, df.computed_resume_state("ema").is_some());
        assert_eq!(df.stale_computed_columns(Some("ema"))[0].state_ready, snapshot.state_ready);
        df.computed.get_mut("ema").unwrap().state_at = usize::MAX;
        assert!(!df.shared_stale_computed_columns(Some("ema")).next().unwrap().state_ready);
        assert!(df.computed_resume_state("ema").is_none());
        assert_eq!(snapshot.state_ready, rows > 0, "a snapshot survives subsequent writes");
        df.set_computed_states("ema", None, None);
        assert!(!df.shared_stale_computed_columns(Some("ema")).next().unwrap().state_ready);
        df.invalidate_computed_on_write("close");
        assert!(!df.shared_stale_computed_columns(Some("ema")).next().unwrap().state_ready);
    }
}

#[test]
fn a_resumed_value_commits_through_its_already_borrowed_cursor() {
    let mut df = frame(5);
    df.set_computed_states("ema", Some(vec![10.0]), Some(vec![9.0]));
    let row = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![3.0])], None).unwrap();
    df.append(&row).unwrap();
    super::computed::WRITE_CURSOR_LOOKUPS.with(|lookups| lookups.set(0));
    let value = df.resume_computed_value("ema", |_, prev, _, next| {
        next[0] = prev[0] + 1.0;
        Some(next[0])
    }).unwrap();
    assert_eq!(value, Some(11.0));
    assert_eq!(df.column("ema").unwrap().as_f64().unwrap()[5], 11.0);
    assert!(!df.has_stale_computed());
    super::computed::WRITE_CURSOR_LOOKUPS.with(|lookups| assert_eq!(lookups.get(), 0,
        "committing the stepped row must not look up that same cache cursor again"));
    df.append(&row).unwrap();
    assert_eq!(df.resume_computed_value("ema", |_, _, _, _| None).unwrap(), None);
    assert_eq!(df.computed_resume_state("ema").unwrap().0, &[11.0]);
    assert!(df.column("ema").unwrap().as_f64().unwrap()[6].is_nan());
    assert!(df.has_stale_computed());
    for name in ["missing", "close"] {
        assert_eq!(df.resume_computed_value(name, |_, _, _, _| panic!("not a cached column")).unwrap(), None);
    }
}

#[test]
fn a_short_live_window_does_not_keep_a_unique_owned_seed_allocation() {
    let (seed, keep) = (65_536, 16);
    let mut df = DataFrame::new(vec!["wide".into(), "narrow".into()], vec![
        Column::f64((0..seed).map(|i| i as f64).collect()),
        Column::f32((0..seed).map(|i| i as f32).collect()),
    ], None).unwrap();
    df.compact_tail(keep, &NoReplay);
    assert_eq!(df.height(), keep);
    assert_eq!(df.columns[0].to_f64_vec()[0], (seed - keep) as f64);
    assert_eq!(df.columns[1].to_f64_vec()[0], (seed - keep) as f64);
    let Column::F64(wide) = &mut df.columns[0] else { unreachable!() };
    assert!(!wide.is_borrowed());
    assert!(wide.make_mut().capacity() <= 2 * keep, "release the unique wide seed capacity");
    let Column::F32(narrow) = &mut df.columns[1] else { unreachable!() };
    assert!(!narrow.is_borrowed());
    assert!(narrow.make_mut().capacity() <= 2 * keep, "release the unique narrow seed capacity");
}
