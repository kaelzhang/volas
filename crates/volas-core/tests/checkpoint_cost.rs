//! Checkpoint lifetime sharing keeps live checkpoints and batch anchors bounded in allocations.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{Anchors, Checkpoint, CheckpointRef, Column, DataFrame, History, Reach, Replay};

struct NoReplay;
impl Replay for NoReplay {
    fn boundary(&self, _: &DataFrame, _: &str, _: usize) -> Option<Checkpoint> { None }
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> { None }
}

#[test]
fn an_empty_history_owner_and_read_allocate_nothing() {
    let (anchors, calls, bytes) = measure(Anchors::default);
    assert_eq!((calls, bytes), (0, 0), "an empty read-only cursor needs no publication owner");
    let (point, calls, bytes) = measure(|| anchors.nearest(usize::MAX));
    assert!(point.is_none());
    assert_eq!((calls, bytes), (0, 0));
}

#[test]
fn a_short_interior_float_selection_has_one_value_owner() {
    let names = (0..12).map(|i| format!("input{i}")).collect();
    let columns = (0..12).map(|_| Column::f64(vec![1.0; 1000])).collect();
    let frame = DataFrame::new(names, columns, None).unwrap();
    let (copy, calls, _) = measure(|| frame.rows(400, 500, &NoReplay));
    assert_eq!(copy.height(), 100);
    assert_eq!(copy.width(), 12);
    assert!(calls <= 6, "short copies should not allocate two buffers per input: {calls}");
}

#[test]
fn selecting_many_float_inputs_copies_their_bounded_history_together() {
    let names = ["open", "high", "low", "close", "volume", "ma"].map(str::to_owned).to_vec();
    let columns = (0..6).map(|_| Column::f64(vec![1.0; 1000])).collect();
    let mut frame = DataFrame::new(names, columns, None).unwrap();
    frame.set_computed("ma", std::sync::Arc::new(()), Reach { lookback: 19, resume: 20, continuation: 20, history: History::Window });
    let (tail, calls, _) = measure(|| frame.rows(900, 1000, &NoReplay));
    assert_eq!(tail.lead_rows(), 20);
    assert!(calls <= 20, "small selections should share one copied history allocation: {calls}");
}

#[test]
fn an_interior_copy_and_its_raw_history_share_their_value_owner() {
    let names = ["open", "high", "low", "close", "volume", "ma"].map(str::to_owned).to_vec();
    let columns = (0..6).map(|_| Column::f64(vec![1.0; 1000])).collect();
    let mut frame = DataFrame::new(names, columns, None).unwrap();
    frame.set_computed("ma", std::sync::Arc::new(()), Reach { lookback: 19, resume: 20, continuation: 20, history: History::Window });
    let (copy, calls, _) = measure(|| frame.rows(900, 950, &NoReplay));
    assert_eq!(copy.lead_rows(), 20);
    assert_eq!(copy.height(), 50);
    assert!(calls <= 6, "visible values and lead should not need separate owners: {calls}");
}

#[test]
fn a_single_shared_stale_snapshot_allocates_nothing() {
    let mut frame = DataFrame::new(vec!["x".into(), "ema".into()],
        vec![Column::f64(vec![1.0; 4]), Column::f64(vec![2.0; 4])], None).unwrap();
    frame.set_computed("ema", std::sync::Arc::new(()), Reach { lookback: 2, resume: 0, continuation: 2, history: History::Carried { operands: 0 } });
    frame.invalidate_computed_on_write("x");
    let (old, calls, _) = measure(|| frame.stale_computed_columns(None));
    assert_eq!(old.len(), 1);
    assert!(calls >= 2, "the compatibility API owns its name and Vec");
    let (mut shared, calls, _) = measure(|| frame.shared_stale_computed_columns(None));
    assert_eq!(calls, 0, "one shared snapshot needs neither a name allocation nor a Vec");
    drop(frame);
    assert_eq!(shared.next().unwrap().name.as_ref(), "ema");
    assert!(shared.next().is_none());
}

#[test]
fn a_live_checkpoint_allocates_one_shared_owner() {
    let point = Checkpoint { at: 7, state: None, raw: vec![] };
    let (shared, calls, _) = measure(|| CheckpointRef::from(point));
    assert_eq!(shared.at, 7);
    assert_eq!(calls, 1, "a live checkpoint needs no intermediate one-element Vec");
}

#[test]
fn a_computing_pass_pools_its_points_and_selections_allocate_nothing() {
    let anchors = Anchors::default();
    let points = (1..=2000).map(|k| Checkpoint { at: k * 512 - 1, state: None, raw: vec![] }).collect();
    let (_, calls, _) = measure(|| anchors.keep_all(points));
    // Independent immutable chunks replace the whole-pass owner. Two allocations
    // per chunk (Vec + Arc), plus the chunk index and memo lock publication.
    let chunks = 2000usize.div_ceil(volas_core::dataframe::ANCHOR_CHUNK_POINTS);
    assert!(calls <= 2 * chunks + 10, "publication allocated {calls} times for {chunks} chunks");
    let (selected, calls, _) = measure(|| anchors.nearest(2000 * 512 - 2));
    assert_eq!(selected.unwrap().at, 1999 * 512 - 1);
    assert_eq!(calls, 0);
}

#[test]
fn grid_publication_and_append_do_not_allocate_per_anchor_handles() {
    let anchors = Anchors::default();
    let batch = |start: usize| (start..start + 2000)
        .map(|k| Checkpoint { at: k * 512 - 1, state: None, raw: vec![] }).collect();
    let first = batch(1);
    let (_, _, bytes) = measure(|| anchors.keep_all(first));
    assert!(bytes < 2000 * 80, "publication should allocate chunks rather than per-point handles: {bytes} bytes");
    let next = batch(2001);
    let (_, _, bytes) = measure(|| anchors.keep_all(next));
    assert!(bytes < 2000 * 80, "grid append should allocate only new chunks: {bytes} bytes");
    assert_eq!(anchors.nearest(2000 * 512).unwrap().at, 2000 * 512 - 1);
    assert_eq!(anchors.nearest(4000 * 512).unwrap().at, 4000 * 512 - 1);
}

#[test]
fn bounded_history_metadata_fits_the_original_hot_column_budget() {
    assert!(std::mem::size_of::<volas_core::dataframe::ComputedMeta>() <= 272,
        "chunk descriptors must not enlarge every cached column: {} bytes",
        std::mem::size_of::<volas_core::dataframe::ComputedMeta>());
}

#[test]
fn first_publication_has_one_lazy_owner_and_one_chunk_index() {
    let platform_lock = std::sync::Mutex::new(());
    let (_, platform_calls, _) = measure(|| drop(platform_lock.lock().unwrap()));
    let anchors = Anchors::default();
    let point = CheckpointRef::from(Checkpoint { at: 511, state: None, raw: vec![] });
    let (_, calls, _) = measure(|| anchors.keep(point));
    assert!(calls <= platform_calls + 2,
        "first publication needs only its owner and chunk index beyond the platform lock: {calls} allocations, platform {platform_calls}");
}

#[test]
fn promoting_a_pending_boundary_allocates_nothing_and_clears_only_its_loss() {
    let mut frame = DataFrame::new(vec!["close".into(), "ema".into()],
        vec![Column::f64(vec![1.0; 2]); 2], None).unwrap();
    frame.set_computed("ema", std::sync::Arc::new(()), Reach {
        lookback: 0, resume: 0, continuation: 0, history: History::Carried { operands: 0 },
    });
    let mut frame = frame.rows(1, 2, &NoReplay);
    frame.set_computed_checkpoint("ema", Err(Some(volas_core::VolasError::DType("old loss".into()))));
    let state = volas_core::StateWords::from(vec![-0.0, f64::from_bits(0x7ff8000000000123)]);
    let words = state.as_ptr();
    frame.set_computed_pending("ema", Some(Checkpoint { at: 0, state: Some(state), raw: vec![] }));
    assert!(!frame.promote_computed_pending("missing", 0));
    assert!(!frame.promote_computed_pending("ema", 1));
    assert!(frame.computed_history_lost("ema").is_some());
    assert!(frame.has_computed_pending("ema", 0));
    let (_, calls, bytes) = measure(|| assert!(frame.promote_computed_pending("ema", 0)));
    assert_eq!((calls, bytes), (0, 0), "a captured checkpoint must not be re-boxed at promotion");
    assert!(frame.computed_history_lost("ema").is_none());
    assert!(!frame.has_computed_pending("ema", 0));
    let state = frame.computed_checkpoint("ema").unwrap().state.as_ref().unwrap();
    assert_eq!(state.as_ptr(), words);
    assert_eq!(state[0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(state[1].to_bits(), 0x7ff8000000000123);
}

#[test]
fn large_float_row_selections_do_not_allocate_per_column() {
    let names = (0..12).map(|i| format!("input{i}")).collect();
    let columns = (0..12).map(|_| Column::f64(vec![1.0; 1_000_000])).collect();
    let mut frame = DataFrame::new(names, columns, None).unwrap();
    for i in 5..12 {
        frame.set_computed(&format!("input{i}"), std::sync::Arc::new(()),
            Reach { lookback: 19, resume: 20, continuation: 20, history: History::Window });
    }
    for (start, end) in [(250_000, 500_000), (750_000, 1_000_000)] {
        let (mut copy, calls, bytes) = measure(|| frame.rows(start, end, &NoReplay));
        assert_eq!((copy.height(), copy.width(), copy.lead_rows()), (250_000, 12, 20));
        assert!(calls <= 8, "selection {start}..{end} must not allocate per column: {calls}");
        assert!(bytes < 24_010_000, "copy only visible values, bounded lead and metadata: {bytes}");
        copy.set_column("input0", Column::f64(vec![2.0; 250_000])).unwrap();
        assert_eq!(frame.column("input0").unwrap().as_f64().unwrap()[start], 1.0);
        assert_eq!(copy.column("input1").unwrap().as_f64().unwrap()[0], 1.0);
    }
}

#[test]
fn a_compacted_float_window_owns_writable_columns_until_its_next_drop() {
    for narrow in [false, true] {
        let raw = |len, value: f64| if narrow { Column::f32(vec![value as f32; len]) }
            else { Column::f64(vec![value; len]) };
        let mut frame = DataFrame::new(vec!["close".into(), "ema".into()],
            vec![raw(32, 1.0), Column::f64(vec![2.0; 32])], None).unwrap();
        frame.set_computed("ema", std::sync::Arc::new(()),
            Reach { lookback: 11, resume: 0, continuation: 11, history: History::Carried { operands: 0 } });
        frame.set_computed_states("ema", Some(vec![1.0]), Some(vec![1.0]));
        frame.set_computed_checkpoint("ema", Ok(Checkpoint { at: 15, state: Some(vec![1.0].into()), raw: vec![] }));
        let (mut kept, retain_calls, retain_bytes) = measure(|| frame.retain_tail(16, &NoReplay));
        let borrowed = ["close", "ema"].iter().filter(|name| {
            matches!(kept.column(name).unwrap(), Column::F64(values) if values.is_borrowed())
        }).count();
        let row = DataFrame::new(vec!["close".into()], vec![raw(1, 3.0)], None).unwrap();
        let (_, calls, bytes) = measure(|| {
            for _ in 0..16 { kept.append(&row).unwrap(); }
        });
        eprintln!("live compact f32={narrow}: retain {retain_calls}/{retain_bytes} B, append {calls}/{bytes} B");
        assert!(retain_calls + calls <= 7, "the whole compaction cycle must not move its allocations elsewhere");
        assert_eq!((borrowed, calls), (0, 0),
            "live retention must not copy its slab into mutable columns or grow them again: {bytes} B");
        assert_eq!(kept.height(), 32);
        assert_eq!(kept.column("close").unwrap().to_f64_vec()[16], 3.0);
        assert_eq!(frame.column("close").unwrap().to_f64_vec()[16], 1.0);
    }
}

#[test]
fn live_retention_keeps_the_recycled_resume_buffer() {
    let mut frame = DataFrame::new(vec!["close".into(), "ema".into()],
        vec![Column::f64(vec![1.0; 32]); 2], None).unwrap();
    frame.set_computed("ema", std::sync::Arc::new(()),
        Reach { lookback: 0, resume: 0, continuation: 0, history: History::Carried { operands: 0 } });
    frame.set_computed_states("ema", Some(vec![2.0]), Some(vec![1.0]));
    let row = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![3.0])], None).unwrap();
    let step = |_: &DataFrame, old: &[f64], _: usize, next: &mut [f64]| {
        next[0] = old[0] + 1.0;
        Some(next[0])
    };
    frame.append(&row).unwrap();
    let value = frame.step_computed_state("ema", step).unwrap();
    frame.update_computed_value("ema", 32, value).unwrap();
    let mut kept = frame.retain_tail(16, &NoReplay);
    drop(frame); // compaction replaces its source, making the recycled buffer unique
    kept.append(&row).unwrap();
    let (value, calls, bytes) = measure(|| kept.step_computed_state("ema", step).unwrap());
    assert_eq!(value, 4.0);
    assert_eq!((calls, bytes), (0, 0), "retention must not discard the live resume scratch");
}

#[test]
fn live_compaction_reuses_unique_float_column_capacity() {
    for narrow in [false, true] {
        let column = || if narrow { Column::f32((0..32).map(|i| i as f32).collect()) }
            else { Column::f64((0..32).map(|i| i as f64).collect()) };
        let mut frame = DataFrame::new(vec!["close".into(), "open".into()],
            vec![column(), column()], None).unwrap();
        let (_, calls, bytes) = measure(|| frame.compact_tail(16, &NoReplay));
        assert_eq!(frame.height(), 16);
        assert_eq!(frame.column("close").unwrap().to_f64_vec()[0], 16.0);
        assert_eq!((calls, bytes), (0, 0), "unique live columns and range index need no new owner");
    }
}

#[test]
fn live_compaction_bounds_shared_borrowed_and_oversized_owners() {
    for size in [32, 65_536] {
        for (borrowed, shared) in [(false, false), (false, true), (true, false), (true, true)] {
            let column = || Column::f64((0..size).map(|i| if i == 0 { f64::NAN } else { i as f64 }).collect());
            let mut frame = DataFrame::new(vec!["close".into(), "open".into()],
                vec![column(), column()], None).unwrap();
            assert!(frame.column("close").unwrap().has_nan());
            if borrowed { frame = frame.rows(0, size, &NoReplay); }
            let parent = shared.then(|| frame.clone());
            let (_, _, bytes) = measure(|| frame.compact_tail(16, &NoReplay));
            assert!(bytes < 4096, "copy only the bounded retained range: {bytes}");
            assert_eq!(frame.column("close").unwrap().to_f64_vec()[0], (size - 16) as f64);
            assert!(!frame.column("close").unwrap().has_nan());
            assert!(!frame.column("close").unwrap().has_nonfinite());
            if let Some(parent) = &parent {
                assert!(parent.column("close").unwrap().to_f64_vec()[0].is_nan());
            }
            drop(parent);
            frame.compact_tail(16, &NoReplay);
            assert_eq!(frame.height(), 16);
            frame.compact_tail(0, &NoReplay);
            assert_eq!(frame.height(), 0);
        }
    }
    let mut ints = DataFrame::new(vec!["x".into()], vec![Column::i64((0..32).collect())], None).unwrap();
    let mut retained = ints.retain_tail(16, &NoReplay);
    assert_eq!((retained.origin(), retained.height()), (16, 16));
    assert_eq!(retained.column("x").unwrap().to_f64_vec(), (16..32).map(f64::from).collect::<Vec<_>>());
    retained.set_column("x", Column::i64(vec![99; 16])).unwrap();
    assert_eq!(ints.column("x").unwrap().to_f64_vec()[16], 16.0);
    ints.compact_tail(16, &NoReplay);
    assert_eq!(ints.column("x").unwrap().to_f64_vec()[0], 16.0);
}
