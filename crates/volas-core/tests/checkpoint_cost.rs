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
    assert!(calls <= 3, "batch points need shared containers, not {calls} per-point allocations");
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
    assert!(bytes < 4096, "publication should retain its bank rather than allocate {bytes} handle bytes");
    let next = batch(2001);
    let (_, _, bytes) = measure(|| anchors.keep_all(next));
    assert!(bytes < 4096, "grid append should retain batches without flattening {bytes} handle bytes");
    assert_eq!(anchors.nearest(2000 * 512).unwrap().at, 2000 * 512 - 1);
    assert_eq!(anchors.nearest(4000 * 512).unwrap().at, 4000 * 512 - 1);
}
