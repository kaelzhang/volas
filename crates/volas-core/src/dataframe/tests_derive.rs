//! The one derivation owner: what a contiguous selection, a projection and a new series
//! keep of their source.

use crate::column::ColumnData;
use std::collections::HashMap;

use super::derive::NoReplay;
use super::*;
use crate::tz::Tz;

/// A replay that has the history: the state of a checkpoint it gives names its row.
struct Replayed;

impl Replay for Replayed {
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> { None }

    fn boundary(&self, frame: &DataFrame, _: &str, start: usize) -> Option<Checkpoint> {
        let row = (frame.origin() + start).checked_sub(1)?;
        Some(Checkpoint { at: row, state: Some(vec![row as f64].into()), raw: vec![] })
    }
}

/// `rows` rows of a data column `close` (its values the row numbers) and two cached
/// columns: `ma`, a trailing window of 3 rows, and `ema`, a recursion carrying its state
/// as of the last row (`[rows - 1]`) and the row before.
fn frame(rows: usize) -> DataFrame {
    let values: Vec<f64> = (0..rows).map(|i| i as f64).collect();
    let mut df = DataFrame::new(
        vec!["close".into(), "ma".into(), "ema".into()],
        vec![Column::f64(values.clone()), Column::f64(values.clone()), Column::f64(values)],
        None,
    )
    .unwrap();
    df.set_computed("ma", std::sync::Arc::new(()), Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
    df.set_computed("ema", std::sync::Arc::new(()), Reach { lookback: 4, resume: 0, continuation: 4, history: History::Carried { operands: 0 } });
    let last = rows as f64 - 1.0;
    df.set_computed_states("ema", Some(vec![last]), Some(vec![last - 1.0]));
    df
}

fn meta(df: &DataFrame, name: &str) -> ComputedMeta {
    df.computed_columns().into_iter().find(|(n, _)| n == name).unwrap().1
}

/// The lead's rows of column `name` (missing where it does not hold it).
fn lead_of(df: &DataFrame, name: &str) -> Vec<f64> {
    let n = df.lead_rows();
    df.led(&df.select(&[name.to_string()]).unwrap(), n).column(name).unwrap().to_f64_vec()[..n].to_vec()
}

fn lead(df: &DataFrame) -> Vec<f64> {
    lead_of(df, "close")
}

#[test]
fn a_multi_column_float_lead_keeps_copied_history_after_writes_and_drops() {
    // An empty selection still needs its raw lead for a future continuation.
    for rows in [0, 5] {
        let mut df = frame(100);
        df.set_column("open", Column::f64((0..100).map(|i| 1000.0 + i as f64).collect())).unwrap();
        df.set_computed_checkpoint("ema", Ok(Checkpoint { at: 89, state: Some(vec![89.0].into()), raw: vec![] }));
        let mut tail = df.rows(90, 90 + rows, &NoReplay);
        assert_eq!(tail.height(), rows);
        assert_eq!(tail.lead_rows(), 3);
        assert_eq!(lead(&tail), vec![87.0, 88.0, 89.0]);
        assert_eq!(lead_of(&tail, "open"), vec![1087.0, 1088.0, 1089.0]);
        df.set_column("open", Column::f64(vec![0.0; 100])).unwrap();
        drop(df);
        tail.set_column("open", Column::f64(vec![7.0; rows])).unwrap();
        let dropped = tail.drop_columns(&["close".into()]).unwrap();
        assert_eq!(lead_of(&dropped, "open"), vec![1087.0, 1088.0, 1089.0]);
        assert_eq!(dropped.column("open").unwrap().to_f64_vec(), vec![7.0; rows]);
    }
}

#[test]
fn interior_float_copies_keep_column_nan_facts_and_independent_writes() {
    let mut df = DataFrame::new(vec!["clean".into(), "mixed".into()],
        vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0]), Column::f64(vec![f64::NAN, 5.0, 6.0, 7.0])], None).unwrap();
    assert!(!df.column("clean").unwrap().has_nan());
    assert!(df.column("mixed").unwrap().has_nan());
    let mut copy = df.rows(1, 3, &NoReplay);
    assert_eq!(copy.column("clean").unwrap().known_has_nan(), Some(false));
    assert_eq!(copy.column("mixed").unwrap().known_has_nan(), None);
    assert!(!copy.column("mixed").unwrap().has_nan());
    assert_eq!(copy.column("mixed").unwrap().known_has_nan(), Some(false));
    df.set_column("mixed", Column::f64(vec![9.0; 4])).unwrap();
    copy.check_writes(vec![CellWrite { col: 0, rows: vec![0], values: Column::f64(vec![8.0]) }]).unwrap().land();
    assert_eq!(copy.column("clean").unwrap().to_f64_vec(), vec![8.0, 3.0]);
    assert_eq!(copy.column("mixed").unwrap().to_f64_vec(), vec![5.0, 6.0]);
    assert_eq!(df.column("clean").unwrap().to_f64_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn an_interior_float_copy_releases_its_source_and_memoizes_only_its_own_bytes() {
    let owner = Arc::new(vec![1.0, 2.0, 3.0, 4.0]);
    let weak = Arc::downgrade(&owner);
    let buffer = crate::buffer::Buffer::from_shared_range(Arc::clone(&owner), 0..4).unwrap();
    let mut df = DataFrame::new(vec!["a".into(), "b".into()], vec![Column { data: ColumnData::F64(buffer) }, Column::f64(vec![0.0; 4])], None).unwrap();
    drop(owner);
    let copy = df.rows(1, 3, &NoReplay);
    assert_eq!(copy.column("a").unwrap().known_has_nan(), None);
    assert!(!copy.column("a").unwrap().has_nan());
    df.check_writes(vec![CellWrite { col: 0, rows: vec![1], values: Column::f64(vec![f64::NAN]) }]).unwrap().land();
    assert_eq!(copy.column("a").unwrap().known_has_nan(), Some(false));
    assert_eq!(copy.column("a").unwrap().to_f64_vec(), vec![2.0, 3.0]);
    drop(df);
    assert!(weak.upgrade().is_none(), "the copied slab does not pin its source allocation");
}

#[test]
fn a_joint_visible_and_lead_slab_preserves_float_bits_and_separate_nan_facts() {
    let payload = f64::from_bits(0x7ff8_0000_0000_0073);
    let mut x = vec![1.0; 100];
    x[88] = payload;
    x[91] = -0.0;
    let mut y = vec![2.0; 100];
    y[91] = payload;
    let mut df = DataFrame::new(vec!["x".into(), "y".into(), "ma".into()],
        vec![Column::f64(x), Column::f64(y), Column::f64(vec![0.0; 100])], None).unwrap();
    assert!(df.column("x").unwrap().has_nan());
    df.set_computed("ma", Arc::new(()), Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
    let copied = df.rows(90, 95, &NoReplay);
    assert_eq!(copied.lead_rows(), 3);
    assert_eq!(lead_of(&copied, "x")[1].to_bits(), payload.to_bits());
    assert_eq!(copied.column("x").unwrap().as_f64().unwrap()[1].to_bits(), (-0.0f64).to_bits());
    assert_eq!(copied.column("y").unwrap().as_f64().unwrap()[1].to_bits(), payload.to_bits());
    assert!(!copied.column("x").unwrap().has_nan());
    assert!(copied.column("y").unwrap().has_nan());
    assert_eq!(copied.column("x").unwrap().known_has_nan(), Some(false));
    assert_eq!(copied.column("y").unwrap().known_has_nan(), Some(true));
}

#[test]
fn a_prefix_copy_writes_only_its_visible_range_and_releases_the_source() {
    let mut df = DataFrame::new(vec!["x".into(), "y".into(), "ma".into()],
        vec![Column::f64((0..100).map(f64::from).collect()), Column::f64(vec![2.0; 100]), Column::f64(vec![0.0; 100])], None).unwrap();
    let Column { data: ColumnData::F64(source) } = df.column("x").unwrap() else { unreachable!() };
    let guard = source.keepalive();
    let weak = Arc::downgrade(&guard);
    drop(guard);
    assert!(!source.has_nan());
    df.set_computed("ma", Arc::new(()), Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
    let mut copied = df.rows(90, 95, &NoReplay);
    assert_eq!(copied.column("x").unwrap().known_has_nan(), Some(false));
    assert_eq!(lead_of(&copied, "x"), vec![87.0, 88.0, 89.0]);
    df.check_writes(vec![CellWrite { col: 0, rows: vec![90], values: Column::f64(vec![f64::NAN]) }]).unwrap().land();
    copied.check_writes(vec![CellWrite { col: 0, rows: vec![0], values: Column::f64(vec![8.0]) }]).unwrap().land();
    assert_eq!(lead_of(&copied, "x"), vec![87.0, 88.0, 89.0]);
    assert_eq!(copied.column("x").unwrap().to_f64_vec(), vec![8.0, 91.0, 92.0, 93.0, 94.0]);
    assert_eq!(copied.column("y").unwrap().to_f64_vec(), vec![2.0; 5]);
    drop(df);
    assert!(weak.upgrade().is_none(), "the prefix copy keeps no source values allocation");
}

#[test]
fn a_tail_keeps_every_cached_column_with_its_states_and_lead() {
    let df = frame(10);
    let tail = df.rows(7, 10, &NoReplay);
    assert_eq!((tail.origin(), tail.height()), (7, 3));
    assert!(Arc::ptr_eq(df.computed_directive("ema").unwrap(), tail.computed_directive("ema").unwrap()));
    let view = tail.view_rows(1, 3);
    assert!(Arc::ptr_eq(df.computed_directive("ema").unwrap(), view.computed_directive("ema").unwrap()));
    assert_eq!((meta(&tail, "ma").valid_rows, meta(&tail, "ema").valid_rows), (3, 3));
    assert_eq!((meta(&tail, "ema").state.as_ref().map(|words| words.to_vec()), meta(&tail, "ema").state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), Some(vec![8.0])));
    assert_eq!(tail.computed_resume_state("ema"), Some((&[9.0][..], 7)));
    // the lead: ema, a recursion carried by its state whose history starts within
    // ANCHOR_SPACING rows before, replays from its start — every row of it; ma's reach (3)
    // is among them
    assert_eq!(lead(&tail), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(tail.lead_rows(), 7);
    assert_eq!(tail.led(&tail, 4).origin(), 3);
    assert!(lead_of(&tail, "ma").iter().all(|v| v.is_nan())); // data rows only
    // a one-row tail keeps no state before its only row
    let one = df.rows(9, 10, &NoReplay);
    assert_eq!((meta(&one, "ema").state.as_ref().map(|words| words.to_vec()), meta(&one, "ema").state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), None));
}

#[test]
fn a_recursion_without_a_state_keeps_its_whole_history() {
    let mut df = frame(10);
    df.set_column("kcw", Column::f64(vec![0.0; 10])).unwrap();
    df.set_computed("kcw", std::sync::Arc::new(()), Reach { lookback: 1, resume: 0, continuation: 1, history: History::Carried { operands: 0 } });
    let tail = df.rows(6, 10, &NoReplay);
    assert_eq!(lead(&tail), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    // a short lead of a long frame is a copy; one of most of its rows shares its buffers
    let again = tail.rows(3, 4, &NoReplay);
    assert_eq!(lead(&again), (0..9).map(f64::from).collect::<Vec<_>>());
    let mut stale = df.clone();
    stale.invalidate_computed_on_write("close"); // every state dropped
    assert_eq!(stale.rows(8, 10, &NoReplay).lead_rows(), 8);
    // a recursion replayed from a checkpoint needs only its reach before the rows
    let mut kept = frame(10);
    kept.set_computed_checkpoint("ema", Ok(Checkpoint { at: 7, state: Some(vec![7.0].into()), raw: vec![] }));
    kept.invalidate_computed_on_write("close");
    assert_eq!(kept.rows(8, 10, &NoReplay).lead_rows(), 3);
}

#[test]
fn a_lead_keeps_its_history_whatever_dtype_a_column_takes() {
    let mut df = frame(10);
    df.set_column("close", Column::f64((0..10).map(|i| i as f64 + 0.123456789).collect())).unwrap();
    df.set_computed("ma", std::sync::Arc::new(()), Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
    let tail = df.rows(7, 10, &NoReplay);
    // a whole column of another dtype: the history and the new values read in their
    // lossless common dtype — never rounded, never missing
    for col in [Column::i64(vec![7, 8, 9]), Column::f32(vec![7.5, 8.5, 9.5])] {
        let mut written = tail.clone();
        written.set_column("close", col).unwrap();
        let led = written.led(&written.select(&["close".to_string()]).unwrap(), 3);
        let close = led.column("close").unwrap();
        assert_eq!((close.dtype(), close.to_f64_vec()[0]), (crate::DType::F64, 4.123456789));
    }
    // a dtype the history cannot be read in is refused before anything is written
    let mut text = tail.clone();
    let before = text.column("close").unwrap().clone();
    assert!(matches!(text.set_column("close", Column::str(vec!["a".into(); 3])), Err(VolasError::DType(_))));
    assert!(text.column("close").unwrap().equals(&before));
}

/// The lead holds only what the cached columns read now: dropping the one that needed
/// the whole history releases the rest, copied out of the buffers it no longer is most of.
#[test]
fn a_lead_shrinks_with_the_columns_that_read_it() {
    let mut df = frame(100);
    df.set_column("kcw", Column::f64(vec![0.0; 100])).unwrap();
    df.set_computed("kcw", std::sync::Arc::new(()), Reach { lookback: 1, resume: 0, continuation: 1, history: History::Carried { operands: 0 } });
    // ema replays from its checkpoint at row 89, which the tail holds the row after
    df.set_computed_checkpoint("ema", Ok(Checkpoint { at: 89, state: Some(vec![89.0].into()), raw: vec![] }));
    let tail = df.rows(90, 100, &NoReplay);
    assert_eq!(tail.lead_rows(), 90);
    // a column removal keeps the others cached, its lead trimmed to what they read
    let dropped = tail.drop_columns(&["kcw".to_string(), "nope".to_string()]).unwrap();
    assert_eq!((dropped.lead_rows(), dropped.is_computed("ema"), dropped.has_column("kcw")), (3, true, false));
    assert_eq!(lead(&dropped), vec![87.0, 88.0, 89.0]);
    assert_eq!(dropped.computed_checkpoint("ema").map(|cp| cp.at), Some(89));
    // a column that stops being cached stops reading before row 0
    let mut written = tail.clone();
    written.invalidate_computed_on_write("kcw");
    assert_eq!(written.lead_rows(), 3);
    assert_eq!(lead(&written), vec![87.0, 88.0, 89.0]);
    // a copied lead shrinks in place while it stays most of its rows; none is left once no
    // cached column reads before row 0
    // Only the rows after ema's checkpoint (90..95); its kernel tails are in state
    let mut short = df.rows(95, 100, &NoReplay);
    short.invalidate_computed_on_write("kcw");
    assert_eq!(short.lead_rows(), 5);
    let mut ma = short.clone();
    ma.invalidate_computed_on_write("ema");
    assert_eq!(lead(&ma), vec![92.0, 93.0, 94.0]);
    // a copied one that stays most of its rows keeps sharing them
    let mut window = frame(10).rows(4, 10, &NoReplay);
    window.invalidate_computed_on_write("ema");
    assert_eq!(lead(&window), vec![1.0, 2.0, 3.0]);
    let mut bare = ma.clone();
    bare.invalidate_computed_on_write("ma");
    assert_eq!(bare.lead_rows(), 0);
    // a data column the lead does not hold takes any dtype
    let mut added = ma.clone();
    added.set_column("x", Column::f64(vec![0.0; 5])).unwrap();
    added.set_column("x", Column::str(vec!["a".into(); 5])).unwrap();
}

/// Copies and selections share only their reachable immutable chunks and
/// publish independently, including future rows after append. Input writes drop it.
#[test]
fn anchors_follow_the_rows_and_a_write_drops_them() {
    let mut df = frame(10);
    let anchors = df.computed_anchors("ema").unwrap();
    for at in [2, 5, 8, 5] {
        anchors.keep(Checkpoint { at, state: Some(vec![at as f64].into()), raw: vec![] });
    }
    assert_eq!(anchors.len(), 3);
    assert_eq!(anchors.nearest(6).map(|cp| cp.at), Some(5));
    assert_eq!(anchors.nearest(1), None);
    let copied = df.clone().computed_anchors("ema").unwrap();
    assert_ne!(copied, anchors);
    assert!(std::ptr::eq(&*copied.nearest(5).unwrap(), &*anchors.nearest(5).unwrap()));
    let sub = df.rows(4, 9, &NoReplay);
    let shared = sub.computed_anchors("ema").unwrap();
    assert_eq!((shared.len(), shared.nearest(9).map(|cp| cp.at)), (1, Some(5)));
    assert_ne!(shared, anchors);
    // A selection owns its reachable anchors independently; a publication cannot
    // make the other frame retain new history, including future publications.
    for at in [6, 8] {
        shared.keep(Checkpoint { at, state: None, raw: vec![] });
    }
    assert_eq!((anchors.len(), shared.len()), (3, 3));
    df.invalidate_computed_on_write("close");
    assert_eq!(df.computed_anchors("ema").unwrap().len(), 0);
    assert!(df.computed_anchors("close").is_none());
}

#[test]
fn a_selection_from_the_history_start_needs_no_lead() {
    let df = frame(10);
    assert!(df.rows(0, 10, &NoReplay).lead.is_none());
    // a lead never reaches before the history it serves
    assert_eq!(lead(&df.rows(2, 10, &NoReplay)), vec![0.0, 1.0]);
}

#[test]
fn a_middle_selection_carries_no_states_and_replays_nothing() {
    // its states as of its last row are computed when the column continues, from what it
    // keeps: its valid rows, the nearest point of the history before it and the lead
    let df = frame(10);
    let counting = Counting(std::cell::Cell::new(0));
    let mid = df.rows(3, 6, &counting);
    assert_eq!((meta(&mid, "ema").state.as_ref().map(|words| words.to_vec()), meta(&mid, "ema").state_before_last.as_ref().map(|words| words.to_vec())), (None, None));
    assert_eq!((mid.computed_resume_state("ema"), meta(&mid, "ema").valid_rows), (None, 3));
    assert_eq!((meta(&mid, "ma").state.as_ref().map(|words| words.to_vec()), counting.0.get()), (None, 0));
    assert_eq!(mid.lead_rows(), 3); // ema's whole history before it
    // The lazy interface is invoked only when continuation is requested; a replay
    // without an executable history declines rather than fabricate end states.
    assert_eq!(counting.continuation(&mid, "ema", mid.height()), None);
    assert_eq!(counting.0.get(), 1);
    assert_eq!(NoReplay.continuation(&mid, "ema", mid.height()), None);
    assert_eq!(Replayed.continuation(&mid, "ema", mid.height()), None);
}

#[test]
fn a_stale_source_keeps_its_valid_rows_and_its_state_where_it_is() {
    let mut df = frame(10);
    df.append(&DataFrame::new(vec!["close".into()], vec![Column::f64(vec![10.0, 11.0])], None).unwrap()).unwrap();
    let tail = df.rows(8, 12, &NoReplay);
    assert_eq!(meta(&tail, "ema").valid_rows, 2);
    assert_eq!(tail.computed_resume_state("ema"), Some((&[9.0][..], 8)));
    // a selection after the last valid row has none valid and no state
    let after = df.rows(10, 12, &NoReplay);
    assert_eq!((meta(&after, "ema").valid_rows, meta(&after, "ema").state.as_ref().map(|words| words.to_vec())), (0, None));
    assert!(after.has_stale_computed());
}

#[test]
fn a_selection_of_a_selection_takes_its_lead_from_both() {
    let df = frame(10);
    let tail = df.rows(6, 10, &NoReplay); // lead rows 0..6 (ema's whole history)
    let again = tail.rows(1, 4, &NoReplay); // rows 7..10: lead rows 0..7
    assert_eq!(again.origin(), 7);
    assert_eq!(lead(&again), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // a data column the source's lead does not hold is missing there
    let mut wide = tail.clone();
    wide.set_column("x", Column::f64(vec![1.0; 4])).unwrap();
    let led = wide.rows(1, 4, &NoReplay);
    let x = lead_of(&led, "x");
    assert!(x[..6].iter().all(|v| v.is_nan()) && x[6] == 1.0);
}

#[test]
fn the_led_frame_puts_the_lead_rows_before_row_0() {
    let df = frame(10);
    let tail = df.rows(7, 10, &NoReplay);
    let data = tail.select(&["close".to_string()]).unwrap();
    let led = tail.led(&data, 2);
    assert_eq!((led.origin(), led.height()), (5, 5));
    assert_eq!(led.column("close").unwrap().to_f64_vec(), vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    // more than it holds: all of it; a column it does not hold is missing
    let led = tail.led(&tail, 9);
    assert_eq!(led.height(), 10);
    assert!(led.column("ma").unwrap().to_f64_vec()[0].is_nan());
    // a frame without a lead is itself
    assert!(df.led(&df, 3).equals(&df));
    assert_eq!(df.lead_rows(), 0);
}

#[test]
fn checkpoints_follow_the_rows_they_cover() {
    let mut df = frame(10);
    let cp = Checkpoint { at: 4, state: Some(vec![4.0].into()), raw: vec![] };
    df.set_computed_checkpoint("ema", Ok(cp.clone()));
    df.set_computed_pending("ema", Some(Checkpoint { at: 7, ..cp.clone() }));
    let kept = df.rows(5, 10, &NoReplay);
    assert_eq!(kept.computed_checkpoint("ema"), Some(&cp));
    assert!(kept.has_computed_pending("ema", 7));
    // a selection after it keeps it too, the rows after it in its lead
    let later = df.rows(6, 10, &NoReplay);
    assert_eq!(later.computed_checkpoint("ema"), Some(&cp));
    assert_eq!(later.lead_rows(), 3);
    assert!(df.rows(9, 10, &NoReplay).take_computed_pending("ema", 7).is_none());
}

/// A selection keeps the nearest point of a recursion's history before it — an anchor, a
/// checkpoint or the history's start — with the rows after it in its lead, when it is
/// within ANCHOR_SPACING rows; a farther one is replayed to the selection's boundary now,
/// and a source that does not hold it keeps none.
#[test]
fn a_selection_keeps_the_nearest_point_of_a_history_within_the_anchor_spacing() {
    let n = ANCHOR_SPACING + 20;
    let df = frame(n);
    let first = ANCHOR_SPACING + 10;
    // the history's start is farther: the boundary is replayed, and the lead is the reach
    let far = df.rows(first, n, &Replayed);
    assert_eq!(far.computed_checkpoint("ema").map(|cp| cp.at), Some(first - 1));
    assert_eq!(far.lead_rows(), 3);
    assert_eq!(df.rows(first, n, &NoReplay).computed_checkpoint("ema"), None);
    // an anchor within it is kept, the rows after it in the lead
    let at = ANCHOR_SPACING - 1;
    df.computed_anchors("ema").unwrap().keep(Checkpoint { at, state: Some(vec![at as f64].into()), raw: vec![] });
    let near = df.rows(first, n, &NoReplay);
    assert_eq!(near.computed_checkpoint("ema").map(|cp| cp.at), Some(at));
    assert_eq!(near.lead_rows(), first - at - 1);
    // a checkpoint later than the anchor is the nearer
    let mut own = df.clone();
    own.set_computed_checkpoint("ema", Ok(Checkpoint { at: first - 3, state: None, raw: vec![] }));
    let nearer = own.rows(first, n, &NoReplay);
    assert_eq!(nearer.computed_checkpoint("ema").map(|cp| cp.at), Some(first - 3));
    assert_eq!(nearer.lead_rows(), 3);
}

#[test]
fn rows_by_position_continue_only_when_contiguous_in_order() {
    let df = frame(10);
    let run = df.select_rows(&[4, 5, 6], &NoReplay).unwrap();
    assert_eq!((run.origin(), run.is_computed("ema")), (4, true));
    for positions in [&[4usize, 6][..], &[5, 4], &[]] {
        let new = df.select_rows(positions, &NoReplay).unwrap();
        assert_eq!(new.origin(), 0);
        assert_eq!(new.names(), &["close".to_string()]);
        assert!(new.lead.is_none());
    }
    assert!(df.select_rows(&[4, 4], &NoReplay).is_err());
    let mask = df.filter_mask(&[false, true, true, false, false, false, false, false, false, false], &NoReplay).unwrap();
    assert_eq!((mask.origin(), mask.height(), mask.is_computed("ma")), (1, 2, true));
    // a read of the rows keeps every column's values, none cached
    let read = df.take(&[2, 9]).unwrap();
    assert_eq!((read.width(), read.computed_columns().len()), (3, 0));
}

#[test]
fn a_projection_keeps_the_rows_and_the_listed_columns_as_data() {
    // a listed cached column keeps its values as plain data, never continued (R40); an
    // unlisted one is dropped; the frame keeps no history before its rows
    let tail = frame(10).rows(7, 10, &NoReplay);
    let p = tail.select(&["ema".to_string(), "close".to_string()]).unwrap();
    assert_eq!((p.origin(), p.is_computed("ema"), p.has_column("ma")), (7, false, false));
    assert!(p.column("ema").unwrap().equals(tail.column("ema").unwrap()));
    assert!(p.computed_columns().is_empty() && p.lead.is_none());
    assert!(tail.select(&["nope".to_string()]).is_err());
    let twice = tail.select(&["close".to_string(), "close".to_string()]).unwrap_err();
    assert!(twice.to_string().contains("duplicate column name \"close\""));
}

#[test]
fn every_other_derivation_is_a_new_series_of_the_data_columns() {
    let tail = frame(10).rows(7, 10, &NoReplay);
    let data = ["close".to_string()];
    let renamed = tail.rename(&HashMap::from([("close".into(), "px".into()), ("ema".into(), "e".into())])).unwrap();
    assert_eq!((renamed.names(), renamed.origin()), (&["px".to_string()][..], 0));
    let cast = tail.astype(&HashMap::from([("close".into(), crate::DType::F32)])).unwrap();
    assert_eq!((cast.names(), cast.columns()[0].dtype()), (&data[..], crate::DType::F32));
    assert!(tail.astype(&HashMap::from([("ema".into(), crate::DType::F32)])).is_err());
    let reset = tail.reset_index(Some("i")).unwrap();
    assert_eq!(reset.names(), &["i".to_string(), "close".to_string()]);
    assert_eq!(reset.column("i").unwrap().as_i64().unwrap(), &[7, 8, 9]);
    assert!(tail.reset_index(Some("close")).is_err());
    assert_eq!(tail.reset_index(None).unwrap().names(), &data[..]);
    let indexed = reset.set_index("i").unwrap();
    assert_eq!((indexed.names(), indexed.index().name()), (&data[..], Some("i")));
    assert!(tail.set_index("ema").is_err());
    assert!(tail.set_index("nope").is_err());
    assert!(tail.with_labels(&Column::i64(vec![1, 2]), Tz::Naive, None).is_err());
    for df in [renamed, cast, reset, indexed] {
        assert!(df.computed_columns().is_empty() && df.lead.is_none());
    }
}

#[test]
fn a_time_zone_change_is_a_new_series() {
    let mut df = DataFrame::new(
        vec!["close".into(), "ma".into()],
        vec![Column::f64(vec![1.0, 2.0]); 2],
        Some(Index::new(None, crate::IndexKind::Datetime(vec![0, 60_000_000_000], Tz::Naive)).unwrap()),
    )
    .unwrap();
    df.set_computed("ma", std::sync::Arc::new(()), Reach { lookback: 1, resume: 0, continuation: 1, history: History::Window });
    let utc = df.tz_localize(Tz::Utc).unwrap();
    assert_eq!((utc.names(), utc.index().tz()), (&["close".to_string()][..], Tz::Utc));
    assert!(df.tz_convert(Tz::Utc).is_err()); // naive
    assert_eq!(utc.tz_convert(Tz::Naive).unwrap().index().tz(), Tz::Naive);
    assert_eq!(df.set_index_tz(Tz::Utc).unwrap().width(), 1);
    assert!(utc.tz_localize(Tz::Utc).is_err()); // already aware
    let plain = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![1.0])], None).unwrap();
    for e in [plain.tz_localize(Tz::Utc), plain.tz_convert(Tz::Utc), plain.set_index_tz(Tz::Utc)] {
        assert!(matches!(e, Err(VolasError::DType(_))));
    }
}

#[test]
fn a_frame_has_unique_column_names_and_labels() {
    let twice = DataFrame::new(vec!["a".into(), "a".into()], vec![Column::f64(vec![1.0]); 2], None);
    assert!(matches!(twice, Err(VolasError::Value(_))));
    let col = Column { data: ColumnData::Datetime(crate::buffer::Buffer::from(vec![5i64, 5])) };
    assert!(matches!(Index::from_column(&col), Err(VolasError::Value(_))));
    let col = Column { data: ColumnData::Datetime(crate::buffer::Buffer::from(vec![5i64, 3])) };
    assert!(Index::from_column(&col).is_err());
    // Missing timestamps fail before a datetime index can be admitted.
    let nat = crate::datetime::NAT;
    assert!(Index::from_column(&Column { data: ColumnData::Datetime(crate::buffer::Buffer::from(vec![nat, 1, nat])) }).is_err());
}

/// A frame's labels never repeat; datetime labels are present and time ordered.
#[test]
fn a_frame_is_labelled_by_unique_labels() {
    let df = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])], None).unwrap();
    let ts = |v: Vec<i64>| Column { data: ColumnData::Datetime(crate::buffer::Buffer::from(v)) };
    let labelled = df.with_labels(&ts(vec![10, 20, 25, 30]), Tz::Utc, Some("t".into())).unwrap();
    assert_eq!((labelled.index().len(), labelled.index().tz(), labelled.index().name()), (4, Tz::Utc, Some("t")));
    for repeated in [vec![10, 20, 20, 30], vec![10, 20, 10, 30]] {
        assert!(matches!(df.with_labels(&ts(repeated), Tz::Naive, None), Err(VolasError::Value(_))));
    }
    assert!(matches!(df.with_labels(&ts(vec![10, 20]), Tz::Naive, None), Err(VolasError::Shape(_))));
    let nat = crate::datetime::NAT;
    assert!(df.with_labels(&ts(vec![nat, nat, 1, 2]), Tz::Naive, None).is_err());
    assert!(df.with_labels(&Column::i64(vec![1, 1, 2, 3]), Tz::Naive, None).is_err());
    assert_eq!(df.with_labels(&Column::i64(vec![4, 3, 2, 1]), Tz::Naive, None).unwrap().height(), 4);
    let mut with_t = df.clone();
    with_t.set_column("t", ts(vec![10, 10, 20, 30])).unwrap();
    assert!(with_t.set_index("t").is_err());
}

/// A replay that counts the boundaries it is asked for, giving what [`Replayed`] gives.
struct Counting(std::cell::Cell<usize>);

impl Replay for Counting {
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> { self.0.set(self.0.get() + 1); None }

    fn boundary(&self, frame: &DataFrame, name: &str, start: usize) -> Option<Checkpoint> {
        self.0.set(self.0.get() + 1);
        Replayed.boundary(frame, name, start)
    }
}

/// With the anchors of a long history, no selection replays its boundary: each keeps the
/// nearest anchor before it — a tail, one starting just after an anchor, a middle one, one
/// of a selection.
#[test]
fn a_selection_of_an_anchored_history_replays_no_boundary() {
    let n = 3 * ANCHOR_SPACING;
    let df = frame(n);
    let anchors = df.computed_anchors("ema").unwrap();
    for k in 1..3 {
        let at = k * ANCHOR_SPACING - 1;
        anchors.keep(Checkpoint { at, state: Some(vec![at as f64].into()), raw: vec![] });
    }
    let counting = Counting(std::cell::Cell::new(0));
    for (start, end) in [(n - 100, n), (2 * ANCHOR_SPACING, n), (ANCHOR_SPACING + 5, n), (ANCHOR_SPACING + 9, n - 9)] {
        let sub = df.rows(start, end, &counting);
        assert!(sub.computed_checkpoint("ema").is_some_and(|cp| cp.at < start), "{start}");
        let again = sub.rows(3, sub.height(), &counting);
        assert!(again.computed_checkpoint("ema").is_some(), "{start}");
    }
    assert_eq!(counting.0.get(), 0);
    // without them, one far from the history's start replays its boundary
    let far = frame(n).rows(n - 100, n, &counting);
    assert_eq!((counting.0.get(), far.computed_checkpoint("ema").map(|cp| cp.at)), (1, Some(n - 101)));
}

#[test]
fn an_integer_replacement_must_share_a_lossless_dtype_with_its_float_lead() {
    let mut source = frame(4);
    let mut tail = source.rows(2, 4, &NoReplay);
    let before = tail.clone();
    let replacement = Column::i64(vec![1 << 60, (1 << 53) + 1]);
    assert!(matches!(tail.set_column("close", replacement), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    assert_eq!(tail.computed_resume_state("ema"), before.computed_resume_state("ema"));
    // Sparse exact integers outside the contiguous math domain remain valid storage.
    tail.set_column("close", Column::i64(vec![1 << 60, -(1 << 60)])).unwrap();
    let joined = tail.led(&tail, tail.lead_rows());
    assert_eq!(joined.column("close").unwrap().get_f64(joined.height() - 2), (1u64 << 60) as f64);
    // A missing integer payload cannot make a lossless join fail.
    source.set_column("close", Column::f32(vec![0.0, 1.0, 2.0, 3.0])).unwrap();
    let mut narrow = source.rows(2, 4, &NoReplay);
    let values = Column { data: ColumnData::I64(crate::buffer::Buffer::from(vec![(1 << 53) + 1, 7]),
        crate::validity::Validity::from_bitmap(crate::validity::Bitmap::from_valid_iter(2, [false, true]))) };
    narrow.set_column("close", values).unwrap();
    let joined = narrow.led(&narrow, narrow.lead_rows());
    assert!(joined.column("close").unwrap().get_f64(joined.height() - 2).is_nan());
}

#[test]
fn incoming_integer_rows_and_cells_must_remain_lossless_with_the_lead() {
    let mut tail = frame(4).rows(2, 4, &NoReplay);
    tail.set_column("close", Column::i64(vec![2, 3])).unwrap();
    let before = tail.clone();
    let bad = (1 << 53) + 1;
    let src = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![bad])], None).unwrap();
    assert!(matches!(tail.append(&src), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    assert!(matches!(tail.replace_last_row(&src, 0), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    let writes = vec![CellWrite { col: 0, rows: vec![0], values: Column::i64(vec![7]) },
        CellWrite { col: 0, rows: vec![1], values: Column::i64(vec![bad]) }];
    assert!(matches!(tail.check_writes(writes), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    let exact = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1 << 60])], None).unwrap();
    tail.append(&exact).unwrap();
    assert_eq!(tail.led(&tail, tail.lead_rows()).height(), tail.height() + tail.lead_rows());
}

#[test]
fn an_integer_fold_cannot_make_the_float_lead_join_inexact() {
    let mut tail = frame(4).rows(3, 4, &NoReplay);
    tail.set_column("close", Column::i64(vec![1 << 53])).unwrap();
    let before = tail.clone();
    let one = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1])], None).unwrap();
    let ops = [(0, 0, crate::CombineOp::Sum)];
    assert!(matches!(tail.fold_forming_row(0, &one, 0, &ops), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    let src = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1 << 53, 1])], None).unwrap();
    assert!(matches!(tail.refold_forming_row(0, &src, &ops), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    assert!(matches!(tail.check_refold(0, &src, &one, &ops), Err(VolasError::DType(_))));
    let mut snapshot = tail.clone();
    assert!(matches!(tail.fold_with_checkpoint(&mut snapshot, &one, &ops, &[false], false), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
    assert!(snapshot.equals(&before));
    let bad = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![(1 << 53) + 1])], None).unwrap();
    assert!(matches!(tail.write_folded_row(&bad, &ops), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
}

#[test]
fn a_selection_shares_immutable_states_and_continues_independently() {
    let source = frame(10);
    let mut tail = source.rows(7, 10, &NoReplay);
    let source_key = source.computed.keys().find(|key| key.as_ref() == "ema").unwrap();
    let selected_key = tail.computed.keys().find(|key| key.as_ref() == "ema").unwrap();
    assert!(Arc::ptr_eq(source_key, selected_key));
    let source_state = source.computed_resume_state("ema").unwrap().0.as_ptr();
    assert_eq!(tail.computed_resume_state("ema").unwrap().0.as_ptr(), source_state);
    let mut sibling = tail.clone();
    for selected in [&mut tail, &mut sibling] {
        let row = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![10.0])], None).unwrap();
        selected.append(&row).unwrap();
        selected.step_computed_state("ema", |_, prior, _, next| {
            next.copy_from_slice(prior);
            next[0] += 1.0;
            Some(next[0])
        });
        selected.update_computed_value("ema", 3, 10.0).unwrap();
        assert_eq!(selected.computed_resume_state("ema").unwrap().0, &[10.0]);
    }
    assert_eq!(source.computed_resume_state("ema").unwrap().0, &[9.0]);
    assert_eq!(source.computed_resume_state("ema").unwrap().0.as_ptr(), source_state);
}

#[test]
fn state_scratch_reuse_handles_unique_shared_and_weak_owners() {
    for sharing in 0..3 {
        for success in [false, true] {
            let mut df = frame(4);
            let scratch = Arc::<[f64]>::from(vec![71.0]);
            let original = scratch.as_ptr();
            let held = (sharing == 1).then(|| Arc::clone(&scratch));
            let weak = (sharing == 2).then(|| Arc::downgrade(&scratch));
            df.computed.get_mut("ema").unwrap().scratch = Some(scratch);
            let output = df.step_computed_state("ema", |_, prior, _, next| {
                assert_eq!(prior, &[3.0]);
                if sharing == 0 { assert_eq!(next.as_ptr(), original); }
                next[0] = 4.0;
                success.then_some(4.0)
            });
            assert_eq!(output, success.then_some(4.0));
            let metadata = df.computed.get("ema").unwrap();
            assert_eq!(metadata.state.as_deref(), Some(if success { &[4.0][..] } else { &[3.0][..] }));
            assert_eq!(metadata.state_before_last.as_deref(), Some(if success { &[3.0][..] } else { &[2.0][..] }));
            assert_eq!(metadata.state_at, if success { 4 } else { 3 });
            if let Some(held) = held { assert_eq!(&*held, &[71.0]); }
            if let Some(weak) = weak { assert!(weak.upgrade().is_none()); }
        }
    }
}

#[test]
fn shared_stale_snapshots_match_owned_snapshots_and_survive_refresh_writes() {
    let mut df = frame(4).rows(2, 4, &NoReplay);
    for invalidated in [false, true] {
        if invalidated { df.invalidate_computed_on_write("close"); }
        for only in [None, Some("ma"), Some("ema"), Some("close"), Some("absent")] {
            let owned = df.stale_computed_columns(only);
            let shared: Vec<_> = df.shared_stale_computed_columns(only).collect();
            assert_eq!(owned.len(), shared.len());
            for (a, b) in owned.iter().zip(&shared) {
                assert_eq!(a.name, b.name.as_ref());
                assert!(Arc::ptr_eq(&a.directive, &b.directive));
                assert_eq!((a.reach, a.valid_rows, a.since, a.warm), (b.reach, b.valid_rows, b.since, b.warm));
            }
        }
    }
    let mut held = df.shared_stale_computed_columns(Some("ma"));
    df.update_computed_tail("ma", 0, &Column::f64(vec![8.0; 2])).unwrap();
    assert_eq!(df.shared_stale_computed_columns(Some("ma")).count(), 0);
    drop(df);
    let before = held.next().unwrap();
    assert_eq!((before.name.as_ref(), before.valid_rows, before.since), ("ma", 0, 0));
    assert!(held.next().is_none());
}

#[test]
fn a_regular_batch_appends_to_sparse_history_without_replacing_existing_points() {
    let point = |at, value| Checkpoint { at, state: Some(vec![value].into()), raw: vec![] };
    let anchors = Anchors::default();
    anchors.keep_all(vec![point(10, 1.0), point(20, 2.0)]);
    anchors.keep_all(vec![point(511, 3.0), point(1023, 4.0)]);
    anchors.keep_all(vec![point(20, 99.0), point(1535, 5.0)]);
    assert_eq!(anchors.len(), 5);
    assert_eq!(anchors.nearest(510).unwrap().state, Some(vec![2.0].into()));
    assert_eq!(anchors.nearest(511).unwrap().state, Some(vec![3.0].into()));
    assert_eq!(anchors.nearest(usize::MAX).unwrap().state, Some(vec![5.0].into()));
}

#[test]
fn interleaved_raw_float_history_preserves_known_clean_facts_and_independent_ranges() {
    let mut values: Vec<_> = (0..1000).map(f64::from).collect();
    values[890] = -0.0;
    let mut df = DataFrame::new(vec!["raw_a".into(), "ma".into(), "raw_b".into()],
        vec![Column::f64(values), Column::f64(vec![0.0; 1000]), Column::f64(vec![2.0; 1000])], None).unwrap();
    let Column { data: ColumnData::F64(source) } = df.column("raw_a").unwrap() else { unreachable!() };
    let guard = source.keepalive();
    let weak = Arc::downgrade(&guard);
    drop(guard);
    assert!(!source.has_nan());
    assert!(!df.column("raw_b").unwrap().has_nan());
    df.set_computed("ma", Arc::new(()), Reach { lookback: 19, resume: 20, continuation: 20, history: History::Window });
    let mut copied = df.rows(900, 950, &NoReplay);
    assert_eq!(copied.lead_rows(), 20);
    assert_eq!(copied.column("raw_a").unwrap().known_has_nan(), Some(false));
    let input = copied.select(&["raw_a".into(), "raw_b".into()]).unwrap();
    let led = copied.led(&input, 20);
    assert_eq!(led.column("raw_a").unwrap().as_f64().unwrap()[10].to_bits(), (-0.0f64).to_bits());
    assert_eq!(led.column("raw_a").unwrap().known_has_nan(), Some(false));
    assert_eq!(led.column("raw_b").unwrap().known_has_nan(), Some(false));
    copied.check_writes(vec![CellWrite { col: 2, rows: vec![0], values: Column::f64(vec![9.0]) }]).unwrap().land();
    assert_eq!(lead_of(&copied, "raw_b"), vec![2.0; 20]);
    assert_eq!(copied.column("raw_a").unwrap().get_f64(0), 900.0);
    assert_eq!(df.column("raw_b").unwrap().get_f64(900), 2.0);
    drop(df);
    assert!(weak.upgrade().is_none());
}

#[test]
fn exact_integer_folds_with_float_history_succeed_and_inexact_folds_are_atomic() {
    let mut tail = frame(4).rows(2, 4, &NoReplay);
    tail.set_column("close", Column::i64(vec![3, 4])).unwrap();
    let bar = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![10])], None).unwrap();
    let ops = [(0, 0, crate::CombineOp::Sum)];
    tail.fold_forming_row(0, &bar, 0, &ops).unwrap();
    assert_eq!(tail.column("close").unwrap().get_i64(0), 13);
    assert_eq!(tail.led(&tail, tail.lead_rows()).column("close").unwrap().get_f64(tail.lead_rows()), 13.0);
    let before = tail.clone();
    let bad = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1 << 53])], None).unwrap();
    assert!(matches!(tail.fold_forming_row(0, &bad, 0, &ops), Err(VolasError::DType(_))));
    assert!(tail.equals(&before));
}

#[test]
fn a_checkpoint_lead_keeps_raw_operands_but_does_not_duplicate_carried_tails() {
    let mut source = frame(100);
    source.set_computed("ema", std::sync::Arc::new(()), Reach {
        lookback: 40, resume: 40, continuation: 40, history: History::Carried { operands: 7 },
    });
    source.set_computed_checkpoint("ema", Ok(Checkpoint { at: 79, state: Some(vec![79.0].into()), raw: vec![] }));
    let tail = source.rows(90, 100, &NoReplay);
    // Raw rows 73..80 feed its stateless operands; 80..90 advance from the checkpoint.
    assert_eq!(lead(&tail), (73..90).map(f64::from).collect::<Vec<_>>());
    let boundary = source.rows(80, 100, &NoReplay);
    assert_eq!(lead(&boundary), (73..80).map(f64::from).collect::<Vec<_>>());
}

#[test]
fn nearest_anchor_predicts_the_regular_grid_and_checks_sparse_or_irregular_rows() {
    let checkpoint = |at| Checkpoint { at, state: Some(vec![at as f64].into()), raw: vec![] };
    for rows in [vec![511, 1023, 1535], vec![511, 1535], vec![511, 515, 1023]] {
        let anchors = Anchors::default();
        for &at in &rows { anchors.keep(checkpoint(at)); }
        for at in [0, 510, 511, 512, 514, 515, 516, 1000, 1023, 1500, 1535, 1536, usize::MAX] {
            let expected = rows.iter().copied().filter(|&row| row <= at).max();
            assert_eq!(anchors.nearest(at).map(|cp| cp.at), expected);
        }
    }
}

#[test]
fn dense_anchor_bounds_survive_duplicates_batches_and_selection_limits() {
    let cp = |at, value| Checkpoint { at, state: Some(vec![value].into()), raw: vec![] };
    let anchors = Anchors::default();
    anchors.keep_all(vec![cp(1023, 1.0), cp(1535, 2.0)]);
    anchors.keep(cp(1535, 9.0)); // old point wins; the grid remains intact
    anchors.keep_all(vec![cp(2047, 3.0), cp(2559, 4.0)]);
    assert!(anchors.nearest(1022).is_none());
    assert_eq!(anchors.nearest(usize::MAX).unwrap().at, 2559);
    assert_eq!(anchors.nearest(1535).unwrap().state, Some(vec![2.0].into()));
    assert!(anchors.bounded(0, 0).nearest(usize::MAX).is_none());
    assert!(anchors.bounded(0, 1023).nearest(usize::MAX).is_none());
    assert_eq!(anchors.bounded(0, 1535).nearest(usize::MAX).unwrap().at, 1023);
    anchors.keep(cp(1200, 5.0)); // invalidate the grid; ordered fallback finds it
    assert_eq!(anchors.nearest(1400).unwrap().at, 1200);
    anchors.keep(cp(511, 6.0)); // a point before the initial bulk remains ordered
    assert_eq!(anchors.nearest(900).unwrap().at, 511);
    let high = Anchors::default();
    high.keep_all(vec![cp(usize::MAX - 1024, 1.0), cp(usize::MAX - 512, 2.0)]);
    assert_eq!(high.nearest(usize::MAX).unwrap().at, usize::MAX - 512);
    assert_eq!(high.bounded(0, usize::MAX - 512).nearest(usize::MAX).unwrap().at, usize::MAX - 1024);
}

#[test]
fn a_retained_batch_point_can_seed_another_grid_and_survive_sparse_conversion() {
    let source = Anchors::default();
    source.keep_all((1..=4).map(|k| Checkpoint { at: k * 512 - 1, state: Some(vec![k as f64].into()), raw: vec![] }).collect());
    let borrowed = source.nearest(1023).unwrap();
    let target = Anchors::default();
    target.keep(borrowed);
    target.keep(Checkpoint { at: 1535, state: Some(vec![5.0].into()), raw: vec![] });
    target.keep_all(vec![Checkpoint { at: 2047, state: Some(vec![6.0].into()), raw: vec![] }]);
    assert_eq!(target.nearest(1023).unwrap().state, Some(vec![2.0].into()));
    let retained = target.nearest(2047).unwrap();
    // Overlap preserves the existing point while an off-grid insertion switches
    // to sparse order. Previously returned points retain their immutable bank.
    target.keep_all(vec![Checkpoint { at: 1535, state: Some(vec![99.0].into()), raw: vec![] },
        Checkpoint { at: 1600, state: Some(vec![7.0].into()), raw: vec![] }]);
    assert_eq!(target.nearest(1535).unwrap().state, Some(vec![5.0].into()));
    assert_eq!(target.nearest(1700).unwrap().at, 1600);
    drop(source);
    drop(target);
    assert_eq!(retained.into_owned().state, Some(vec![6.0].into()));
}

#[test]
fn shared_checkpoint_row_metadata_matches_its_opaque_payload_across_owners() {
    let point = |at: usize| Checkpoint { at, state: Some(vec![at as f64].into()), raw: vec![] };
    let single = CheckpointRef::from(point(17));
    let mut external = Arc::new(point(19));
    let shared = CheckpointRef::from(Arc::clone(&external));
    Arc::make_mut(&mut external).at = 20;
    assert_eq!(shared.position(), 19);
    for cp in [single, shared] {
        assert_eq!(cp.position(), cp.at);
        let at = cp.position();
        assert_eq!(cp.into_owned().state, Some(vec![at as f64].into()));
    }
    let anchors = Anchors::default();
    anchors.keep_all((1..=4).map(|k| point(k * 512 - 1)).collect());
    anchors.keep_all((5..=8).map(|k| point(k * 512 - 1)).collect());
    let moved = Anchors::default();
    moved.keep(anchors.nearest(1535).unwrap()); // original bank's nonzero slot
    moved.keep(point(2047));
    anchors.keep(point(1800)); // convert the dense banks to sparse handles
    for store in [&anchors, &moved] {
        for row in [1535, 1800, 2047, 4095, usize::MAX] {
            let cp = store.nearest(row).unwrap();
            assert_eq!(cp.position(), cp.at);
            let at = cp.position();
            assert_eq!(cp.into_owned().state, Some(vec![at as f64].into()));
        }
    }
}

#[test]
fn irregular_anchor_batches_append_and_overlapping_grid_batches_keep_existing_values() {
    let point = |at, value| Checkpoint { at, state: Some(vec![value].into()), raw: vec![] };
    let irregular = Anchors::default();
    irregular.keep_all(vec![point(10, 1.0), point(20, 2.0)]);
    irregular.keep_all(vec![point(30, 3.0), point(40, 4.0)]);
    assert_eq!(irregular.len(), 4);
    assert_eq!(irregular.nearest(35).unwrap().state, Some(vec![3.0].into()));
    assert_eq!(irregular.bounded(0, 30).nearest(usize::MAX).unwrap().at, 20);
    let grid = Anchors::default();
    grid.keep_all(vec![point(511, 1.0), point(1023, 2.0)]);
    grid.keep_all(vec![point(511, 9.0), point(1023, 10.0)]);
    assert_eq!(grid.len(), 2);
    assert_eq!(grid.nearest(1023).unwrap().state, Some(vec![2.0].into()));
    let gap = Anchors::default();
    gap.keep_all(vec![point(511, 1.0), point(1023, 2.0)]);
    gap.keep_all(vec![point(2047, 3.0), point(2559, 4.0)]);
    assert_eq!(gap.nearest(1535).unwrap().at, 1023);
    assert_eq!(gap.nearest(2047).unwrap().state, Some(vec![3.0].into()));
}

#[test]
fn an_integer_history_keeps_native_integer_folds_outside_float_precision() {
    let mut source = DataFrame::new(vec!["volume".into(), "ma".into()],
        vec![Column::i64(vec![1, 2, 3, 4]), Column::f64(vec![0.0; 4])], None).unwrap();
    source.set_computed("ma", Arc::new(()), Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
    let mut tail = source.rows(2, 4, &NoReplay);
    let bar = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![1 << 53])], None).unwrap();
    tail.fold_forming_row(0, &bar, 0, &[(0, 0, crate::CombineOp::Sum)]).unwrap();
    assert_eq!(tail.column("volume").unwrap().get_i64(0), (1 << 53) + 3);
    let joined = tail.led(&tail, tail.lead_rows());
    assert_eq!(joined.column("volume").unwrap().dtype(), crate::DType::I64);
    assert_eq!(joined.column("volume").unwrap().get_i64(tail.lead_rows()), (1 << 53) + 3);
}

#[test]
fn replacing_a_row_ignores_supplied_cached_values_and_invalidates_their_cursor() {
    let mut df = frame(4);
    let bar = DataFrame::new(vec!["close".into(), "ma".into(), "ema".into()],
        vec![Column::f64(vec![7.0]), Column::i64(vec![99]), Column::f64(vec![88.0])], None).unwrap();
    df.replace_last_row(&bar, 0).unwrap();
    assert_eq!(df.column("close").unwrap().get_f64(3), 7.0);
    assert_eq!(df.column("ma").unwrap().get_f64(3), 3.0);
    assert_eq!(df.column("ema").unwrap().get_f64(3), 3.0);
    assert_eq!(meta(&df, "ma").valid_rows, 3);
    assert_eq!(meta(&df, "ema").valid_rows, 3);
}

#[test]
fn batch_anchors_keep_sorted_sparse_and_duplicate_points_with_the_same_owner() {
    let cp = |at, value| Checkpoint { at, state: Some(vec![value].into()), raw: vec![] };
    let anchors = Anchors::default();
    anchors.keep_all(vec![cp(511, 1.0), cp(1023, 2.0)]);
    anchors.keep_all(vec![cp(1535, 3.0), cp(2047, 4.0)]);
    assert_eq!(anchors.len(), 4);
    assert_eq!(anchors.nearest(1600).unwrap().state, Some(vec![3.0].into()));
    // Unordered points are sorted stably; both old and intra-batch duplicates keep
    // the first value exactly as repeated `keep` calls do.
    anchors.keep_all(vec![cp(1400, 5.0), cp(700, 6.0), cp(700, 7.0), cp(1535, 8.0)]);
    assert_eq!(anchors.len(), 6);
    assert_eq!(anchors.nearest(700).unwrap().state, Some(vec![6.0].into()));
    assert_eq!(anchors.nearest(1535).unwrap().state, Some(vec![3.0].into()));
    let selected = anchors.bounded(0, 1000);
    selected.keep_all(vec![cp(800, 9.0), cp(1100, 10.0)]);
    assert_eq!(selected.nearest(2000).unwrap().at, 1100);
    assert_eq!(anchors.nearest(1100).unwrap().at, 1023);
    // Empty and future publications never expand the parent's owner.
    selected.keep_all(vec![]);
    selected.keep_all(vec![cp(1200, 11.0)]);
    assert_eq!(anchors.len(), 6);
}

#[test]
fn batch_checkpoint_references_share_the_pool_and_keep_it_alive() {
    let anchors = Anchors::default();
    anchors.keep_all(vec![Checkpoint { at: 511, state: Some(vec![1.0].into()), raw: vec![] },
        Checkpoint { at: 1023, state: Some(vec![2.0].into()), raw: vec![] }]);
    let selected = anchors.nearest(600).unwrap();
    assert_eq!(std::ops::Deref::deref(&selected) as *const Checkpoint,
        std::ops::Deref::deref(&anchors.nearest(600).unwrap()) as *const Checkpoint);
    let clone = selected.clone();
    drop(anchors);
    assert_eq!(selected, clone);
    assert!(format!("{clone:?}").contains("511"));
    assert_eq!(clone.into_owned().state, Some(vec![1.0].into()));
    let cp = Arc::new(Checkpoint { at: 7, state: None, raw: vec![] });
    let borrowed = CheckpointRef::from(Arc::clone(&cp));
    assert_eq!(borrowed.at, cp.at);
    assert_eq!(CheckpointRef::from(cp).into_owned().at, 7);
    let one = CheckpointRef::from(Checkpoint { at: 8, state: None, raw: vec![] });
    let one_alias = one.clone();
    assert_eq!(one.into_owned().at, 8); // shared single-point owner
    assert_eq!(one_alias.into_owned().at, 8); // uniquely owned single point
    let pool = Anchors::default();
    pool.keep_all(vec![Checkpoint { at: 511, state: Some(vec![3.0].into()), raw: vec![] },
        Checkpoint { at: 1023, state: Some(vec![4.0].into()), raw: vec![] }]);
    let sole_point = pool.nearest(1023).unwrap();
    drop(pool);
    assert_eq!(sole_point.into_owned().state, Some(vec![4.0].into())); // unique batch owner
}

#[test]
fn public_projection_owns_coordinates_but_internal_data_view_keeps_them() {
    let tail = frame(10).rows(7, 10, &NoReplay);
    let names = ["ema".to_string(), "close".to_string()];
    let internal = tail.select(&names).unwrap();
    assert_eq!(internal.origin(), 7);
    let public = tail.project(&names, 1, 3).unwrap();
    assert_eq!(public.origin(), 0);
    assert_eq!(public.index().label_at(0), tail.index().label_at(1));
    assert!(public.computed_columns().is_empty() && public.lead.is_none());
    assert!(public.column("ema").unwrap().equals(&tail.column("ema").unwrap().slice(1, 3)));
    assert_eq!(tail.project(&names, 0, 3).unwrap().origin(), 0);
    assert_eq!(tail.project(&names, 20, 1).unwrap().height(), 0);
    assert!(tail.project(&["absent".into()], 0, 3).is_err());
}

#[test]
fn reusing_a_dropped_label_keeps_the_compacted_recursive_checkpoint() {
    let mut target = frame(10);
    target.set_computed_checkpoint("ema", Ok(Checkpoint { at: 3, state: Some(vec![3.].into()), raw: vec![] }));
    let rows = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![100., 101.])],
        Some(Index::new(None, crate::IndexKind::Int64(vec![1, 11])).unwrap())).unwrap();
    target.append_retaining(&rows, Retained { from: 4, shown: 6 }, &Replayed).unwrap();
    assert_eq!((target.origin(), target.height()), (4, 8));
    assert_eq!(target.index().to_i64_labels(), vec![4, 5, 6, 7, 8, 9, 1, 11]);
    assert_eq!(target.computed_checkpoint("ema").map(|cp| cp.at), Some(3));
    assert!(target.is_computed("ema"));
    assert_eq!(meta(&target, "ema").valid_rows, 6);
}
