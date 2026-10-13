//! The one derivation owner: what a contiguous selection, a projection and a new series
//! keep of their source.

use std::collections::HashMap;

use super::derive::NoReplay;
use super::*;
use crate::tz::Tz;

/// A replay that has the history: the state of a checkpoint it gives names its row.
struct Replayed;

impl Replay for Replayed {
    fn boundary(&self, frame: &DataFrame, _: &str, start: usize) -> Option<Checkpoint> {
        let row = (frame.origin() + start).checked_sub(1)?;
        Some(Checkpoint { at: row, state: Some(vec![row as f64]), inputs: vec![] })
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
    df.set_computed("ma", Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
    df.set_computed("ema", Reach { lookback: 4, resume: 0, continuation: 4, history: History::Carried { operands: 0 } });
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
fn a_tail_keeps_every_cached_column_with_its_states_and_lead() {
    let df = frame(10);
    let tail = df.rows(7, 10, &NoReplay);
    assert_eq!((tail.origin(), tail.height()), (7, 3));
    assert_eq!((meta(&tail, "ma").valid_rows, meta(&tail, "ema").valid_rows), (3, 3));
    assert_eq!((meta(&tail, "ema").state, meta(&tail, "ema").state_before_last), (Some(vec![9.0]), Some(vec![8.0])));
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
    assert_eq!((meta(&one, "ema").state, meta(&one, "ema").state_before_last), (Some(vec![9.0]), None));
}

#[test]
fn a_recursion_without_a_state_keeps_its_whole_history() {
    let mut df = frame(10);
    df.set_column("kcw", Column::f64(vec![0.0; 10])).unwrap();
    df.set_computed("kcw", Reach { lookback: 1, resume: 0, continuation: 1, history: History::Unkept });
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
    kept.set_computed_checkpoint("ema", Ok(Checkpoint { at: 7, state: Some(vec![7.0]), inputs: vec![] }));
    kept.invalidate_computed_on_write("close");
    assert_eq!(kept.rows(8, 10, &NoReplay).lead_rows(), 5);
}

#[test]
fn a_lead_keeps_its_history_whatever_dtype_a_column_takes() {
    let mut df = frame(10);
    df.set_column("close", Column::f64((0..10).map(|i| i as f64 + 0.123456789).collect())).unwrap();
    df.set_computed("ma", Reach { lookback: 2, resume: 3, continuation: 2, history: History::Window });
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
    df.set_computed("kcw", Reach { lookback: 1, resume: 0, continuation: 1, history: History::Unkept });
    // ema replays from its checkpoint at row 89, which the tail holds the row after
    df.set_computed_checkpoint("ema", Ok(Checkpoint { at: 89, state: Some(vec![89.0]), inputs: vec![] }));
    let tail = df.rows(90, 100, &NoReplay);
    assert_eq!(tail.lead_rows(), 90);
    let names: Vec<String> = ["close", "ma", "ema"].map(String::from).to_vec();
    let kept = tail.select(&names).unwrap();
    assert_eq!(kept.lead_rows(), 5);
    assert_eq!(lead(&kept), vec![85.0, 86.0, 87.0, 88.0, 89.0]);
    // so does dropping a column's cached status
    let mut written = tail.clone();
    written.invalidate_computed_on_write("kcw");
    assert_eq!(written.lead_rows(), 5);
    // a copied lead shrinks in place while it stays most of its rows; none is left once no
    // cached column reads before row 0
    // the rows after ema's checkpoint (90..95) and the reach before them (5)
    let short = df.rows(95, 100, &NoReplay).select(&names).unwrap();
    assert_eq!(short.lead_rows(), 10);
    let ma = short.select(&["close".to_string(), "ma".to_string()]).unwrap();
    assert_eq!(lead(&ma), vec![92.0, 93.0, 94.0]);
    // a copied one that stays most of its rows keeps sharing them
    let window = frame(10).rows(4, 10, &NoReplay).select(&["close".to_string(), "ma".to_string()]).unwrap();
    assert_eq!(lead(&window), vec![1.0, 2.0, 3.0]);
    let mut bare = ma.clone();
    bare.invalidate_computed_on_write("ma");
    assert_eq!(bare.lead_rows(), 0);
    // a data column the lead does not hold takes any dtype
    let mut added = ma.clone();
    added.set_column("x", Column::f64(vec![0.0; 5])).unwrap();
    added.set_column("x", Column::str(vec!["a".into(); 5])).unwrap();
}

/// Anchors are a memo of a recursion's history: shared by copies, shared by a selection
/// for the rows before its last (where its history agrees with its source's), dropped by
/// a write to the inputs.
#[test]
fn anchors_follow_the_rows_and_a_write_drops_them() {
    let mut df = frame(10);
    let anchors = df.computed_anchors("ema").unwrap();
    for at in [2, 5, 8, 5] {
        anchors.keep(Checkpoint { at, state: Some(vec![at as f64]), inputs: vec![] });
    }
    assert_eq!(anchors.len(), 3);
    assert_eq!(anchors.nearest(6).map(|cp| cp.at), Some(5));
    assert_eq!(anchors.nearest(1), None);
    assert_eq!(df.clone().computed_anchors("ema").unwrap(), anchors);
    let sub = df.rows(4, 9, &NoReplay);
    let shared = sub.computed_anchors("ema").unwrap();
    assert_eq!((shared.len(), shared.nearest(9).map(|cp| cp.at)), (2, Some(5)));
    assert_ne!(shared, anchors);
    // one the selection forms before its last row is its source's too; one past it, on
    // rows that may come to differ, is not kept
    for at in [6, 8] {
        shared.keep(Checkpoint { at, state: None, inputs: vec![] });
    }
    assert_eq!((anchors.len(), shared.len()), (4, 3));
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
    assert_eq!((meta(&mid, "ema").state, meta(&mid, "ema").state_before_last), (None, None));
    assert_eq!((mid.computed_resume_state("ema"), meta(&mid, "ema").valid_rows), (None, 3));
    assert_eq!((meta(&mid, "ma").state, counting.0.get()), (None, 0));
    assert_eq!(mid.lead_rows(), 3); // ema's whole history before it
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
    assert_eq!((meta(&after, "ema").valid_rows, meta(&after, "ema").state), (0, None));
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
    let cp = Checkpoint { at: 4, state: Some(vec![4.0]), inputs: vec![] };
    df.set_computed_checkpoint("ema", Ok(cp.clone()));
    df.set_computed_pending("ema", Some(Checkpoint { at: 7, ..cp.clone() }));
    let kept = df.rows(5, 10, &NoReplay);
    assert_eq!(kept.computed_checkpoint("ema"), Some(&cp));
    assert!(kept.has_computed_pending("ema", 7));
    // a selection after it keeps it too, the rows after it in its lead
    let later = df.rows(6, 10, &NoReplay);
    assert_eq!(later.computed_checkpoint("ema"), Some(&cp));
    assert_eq!(later.lead_rows(), 6);
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
    assert_eq!(far.lead_rows(), 5);
    assert_eq!(df.rows(first, n, &NoReplay).computed_checkpoint("ema"), None);
    // an anchor within it is kept, the rows after it in the lead
    let at = ANCHOR_SPACING - 1;
    df.computed_anchors("ema").unwrap().keep(Checkpoint { at, state: Some(vec![at as f64]), inputs: vec![] });
    let near = df.rows(first, n, &NoReplay);
    assert_eq!(near.computed_checkpoint("ema").map(|cp| cp.at), Some(at));
    assert_eq!(near.lead_rows(), first - at - 1 + 5);
    // a checkpoint later than the anchor is the nearer
    let mut own = df.clone();
    own.set_computed_checkpoint("ema", Ok(Checkpoint { at: first - 3, state: None, inputs: vec![] }));
    let nearer = own.rows(first, n, &NoReplay);
    assert_eq!(nearer.computed_checkpoint("ema").map(|cp| cp.at), Some(first - 3));
    assert_eq!(nearer.lead_rows(), 2 + 5);
}

#[test]
fn rows_by_position_continue_only_when_contiguous_in_order() {
    let df = frame(10);
    let run = df.select_rows(&[4, 5, 6], &NoReplay);
    assert_eq!((run.origin(), run.is_computed("ema")), (4, true));
    for positions in [&[4usize, 6][..], &[5, 4], &[4, 4], &[]] {
        let new = df.select_rows(positions, &NoReplay);
        assert_eq!(new.origin(), 0);
        assert_eq!(new.names(), &["close".to_string()]);
        assert!(new.lead.is_none());
    }
    let mask = df.filter_mask(&[false, true, true, false, false, false, false, false, false, false], &NoReplay).unwrap();
    assert_eq!((mask.origin(), mask.height(), mask.is_computed("ma")), (1, 2, true));
    // a read of the rows keeps every column's values, none cached
    let read = df.take(&[2, 9]);
    assert_eq!((read.width(), read.computed_columns().len()), (3, 0));
}

#[test]
fn a_projection_keeps_the_series_and_the_listed_cached_columns() {
    let tail = frame(10).rows(7, 10, &NoReplay);
    let p = tail.select(&["ema".to_string(), "close".to_string()]).unwrap();
    assert_eq!((p.origin(), p.is_computed("ema"), p.has_column("ma")), (7, true, false));
    assert_eq!(lead(&p), lead(&tail));
    // without its data columns the lead keeps none; with no cached column there is none
    assert_eq!(tail.select(&["ema".to_string()]).unwrap().lead_rows(), 7);
    assert!(tail.select(&["close".to_string()]).unwrap().lead.is_none());
    assert!(tail.select(&["nope".to_string()]).is_err());
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
        Some(Index::datetime(vec![0, 60_000_000_000], Tz::Naive)),
    )
    .unwrap();
    df.set_computed("ma", Reach { lookback: 1, resume: 0, continuation: 1, history: History::Window });
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
    let col = Column::Datetime(crate::buffer::Buffer::from(vec![5i64, 5]));
    assert!(matches!(Index::from_column(&col), Err(VolasError::Value(_))));
    let col = Column::Datetime(crate::buffer::Buffer::from(vec![5i64, 3]));
    assert!(Index::from_column(&col).is_ok());
    // NaT is no label: it never repeats one
    let nat = crate::datetime::NAT;
    assert!(Index::from_column(&Column::Datetime(crate::buffer::Buffer::from(vec![nat, 1, nat]))).is_ok());
}

/// A frame's labels never repeat: any repeated label raises, consecutive or not; `NaT`
/// is no label.
#[test]
fn a_frame_is_labelled_by_unique_labels() {
    let df = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])], None).unwrap();
    let ts = |v: Vec<i64>| Column::Datetime(crate::buffer::Buffer::from(v));
    let labelled = df.with_labels(&ts(vec![10, 20, 25, 30]), Tz::Utc, Some("t".into())).unwrap();
    assert_eq!((labelled.index().len(), labelled.index().tz(), labelled.index().name()), (4, Tz::Utc, Some("t")));
    for repeated in [vec![10, 20, 20, 30], vec![10, 20, 10, 30]] {
        assert!(matches!(df.with_labels(&ts(repeated), Tz::Naive, None), Err(VolasError::Value(_))));
    }
    assert!(matches!(df.with_labels(&ts(vec![10, 20]), Tz::Naive, None), Err(VolasError::Shape(_))));
    let nat = crate::datetime::NAT;
    assert_eq!(df.with_labels(&ts(vec![nat, nat, 1, 2]), Tz::Naive, None).unwrap().height(), 4);
    assert!(df.with_labels(&Column::i64(vec![1, 1, 2, 3]), Tz::Naive, None).is_err());
    assert_eq!(df.with_labels(&Column::i64(vec![4, 3, 2, 1]), Tz::Naive, None).unwrap().height(), 4);
    let mut with_t = df.clone();
    with_t.set_column("t", ts(vec![10, 10, 20, 30])).unwrap();
    assert!(with_t.set_index("t").is_err());
}

/// A replay that counts the boundaries it is asked for, giving what [`Replayed`] gives.
struct Counting(std::cell::Cell<usize>);

impl Replay for Counting {
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
        anchors.keep(Checkpoint { at, state: Some(vec![at as f64]), inputs: vec![] });
    }
    let counting = Counting(std::cell::Cell::new(0));
    for (start, end) in [(n - 100, n), (2 * ANCHOR_SPACING, n), (ANCHOR_SPACING + 5, n), (ANCHOR_SPACING + 9, n - 9)] {
        let sub = df.rows(start, end, &counting);
        assert!(sub.computed_checkpoint("ema").is_some_and(|cp| cp.at < start), "{start}");
        let again = sub.rows(3, sub.height(), &counting);
        assert!(again.computed_checkpoint("ema").is_some(), "{start}");
    }
    assert_eq!(counting.0.get(), 0);
}
