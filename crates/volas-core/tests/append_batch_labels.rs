//! Batch routing preserves last-row replacement, first error and atomicity.
use volas_core::{Column, DataFrame, Index, IndexKind, Result, Retained, Replay, Checkpoint};

struct NoReplay;
impl Replay for NoReplay {
    fn boundary(&self, _: &DataFrame, _: &str, _: usize) -> Option<Checkpoint> { None }
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> { None }
}

fn checked_frame(labels: &[i64], strings: bool) -> Result<DataFrame> {
    let kind = if strings {
        IndexKind::Str(labels.iter().map(|n| n.to_string()).collect())
    } else {
        IndexKind::Int64(labels.to_vec())
    };
    DataFrame::new(
        vec!["x".into()],
        vec![Column::f64((0..labels.len()).map(|i| i as f64).collect())],
        Some(Index::new(None, kind)?),
    )
}

fn frame(labels: &[i64], strings: bool) -> DataFrame {
    checked_frame(labels, strings).unwrap()
}

#[test]
fn ordered_and_unordered_batches_replace_consecutive_resends() {
    for strings in [false, true] {
        // The existing last row may be replaced by the first incoming row.
        // Repeated labels inside the incoming frame fail at its construction.
        for labels in [vec![2, 3, 4], vec![2, 1, 0], vec![2, 4, 3]] {
            let mut target = frame(&[8, 2], strings);
            target.append(&frame(&labels, strings)).unwrap();
            let kept: Vec<i64> = [vec![8], labels.clone()].concat();
            assert!(target.index().label_eq(frame(&kept, strings).index()));
            let values: Vec<f64> = [vec![0.], (0..labels.len()).map(|i| i as f64).collect()].concat();
            assert_eq!(target.columns()[0].as_f64().unwrap(), values);
        }
        for labels in [&[2, 2, 3, 3, 4][..], &[2, 2, 1, 1, 0], &[2, 2, 4, 4, 3, 3]] {
            assert!(checked_frame(labels, strings).unwrap_err().to_string().contains("duplicate labels"));
        }
    }
}

#[test]
fn separated_duplicates_reject_the_whole_batch() {
    for strings in [false, true] {
        for labels in [&[2, 3, 2][..], &[4, 4, 3, 4], &[4, 3, 3, 4]] {
            assert!(checked_frame(labels, strings).unwrap_err().to_string().contains("duplicate labels"));
        }
        let mut target = frame(&[8, 2], strings);
        let before = target.clone();
        let err = target.append(&frame(&[4, 3, 8], strings)).unwrap_err();
        assert!(err.to_string().contains("an earlier row has that label"), "{err}");
        assert!(target.equals(&before));
    }
}

#[test]
fn retained_margin_errors_and_conversion_errors_keep_their_priority() {
    for strings in [false, true] {
        let mut target = frame(&[1, 2, 3], strings);
        let before = target.clone();
        let err = target
            .append_retaining(&frame(&[4, 1], strings), Retained { from: 0, shown: 1 }, &NoReplay)
            .unwrap_err();
        assert!(err.to_string().contains("hidden margin"), "{err}");
        assert!(target.equals(&before));
        assert!(checked_frame(&[4, 1, 4], strings).is_err());
        target
            .append_retaining(&frame(&[1, 4], strings), Retained { from: 1, shown: 1 }, &NoReplay)
            .unwrap();
        assert_eq!(target.height(), 4);
        assert_eq!(target.origin(), 1);
    }
    let mut target = frame(&[0, 1], false);
    let rows = DataFrame::new(
        vec!["x".into()],
        vec![Column::i64(vec![(1 << 53) + 1, 1])],
        Some(Index::new(None, IndexKind::Int64(vec![2, 3])).unwrap()),
    ).unwrap();
    let before = target.clone();
    let err = target.append(&rows).unwrap_err();
    assert!(err.to_string().contains("not exactly representable"), "{err}");
    assert!(target.equals(&before));
}

#[test]
fn incoming_resends_still_count_when_no_old_labels_are_retained() {
    // An ordinary input frame cannot contain even consecutive re-sends.
    let err = checked_frame(&[2, 3, 2], false).unwrap_err();
    assert!(err.to_string().contains("duplicate labels"));
    let mut target = frame(&[1, 2], false);
    target.append_retaining(&frame(&[2, 3], false), Retained { from: 2, shown: 2 }, &NoReplay).unwrap();
    assert_eq!(target.height(), 3);
}

#[test]
fn dropped_prefix_reuse_compacts_only_when_needed() {
    for strings in [false, true] {
        let mut target = frame(&[1, 2, 3], strings);
        target.append_retaining(&frame(&[3, 1, 4], strings), Retained { from: 1, shown: 1 }, &NoReplay).unwrap();
        assert!(target.index().label_eq(frame(&[2, 3, 1, 4], strings).index()));
        assert_eq!((target.origin(), target.height()), (1, 4));
        assert_eq!(target.columns()[0].as_f64().unwrap(), &[1., 0., 1., 2.]);

        let mut all_dropped = frame(&[1, 2, 3], strings);
        all_dropped.append_retaining(&frame(&[4, 1], strings), Retained { from: 3, shown: 3 }, &NoReplay).unwrap();
        assert!(all_dropped.index().label_eq(frame(&[4, 1], strings).index()));
        assert_eq!(all_dropped.origin(), 3);
    }
}

#[test]
fn dropped_integer_continuation_keeps_the_original_next_label() {
    let mut target = frame(&[1, 0], false);
    let unlabelled = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![9.])], None).unwrap();
    target.append_retaining(&unlabelled, Retained { from: 2, shown: 2 }, &NoReplay).unwrap();
    assert_eq!(target.index().to_i64_labels(), vec![1]);
    assert_eq!((target.origin(), target.height()), (2, 1));
}

#[test]
fn retention_bounds_are_checked_before_writing() {
    let mut target = frame(&[1, 2], false);
    let before = target.clone();
    for retained in [Retained { from: 2, shown: 1 }, Retained { from: 0, shown: 3 }] {
        assert!(target.append_retaining(&frame(&[3], false), retained, &NoReplay).is_err());
        assert!(target.equals(&before));
    }
}
