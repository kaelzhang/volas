//! Batch routing preserves last-row replacement, first error and atomicity.
use volas_core::{Column, DataFrame, Index, Retained};

fn frame(labels: &[i64], strings: bool) -> DataFrame {
    let index = if strings {
        Index::str(labels.iter().map(|n| n.to_string()).collect())
    } else {
        Index::int64(labels.to_vec())
    };
    DataFrame::new(
        vec!["x".into()],
        vec![Column::f64((0..labels.len()).map(|i| i as f64).collect())],
        Some(index),
    )
    .unwrap()
}

#[test]
fn ordered_and_unordered_batches_replace_consecutive_resends() {
    for strings in [false, true] {
        for labels in [
            vec![2, 2, 3, 3, 4],
            vec![2, 2, 1, 1, 0],
            vec![2, 2, 4, 4, 3, 3],
        ] {
            let mut target = frame(&[8, 2], strings);
            target.append(&frame(&labels, strings)).unwrap();
            let mut kept = vec![8];
            let mut values = vec![0.];
            for (i, &label) in labels.iter().enumerate() {
                if kept.last() == Some(&label) {
                    *values.last_mut().unwrap() = i as f64;
                } else {
                    kept.push(label);
                    values.push(i as f64);
                }
            }
            assert!(target.index().label_eq(frame(&kept, strings).index()));
            assert_eq!(target.columns()[0].as_f64().unwrap(), values);
        }
    }
}

#[test]
fn separated_duplicates_reject_the_whole_batch() {
    for strings in [false, true] {
        for labels in [&[2, 3, 2][..], &[4, 4, 3, 4], &[4, 3, 8], &[4, 3, 3, 4]] {
            let mut target = frame(&[8, 2], strings);
            let before = target.clone();
            let err = target.append(&frame(labels, strings)).unwrap_err();
            assert!(
                err.to_string().contains("an earlier row has that label"),
                "{err}"
            );
            assert!(target.equals(&before));
        }
    }
}

#[test]
fn retained_margin_errors_and_conversion_errors_keep_their_priority() {
    for strings in [false, true] {
        let mut target = frame(&[1, 2, 3], strings);
        let before = target.clone();
        let err = target
            .append_retaining(&frame(&[4, 1, 4], strings), Retained { from: 0, shown: 1 })
            .unwrap_err();
        assert!(err.to_string().contains("hidden margin"), "{err}");
        assert!(target.equals(&before));
        // Dropped labels are available again, but later incoming duplicates aren't.
        target
            .append_retaining(&frame(&[1, 4], strings), Retained { from: 1, shown: 1 })
            .unwrap();
        assert_eq!(target.height(), 5);
    }
    let mut target = frame(&[0, 1], false);
    let rows = DataFrame::new(
        vec!["x".into()],
        vec![Column::i64(vec![(1 << 53) + 1, 1])],
        Some(Index::int64(vec![0, 0])),
    )
    .unwrap();
    let before = target.clone();
    let err = target.append(&rows).unwrap_err();
    assert!(
        err.to_string().contains("not exactly representable"),
        "{err}"
    );
    assert!(target.equals(&before));
}

#[test]
fn incoming_resends_still_count_when_no_old_labels_are_retained() {
    let mut target = frame(&[1, 2], false);
    let before = target.clone();
    let err = target
        .append_retaining(&frame(&[2, 3, 2], false), Retained { from: 2, shown: 2 })
        .unwrap_err();
    assert!(err.to_string().contains("an earlier row has that label"));
    assert!(target.equals(&before));
}
