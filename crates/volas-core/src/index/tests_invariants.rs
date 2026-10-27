//! Index owns construction and derivation invariants before a frame can exist.
use super::*;
use crate::DataFrame;
use crate::dataframe::NoReplay;

fn frame(kind: IndexKind) -> Result<DataFrame> {
    let index = Index::new(None, kind)?;
    DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1; index.len()])], Some(index))
}

#[test]
fn datetime_column_rejects_out_of_order_labels() {
    let err = Index::from_column(&Column::datetime(vec![2, 1])).unwrap_err();
    assert!(err.to_string().contains("monotonic"));
}

#[test]
fn datetime_column_rejects_nat() {
    let err = Index::from_column(&Column::datetime(vec![NAT])).unwrap_err();
    assert!(err.to_string().contains("without NaT"));
}

#[test]
fn frame_rejects_direct_datetime_disorder() {
    assert!(frame(IndexKind::Datetime(vec![2, 1], Tz::Utc)).is_err());
}

#[test]
fn frame_rejects_direct_datetime_nat() {
    assert!(frame(IndexKind::Datetime(vec![NAT], Tz::Utc)).is_err());
}

#[test]
fn frame_rejects_direct_datetime_duplicates() {
    let err = frame(IndexKind::Datetime(vec![1, 1], Tz::Utc)).unwrap_err();
    assert!(err.to_string().contains("duplicate labels"));
}

#[test]
fn frame_rejects_direct_integer_duplicates() {
    assert!(frame(IndexKind::Int64(vec![1, 1])).is_err());
}

#[test]
fn frame_rejects_direct_string_duplicates() {
    assert!(frame(IndexKind::Str(vec!["x".into(), "x".into()])).is_err());
}

#[test]
fn unique_unordered_integer_and_string_labels_are_valid() {
    assert!(frame(IndexKind::Int64(vec![2, 0, 1])).is_ok());
    assert!(frame(IndexKind::Str(vec!["z".into(), "a".into(), "m".into()])).is_ok());
}

#[test]
fn take_rechecks_labels_before_constructing_a_frame() {
    let datetime = frame(IndexKind::Datetime(vec![1, 2, 3], Tz::Utc)).unwrap();
    let invalid = [&[1usize, 0][..], &[0, 0], &[9]];
    for positions in invalid {
        assert!(datetime.index().take(positions).is_err());
        assert!(datetime.take(positions).is_err());
        assert!(datetime.take_data(positions).is_err());
        assert!(datetime.select_rows(positions, &NoReplay).is_err());
    }
    assert!(datetime.take(&[0, 2]).unwrap().index().is_strict_datetime());
    assert!(Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap().take(&[0, 0]).is_err());
    let string = Index::new(None, IndexKind::Str(vec!["a".into(), "b".into()])).unwrap();
    assert!(string.take(&[0, 0]).is_err());
}

#[test]
fn rejected_extension_leaves_each_kind_unchanged() {
    let kinds = [
        (IndexKind::Datetime(vec![2, 3], Tz::Utc), IndexKind::Datetime(vec![1, 4], Tz::Utc)),
        (IndexKind::Int64(vec![2, 1]), IndexKind::Int64(vec![1, 4])),
        (IndexKind::Str(vec!["b".into(), "a".into()]), IndexKind::Str(vec!["a".into(), "c".into()])),
        (IndexKind::Range { start: 0, len: 2 }, IndexKind::Int64(vec![1])),
        (IndexKind::Int64(vec![2, 1]), IndexKind::Range { start: 0, len: 1 }),
    ];
    for (mine, added) in kinds {
        let mut index = Index::new(None, mine).unwrap();
        let before = index.clone();
        assert!(index.extend(&Index::new(None, added).unwrap()).is_err());
        assert_eq!(index, before);
    }
}

#[test]
fn frame_admission_checks_the_stored_order_fact() {
    // Only the owning crate can forge this representation. Admission still
    // refuses a corrupt construction fact without scanning all labels again.
    let index = Index {
        name: None,
        kind: IndexKind::Datetime(vec![2, 1], Tz::Utc),
        order: LabelOrder::NONE,
    };
    assert!(DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1, 2])], Some(index)).is_err());
}

#[test]
fn explicit_ranges_check_label_overflow() {
    assert!(Index::new(None, IndexKind::Range { start: i64::MAX, len: 2 }).is_err());
    assert!(Index::new(None, IndexKind::Range { start: i64::MAX, len: 0 }).is_ok());
    assert!(!Index::new(None, crate::IndexKind::Range { start: 0, len: 0 }).unwrap().is_strict_datetime());
}

#[test]
fn range_extension_checks_overflow_without_mutating() {
    let mut index = Index::new(None, IndexKind::Range { start: i64::MAX, len: 1 }).unwrap();
    assert_eq!(index.to_i64_labels(), vec![i64::MAX]);
    assert_eq!(index.to_column().as_i64().unwrap(), &[i64::MAX]);
    let before = index.clone();
    assert!(index.extend(&Index::new(None, IndexKind::Range { start: 0, len: 1 }).unwrap()).is_err());
    assert_eq!(index, before);
    assert!(Index::new(None, IndexKind::Range { start: 0, len: usize::MAX }).is_err());
}

#[test]
fn contiguous_slices_inherit_valid_labels_and_clamp_bounds() {
    let datetime = Index::new(None, IndexKind::Datetime(vec![1, 2, 3], Tz::Utc)).unwrap();
    assert_eq!(datetime.slice(usize::MAX, 0).len(), 0);
    assert_eq!(datetime.slice(0, usize::MAX), datetime);
    assert!(datetime.slice(1, 0).is_strict_datetime());
    let last = Index::new(None, IndexKind::Range { start: i64::MAX, len: 1 }).unwrap();
    assert!(last.slice(1, 1).is_empty());
    assert!(last.slice(usize::MAX, 0).is_empty());
    let frame = frame(IndexKind::Datetime(vec![1], Tz::Utc)).unwrap();
    assert!(frame.select_rows(&[usize::MAX, 0], &NoReplay).is_err());
}

#[test]
fn range_label_slicing_handles_the_inclusive_last_int64_label() {
    let index = Index::new(None, IndexKind::Range { start: i64::MAX, len: 1 }).unwrap();
    let last = SliceBound::exact(Label::I64(i64::MAX));
    assert_eq!(index.label_slice(Some(&last), Some(&last)).unwrap(), (0, 1));
    let len = i64::MAX as usize + 1;
    let wide = Index::new(None, IndexKind::Range { start: 0, len }).unwrap();
    assert_eq!(wide.label_slice(None, Some(&last)).unwrap(), (0, len));
}

#[test]
fn checked_extension_rejects_incompatible_index_kinds() {
    let integer = Index::new(None, IndexKind::Int64(vec![1, 2])).unwrap();
    let string = Index::new(None, IndexKind::Str(vec!["a".into()])).unwrap();
    assert!(integer.plan_append(&string, AppendMode::Extend).is_err());
    assert!(string.plan_append(&integer, AppendMode::Extend).is_err());
    let explicit_range = Index::new(None, IndexKind::Range { start: 1, len: 1 }).unwrap();
    assert!(integer.plan_append(&explicit_range, AppendMode::Extend).is_err());
}

#[test]
fn integer_extension_rejects_unmaterializable_range_without_mutation() {
    // This nominal Range is valid (its last label is MAX) and needs no buffer.
    // Casting its length to i64 used to make the append iterator empty, so the
    // extension silently reported success without appending any labels.
    let added = i64::MAX as usize + 1;
    let range = Index::new(None, IndexKind::Range { start: 0, len: added }).unwrap();
    let mut integer = Index::new(None, IndexKind::Int64(vec![])).unwrap();
    let before = integer.clone();
    let result = integer.extend(&range);
    assert!(matches!(result, Err(VolasError::Overflow(_))), "{result:?}");
    assert_eq!(integer, before);
}

#[test]
fn wide_integer_continuation_checks_its_inclusive_end_without_truncation() {
    let added = i64::MAX as usize + 1;
    let integer = Index::new(None, IndexKind::Int64(vec![1, -1])).unwrap();
    assert_eq!(integer.continuation_clash(added, 0..2), Some(1));
    let range = Index::new(None, IndexKind::Range { start: 0, len: added }).unwrap();
    assert!(matches!(integer.plan_append(&range, AppendMode::Rows { from: 2, shown: 2 }), Err(VolasError::Overflow(_))));
}

#[test]
fn zero_column_frame_rejects_unmaterializable_append_without_mutation() {
    let added = i64::MAX as usize + 1;
    let index = Index::new(None, IndexKind::Range { start: 0, len: added }).unwrap();
    let rows = DataFrame::new(vec![], vec![], Some(index)).unwrap();
    let mut target = DataFrame::new(vec![], vec![], Some(Index::new(None, IndexKind::Int64(vec![])).unwrap())).unwrap();
    let before = target.clone();
    assert!(matches!(target.append(&rows), Err(VolasError::Overflow(_))));
    assert!(target.equals(&before));
    assert_eq!((target.height(), target.index().len()), (0, 0));
}

#[test]
fn wide_range_to_integer_extension_rejects_capacity_without_mutation() {
    let mut index = Index::new(None, IndexKind::Range { start: 0, len: i64::MAX as usize + 1 }).unwrap();
    let before = index.clone();
    let added = Index::new(None, IndexKind::Int64(vec![-1])).unwrap();
    assert!(matches!(index.extend(&added), Err(VolasError::Overflow(_))));
    assert_eq!(index, before);
}

#[test]
fn nonzero_wide_range_normalization_rejects_capacity_without_mutation() {
    let added = Index::new(None, IndexKind::Range { start: -1, len: i64::MAX as usize + 1 }).unwrap();
    let mut integer = Index::new(None, IndexKind::Int64(vec![])).unwrap();
    let before = integer.clone();
    assert!(matches!(integer.extend(&added), Err(VolasError::Overflow(_))));
    assert_eq!(integer, before);
}

#[test]
fn decreasing_integer_prefix_does_not_hide_late_duplicate_labels() {
    let error = Index::new(None, IndexKind::Int64(vec![4, 3, 2, 1, 2])).unwrap_err();
    assert!(error.to_string().contains("duplicate labels"));
}

#[test]
fn decreasing_string_prefix_does_not_hide_late_duplicate_labels() {
    let labels = ["e", "d", "c", "b", "c"].into_iter().map(String::from).collect();
    let error = Index::new(None, IndexKind::Str(labels)).unwrap_err();
    assert!(error.to_string().contains("duplicate labels"));
}

#[test]
fn range_append_column_capacity_error_preserves_frame_and_caches() {
    let rows = DataFrame::from_columns(vec![], vec![], 1usize << 60).unwrap();
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![1.])], None).unwrap();
    target.set_computed("x", std::sync::Arc::new("x".to_string()), crate::Reach { lookback: 0, resume: 0, continuation: 0, history: crate::History::Window });
    let before = target.clone();
    let result = target.append(&rows);
    assert!(matches!(result, Err(VolasError::Overflow(_))), "{result:?}");
    assert!(target.equals(&before));
    assert_eq!((target.height(), target.index().len()), (1, 1));
    assert!(std::sync::Arc::ptr_eq(target.index(), before.index()));
    assert_eq!(target.has_stale_computed(), before.has_stale_computed());
    assert_eq!(target.computed_since("x"), before.computed_since("x"));
    assert_eq!(target.series("x").unwrap().len(), 1);
    assert_eq!(target.index().len(), target.series("x").unwrap().len());
}

#[test]
fn typed_same_last_route_borrows_datetime_and_string_labels() {
    for (held, incoming, expected) in [
        (IndexKind::Datetime(vec![1, 2], Tz::Utc), IndexKind::Datetime(vec![2], Tz::Utc), true),
        (IndexKind::Datetime(vec![1, 2], Tz::Utc), IndexKind::Datetime(vec![3], Tz::Utc), false),
        (IndexKind::Str(vec!["a".into()]), IndexKind::Str(vec!["a".into()]), true),
        (IndexKind::Range { start: -2, len: 2 }, IndexKind::Int64(vec![-1]), true),
        (IndexKind::Int64(vec![2]), IndexKind::Range { start: 2, len: 1 }, true),
    ] {
        let held = Index::new(None, held).unwrap();
        let incoming = Index::new(None, incoming).unwrap();
        assert_eq!(held.admit_input(&incoming).unwrap().same_last(), expected);
    }
}

#[test]
fn every_storage_capacity_returns_error_without_publication() {
    let rows = DataFrame::from_columns(vec![], vec![], isize::MAX as usize).unwrap();
    let columns = [
        Column::f64(vec![1.]), Column::f32(vec![1.]), Column::i64(vec![1]),
        Column::i32(vec![1]), Column::bool(vec![true]), Column::str(vec!["x".into()]),
        Column::datetime(vec![1]), Column::i64_from_options([None]),
        Column::bool_from_options([None]), Column::str_from_options([None]),
    ];
    for column in columns {
        let mut target = DataFrame::new(vec!["x".into()], vec![column], None).unwrap();
        let before = target.clone();
        assert!(matches!(target.append(&rows), Err(VolasError::Overflow(_))));
        assert!(target.equals(&before));
        assert!(std::sync::Arc::ptr_eq(target.index(), before.index()));
        assert_eq!(target.height(), target.index().len());
    }
}

#[test]
fn physical_column_extent_carries_positional_capacity() {
    let column = Column::i64(vec![4, 5]);
    let extent = column.extent();
    assert_eq!(extent.len(), 2);
    let index = Index::for_extent(extent);
    assert_eq!(index.to_i64_labels(), [0, 1]);
    let empty = Column::str(vec![]);
    assert!(Index::for_extent(empty.extent()).is_empty());
    assert!(Index::positional(usize::MAX).is_err());
}

#[test]
fn completing_partial_rows_admits_missing_column_capacity() {
    let rows = DataFrame::from_columns(vec![], vec![], 1usize << 60).unwrap();
    let target = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![1.])], None).unwrap();
    let before = target.clone();
    assert!(matches!(target.admit_data_rows(&rows), Err(VolasError::Overflow(_))));
    assert!(target.equals(&before));
    assert!(std::sync::Arc::ptr_eq(target.index(), before.index()));
}
