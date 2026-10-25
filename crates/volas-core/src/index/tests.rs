//! Index behaviour: construction, lookup, slicing, alignment and the append rule.

use super::*;

#[test]
fn from_datetime_and_int_columns() {
    assert_eq!(
        Index::from_column(&Column::datetime(vec![5, 6])).unwrap(),
        Index::new(None, crate::IndexKind::Datetime(vec![5, 6], Tz::Naive)).unwrap()
    );
    assert_eq!(
        Index::from_column(&Column::i64(vec![1, 2])).unwrap(),
        Index::new(None, crate::IndexKind::Int64(vec![1, 2])).unwrap()
    );
}

#[test]
fn from_unsupported_column_errors() {
    // float / bool columns are not valid labels (string is — see below)
    assert!(Index::from_column(&Column::f64(vec![1.0])).is_err());
    assert!(Index::from_column(&Column::bool(vec![true])).is_err());
}

#[test]
fn is_empty_labels_and_position_of() {
    assert!(Index::new(None, crate::IndexKind::Range { start: 0, len: 0 }).unwrap().is_empty());
    assert!(!Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().is_empty());

    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().to_i64_labels(), vec![0, 1, 2]);
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![5, 6])).unwrap().to_i64_labels(), vec![5, 6]);
    assert_eq!(
        Index::new(None, crate::IndexKind::Datetime(vec![10, 20], Tz::Utc)).unwrap().to_i64_labels(),
        vec![10, 20]
    );

    let i64 = Label::I64;
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 5 }).unwrap().position_of(&i64(3)), Some(3));
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 5 }).unwrap().position_of(&i64(9)), None);
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 5 }).unwrap().position_of(&i64(-1)), None);
    assert_eq!(
        Index::new(None, crate::IndexKind::Int64(vec![10, 20, 30])).unwrap().position_of(&i64(20)),
        Some(1)
    );
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![10, 20])).unwrap().position_of(&i64(99)), None);
    assert_eq!(
        Index::new(None, crate::IndexKind::Datetime(vec![100, 200], Tz::Utc)).unwrap().position_of(&i64(200)),
        Some(1)
    );

    // take() on an Int64 index gathers the labels at those positions
    assert_eq!(
        Index::new(None, crate::IndexKind::Int64(vec![10, 20, 30])).unwrap().take(&[2, 0]).unwrap(),
        Index::new(None, crate::IndexKind::Int64(vec![30, 10])).unwrap()
    );
}

fn str_index(labels: &[&str]) -> Index {
    Index::new(None, crate::IndexKind::Str(labels.iter().map(|s| s.to_string()).collect())).unwrap()
}

#[test]
fn string_index_construction_and_ops() {
    // a string column becomes a string index
    let ix = Index::from_column(&Column::str(vec!["a".into(), "b".into()])).unwrap();
    assert_eq!(ix, str_index(&["a", "b"]));

    let ix = str_index(&["a", "b", "c", "d"]);
    assert_eq!(ix.len(), 4);
    assert!(!ix.is_empty());
    assert_eq!(ix.label_at(2), Label::Str("c".into()));
    assert_eq!(ix.slice(1, 3), str_index(&["b", "c"]));
    assert_eq!(ix.take(&[3, 0]).unwrap(), str_index(&["d", "a"]));
}

#[test]
fn string_index_lookup_and_slice() {
    let ix = str_index(&["aa", "bb", "cc", "dd"]);
    // exact lookup, kind-matched
    assert_eq!(ix.position_of(&Label::Str("cc".into())), Some(2));
    assert_eq!(ix.position_of(&Label::Str("zz".into())), None);
    // a numeric label against a string index never matches
    assert_eq!(ix.position_of(&Label::I64(1)), None);
    // lexicographic slice [bb, cc]
    let lo = Label::Str("bb".into());
    let hi = Label::Str("cc".into());
    let (lo, hi) = (SliceBound::text(lo), SliceBound::text(hi));
    assert_eq!(ix.label_slice(Some(&lo), Some(&hi)).unwrap(), (1, 3));
    // open-ended upper bound: [cc, end)
    assert_eq!(ix.label_slice(Some(&lo), None).unwrap(), (1, 4));
}

#[test]
fn append_rule_per_kind() {
    let ny = Tz::parse("America/New_York").unwrap();
    let aware = Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap();
    let naive = Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Naive)).unwrap();
    let msg = |a: &Index, b: &Index| a.check_append(b).unwrap_err().to_string();
    // the same instants in another zone are fine; awareness must match
    assert!(aware.check_append(&Index::new(None, crate::IndexKind::Datetime(vec![3], ny)).unwrap()).is_ok());
    assert!(msg(&aware, &Index::new(None, crate::IndexKind::Datetime(vec![3], Tz::Naive)).unwrap()).contains("tz_localize the rows"));
    assert!(msg(&naive, &Index::new(None, crate::IndexKind::Datetime(vec![3], Tz::Utc)).unwrap()).contains("tz_localize the frame"));
    // NaT has no place in an index, so rejection precedes append admission.
    assert!(Index::new(None, IndexKind::Datetime(vec![NAT], Tz::Utc)).is_err());
    // an empty RangeIndex is still a RangeIndex
    assert!(msg(&Index::new(None, crate::IndexKind::Range { start: 0, len: 0 }).unwrap(), &aware).contains("RangeIndex"));
    // a datetime index takes datetime rows only, and the reverse
    assert!(msg(&aware, &Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).contains("label the rows with their timestamps"));
    assert!(msg(&aware, &Index::new(None, crate::IndexKind::Int64(vec![3])).unwrap()).contains("integer index"));
    assert!(msg(&Index::new(None, crate::IndexKind::Int64(vec![1])).unwrap(), &aware).contains("tz-aware DatetimeIndex"));
    assert!(msg(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap(), &naive).contains("tz-naive DatetimeIndex"));
    // a string index takes string rows only, and the reverse
    assert!(msg(&str_index(&["x"]), &Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).contains("string index"));
    assert!(msg(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap(), &str_index(&["x"])).contains("RangeIndex"));
    assert!(str_index(&["x"]).check_append(&str_index(&["y"])).is_ok());
    assert!(Index::new(None, crate::IndexKind::Range { start: 0, len: 0 }).unwrap().check_append(&Index::new(None, crate::IndexKind::Int64(vec![4])).unwrap()).is_ok());
}

#[test]
fn unlabelled_rows_continue_unique_checked_integer_labels() {
    // continuing onto a label already present (the labels need not be sorted)
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![1, 0])).unwrap().continuation_clash(1, 0..2), Some(1));
    // only the given rows count (a window's retained rows)
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![1, 0])).unwrap().continuation_clash(1, 1..2), None);
    // continuing past i64::MAX overflows instead of wrapping
    let mut ix = Index::new(None, crate::IndexKind::Int64(vec![i64::MAX])).unwrap();
    let err = ix.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).unwrap_err();
    assert!(matches!(err, VolasError::Overflow(_)), "{err}");
    assert_eq!(ix, Index::new(None, crate::IndexKind::Int64(vec![i64::MAX])).unwrap());
    let err = Index::new(None, crate::IndexKind::Int64(vec![i64::MAX - 1])).unwrap()
        .check_append(&Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap())
        .unwrap_err();
    assert!(matches!(err, VolasError::Overflow(_)), "{err}");
    // the last continued label may be i64::MAX itself; no rows continue nothing
    let mut top = Index::new(None, crate::IndexKind::Int64(vec![i64::MAX - 1])).unwrap();
    top.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).unwrap();
    assert_eq!(top, Index::new(None, crate::IndexKind::Int64(vec![i64::MAX - 1, i64::MAX])).unwrap());
    top.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 0 }).unwrap()).unwrap();
    assert_eq!(top.len(), 2);
    // an empty Int64 index continues from 0
    assert!(Index::new(None, crate::IndexKind::Int64(vec![])).unwrap().check_append(&Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap()).is_ok());
    // the largest label follows every extension, so a repeat is still found after
    // the labels stop increasing
    let mut ix = Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap();
    ix.extend(&Index::new(None, crate::IndexKind::Int64(vec![9, 5])).unwrap()).unwrap(); // [0, 1, 9, 5]: 9 is the largest
    assert_eq!(ix.continuation_clash(4, 0..4), Some(9));
    ix.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap()).unwrap(); // continues 6, 7, 8
    assert_eq!(ix, Index::new(None, crate::IndexKind::Int64(vec![0, 1, 9, 5, 6, 7, 8])).unwrap());
    assert_eq!(ix.continuation_clash(1, 0..ix.len()), Some(9)); // 9 again
    ix.extend(&Index::new(None, crate::IndexKind::Int64(vec![20])).unwrap()).unwrap();
    ix.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).unwrap(); // 21: past the largest, no scan needed
    assert_eq!(ix.to_i64_labels().last(), Some(&21));
    // derived indexes keep their own largest label
    assert_eq!(ix.slice(0, 4).continuation_clash(4, 0..4), Some(9));
    assert_eq!(ix.take(&[0, 1]).unwrap().continuation_clash(1, 0..2), None);
    // other kinds, no rows, and an overflowing continuation find no clash here
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().continuation_clash(1, 0..3), None);
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![1, 0])).unwrap().continuation_clash(0, 0..2), None);
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![5, 3, i64::MAX])).unwrap().continuation_clash(1, 0..3), None);
}

#[test]
fn extend_grows_in_place_per_kind() {
    // same-kind grows the buffer in place (the live append hot path)
    let mut r = Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap();
    r.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap()).unwrap();
    assert_eq!(r, Index::new(None, crate::IndexKind::Range { start: 0, len: 5 }).unwrap());

    let mut d = Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap();
    d.extend(&Index::new(None, crate::IndexKind::Datetime(vec![3], Tz::Utc)).unwrap()).unwrap();
    assert_eq!(d, Index::new(None, crate::IndexKind::Datetime(vec![1, 2, 3], Tz::Utc)).unwrap());

    let mut s = str_index(&["a", "b"]);
    s.extend(&str_index(&["c"])).unwrap();
    assert_eq!(s, str_index(&["a", "b", "c"]));

    // a rejected extension leaves the index unchanged
    let mut keep = Index::new(None, crate::IndexKind::Datetime(vec![1], Tz::Utc)).unwrap();
    assert!(keep.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).is_err());
    assert_eq!(keep, Index::new(None, crate::IndexKind::Datetime(vec![1], Tz::Utc)).unwrap());

    // explicit integer labels after implicit positions: the positions become labels
    let mut m = Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap();
    m.extend(&Index::new(None, crate::IndexKind::Int64(vec![5, 6])).unwrap()).unwrap();
    assert_eq!(m, Index::new(None, crate::IndexKind::Int64(vec![0, 1, 5, 6])).unwrap());

    // an implicit Range appended to explicit Int64 labels CONTINUES the sequence
    // (last + 1, ...) rather than restarting at the Range's own 0 — the
    // sliced/compacted-frame + bare-bar append path (windowed rolling buffers).
    let mut c = Index::new(None, crate::IndexKind::Int64(vec![44, 45, 46])).unwrap();
    c.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap()).unwrap();
    assert_eq!(c, Index::new(None, crate::IndexKind::Int64(vec![44, 45, 46, 47, 48])).unwrap());
    // continuing from an empty Int64 starts at 0.
    let mut e = Index::new(None, crate::IndexKind::Int64(vec![])).unwrap();
    e.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap()).unwrap();
    assert_eq!(e, Index::new(None, crate::IndexKind::Int64(vec![0, 1])).unwrap());

    // mixing string with numeric is an error, either way
    assert!(str_index(&["x"]).extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap()).is_err());
    assert!(Index::new(None, crate::IndexKind::Range { start: 0, len: 1 }).unwrap().extend(&str_index(&["x"])).is_err());
}

#[test]
fn alignment_is_equal_labels_with_matching_awareness() {
    let ny = Tz::parse("America/New_York").unwrap();
    let named = Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap().with_name(Some("t".into()));
    // same instants: the name and the display zone do not matter
    assert!(named
        .require_aligned(&Index::new(None, crate::IndexKind::Datetime(vec![1, 2], ny)).unwrap())
        .is_ok());
    assert!(Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap()
        .require_aligned(&Index::new(None, crate::IndexKind::Int64(vec![0, 1])).unwrap())
        .is_ok());
    // naive vs aware never aligns, even over the same stored values
    let err = Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Naive)).unwrap()
        .require_aligned(&named)
        .unwrap_err();
    assert!(err.to_string().contains("tz-naive"), "{err}");
    let err = Index::new(None, crate::IndexKind::Str(vec!["a".into()])).unwrap().require_aligned(&Index::new(None, crate::IndexKind::Str(vec!["b".into()])).unwrap());
    assert!(err.unwrap_err().to_string().contains("different indexes"));
}

#[test]
fn name_set_and_propagates_through_ops() {
    let ix = Index::new(None, crate::IndexKind::Datetime(vec![1, 2, 3], Tz::Utc)).unwrap().with_name(Some("date".into()));
    assert_eq!(ix.name(), Some("date"));
    // an unnamed index reports None
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().name(), None);
    // the name rides through identity-preserving ops
    assert_eq!(ix.slice(0, 2).name(), Some("date"));
    assert_eq!(ix.take(&[0, 2]).unwrap().name(), Some("date"));
    assert_eq!(ix.clone().with_tz(Tz::Offset(28800)).name(), Some("date"));
    // extend keeps the growing index's name
    let mut g = ix.clone();
    g.extend(&Index::new(None, crate::IndexKind::Datetime(vec![4], Tz::Utc)).unwrap()).unwrap();
    assert_eq!(g.name(), Some("date"));
    // with_name(None) clears it
    assert_eq!(ix.with_name(None).name(), None);
}

#[test]
fn label_accessors_and_numeric_label_at() {
    // accessors return None on the other variant
    assert_eq!(Label::I64(5).as_i64(), Some(5));
    assert_eq!(Label::I64(5).as_str(), None);
    assert_eq!(Label::Str("x".into()).as_str(), Some("x"));
    assert_eq!(Label::Str("x".into()).as_i64(), None);
    // label_at over the numeric index kinds
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().label_at(2), Label::I64(2));
    assert_eq!(Index::new(None, crate::IndexKind::Int64(vec![10, 20])).unwrap().label_at(1), Label::I64(20));
    assert_eq!(
        Index::new(None, crate::IndexKind::Datetime(vec![100, 200], Tz::Utc)).unwrap().label_at(0),
        Label::I64(100)
    );
}

#[test]
fn index_kind_branch_coverage() {
    // tz() / with_tz() are no-ops on a non-datetime index.
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().tz(), Tz::Naive);
    assert!(matches!(
        Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().with_tz(Tz::Utc).kind,
        IndexKind::Range { start: 0, len: 3 }
    ));
    // slice over the non-range kinds.
    assert_eq!(
        Index::new(None, crate::IndexKind::Int64(vec![1, 2, 3])).unwrap().slice(0, 2),
        Index::new(None, crate::IndexKind::Int64(vec![1, 2])).unwrap()
    );
    assert!(matches!(
        Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap().slice(0, 1).kind,
        IndexKind::Datetime(_, _)
    ));
    assert_eq!(
        Index::new(None, crate::IndexKind::Str(vec!["a".into(), "b".into()])).unwrap().slice(1, 2),
        Index::new(None, crate::IndexKind::Str(vec!["b".into()])).unwrap()
    );
    // argsort lexicographically over a string index.
    assert_eq!(
        Index::new(None, crate::IndexKind::Str(vec!["b".into(), "a".into()])).unwrap().argsort(true),
        vec![1, 0]
    );
    // to_column over every kind.
    assert_eq!(Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap().to_column().len(), 2);
    assert_eq!(Index::new(None, crate::IndexKind::Datetime(vec![5], Tz::Utc)).unwrap().to_column().len(), 1);
    assert_eq!(Index::new(None, crate::IndexKind::Str(vec!["x".into()])).unwrap().to_column().len(), 1);
}

#[test]
fn label_eq_value_semantics() {
    // a RangeIndex equals the same integer labels materialized as Int64
    assert!(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().label_eq(&Index::new(None, crate::IndexKind::Int64(vec![0, 1, 2])).unwrap()));
    assert!(!Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().label_eq(&Index::new(None, crate::IndexKind::Int64(vec![0, 1, 9])).unwrap()));
    // datetime and string indexes compare by value
    assert!(Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap().label_eq(&Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap()));
    assert!(!Index::new(None, crate::IndexKind::Datetime(vec![1, 2], Tz::Utc)).unwrap().label_eq(&Index::new(None, crate::IndexKind::Datetime(vec![1, 9], Tz::Utc)).unwrap()));
    assert!(Index::new(None, crate::IndexKind::Str(vec!["a".into()])).unwrap().label_eq(&Index::new(None, crate::IndexKind::Str(vec!["a".into()])).unwrap()));
    assert!(!Index::new(None, crate::IndexKind::Str(vec!["a".into()])).unwrap().label_eq(&Index::new(None, crate::IndexKind::Str(vec!["b".into()])).unwrap()));
    // different label kinds are never equal
    assert!(!Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap().label_eq(&Index::new(None, crate::IndexKind::Datetime(vec![0, 1], Tz::Utc)).unwrap()));
    assert!(!Index::new(None, crate::IndexKind::Str(vec!["a".into()])).unwrap().label_eq(&Index::new(None, crate::IndexKind::Int64(vec![0])).unwrap()));
}

#[test]
fn a_sliced_range_keeps_its_start_and_its_labels() {
    // a slice of positions stays positional, from its start (pandas RangeIndex)
    let tail = Index::new(None, crate::IndexKind::Range { start: 0, len: 10 }).unwrap().slice(7, 10);
    assert!(matches!(tail.kind(), IndexKind::Range { start: 7, len: 3 }));
    assert_eq!(tail.label_at(0), Label::I64(7));
    assert_eq!(tail.position_of(&Label::I64(9)), Some(2));
    assert_eq!(tail.position_of(&Label::I64(6)), None);
    assert_eq!(tail.to_i64_labels(), vec![7, 8, 9]);
    // as a frame it continues with fresh (unlabelled) rows
    let mut grown = tail.clone();
    grown.extend(&Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap()).unwrap();
    assert!(matches!(
        grown.kind(),
        IndexKind::Range { start: 7, len: 5 }
    ));
    // as rows it carries its labels: explicit integer labels, never a continuation
    assert!(!tail.is_unlabelled() && Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap().is_unlabelled());
    let mut ints = Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap();
    ints.extend(&tail).unwrap();
    assert_eq!(ints, Index::new(None, crate::IndexKind::Int64(vec![0, 1, 2, 7, 8, 9])).unwrap());
    let err = Index::new(None, crate::IndexKind::Datetime(vec![1], Tz::Utc)).unwrap()
        .check_append(&tail)
        .unwrap_err();
    assert!(err.to_string().contains("integer index"), "{err}");
    // a label slice reads the labels, not the positions
    assert_eq!(
        tail.label_slice(Some(&SliceBound::exact(Label::I64(8))), None)
            .unwrap(),
        (1, 3)
    );
}
