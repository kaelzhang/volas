//! The index's label order and the label slice that searches by it (pandas
//! `slice_locs`): ordered search on a monotonic index, exact bounds otherwise.

use super::*;

fn int(v: &[i64]) -> Index {
    Index::int64(v.to_vec())
}

fn strs(v: &[&str]) -> Index {
    Index::str(v.iter().map(|s| s.to_string()).collect())
}

fn order(ix: &Index) -> (bool, bool) {
    (ix.order.increasing, ix.order.decreasing)
}

fn sl(ix: &Index, lo: Option<i64>, hi: Option<i64>) -> Result<(usize, usize)> {
    let b = |x: Option<i64>| x.map(|x| SliceBound::exact(Label::I64(x)));
    ix.label_slice(b(lo).as_ref(), b(hi).as_ref())
}

fn text(s: &str) -> SliceBound {
    SliceBound::text(Label::Str(s.into()))
}

fn key_err(r: Result<(usize, usize)>, needle: &str) -> bool {
    matches!(r, Err(VolasError::Key(m)) if m.contains(needle))
}

#[test]
fn order_is_computed_for_every_kind() {
    assert_eq!(order(&Index::range(0)), (true, true));
    assert_eq!(order(&Index::range(1)), (true, true));
    assert_eq!(order(&Index::range(4)), (true, false));
    assert_eq!(order(&int(&[1, 2, 2, 5])), (true, false));
    assert_eq!(order(&int(&[5, 2, 2, 1])), (false, true));
    assert_eq!(order(&int(&[2, 0, 1])), (false, false));
    assert_eq!(order(&int(&[7, 7])), (true, true));
    assert_eq!(order(&strs(&["a", "b"])), (true, false));
    assert_eq!(order(&strs(&["c", "a", "b"])), (false, false));
    assert_eq!(order(&Index::datetime(vec![1, 2], Tz::Utc)), (true, false));
    // a NaT makes a datetime index unordered, though it is the smallest i64
    assert_eq!(order(&Index::datetime(vec![NAT, 1, 2], Tz::Utc)), (false, false));
    assert_eq!(order(&Index::datetime(vec![NAT], Tz::Utc)), (false, false));
}

#[test]
fn derived_indexes_recompute_their_order() {
    let ix = int(&[3, 0, 1, 2]);
    assert_eq!(order(&ix.slice(1, 4)), (true, false));
    assert_eq!(order(&ix.take(&[0, 1])), (false, true));
    assert_eq!(order(&Index::range(5).slice(2, 4)), (true, false));
    let mut grown = int(&[1, 2]);
    grown.extend(&int(&[0])).unwrap();
    assert_eq!(order(&grown), (false, false));
    let from = Index::from_column(&Column::i64(vec![3, 1])).unwrap();
    assert_eq!(order(&from), (false, true));
    // the order is a fact of the labels: equal labels give equal indexes
    assert_eq!(int(&[0, 1, 2]).take(&[0, 1, 2]), int(&[0, 1, 2]));
}

#[test]
fn extend_carries_the_order_forward() {
    let mut ix = int(&[1, 2]);
    ix.extend(&int(&[3])).unwrap();
    assert_eq!(order(&ix), (true, false));
    ix.extend(&int(&[0])).unwrap();
    assert_eq!(order(&ix), (false, false));
    // once unordered, appending in order cannot restore it
    ix.extend(&int(&[10, 11])).unwrap();
    assert_eq!(order(&ix), (false, false));

    let mut d = int(&[9]);
    d.extend(&int(&[8, 8])).unwrap();
    assert_eq!(order(&d), (false, true));

    let mut r = Index::range(1);
    assert_eq!(order(&r), (true, true));
    r.extend(&Index::range(2)).unwrap();
    assert_eq!(order(&r), (true, false));

    // an implicit Range bar continues Int64 labels upward (from the last, 5: unused)
    let mut c = int(&[7, 4]);
    c.extend(&Index::range(1)).unwrap();
    assert_eq!(order(&c), (false, false));
    let mut e = int(&[]);
    e.extend(&Index::range(2)).unwrap();
    assert_eq!(order(&e), (true, false));

    let mut s = strs(&["a"]);
    s.extend(&strs(&["c", "b"])).unwrap();
    assert_eq!(order(&s), (false, false));

    let mut t = Index::datetime(vec![1], Tz::Utc);
    t.extend(&Index::datetime(vec![2], Tz::Utc)).unwrap();
    assert_eq!(order(&t), (true, false));
    // a NaT label is never appended (it has no place in the order); built in, it
    // leaves the index unordered
    assert!(t.extend(&Index::datetime(vec![NAT], Tz::Utc)).is_err());
    assert_eq!(order(&t), (true, false));
    assert_eq!(order(&Index::datetime(vec![1, 2, NAT], Tz::Utc)), (false, false));

    // explicit integer labels after implicit positions recompute over the collapsed
    // Int64 labels (a datetime index is another kind: never mixed in)
    let mut m = Index::range(2);
    m.extend(&int(&[-1])).unwrap();
    assert_eq!(order(&m), (false, false));
    assert!(Index::range(2).extend(&Index::datetime(vec![-1], Tz::Utc)).is_err());
}

#[test]
fn increasing_index_searches_by_order() {
    let ix = int(&[1, 2, 2, 5]);
    assert_eq!(sl(&ix, Some(2), Some(4)).unwrap(), (1, 3));
    assert_eq!(sl(&ix, Some(0), Some(9)).unwrap(), (0, 4)); // bounds need not exist
    assert_eq!(sl(&ix, Some(5), Some(1)).unwrap(), (3, 3)); // empty, never inverted
    assert_eq!(sl(&ix, None, None).unwrap(), (0, 4));
    let r = Index::range(4);
    assert_eq!(sl(&r, Some(1), Some(2)).unwrap(), (1, 3));
    assert_eq!(sl(&r, Some(-5), Some(i64::MAX)).unwrap(), (0, 4));
    assert_eq!(sl(&r, Some(9), None).unwrap(), (4, 4));
    assert_eq!(sl(&r, None, Some(-1)).unwrap(), (0, 0));
}

#[test]
fn decreasing_index_searches_in_reverse() {
    let ix = int(&[3, 2, 1]);
    assert_eq!(sl(&ix, Some(2), Some(1)).unwrap(), (1, 3));
    assert_eq!(sl(&ix, Some(5), Some(0)).unwrap(), (0, 3));
    assert_eq!(sl(&ix, Some(1), Some(2)).unwrap(), (2, 2));
    assert_eq!(sl(&ix, None, Some(2)).unwrap(), (0, 2));
    let s = strs(&["c", "b", "a"]);
    assert_eq!(s.label_slice(Some(&text("bb")), Some(&text(""))).unwrap(), (1, 3));
}

#[test]
fn unordered_index_locates_exact_bounds() {
    let ix = int(&[2, 0, 1]);
    assert_eq!(sl(&ix, Some(0), Some(1)).unwrap(), (1, 3));
    assert_eq!(sl(&ix, Some(1), Some(0)).unwrap(), (2, 2));
    assert_eq!(sl(&ix, None, Some(0)).unwrap(), (0, 2));
    assert_eq!(sl(&ix, Some(0), None).unwrap(), (1, 3));
    assert!(key_err(
        sl(&ix, Some(0), Some(5)),
        "Cannot get right slice bound for non-monotonic index with a missing label 5."
    ));
    assert!(key_err(sl(&ix, Some(-1), None), "left slice bound"));
    let s = strs(&["c", "a", "b"]);
    assert_eq!(s.label_slice(Some(&text("a")), None).unwrap(), (1, 3));
    assert!(matches!(
        s.label_slice(Some(&text("a")), Some(&text("z"))),
        Err(VolasError::Key(m)) if m.contains("missing label 'z'")
    ));
    // a repeated bound on an unordered datetime index is ambiguous
    let t = Index::datetime(vec![30, 10, 20, 10], Tz::Utc);
    assert!(key_err(sl(&t, Some(10), None), "non-unique label: 1970-01-01 00:00:00.000000010+00:00"));
    assert_eq!(sl(&t, Some(30), Some(20)).unwrap(), (0, 3));
    let n = Index::datetime(vec![NAT, 1, 2], Tz::Naive);
    assert_eq!(sl(&n, Some(1), Some(2)).unwrap(), (1, 3));
    assert!(key_err(sl(&n, Some(1), Some(5)), "missing label 1970-01-01 00:00:00.000000005."));
}

#[test]
fn a_bound_of_the_other_label_kind_is_a_dtype_error() {
    let s = text("a");
    for ix in [Index::range(2), int(&[1]), Index::datetime(vec![1], Tz::Utc)] {
        assert!(matches!(
            ix.label_slice(Some(&s), None),
            Err(VolasError::DType(m)) if m.contains("indexers [a] of type str")
        ));
        assert!(matches!(ix.label_slice(None, Some(&s)), Err(VolasError::DType(_))));
    }
    let i = SliceBound::exact(Label::I64(1));
    assert!(matches!(
        strs(&["a"]).label_slice(Some(&i), None),
        Err(VolasError::DType(m)) if m.contains("indexers [1] of type int")
    ));
    assert!(matches!(strs(&["a"]).label_slice(None, Some(&i)), Err(VolasError::DType(_))));
}

#[test]
fn a_text_bound_on_a_non_increasing_datetime_index_fails_closed() {
    let day = |d: i64| SliceBound::text(Label::I64(d));
    let partial = |r: Result<(usize, usize)>| {
        matches!(r, Err(VolasError::Key(m)) if m.starts_with("Value based partial slicing"))
    };
    for labels in [vec![3, 2, 1], vec![2, 1, 3], vec![NAT, 1, 2]] {
        let ix = Index::datetime(labels, Tz::Utc);
        assert!(partial(ix.label_slice(Some(&day(2)), None)));
        assert!(partial(ix.label_slice(None, Some(&day(2)))));
        // an exact (non-text) bound keeps the exact-label rule
        let exact = SliceBound::exact(Label::I64(2));
        assert!(ix.label_slice(Some(&exact), None).is_ok());
    }
    // an increasing datetime index searches text bounds by order, as before
    let inc = Index::datetime(vec![1, 2, 4], Tz::Utc);
    assert_eq!(inc.label_slice(Some(&day(2)), Some(&day(3))).unwrap(), (1, 2));
}
