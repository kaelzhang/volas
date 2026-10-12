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
    ix.label_slice(lo.map(Label::I64).as_ref(), hi.map(Label::I64).as_ref())
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
    assert_eq!(order(&int(&[1, 2]).append(&int(&[0])).unwrap()), (false, false));
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

    // an implicit Range bar continues Int64 labels upward
    let mut c = int(&[5, 4]);
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
    t.extend(&Index::datetime(vec![NAT], Tz::Utc)).unwrap();
    assert_eq!(order(&t), (false, false));

    // a numeric-kind mix recomputes over the collapsed Int64 labels
    let mut m = Index::range(2);
    m.extend(&Index::datetime(vec![-1], Tz::Utc)).unwrap();
    assert_eq!(order(&m), (false, false));
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
    let (lo, hi) = (Label::Str("bb".into()), Label::Str(String::new()));
    assert_eq!(s.label_slice(Some(&lo), Some(&hi)).unwrap(), (1, 3));
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
    let (a, z) = (Label::Str("a".into()), Label::Str("z".into()));
    assert_eq!(s.label_slice(Some(&a), None).unwrap(), (1, 3));
    assert!(matches!(
        s.label_slice(Some(&a), Some(&z)),
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
    let s = Label::Str("a".into());
    for ix in [Index::range(2), int(&[1]), Index::datetime(vec![1], Tz::Utc)] {
        assert!(matches!(ix.label_slice(Some(&s), None), Err(VolasError::DType(_))));
        assert!(matches!(ix.label_slice(None, Some(&s)), Err(VolasError::DType(_))));
    }
    let i = Label::I64(1);
    assert!(matches!(strs(&["a"]).label_slice(Some(&i), None), Err(VolasError::DType(_))));
    assert!(matches!(strs(&["a"]).label_slice(None, Some(&i)), Err(VolasError::DType(_))));
}
