//! Count actual label materializations during admission, excluding the necessary
//! label copy at commit. No timer or machine-load-dependent assertion is used.
use super::*;
use std::cell::Cell;

thread_local! { static LABEL_READS: Cell<usize> = const { Cell::new(0) }; }
pub(super) fn record_label_read() { LABEL_READS.with(|n| n.set(n.get() + 1)); }

#[test]
fn datetime_new_rows_admission_reads_only_the_boundary() {
    for held in [0, 100_000] {
        let target = Index::new(None, IndexKind::Datetime((0..held).collect(), Tz::Utc)).unwrap();
        for count in [0, 1, 10_000] {
            let incoming = Index::new(None, IndexKind::Datetime((held..held + count).collect(), Tz::Utc)).unwrap();
            LABEL_READS.with(|n| n.set(0));
            let plan = target.plan_append(&incoming, AppendMode::NewRows).unwrap();
            assert_eq!(plan.appended(), count as usize);
            let reads = LABEL_READS.with(Cell::get);
            assert!(reads <= 2, "a checked datetime batch needs only its boundary: {count} incoming labels caused {reads} label reads");
        }
    }
}

#[test]
fn new_rows_keep_non_datetime_order_and_error_rules() {
    let target = Index::new(None, IndexKind::Int64(vec![1, 2])).unwrap();
    let rows = Index::new(None, IndexKind::Int64(vec![3, 4])).unwrap();
    assert!(target.plan_append(&rows, AppendMode::NewRows).is_ok());
    for labels in [vec![4, 3], vec![1], vec![2]] {
        let rows = Index::new(None, IndexKind::Int64(labels)).unwrap();
        assert!(target.plan_append(&rows, AppendMode::NewRows).is_err());
    }
    let empty = Index::new(None, IndexKind::Str(vec![])).unwrap();
    let first = Index::new(None, IndexKind::Str(vec!["a".into()])).unwrap();
    assert!(empty.plan_append(&first, AppendMode::NewRows).is_ok());
    let second = Index::new(None, IndexKind::Str(vec!["b".into()])).unwrap();
    let result = first.plan_append(&second, AppendMode::NewRows);
    assert!(matches!(result, Err(VolasError::Value(message)) if message.contains("does not follow the last row")));
}
