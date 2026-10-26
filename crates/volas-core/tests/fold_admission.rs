//! A bar carries its checked schema through displayed and checkpoint updates.
use volas_core::{Column, CombineOp, DataFrame, FoldLayout, Index, IndexKind, Tz, History, Reach};
use std::sync::Arc;
fn frame(values: Vec<f64>) -> DataFrame {
    let n = values.len();
    DataFrame::new(vec!["x".into()], vec![Column::f64(values)], Some(Index::new(None, IndexKind::Datetime((0..n as i64).collect(), Tz::Utc)).unwrap())).unwrap()
}

#[test]
fn schema_changes_retire_proofs_but_value_writes_keep_them() {
    let mut target = frame(vec![1.]);
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    target.set_column("x", Column::f64(vec![2.])).unwrap();
    assert!(plan.matches(&target));
    target.set_column("x", Column::i64(vec![2])).unwrap();
    assert!(!plan.matches(&target));
    assert!(plan.admit(&target, &frame(vec![1.])).is_err());
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    target.set_computed("x", Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    assert!(!plan.matches(&target));
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    assert!(!plan.matches(&target.take(&[0]).unwrap()));
    target.invalidate_computed_on_write("x");
    assert!(!plan.matches(&target));
}

#[test]
fn admitted_bar_updates_without_allocating_or_rechecking_positions() {
    let mut target = frame(vec![2.]);
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let rows = frame(vec![3.]);
    let batch = plan.admit(&target, &rows).unwrap();
    assert!(batch.bar(1).is_err());
    let bar = batch.bar(0).unwrap();
    let mut current = frame(vec![2.]);
    let mut before = current.clone();
    let p = bar.project(&current).unwrap();
    let b = bar.project(&before).unwrap();
    bar.check_from(&target, Some((&current, &p))).unwrap();
    bar.checkpoint(&mut current, &p, &mut before, &b, &[false], false).unwrap();
    bar.publish(&mut target, &current, &p).unwrap();
    assert_eq!(target.column("x").unwrap().as_f64().unwrap(), &[5.]);
    bar.fold(&mut target).unwrap();
    assert_eq!(target.column("x").unwrap().as_f64().unwrap(), &[8.]);
    bar.reduce(&mut current, &p).unwrap();
    assert_eq!(current.column("x").unwrap().as_f64().unwrap(), &[3.]);
    before.set_column("x", Column::f64(vec![f64::NAN])).unwrap();
    bar.checkpoint(&mut current, &p, &mut before, &b, &[true], true).unwrap();
    assert!(current.column("x").unwrap().get_f64(0).is_nan());
    let q = bar.project(&target).unwrap();
    bar.append(&mut target, &q).unwrap(); // same timestamp replaces
    assert_eq!(target.height(), 1);
    assert_eq!(target.column("x").unwrap().get_f64(0), 3.);
}

#[test]
fn foreign_and_retired_layouts_are_rejected_before_writing() {
    let target = frame(vec![1.]);
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let rows = frame(vec![2., 3.]);
    let batch = plan.admit(&target, &rows).unwrap();
    let bar = batch.bar(0).unwrap();
    let mut foreign = frame(vec![9.]);
    let other = FoldLayout::new(&foreign, |_| CombineOp::Sum).unwrap();
    assert!(bar.fold(&mut foreign).is_err());
    assert!(bar.publish(&mut foreign, &target, &plan).is_err());
    assert!(bar.check_from(&target, Some((&foreign, &other))).is_err());
    let p = bar.project(&foreign).unwrap();
    let mut before = foreign.clone();
    assert!(bar.checkpoint(&mut foreign, &p, &mut before, &p, &[], false).is_err());
    let empty = frame(vec![]);
    let empty_plan = bar.project(&empty).unwrap();
    assert!(bar.check_from(&target, Some((&empty, &empty_plan))).is_err());
    assert!(bar.publish(&mut foreign, &empty, &empty_plan).is_err());
    foreign.set_column("x", Column::str(vec!["bad".into()])).unwrap();
    assert!(bar.project(&foreign).is_err());
    assert!(bar.append(&mut foreign, &p).is_err());
    let empty = DataFrame::new(vec![], vec![], Some(Index::positional(1).unwrap())).unwrap();
    assert!(bar.project(&empty).is_err());
    let unsupported = DataFrame::new(vec!["x".into()], vec![Column::str(vec!["s".into()])], None).unwrap();
    assert!(FoldLayout::new(&unsupported, |_| CombineOp::Sum).is_err());
}

#[test]
fn integer_overflow_is_atomic_across_checkpoint_and_destination() {
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![i64::MAX])], None).unwrap();
    let mut current = target.clone();
    let mut before = target.clone();
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let rows = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], None).unwrap();
    let batch = plan.admit(&target, &rows).unwrap();
    let bar = batch.bar(0).unwrap();
    let p = bar.project(&current).unwrap();
    assert!(bar.check_from(&target, Some((&current, &p))).is_err());
    assert!(bar.checkpoint(&mut current, &p, &mut before, &p, &[false], false).is_err());
    assert_eq!(current.column("x").unwrap().as_i64().unwrap(), &[i64::MAX]);
    assert_eq!(before.column("x").unwrap().as_i64().unwrap(), &[i64::MAX]);
    bar.fold(&mut target).unwrap_err();
    assert_eq!(target.column("x").unwrap().as_i64().unwrap(), &[i64::MAX]);
}

struct NoReplay;
impl volas_core::Replay for NoReplay {
    fn boundary(&self, _: &DataFrame, _: &str, _: usize) -> Option<volas_core::Checkpoint> { None }
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> { None }
}

#[test]
fn projected_writes_admit_their_own_retained_history() {
    let mut parent = DataFrame::new(vec!["x".into(), "cache".into()], vec![Column::f64(vec![1., 2., 3.]), Column::f64(vec![1., 2., 3.])], None).unwrap();
    parent.set_computed("cache", Arc::new(()), Reach { lookback: 1, resume: 1, continuation: 1, history: History::Window });
    let slice = parent.rows(2, 3, &NoReplay);
    let mut before = slice.forming_fold_probe();
    assert!(before.lead_rows() > 0);
    let mut current = before.clone();
    let plan = FoldLayout::new(&current, |_| CombineOp::Sum).unwrap();
    let rows = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![4.])], None).unwrap();
    let batch = plan.admit(&current, &rows).unwrap();
    let bar = batch.bar(0).unwrap();
    let p = bar.project(&current).unwrap();
    let b = bar.project(&before).unwrap();
    bar.checkpoint(&mut current, &p, &mut before, &b, &[false], false).unwrap();
    assert_eq!(current.columns()[0].get_f64(0), 7.);
    bar.append(&mut before, &b).unwrap();
    assert_eq!(before.height(), 2);
    let next = bar.reduced(Index::new(None, IndexKind::Range { start: 4, len: 1 }).unwrap()).unwrap();
    next.append_new(&mut before).unwrap();
    assert_eq!(before.height(), 3);
}

#[test]
fn independent_prefix_admission_retains_its_full_checks() {
    let target = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![i64::MAX])], None).unwrap();
    let source = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], None).unwrap();
    let ops = [(0, 0, CombineOp::Sum)];
    assert!(target.check_fold_from(Some(&target), &source, &ops).is_err());
    target.check_fold_from(None, &source, &ops).unwrap();
    assert!(target.check_fold_from(None, &source, &[(0, 1, CombineOp::Sum)]).is_err());
    let empty = frame(vec![]);
    assert!(empty.check_fold_from(None, &frame(vec![1.]), &ops).is_err());
    let floats = frame(vec![1.]);
    floats.check_fold_from(Some(&floats), &frame(vec![2.]), &ops).unwrap();
}

#[path = "support/allocation.rs"]
mod allocation;
#[test]
fn warm_admitted_float_fold_has_no_per_bar_allocation() {
    let mut target = frame(vec![1.]);
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let rows = frame(vec![2.]);
    let admitted = plan.admit(&target, &rows).unwrap();
    let bar = admitted.bar(0).unwrap();
    bar.fold(&mut target).unwrap();
    let (_, allocations, bytes) = allocation::measure(|| {
        for _ in 0..100 { bar.fold(&mut target).unwrap(); }
    });
    assert_eq!((allocations, bytes), (0, 0));
    assert_eq!(target.columns()[0].get_f64(0), 203.);
}

#[test]
fn range_step_rejects_capacity_before_publishing_the_index() {
    for (start, held, added, message) in [
        (0, isize::MAX as usize + 1, isize::MAX as usize + 1, "range index length overflow"),
        (i64::MAX, 1, 1, "range index labels overflow int64"),
    ] {
        let mut target = DataFrame::new(vec![], vec![], Some(Index::new(None, IndexKind::Range { start, len: held }).unwrap())).unwrap();
        let rows = DataFrame::new(vec![], vec![], Some(Index::positional(added).unwrap())).unwrap();
        let index = target.index().clone();
        assert!(target.append(&rows).unwrap_err().to_string().contains(message));
        assert_eq!(target.height(), held);
        assert_eq!(target.index(), &index);
        let retained = volas_core::Retained { from: 0, shown: 0 };
        assert!(target.append_retaining(&rows, retained, &NoReplay).unwrap_err().to_string().contains(message));
        assert_eq!(target.index(), &index);
    }
}

#[test]
fn dropping_a_computed_cell_retires_its_previous_fold_layout() {
    let mut target = DataFrame::new(vec!["x".into(), "cache".into()], vec![Column::f64(vec![1.]), Column::f64(vec![2.])], None).unwrap();
    target.set_computed("cache", Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    let layout = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    target.check_writes(vec![volas_core::CellWrite { col: 1, rows: vec![0], values: Column::f64(vec![3.]) }]).unwrap().land();
    assert!(!target.is_computed("cache"));
    assert!(!layout.matches(&target));
    assert_eq!(target.columns()[1].get_f64(0), 3.);
}

#[test]
fn incoming_computed_values_become_complete_admitted_data() {
    let mut source = DataFrame::new(vec!["x".into(), "y".into()], vec![Column::f64(vec![2., 3.]), Column::f64(vec![5., 7.])], None).unwrap();
    source.set_computed("y", Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    let mut target = DataFrame::new(vec!["x".into(), "y".into()], vec![Column::f64(vec![0.]), Column::f64(vec![0.])], None).unwrap();
    let layout = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let admitted = layout.admit(&target, &source).unwrap();
    assert!(source.is_computed("y"));
    assert!(!admitted.rows().is_computed("y"));
    for i in 0..2 { admitted.bar(i).unwrap().fold(&mut target).unwrap(); }
    assert_eq!(target.columns()[0].get_f64(0), 5.);
    assert_eq!(target.columns()[1].get_f64(0), 12.);
}

#[test]
fn virtual_range_replacement_needs_no_label_materialization() {
    let held = isize::MAX as usize + 1;
    let mut target = DataFrame::new(vec![], vec![], Some(Index::new(None, IndexKind::Range { start: 0, len: held }).unwrap())).unwrap();
    let last = i64::try_from(held - 1).unwrap();
    let row = DataFrame::new(vec![], vec![], Some(Index::new(None, IndexKind::Int64(vec![last])).unwrap())).unwrap();
    let index = target.index().clone();
    target.append(&row).unwrap();
    assert_eq!(target.height(), held);
    assert!(std::sync::Arc::ptr_eq(target.index(), &index));
}

#[test]
fn admitted_numeric_proofs_keep_float32_and_mixed_overflow_atomicity() {
    let make = |a, b| DataFrame::new(vec!["a".into(), "b".into()], vec![a, b], None).unwrap();
    let mut target = make(Column::f32(vec![2.]), Column::f64(vec![3.]));
    let rows = make(Column::f32(vec![4.]), Column::f64(vec![5.]));
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let batch = plan.admit(&target, &rows).unwrap();
    let bar = batch.bar(0).unwrap();
    let mut current = target.clone();
    let mut before = current.clone();
    let projected = bar.project(&current).unwrap();
    bar.fold(&mut target).unwrap();
    bar.checkpoint(&mut current, &projected, &mut before, &projected, &[false, false], false).unwrap();
    assert!(current.equals(&target));
    assert_eq!(target.columns()[0].get_f64(0), 6.);
    assert_eq!(target.columns()[1].get_f64(0), 8.);
    bar.check_from(&target, Some((&current, &projected))).unwrap();
    bar.reduce(&mut current, &projected).unwrap();
    assert_eq!(current.columns()[0].get_f64(0), 4.);

    let mut target = make(Column::f64(vec![2.]), Column::i64(vec![i64::MAX]));
    let snapshot = target.clone();
    let rows = make(Column::f64(vec![4.]), Column::i64(vec![1]));
    let plan = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let batch = plan.admit(&target, &rows).unwrap();
    let bar = batch.bar(0).unwrap();
    let mut current = target.clone();
    let mut before = current.clone();
    let projected = bar.project(&current).unwrap();
    assert!(bar.fold(&mut target).is_err());
    assert!(bar.check_from(&target, Some((&current, &projected))).is_err());
    assert!(bar.checkpoint(&mut current, &projected, &mut before, &projected, &[false, false], false).is_err());
    assert!(target.equals(&snapshot));
    assert!(current.equals(&snapshot));
    assert!(before.equals(&snapshot));
    bar.reduce(&mut current, &projected).unwrap();
    assert_eq!(current.columns()[0].get_f64(0), 4.);
    assert_eq!(current.columns()[1].as_i64().unwrap(), &[1]);
}

#[test]
fn checked_replacement_lands_converted_cells_once_and_keeps_missing_payloads() {
    let mut target = DataFrame::new(vec!["y".into(), "x".into(), "z".into(), "cache".into()],
        vec![Column::f64(vec![1.]), Column::i64(vec![2]), Column::bool(vec![false]), Column::f64(vec![99.])],
        Some(Index::new(None, IndexKind::Datetime(vec![17], Tz::Utc)).unwrap())).unwrap();
    target.set_computed("cache", Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    let source = DataFrame::new(vec!["extra".into(), "z".into(), "y".into(), "x".into(), "cache".into()],
        vec![Column::str(vec!["a".into(), "b".into()]), Column::bool(vec![false, true]), Column::i64(vec![7, 8]),
             Column::f64(vec![3., 4.]), Column::str(vec!["ignored".into(), "ignored".into()])],
        Some(Index::new(None, IndexKind::Str(vec!["a".into(), "b".into()])).unwrap())).unwrap();
    let index = target.index().clone();
    target.replace_last_row(&source, 1).unwrap();
    assert_eq!(target.columns()[0].get_f64(0), 8.);
    assert_eq!(target.columns()[1].as_i64().unwrap(), &[4]);
    assert_eq!(target.columns()[2].as_bool().unwrap(), &[true]);
    assert_eq!(target.columns()[3].get_f64(0), 99.);
    assert!(target.is_computed("cache"));
    assert!(Arc::ptr_eq(target.index(), &index));
    let before = target.clone();
    let bad = DataFrame::new(vec!["y".into(), "x".into()],
        vec![Column::i64(vec![9]), Column::f64(vec![4.5])], None).unwrap();
    assert!(target.replace_last_row(&bad, 0).is_err());
    assert!(target.equals(&before));
    assert!(target.is_computed("cache"));
    let absent = DataFrame::new(vec!["extra".into()], vec![Column::i64(vec![0])], None).unwrap();
    target.replace_last_row(&absent, 0).unwrap();
    assert!(target.columns()[0].get_f64(0).is_nan());
    assert!(!target.columns()[1].is_valid(0));
    assert_eq!(target.columns()[1].as_i64().unwrap(), &[4]);
    assert!(!target.columns()[2].is_valid(0));
    assert_eq!(target.columns()[2].as_bool().unwrap(), &[true]);
    assert_eq!(target.columns()[3].get_f64(0), 99.);
    assert!(Arc::ptr_eq(target.index(), &index));
}

#[test]
fn standalone_last_cell_admission_keeps_prefix_and_na_bits() {
    let mut target = Column::i64(vec![1, 2]);
    let source = Column::i64(vec![9]);
    target.set_last_from(&source, 0).unwrap();
    assert_eq!(target.as_i64().unwrap(), &[1, 9]);
    let before = target.clone();
    assert!(target.set_last_from(&source, 1).is_err());
    assert_eq!(target, before);
    assert!(Column::i64(vec![]).set_last_from(&source, 0).is_err());
    let missing = Column::na_of(volas_core::DType::I64, 1);
    target.set_last_from(&missing, 0).unwrap();
    assert!(target.is_valid(0));
    assert!(!target.is_valid(1));
    assert_eq!(target.as_i64().unwrap()[0], 1);
    assert_eq!(target.as_i64().unwrap()[1], missing.as_i64().unwrap()[0]);
}

#[test]
fn schema_preserving_derivation_reuses_proof_and_coercion_retires_it() {
    let source = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![3, 4])], None).unwrap();
    let proof = FoldLayout::new(&source, |_| CombineOp::Sum).unwrap();
    assert!(proof.matches(&source.take(&[1]).unwrap()));
    assert!(proof.matches(&source.slice_data(1, 2)));
    let target = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![0.])], None).unwrap();
    let converted = target.admit_data_rows(&source).unwrap();
    assert_eq!(converted.columns()[0].as_f64().unwrap(), &[3., 4.]);
    assert!(!proof.matches(&converted));
    assert!(proof.matches(&source));
    assert_eq!(source.columns()[0].as_i64().unwrap(), &[3, 4]);
}
