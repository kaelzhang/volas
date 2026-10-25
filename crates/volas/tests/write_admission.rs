use volas::{Column, DataFrame, Index, IndexKind, Tz};
use volas::time::OpenPeriod;

#[test]
fn rejected_append_cannot_publish_an_extended_index() {
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], None).unwrap();
    let rows = DataFrame::new(vec!["x".into()], vec![Column::str(vec!["bad".into()])], None).unwrap();
    assert!(target.append(&rows).is_err());
    assert_eq!(target.index().len(), target.height());
    assert_eq!(target.column("x").unwrap().as_i64(), Some(&[1][..]));
}

#[test]
fn column_scatter_rejects_all_invalid_positions_before_writing() {
    let mut col = Column::i64(vec![1, 2]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| col.scatter(&[0, 2], &Column::i64(vec![9]))));
    assert!(result.is_ok(), "scatter must return an error rather than panic");
    assert!(result.unwrap().is_err());
    assert_eq!(col.as_i64(), Some(&[1, 2][..]));
}

#[test]
fn column_scatter_rejects_short_sources_before_writing() {
    let mut col = Column::i64(vec![1, 2, 3]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| col.scatter(&[0, 1, 2], &Column::i64(vec![9, 8]))));
    assert!(result.is_ok(), "scatter must check source length before writing");
    assert!(result.unwrap().is_err());
    assert_eq!(col.as_i64(), Some(&[1, 2, 3][..]));
}

#[test]
fn open_period_rejects_unadmitted_schema_atomically() {
    let index = |t| Some(Index::new(None, IndexKind::Datetime(vec![t], Tz::Utc)).unwrap());
    let seed = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], index(1)).unwrap();
    let bad = DataFrame::new(vec!["x".into()], vec![Column::str(vec!["bad".into()])], index(2)).unwrap();
    let mut open = OpenPeriod::new(seed).unwrap();
    assert!(open.push(&bad).is_err());
    assert_eq!(open.bars().index().len(), open.bars().height());
    assert_eq!(open.last_ts(), 1);
    assert_eq!(open.bars().column("x").unwrap().as_i64(), Some(&[1][..]));
}

#[test]
fn folded_rows_check_every_column_before_publishing_any_value() {
    let mut target = DataFrame::new(vec!["x".into(), "y".into()], vec![Column::i64(vec![1]), Column::i64(vec![2])], None).unwrap();
    let bad = DataFrame::new(vec!["x".into(), "y".into()], vec![Column::i64(vec![9]), Column::str(vec!["bad".into()])], None).unwrap();
    let op = volas::time::Agg::Last.as_combine_op();
    assert!(target.check_folded_row(&bad, &[(0, 0, op), (1, 1, op)]).is_err());
    assert_eq!(target.column("x").unwrap().as_i64(), Some(&[1][..]));
}

fn invalid_column_write(action: usize) {
    let op = volas::time::Agg::Sum.as_combine_op();
        let mut col = Column::i64(vec![1, 2]);
        let src = Column::i64(vec![9]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match action {
            0 => col.set_last_from(&src, 1),
            1 => col.write_rows(3, &src),
            2 => col.check_fold_cell(0, op, &src, 1, true).and_then(|write| write.land()),
            3 => col.check_fold_cell(2, op, &src, 0, false).and_then(|write| write.land()),
            _ => col.check_reduced_cell(0, op, &src, 1, false).and_then(|write| write.land()),
        }));
        assert!(result.is_ok(), "action {action} panicked");
        assert!(result.unwrap().is_err());
        assert_eq!(col.as_i64(), Some(&[1, 2][..]));
}

#[test] fn last_write_rejects_invalid_source() { invalid_column_write(0); }
#[test] fn range_write_rejects_invalid_start() { invalid_column_write(1); }
#[test] fn init_rejects_invalid_source() { invalid_column_write(2); }
#[test] fn combine_rejects_invalid_target() { invalid_column_write(3); }
#[test] fn reduced_combine_rejects_invalid_source() { invalid_column_write(4); }

#[test]
fn aggregate_push_does_not_treat_another_schema_as_admitted_bars() {
    let make = |name: &str, value, time| DataFrame::new(vec![name.into()], vec![Column::i64(vec![value])],
        Some(Index::new(None, IndexKind::Datetime(vec![time], Tz::Utc)).unwrap())).unwrap();
    let seed = make("close", 1, 1);
    let (_, _, aggregate) = volas::time::Cumulator::from_frame(&seed, volas::TimeFrame::Min1, volas::time::AggSpec::ohlcv()).unwrap().into_parts().unwrap();
    let mut aggregate = aggregate.unwrap();
    let before = aggregate.current().clone();
    assert!(aggregate.check_push(&make("other", 9, 2), false).is_err());
    assert!(aggregate.current().equals(&before));
}

#[test]
fn aggregate_restart_cannot_publish_an_unchecked_shape() {
    let index = || Some(Index::new(None, IndexKind::Datetime(vec![1], Tz::Utc)).unwrap());
    let seed = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1])], index()).unwrap();
    let (_, _, aggregate) = volas::time::Cumulator::from_frame(&seed, volas::TimeFrame::Min1, volas::time::AggSpec::ohlcv()).unwrap().into_parts().unwrap();
    let mut aggregate = aggregate.unwrap();
    let before = aggregate.current().clone();
    let bad = DataFrame::new(vec!["other".into()], vec![Column::i64(vec![9])], index()).unwrap();
    assert!(aggregate.check_restart(&bad, &volas::time::AggSpec::ohlcv()).is_err());
    assert!(aggregate.current().equals(&before));
}

fn invalid_aggregate_timestamp(timestamp: i64, resent: bool) {
    let make = |time, value| DataFrame::new(vec!["close".into()], vec![Column::i64(vec![value])],
        Some(Index::new(None, IndexKind::Datetime(vec![time], Tz::Utc)).unwrap())).unwrap();
    let seed = make(1, 1);
    let (_, _, aggregate) = volas::time::Cumulator::from_frame(&seed, volas::TimeFrame::Min1, volas::time::AggSpec::ohlcv()).unwrap().into_parts().unwrap();
    let mut aggregate = aggregate.unwrap();
    let before = aggregate.current().clone();
    assert!(aggregate.check_push(&make(timestamp, 9), resent).is_err());
    assert!(aggregate.current().equals(&before));
}
#[test] fn aggregate_push_rejects_a_different_period() { invalid_aggregate_timestamp(60_000_000_001, false); }
#[test] fn aggregate_push_rejects_a_false_resend() { invalid_aggregate_timestamp(2, true); }
#[test] fn aggregate_push_rejects_an_older_bar() { invalid_aggregate_timestamp(0, false); }

#[test]
fn checked_fold_plans_reject_bad_shapes_and_positions_atomically() {
    let make = |name: &str, values| DataFrame::new(vec![name.into()], vec![Column::i64(values)], None).unwrap();
    let mut target = make("x", vec![1]);
    let src = make("x", vec![2]);
    let op = volas::time::Agg::Last.as_combine_op();
    let before = target.clone();
    for ops in [vec![(1, 0, op)], vec![(0, 1, op)], vec![(0, 0, op), (0, 0, op)]] {
        assert!(target.check_folded_row(&src, &ops).is_err());
        assert!(target.equals(&before));
    }
    assert!(target.check_forming_fold(1, &src, 0, &[(0, 0, op)]).is_err());
    assert!(target.check_folded_row(&make("x", vec![]), &[(0, 0, op)]).is_err());
    let mut empty = make("x", vec![]);
    assert!(empty.check_folded_row(&src, &[(0, 0, op)]).is_err());
    for mut checkpoint in [make("y", vec![1]), make("x", vec![1, 2])] {
        assert!(target.fold_with_checkpoint(&mut checkpoint, &src, &[(0, 0, op)], &[false], false).is_err());
        assert!(target.equals(&before));
    }
    assert!(target.fold_with_checkpoint(&mut before.clone(), &src, &[(0, 0, op)], &[], false).is_err());
    target.check_folded_row(&src, &[(0, 0, op)]).unwrap().land().unwrap();
    assert_eq!(target.columns()[0].as_i64(), Some(&[2][..]));
    let sum = volas::time::Agg::Sum.as_combine_op();
    target.check_forming_fold(0, &src, 0, &[(0, 0, sum)]).unwrap().land().unwrap();
    assert_eq!(target.columns()[0].as_i64(), Some(&[4][..]));
}

#[test]
fn replacing_then_appending_materializes_a_range_only_once() {
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1, 2, 3])], None).unwrap();
    let rows = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![9, 8, 7])],
        Some(Index::new(None, IndexKind::Int64(vec![2, 3, 4])).unwrap())).unwrap();
    target.append(&rows).unwrap();
    assert_eq!(target.index().to_i64_labels(), [0, 1, 2, 3, 4]);
    assert_eq!(target.columns()[0].as_i64(), Some(&[1, 2, 9, 8, 7][..]));
}

#[test]
fn aggregate_admissions_preserve_schema_grid_and_latest_bar() {
    use volas::time::{Agg, AggSpec, Cumulator};
    let make = |time, values: Vec<i64>| DataFrame::new(vec!["close".into()], vec![Column::i64(values.clone())],
        Some(Index::new(None, IndexKind::Datetime((0..values.len()).map(|i| time + i as i64).collect(), Tz::Utc)).unwrap())).unwrap();
    let spec = AggSpec::ohlcv();
    let (_, _, aggregate) = Cumulator::from_frame(&make(1, vec![1]), volas::TimeFrame::Min1, spec.clone()).unwrap().into_parts().unwrap();
    let mut aggregate = aggregate.unwrap();
    assert!(aggregate.check_push(&make(2, vec![]), false).is_err());
    assert!(aggregate.check_push(&make(2, vec![2, 3]), false).is_err());
    let mut different = spec.clone(); different.set("close", Agg::Sum);
    assert!(aggregate.check_restart(&make(60_000_000_001, vec![3]), &different).is_err());
    assert!(aggregate.check_restart(&make(2, vec![3]), &spec).is_err());
    aggregate.check_push(&make(1, vec![2]), true).unwrap().land().unwrap();
    aggregate.check_push(&make(2, vec![3]), false).unwrap().land().unwrap();
    aggregate.check_push(&make(2, vec![4]), true).unwrap().land().unwrap();
    assert_eq!(aggregate.current().columns()[0].as_i64(), Some(&[4][..]));
    let bar = make(60_000_000_001, vec![5]);
    let restart = aggregate.check_restart(&bar, &spec).unwrap();
    assert_eq!(restart.current().index().to_i64_labels(), [60_000_000_000]);
    restart.land();
    aggregate.check_push(&bar, true).unwrap().land().unwrap();
    assert_eq!(aggregate.current().columns()[0].as_i64(), Some(&[5][..]));
}

#[test]
fn checked_reduction_validates_the_whole_selection_before_publication() {
    let op = volas::time::Agg::Sum.as_combine_op();
    let source = Column::i64(vec![2, 3]);
    let mut target = Column::i64(vec![10]);
    assert!(target.check_reduction(0, op, &source, &[0, 2]).is_err());
    assert_eq!(target.as_i64(), Some(&[10][..]));
    let overflow = Column::i64(vec![i64::MAX, 1]);
    assert!(target.check_reduction(0, op, &overflow, 0..2).is_err());
    assert_eq!(target.as_i64(), Some(&[10][..]));
    target.check_reduction(0, op, &source, 0..2).unwrap().land().unwrap();
    assert_eq!(target.as_i64(), Some(&[5][..]));
}
