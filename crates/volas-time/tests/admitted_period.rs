//! Composed period folds retain the checked entry's timestamp/spec rules.
use volas_core::{Column, CombineOp, DataFrame, FoldLayout, Index, IndexKind, Tz};
use volas_time::{Agg, AggSpec, OpenPeriod, PeriodAggregate, TimeFrame, TimedBar};
const MIN: i64 = 60_000_000_000;
fn bar(t: i64, v: f64) -> DataFrame {
    DataFrame::new(vec!["volume".into()], vec![Column::f64(vec![v])], Some(Index::new(None, IndexKind::Datetime(vec![t], Tz::Utc)).unwrap())).unwrap()
}
#[test]
fn admitted_push_and_correction_match_the_raw_period() {
    let seed = bar(0, 1.);
    let mut open = OpenPeriod::new(seed.clone()).unwrap();
    let spec = AggSpec::ohlcv();
    let mut aggregate = PeriodAggregate::new(&open, &spec, TimeFrame::Min5, &Tz::Utc).unwrap();
    let mut target = aggregate.current().clone();
    assert!(aggregate.prefix(false).is_some());
    assert!(aggregate.prefix(true).is_none());
    assert_eq!(open.select(&["volume".into()]).unwrap().bars().width(), 1);
    let layout = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    for (t, value, resent) in [(MIN, 2., false), (MIN, 4., true), (2*MIN, 8., false)] {
        let rows = bar(t, value);
        let admitted = layout.admit(&target, &rows).unwrap();
        let proof = admitted.bar(0).unwrap();
        let timed = TimedBar::new(&proof, TimeFrame::Min5, &Tz::Utc).unwrap();
        assert_eq!(timed.key(), 0);
        let push = open.check_timed(&timed).unwrap();
        assert_eq!(push.replaces(), resent);
        aggregate.refold_bar(&mut target, &push).unwrap();
        push.land().unwrap();
        assert!(target.equals(aggregate.current()));
        assert!(aggregate.prefix(true).is_some());
    }
    assert_eq!(target.columns()[0].get_f64(0), 13.);
    assert_eq!(open.bars().height(), 3);
}
#[test]
fn first_correction_and_invalid_time_or_spec_preserve_checkpoints() {
    let seed = bar(0, 1.);
    let mut open = OpenPeriod::new(seed.clone()).unwrap();
    let spec = AggSpec::ohlcv();
    let mut aggregate = PeriodAggregate::new(&open, &spec, TimeFrame::Min5, &Tz::Utc).unwrap();
    let mut target = aggregate.current().clone();
    let layout = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let rows = bar(0, 4.);
    let admitted = layout.admit(&target, &rows).unwrap();
    let proof = admitted.bar(0).unwrap();
    let timed = TimedBar::new(&proof, TimeFrame::Min5, &Tz::Utc).unwrap();
    let push = open.check_timed(&timed).unwrap();
    assert!(push.replaces());
    aggregate.refold_bar(&mut target, &push).unwrap();
    push.land().unwrap();
    open.push_bar(&proof).unwrap();
    assert_eq!(target.columns()[0].get_f64(0), 4.);
    let wrong_grid = TimedBar::new(&proof, TimeFrame::Min1, &Tz::Utc).unwrap();
    assert!(aggregate.push_bar(&open.check_timed(&wrong_grid).unwrap()).is_err());
    let late = bar(5*MIN, 6.);
    let admitted = layout.admit(&target, &late).unwrap();
    let proof = admitted.bar(0).unwrap();
    let timed = TimedBar::new(&proof, TimeFrame::Min5, &Tz::Utc).unwrap();
    assert!(aggregate.push_bar(&open.check_timed(&timed).unwrap()).is_err());
    assert_eq!(aggregate.current().columns()[0].get_f64(0), 4.);
    let wrong_layout = FoldLayout::new(&target, |_| CombineOp::Replace).unwrap();
    let admitted = wrong_layout.admit(&target, &rows).unwrap();
    let proof = admitted.bar(0).unwrap();
    let timed = TimedBar::new(&proof, TimeFrame::Min5, &Tz::Utc).unwrap();
    assert!(aggregate.push_bar(&open.check_timed(&timed).unwrap()).is_err());
    let ranged = DataFrame::new(vec!["volume".into()], vec![Column::f64(vec![1.])], None).unwrap();
    let ranged_layout = FoldLayout::new(&ranged, |_| CombineOp::Sum).unwrap();
    let admitted = ranged_layout.admit(&ranged, &ranged).unwrap();
    let proof = admitted.bar(0).unwrap();
    assert!(TimedBar::new(&proof, TimeFrame::Min5, &Tz::Utc).is_err());
    let rows = bar(0, 2.);
    let admitted = layout.admit(&target, &rows).unwrap();
    let proof = admitted.bar(0).unwrap();
    assert!(TimedBar::new(&proof, TimeFrame::Min5, &Tz::Naive).is_err());
    // An independent checked restart retires the cached projections.
    let mut spec = spec;
    spec.set("volume", Agg::Sum);
    aggregate.check_restart(&bar(5*MIN, 1.), &spec).unwrap().land();
    assert_eq!(aggregate.current().columns()[0].get_f64(0), 1.);
}

#[test]
fn rollover_consumes_the_existing_bar_and_grid_admissions() {
    let seed = bar(0, 1.);
    let open = OpenPeriod::new(seed).unwrap();
    let mut aggregate = PeriodAggregate::new(&open, &AggSpec::ohlcv(), TimeFrame::Min5, &Tz::Utc).unwrap();
    let mut target = aggregate.current().clone();
    let layout = FoldLayout::new(&target, |_| CombineOp::Sum).unwrap();
    let rows = bar(5*MIN, f64::INFINITY);
    let admitted = layout.admit(&target, &rows).unwrap();
    let proof = admitted.bar(0).unwrap();
    let timed = TimedBar::new(&proof, TimeFrame::Min5, &Tz::Utc).unwrap();
    let wrong = TimedBar::new(&proof, TimeFrame::Min1, &Tz::Utc).unwrap();
    assert!(aggregate.restart_bar(&mut target, &wrong).is_err());
    aggregate.restart_bar(&mut target, &timed).unwrap();
    assert_eq!(target.height(), 2);
    assert!(target.columns()[0].get_f64(1).is_nan());
    assert!(aggregate.current().columns()[0].get_f64(0).is_nan());
    assert!(aggregate.restart_bar(&mut target, &timed).is_err());
    assert!(proof.reduced(Index::positional(0).unwrap()).is_err());
}

#[test]
fn admitted_cumulator_rejects_late_bars_before_rollover_or_fold() {
    for timestamp in [-MIN, MIN] {
        let mut cumulator = volas_time::Cumulator::new(TimeFrame::Min5, AggSpec::ohlcv());
        cumulator.append(&bar(0, 1.)).unwrap();
        cumulator.append(&bar(2 * MIN, 2.)).unwrap();
        let before = cumulator.frame().unwrap();
        assert!(cumulator.append(&bar(timestamp, 9.)).is_err());
        assert!(before.equals(&cumulator.frame().unwrap()));
    }
}
