use volas_core::{Column, DataFrame, Index, Tz};
use volas_time::{aggregate_period, Agg, AggSpec, TimeFrame};

#[test]
fn empty_reduction_returns_an_error() {
    assert!(Agg::First.reduce(&Column::f64(vec![]), &[]).is_err());
}

#[test]
fn one_period_aggregator_rejects_multiple_periods() {
    let frame = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![10., 20.])],
        Some(Index::datetime(vec![0, 300_000_000_000], Tz::Utc))).unwrap();
    assert!(aggregate_period(&frame, &AggSpec::ohlcv(), TimeFrame::Min5, &Tz::Utc).is_err());
}

#[test]
fn reduction_rejects_out_of_range_positions_for_every_aggregator() {
    for agg in [Agg::First, Agg::Last, Agg::Min, Agg::Max, Agg::Sum] {
        let col = Column::i64(vec![1,2]);
        assert!(agg.reduce(&col, &[]).is_err());
        assert!(agg.reduce(&col, &[0,2]).is_err());
        assert!(agg.reduce(&col, &[usize::MAX]).is_err());
    }
}

#[test]
fn one_period_aggregator_rejects_unordered_and_missing_timestamps() {
    for ts in [vec![60_000_000_000, 0], vec![0,i64::MIN]] {
        let frame = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![10.,20.])], Some(Index::datetime(ts,Tz::Utc))).unwrap();
        for tf in [TimeFrame::Min5, TimeFrame::Day1] {
            assert!(aggregate_period(&frame,&AggSpec::ohlcv(),tf,&Tz::Utc).is_err());
        }
    }
}

#[test]
fn one_period_aggregator_keeps_last_duplicate_and_grid_label() {
    let frame = DataFrame::new(vec!["open".into(),"close".into()],
        vec![Column::f64(vec![1.,2.,3.]),Column::f64(vec![10.,20.,30.])],
        Some(Index::datetime(vec![60_000_000_000,60_000_000_000,120_000_000_000],Tz::Utc))).unwrap();
    let out = aggregate_period(&frame,&AggSpec::ohlcv(),TimeFrame::Min5,&Tz::Utc).unwrap();
    assert_eq!(out.column("open").unwrap().to_f64_vec(),[2.]);
    assert_eq!(out.column("close").unwrap().to_f64_vec(),[30.]);
    assert_eq!(out.index().to_column().to_f64_vec(),[0.]);
}
