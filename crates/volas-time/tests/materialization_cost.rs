//! One-shot output owns its closed buffers, with room for the forming row.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{Column, DataFrame, DType, Index, Tz};
use volas_time::{cumulate, AggSpec, Cumulator, TimeFrame};

#[test]
fn one_shot_cumulation_does_not_copy_and_regrow_closed_output() {
    let n = 30_000;
    let names = ["open", "high", "low", "close", "volume"].map(String::from).to_vec();
    let columns = names.iter().map(|_| Column::f64(vec![1.; n])).collect();
    let fine = DataFrame::new(names, columns, Some(Index::datetime(
        (0..n).map(|i| i as i64 * 60_000_000_000).collect(), Tz::Utc,
    ))).unwrap();
    let (out, calls, bytes) = measure(|| cumulate(&fine, TimeFrame::Min1, &AggSpec::ohlcv()).unwrap());
    assert!(out.equals(&fine));
    assert!(bytes < n * 170, "{calls} allocations, {bytes} bytes");
}

#[test]
fn first_append_after_materialization_retains_live_capacity() {
    for dtype in [DType::F64, DType::F32, DType::I64, DType::I32, DType::Bool, DType::Utf8, DType::Datetime] {
        let n = 30_000;
        let col = match dtype {
            DType::Utf8 => Column::str(vec!["bar".into(); n]),
            DType::Datetime => Column::datetime(vec![0; n]),
            _ => Column::f64(vec![1.; n]).cast(dtype).unwrap(),
        };
        let fine = DataFrame::new(vec!["v".into()], vec![col], Some(Index::datetime(
            (0..n).map(|i| i as i64 * 60_000_000_000).collect(), Tz::Utc,
        ))).unwrap();
        let bar = DataFrame::new(vec!["v".into()], vec![fine.columns()[0].slice(0, 1)],
            Some(Index::datetime(vec![n as i64 * 60_000_000_000], Tz::Utc))).unwrap();
        let mut out = cumulate(&fine, TimeFrame::Min1, &AggSpec::ohlcv()).unwrap();
        let (_, calls, bytes) = measure(|| out.append(&bar).unwrap());
        assert_eq!(out.height(), n + 1);
        assert_eq!(out.columns()[0].dtype(), dtype);
        assert_eq!(calls, 0, "{dtype}: {calls} allocations / {bytes} bytes on first append");
    }
}

#[test]
fn incremental_snapshot_keeps_its_original_allocation_budget() {
    let n = 30_000;
    let names = ["open", "high", "low", "close", "volume"].map(String::from).to_vec();
    let fine = DataFrame::new(names.clone(), names.iter().map(|_| Column::f64(vec![1.; n])).collect(),
        Some(Index::datetime((0..n).map(|i| i as i64 * 60_000_000_000).collect(), Tz::Utc))).unwrap();
    let (out, calls, bytes) = measure(|| {
        let mut cum = Cumulator::new(TimeFrame::Min1, AggSpec::ohlcv());
        cum.append(&fine).unwrap();
        cum.frame().unwrap()
    });
    assert!(out.equals(&fine));
    assert!(bytes < n * 280, "{calls} allocations / {bytes} bytes");
}
