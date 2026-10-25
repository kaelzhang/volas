use volas_core::ColumnView;
use crate::{aggregate_period, cumulate, Agg, AggSpec, Cumulator, TimeFrame};
use volas_core::{Column, DataFrame, Index, Tz, Validity, VolasError};

const MINUTE: i64 = 60_000_000_000;

fn frame(stamps: &[i64], columns: Vec<Column>) -> DataFrame {
    DataFrame::new(
        (0..columns.len()).map(|i| format!("v{i}")).collect(), columns,
        Some(Index::new(None, volas_core::IndexKind::Datetime(stamps.to_vec(), Tz::Utc)).unwrap().with_name(Some("time".into()))),
    ).unwrap()
}

fn spec(agg: Agg, count: usize) -> AggSpec {
    let mut spec = AggSpec::ohlcv();
    for i in 0..count { spec.set(format!("v{i}"), agg); }
    spec
}

fn same_bits(got: &DataFrame, want: &DataFrame) {
    assert_eq!(got.index(), want.index());
    assert_eq!(got.names(), want.names());
    for (got, want) in got.columns().iter().zip(want.columns()) {
        assert_eq!(got.dtype(), want.dtype());
        assert!(got.equals(want), "{got:?} != {want:?}");
        match (got.view(), want.view()) {
            (ColumnView::F64(a), ColumnView::F64(b)) => assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
            (ColumnView::F32(a), ColumnView::F32(b)) => assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
            _ => {}
        }
    }
}

fn columns() -> Vec<Column> {
    let valid = Validity::from_valid_iter(8, [true, true, false, true, true, true, true, true]);
    vec![
        Column::f64(vec![1e20, 3., f64::NAN, 4., -0., 0., -1e20, 1.]),
        Column::f32(vec![1e20, 3., f32::NAN, 4., -0., 0., -1e20, 1.]),
        Column::i64_with(vec![1, 3, i64::MAX, 4, 5, 6, 7, 8], valid.clone()).unwrap(),
        Column::i32_with(vec![1, 3, i32::MIN, 4, 5, 6, 7, 8], valid.clone()).unwrap(),
        Column::datetime(vec![8, 7, i64::MIN, 5, 4, 3, 2, 1]),
        Column::bool_with(vec![true, false, true, false, true, false, true, false], valid.clone()).unwrap(),
        Column::str_from_options([Some("a".into()), Some("长".into()), None, Some("four".into()), Some("".into()), Some("z".into()), Some("longer text".into()), Some("last".into())]),
    ]
}

#[test]
fn column_builders_match_the_period_fold_for_every_dtype_and_aggregator() {
    for agg in [Agg::First, Agg::Last, Agg::Max, Agg::Min, Agg::Sum] {
        for col in columns() {
            let raw = crate::RawFineInput::new(vec!["v0".into()], vec![col],
                vec![0, 0, MINUTE, MINUTE, 5 * MINUTE, 6 * MINUTE, 7 * MINUTE, 10 * MINUTE], Tz::Utc, Some("time".into())).unwrap();
            let df = &raw.frame;
            let spec = spec(agg, 1);
            if !agg.as_combine_op().supports(df.columns()[0].dtype()) {
                assert!(cumulate(df, TimeFrame::Min5, &spec).is_err());
                continue;
            }
            let mut want = aggregate_period(&df.slice_data(0, 2), &spec, TimeFrame::Min5, &Tz::Utc).unwrap();
            for (a, b) in [(2, 5), (5, 6)] {
                want.append(&aggregate_period(&df.slice_data(a, b), &spec, TimeFrame::Min5, &Tz::Utc).unwrap()).unwrap();
            }
            same_bits(&cumulate(df, TimeFrame::Min5, &spec).unwrap(), &want);
            let mut live = Cumulator::new(TimeFrame::Min5, spec.clone());
            live.append_raw(&raw).unwrap();
            let snapshot = live.frame().unwrap();
            let (owned, open, aggregate) = live.into_parts().unwrap();
            same_bits(&owned, &snapshot);
            assert!(open.is_some());
            assert!(aggregate.is_some());
        }
    }
}

#[test]
fn grouped_reductions_keep_missing_endpoints_and_the_first_failure_position() {
    let rows = [vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
    for col in columns() {
        for agg in [Agg::First, Agg::Last, Agg::Sum, Agg::Max, Agg::Min] {
            let got = col.reduce_groups(agg.as_combine_op(), rows.iter().map(Vec::as_slice));
            if !agg.as_combine_op().supports(col.dtype()) {
                assert_eq!(got.unwrap_err().0, 0);
                continue;
            }
            let mut want = agg.reduce(&col, &rows[0]).unwrap();
            want.append(&agg.reduce(&col, &rows[1]).unwrap()).unwrap();
            same_bits(&frame(&[0, MINUTE], vec![got.unwrap()]), &frame(&[0, MINUTE], vec![want]));
        }
    }
    let mut assignment = spec(Agg::Sum, 2);
    let same_period = frame(&[0, MINUTE, 5 * MINUTE], vec![Column::i64(vec![i64::MAX, 1, 0]), Column::i32(vec![i32::MAX, 1, 0])]);
    assert!(cumulate(&same_period, TimeFrame::Min5, &assignment).unwrap_err().to_string().contains("int64"));
    assignment.set("v0", Agg::Last);
    assert!(cumulate(&same_period, TimeFrame::Min5, &assignment).unwrap_err().to_string().contains("int32"));
}

#[test]
fn live_checkpoints_match_refolding_after_first_last_and_middle_corrections() {
    for agg in [Agg::First, Agg::Last, Agg::Max, Agg::Min, Agg::Sum] {
        for col in columns() {
            if !agg.as_combine_op().supports(col.dtype()) { continue; }
            let values = frame(&(0..8).map(|i| i * MINUTE).collect::<Vec<_>>(), vec![col]);
            let spec = spec(agg, 1);
            let mut live = Cumulator::new(TimeFrame::Day1, spec.clone());
            let mut raw = values.slice_data(0, 1);
            live.append(&raw).unwrap();
            for i in 0..8 {
                let bar = values.slice_data(i, i + 1);
                raw.append(&bar).unwrap();
                live.append(&bar).unwrap();
                // Revise first, middle, and last bars while each is latest, then
                // restore it: a missing/NaN/extreme last value must be removable.
                for j in [7, 2, i] {
                    let correction = frame(&[i as i64 * MINUTE], vec![values.columns()[0].slice(j, j + 1)]);
                    raw.append(&correction).unwrap();
                    live.append(&correction).unwrap();
                    same_bits(&live.last().unwrap().unwrap(), &aggregate_period(&raw, &spec, TimeFrame::Day1, &Tz::Utc).unwrap());
                }
            }
            let before = live.frame().unwrap();
            for i in [0, 3] {
                assert!(live.append(&values.slice_data(i, i + 1)).is_err());
                same_bits(&live.frame().unwrap(), &before);
            }
        }
    }
}

#[test]
fn overflow_priority_is_period_then_column_and_open_errors_stay_deferred() {
    let df = frame(&[0, MINUTE, 5 * MINUTE, 6 * MINUTE, 10 * MINUTE], vec![
        Column::i64(vec![0, 1, i64::MAX, 1, 0]),
        Column::i32(vec![i32::MAX, 1, 0, 1, 0]),
    ]);
    let spec = spec(Agg::Sum, 2);
    let err = cumulate(&df, TimeFrame::Min5, &spec).unwrap_err();
    assert!(err.to_string().contains("int32"), "{err}");
    let mut live = Cumulator::new(TimeFrame::Min5, spec.clone());
    live.append(&df.slice_data(0, 1)).unwrap();
    live.append(&df.slice_data(1, 2)).unwrap(); // only the open aggregate overflows
    assert!(matches!(live.last(), Err(VolasError::Overflow(_))));
    let mut failing = Cumulator::new(TimeFrame::Min5, spec.clone());
    failing.append(&df.slice_data(0, 2)).unwrap();
    assert!(matches!(failing.into_parts(), Err(VolasError::Overflow(_))));
    let batch = Cumulator::from_frame(&df.slice_data(0, 2), TimeFrame::Min5, spec.clone()).unwrap();
    assert!(matches!(batch.into_parts(), Err(VolasError::Overflow(_))));
    let held = live.open_period().unwrap().bars().clone();
    assert!(live.append(&df.slice_data(2, 3)).is_err()); // cannot close it
    assert!(live.open_period().unwrap().bars().equals(&held));
    let fix = frame(&[MINUTE], vec![Column::i64(vec![1]), Column::i32(vec![-1])]);
    live.append(&fix).unwrap();
    assert!(live.last().is_ok());
    live.append(&df.slice_data(2, 3)).unwrap();
    assert_eq!(live.frame().unwrap().height(), 2);
}

#[test]
fn float_fold_order_nan_payloads_and_signed_zeros_are_unchanged() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0042);
    for agg in [Agg::First, Agg::Last, Agg::Max, Agg::Min, Agg::Sum] {
        for values in [vec![-0., 0., -0.], vec![1e20, 1., -1e20], vec![f64::INFINITY, f64::NEG_INFINITY, nan], vec![nan, 2., 3.]] {
            let raw = frame(&[0, MINUTE, 2 * MINUTE], vec![Column::f32(values.iter().map(|&x| x as f32).collect()), Column::f64(values)]);
            let spec = spec(agg, 2);
            let mut live = Cumulator::new(TimeFrame::Day1, spec.clone());
            for i in 0..3 {
                live.append(&raw.slice_data(i, i + 1)).unwrap();
                same_bits(&live.last().unwrap().unwrap(), &aggregate_period(&raw.slice_data(0, i + 1), &spec, TimeFrame::Day1, &Tz::Utc).unwrap());
            }
        }
    }
}

#[test]
fn rollover_keeps_the_original_grid_when_incoming_bars_use_another_zone() {
    let spec = spec(Agg::Sum, 1);
    let seed = frame(&[0], vec![Column::f64(vec![1.])]);
    let mut live = Cumulator::new(TimeFrame::Day1, spec.clone());
    live.append(&seed).unwrap();
    let mut raw = seed;
    for stamp in [1_440 * MINUTE, 2_500 * MINUTE, 2_880 * MINUTE] {
        let bar = frame(&[stamp], vec![Column::f64(vec![2.])]).set_index_tz(Tz::parse("+08:00").unwrap()).unwrap();
        raw.append(&bar).unwrap();
        live.append(&bar).unwrap();
        same_bits(&live.frame().unwrap(), &cumulate(&raw, TimeFrame::Day1, &spec).unwrap());
    }
}

#[test]
fn nonfinite_inputs_are_missing_in_initial_fold_and_resend_checkpoints() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for col in [Column::f64(vec![bad, 2., 3.]), Column::f32(vec![bad as f32, 2., 3.])] {
            for agg in [Agg::First, Agg::Last, Agg::Max, Agg::Min, Agg::Sum] {
                let op = agg.as_combine_op();
                let mut initial = col.take(&[1]);
                initial.init_at(0, op, &col, 0).unwrap();
                assert!(initial.get_f64(0).is_nan(), "initial {agg:?} {bad}");
                let mut fold = col.take(&[1]);
                fold.combine_at(0, op, &col, 0).unwrap();
                assert!(fold.get_f64(0).is_nan(), "fold {agg:?} {bad}");
                let mut checkpoint = col.take(&[1]);
                checkpoint.combine_reduced_at(0, op, &col, 0, false).unwrap();
                assert!(checkpoint.get_f64(0).is_nan(), "checkpoint {agg:?} {bad}");
                let raw = frame(&[0, MINUTE, 2 * MINUTE], vec![col.clone()]);
                let spec = spec(agg, 1);
                let mut live = Cumulator::from_frame(&raw, TimeFrame::Min5, spec.clone()).unwrap();
                live.append(&frame(&[2 * MINUTE], vec![col.slice(1, 2)])).unwrap();
                assert!(live.last().unwrap().unwrap().columns()[0].get_f64(0).is_nan());
            }
        }
    }
}
