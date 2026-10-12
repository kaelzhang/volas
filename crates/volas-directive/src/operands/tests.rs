use std::borrow::Cow;

use volas_core::{Column, DataFrame, VolasError};

use super::Operands;
use crate::parse;

fn frame() -> DataFrame {
    DataFrame::new(vec!["close".into()], vec![Column::f64(vec![1.0, 2.0, 3.0])], None).unwrap()
}

/// Every input a rerun hands out is as long as the others; a shorter one is a shape
/// error, never a slice a kernel reads past.
#[test]
fn a_rerun_input_of_another_length_is_an_error() {
    let df = frame();
    let (a, b) = ([1.0, 2.0, 3.0], [1.0, 2.0]);
    let views = vec![Some(Cow::Borrowed(&a[..])), Some(Cow::Borrowed(&b[..]))];
    let ops = Operands::over(&df, &[], 0, views);
    assert_eq!(ops.f64(0, "close").unwrap().len(), 3);
    let err = ops.f64(1, "close").unwrap_err();
    assert!(matches!(err, VolasError::Shape(ref m) if m.contains("#1 has 2 rows") && m.ends_with("have 3")));
}

/// A rerun reads only what the first run recorded: an unrecorded slot is an internal
/// error, not an empty series.
#[test]
fn a_rerun_reads_only_recorded_inputs() {
    let df = frame();
    let ops = Operands::over(&df, &[], 0, vec![None]);
    assert!(matches!(ops.f64(0, "close"), Err(VolasError::Value(m)) if m.contains("internal error") && m.contains("not read before")));
    assert!(Operands::over(&df, &[], 0, vec![]).f64(3, "close").is_err());
    assert!(Operands::over(&df, &[], 0, vec![]).bool(0).is_err());
}

/// A read that fails is remembered — the record lacks that input — and a later
/// successful read does not clear it.
#[test]
fn a_failed_read_is_remembered() {
    let df = frame();
    let ops = Operands::new(&df, &[]);
    assert!(!ops.failed());
    assert!(matches!(ops.f64(0, "volume"), Err(VolasError::ColumnNotFound(_))));
    assert!(ops.failed());
    assert!(ops.f64(1, "close").is_ok());
    assert!(ops.failed());
}

/// A boolean operand is read once and recorded like a numeric one: its truths as 1 / 0,
/// a missing truth as NaN, and a rerun hands the recorded truths back.
#[test]
fn a_boolean_read_is_recorded() {
    let df = frame();
    let node = parse("repeat:1@(close > 1)").unwrap();
    let crate::types::Node::Command(cmd) = &*node else { unreachable!() };
    let ops = Operands::new(&df, &cmd.series);
    assert_eq!(ops.bool(0).unwrap(), vec![Some(false), Some(true), Some(true)]);
    assert_eq!(ops.bool(0).unwrap().len(), 3); // the second read takes the record
    let recorded: Vec<_> = ops.reads().map(|(slot, r)| (slot, r.view().to_vec())).collect();
    assert_eq!(recorded, vec![(0, vec![0.0, 1.0, 1.0])]);
    let truths = [1.0, f64::NAN];
    let rerun = Operands::over(&df, &cmd.series, 0, vec![Some(Cow::Borrowed(&truths[..]))]);
    assert_eq!(rerun.bool(0).unwrap(), vec![Some(true), None]);
    assert!(Operands::new(&df, &[]).bool(0).is_err()); // no operand given
}

/// A window reads the frame's rows from its start on — converted only there, an `@`
/// operand evaluated over them alone — and keeps their absolute positions.
#[test]
fn a_window_reads_only_its_rows() {
    let n = Column::i64(vec![1, 2, 3, 4]);
    let df = DataFrame::new(vec!["close".into(), "n".into()], vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0]), n], None)
        .unwrap()
        .with_origin(10);
    let node = parse("sum:2@(close + n)").unwrap();
    let crate::types::Node::Command(cmd) = &*node else { unreachable!() };
    let ops = Operands::window(&df, &cmd.series, 2);
    assert_eq!((ops.start(), ops.origin()), (2, 12));
    assert_eq!(&*ops.f64(1, "n").unwrap(), &[3.0, 4.0]);
    assert_eq!(&*ops.f64(2, "close").unwrap(), &[3.0, 4.0]);
    assert_eq!(&*ops.f64(0, "close").unwrap(), &[6.0, 8.0]);
    assert!(ops.reads().all(|(_, r)| !r.has_nan()));
    // a window of a column known to hold a NaN does not know whether it holds one
    let gap = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![f64::NAN, 1.0, 2.0])], None).unwrap();
    assert!(gap.column("close").unwrap().has_nan());
    let ops = Operands::window(&gap, &[], 1);
    assert_eq!(&*ops.f64(0, "close").unwrap(), &[1.0, 2.0]);
    let (_, read) = ops.reads().next().unwrap();
    assert_eq!(read.known_has_nan(), None);
    assert!(!read.has_nan());
    // an int64 past 2^53 inside the window is rejected; outside it, not read
    let big = DataFrame::new(vec!["v".into()], vec![Column::i64(vec![1 << 60, 1, 2])], None).unwrap();
    assert!(Operands::window(&big, &[], 0).f64(0, "v").is_err());
    assert_eq!(&*Operands::window(&big, &[], 1).f64(0, "v").unwrap(), &[1.0, 2.0]);
}
