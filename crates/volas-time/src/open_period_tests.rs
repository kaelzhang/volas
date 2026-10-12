use super::*;
use volas_core::{Column, Index, Tz};

const MIN: i64 = 60_000_000_000;

fn bars(ts: &[i64], volume: &[f64]) -> DataFrame {
    DataFrame::new(
        vec!["volume".into()],
        vec![Column::f64(volume.to_vec())],
        Some(Index::datetime(ts.to_vec(), Tz::Utc)),
    )
    .unwrap()
}

fn volumes(open: &OpenPeriod) -> Vec<f64> {
    open.bars().columns()[0].as_f64().unwrap().to_vec()
}

fn stamps_of(open: &OpenPeriod) -> Vec<i64> {
    stamps(open.bars()).to_vec()
}

/// A re-sent bar inside the given bars collapses to its last copy, wherever the run
/// sits — so the period never holds a superseded bar.
#[test]
fn new_keeps_the_last_bar_of_each_timestamp() {
    let open = OpenPeriod::new(bars(&[0, MIN, MIN, 2 * MIN, 2 * MIN], &[1.0, 30.0, 7.0, 4.0, 5.0])).unwrap();
    assert_eq!(stamps_of(&open), vec![0, MIN, 2 * MIN]);
    assert_eq!(volumes(&open), vec![1.0, 7.0, 5.0]);
    assert_eq!((open.first_ts(), open.last_ts()), (0, 2 * MIN));
    // bars without a repeat are kept as they are
    let open = OpenPeriod::new(bars(&[0, MIN], &[1.0, 2.0])).unwrap();
    assert_eq!(volumes(&open), vec![1.0, 2.0]);
}

#[test]
fn new_rejects_what_is_not_an_open_period() {
    let empty = OpenPeriod::new(bars(&[], &[])).unwrap_err();
    assert!(empty.to_string().contains("at least one bar"), "{empty}");
    let disorder = OpenPeriod::new(bars(&[MIN, 0], &[1.0, 2.0])).unwrap_err();
    assert!(disorder.to_string().contains("time order"), "{disorder}");
    let ranged = DataFrame::new(vec!["volume".into()], vec![Column::f64(vec![1.0])], None).unwrap();
    let err = OpenPeriod::new(ranged).unwrap_err();
    assert!(err.to_string().contains("DatetimeIndex"), "{err}");
}

/// A pushed bar stamped like the last one replaces it; a later one is appended; an
/// earlier one is refused and the period is unchanged.
#[test]
fn push_keeps_one_bar_per_timestamp() {
    let mut open = OpenPeriod::new(bars(&[0], &[1.0])).unwrap();
    open.push(&bars(&[MIN], &[2.0])).unwrap();
    open.push(&bars(&[MIN], &[3.0])).unwrap();
    assert_eq!(stamps_of(&open), vec![0, MIN]);
    assert_eq!(volumes(&open), vec![1.0, 3.0]);
    assert!(open.push(&bars(&[0], &[9.0])).is_err());
    assert_eq!(volumes(&open), vec![1.0, 3.0]);
}

#[test]
fn select_projects_every_bar() {
    let two = DataFrame::new(
        vec!["close".into(), "volume".into()],
        vec![Column::f64(vec![1.0, 2.0]), Column::f64(vec![3.0, 4.0])],
        Some(Index::datetime(vec![0, MIN], Tz::Utc)),
    )
    .unwrap();
    let open = OpenPeriod::new(two).unwrap().select(&["volume".to_string()]).unwrap();
    assert_eq!(open.bars().names(), &["volume".to_string()]);
    assert_eq!(volumes(&open), vec![3.0, 4.0]);
    assert!(OpenPeriod::new(bars(&[0], &[1.0])).unwrap().select(&["nope".to_string()]).is_err());
}
