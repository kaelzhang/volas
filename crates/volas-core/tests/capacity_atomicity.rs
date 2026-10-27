//! Failed reservations are ordinary errors before any logical frame publication.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr;
use std::sync::Arc;
use volas_core::{Column, DataFrame, History, Reach, VolasError};
thread_local! {
    static FAIL_LARGE: Cell<usize> = const { Cell::new(0) };
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}
fn record(size: usize) {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        LARGEST.with(|n| n.set(n.get().max(size)));
    }
}
fn start_recording() {
    ALLOCATIONS.with(|n| n.set(0));
    LARGEST.with(|n| n.set(0));
    TRACK.with(|n| n.set(true));
}
fn stop_recording() -> (usize, usize) {
    TRACK.with(|n| n.set(false));
    (ALLOCATIONS.with(Cell::get), LARGEST.with(Cell::get))
}
struct FailReservation;
fn fail(size: usize) -> bool {
    if size < 65536 { return false; }
    FAIL_LARGE.try_with(|left| {
        let n = left.get();
        if n == 0 { return false; }
        left.set(n - 1);
        n == 1
    }).unwrap_or(false)
}
unsafe impl GlobalAlloc for FailReservation {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        if fail(layout.size()) { ptr::null_mut() } else { System.alloc(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        if fail(size) { ptr::null_mut() } else { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { System.dealloc(ptr, layout); }
}
#[global_allocator]
static ALLOCATOR: FailReservation = FailReservation;

#[test]
fn later_column_reservation_error_publishes_no_index_values_or_cache() {
    let mut target = DataFrame::new(vec!["a".into(), "b".into()], vec![Column::i64(vec![1]), Column::f64(vec![2.])], None).unwrap();
    target.set_computed("b", Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    target.set_computed_states("b", Some(vec![7.]), None);
    let before = target.clone();
    let rows = DataFrame::from_columns(vec![], vec![], 8192).unwrap();
    // The first value buffer and its first missing bitmap are prepared; the
    // second value reservation fails. All logical state must remain unchanged.
    FAIL_LARGE.with(|n| n.set(2));
    let error = target.append(&rows).unwrap_err();
    FAIL_LARGE.with(|n| n.set(0));
    assert!(matches!(error, VolasError::Overflow(_)));
    assert!(target.equals(&before));
    assert_eq!((target.height(), target.index().len()), (1, 1));
    assert!(Arc::ptr_eq(target.index(), before.index()));
    assert_eq!(target.computed_resume_state("b"), before.computed_resume_state("b"));
    assert_eq!(target.has_stale_computed(), before.has_stale_computed());
    assert_eq!(target.stale_cause(), before.stale_cause());
    assert_eq!(target.series("a").unwrap().data().null_count(), 0);
    // A retry uses the prepared capacity safely, including the restored mask.
    target.append(&rows).unwrap();
    assert_eq!(target.height(), 8193);
    assert_eq!(target.column("a").unwrap().null_count(), 8192);
}

#[test]
fn replacement_reserves_later_string_payload_before_the_first_value_write() {
    let mut target = DataFrame::new(vec!["x".into(), "s".into()], vec![Column::f64(vec![1.]), Column::str(vec!["a".into()])], None).unwrap();
    let source = DataFrame::new(vec!["x".into(), "s".into()], vec![Column::f64(vec![2.]), Column::str(vec!["b".repeat(65536)])], None).unwrap();
    let before = target.clone();
    FAIL_LARGE.with(|n| n.set(1));
    let result = target.replace_last_row(&source, 0);
    FAIL_LARGE.with(|n| n.set(0));
    assert!(matches!(result, Err(VolasError::Overflow(_))));
    assert!(target.equals(&before));
    assert!(Arc::ptr_eq(target.index(), before.index()));
    target.replace_last_row(&source, 0).unwrap();
    assert_eq!(target.column("x").unwrap().get_f64(0), 2.);
    assert_eq!(target.column("s").unwrap().str_at(0).unwrap().len(), 65536);
}

#[test]
fn warmed_shared_computed_append_does_not_allocate_again() {
    let mut target = DataFrame::new(vec!["x".into(), "cached".into()], vec![Column::f64(vec![1.; 128]), Column::f64(vec![7.; 128])], None).unwrap();
    target.set_computed("cached", Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
    let snapshot = target.series("cached").unwrap();
    let row = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![2.])], None).unwrap();
    target.append(&row).unwrap();
    start_recording();
    let result = target.append(&row);
    let (allocations, _) = stop_recording();
    result.unwrap();
    assert_eq!(allocations, 0, "the warm COW append must retain the next append's spare capacity");
    assert_eq!(snapshot.data().len(), 128);
    assert_eq!(snapshot.data().get_f64(127), 7.);
    assert_eq!(target.height(), 130);
}

#[test]
fn last_string_replacement_reserves_only_its_net_payload_growth() {
    let mut target = DataFrame::new(vec!["s".into()], vec![Column::str(vec!["a".repeat(2048); 32])], None).unwrap();
    let source = DataFrame::new(vec!["s".into()], vec![Column::str(vec!["b".repeat(2048)])], None).unwrap();
    let snapshot = target.series("s").unwrap();
    start_recording();
    let result = target.replace_last_row(&source, 0);
    let (_, largest) = stop_recording();
    result.unwrap();
    assert!(largest <= 65536, "a replacement that fits the original payload reserved {largest} bytes");
    assert_eq!(snapshot.data().str_at(31).unwrap(), "a".repeat(2048));
    assert_eq!(target.column("s").unwrap().str_at(31).unwrap(), "b".repeat(2048));
}

#[test]
fn warm_range_materialization_keeps_the_next_labels_spare() {
    use volas_core::{Index, IndexKind};
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![1.; 128])], None).unwrap();
    let snapshot = target.series("x").unwrap();
    let row = |label| DataFrame::new(vec!["x".into()], vec![Column::f64(vec![2.])], Some(Index::new(None, IndexKind::Int64(vec![label])).unwrap())).unwrap();
    let warm = row(128);
    let next = row(129);
    target.append(&warm).unwrap();
    start_recording();
    let result = target.append(&next);
    let (allocations, _) = stop_recording();
    result.unwrap();
    assert_eq!(allocations, 0, "materializing Range labels must retain the next append's spare capacity");
    assert_eq!(snapshot.index().len(), 128);
    assert_eq!(target.index().len(), 130);
    assert!(matches!(target.index().kind(), IndexKind::Int64(v) if v.last() == Some(&129)));
}
