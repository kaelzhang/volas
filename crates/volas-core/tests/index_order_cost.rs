//! Ordered labels prove uniqueness without auxiliary storage.
#[path = "support/allocation.rs"]
mod allocation;

use allocation::measure;
use volas_core::{Index, IndexKind};

const ROWS: usize = 2000;

fn assert_no_auxiliary_allocations(kind: IndexKind) {
    // The caller prepared all owned labels before the measured constructor.
    let (result, calls, bytes) = measure(|| Index::new(None, kind));
    let index = result.unwrap();
    assert_eq!(index.len(), ROWS);
    assert_eq!((calls, bytes), (0, 0), "ordered labels already prove uniqueness");
}

#[test]
fn increasing_integer_construction_needs_no_auxiliary_allocation() {
    assert_no_auxiliary_allocations(IndexKind::Int64((0..ROWS).map(|i| i as i64).collect()));
}

#[test]
fn decreasing_integer_construction_needs_no_auxiliary_allocation() {
    assert_no_auxiliary_allocations(IndexKind::Int64((0..ROWS).rev().map(|i| i as i64).collect()));
}

#[test]
fn increasing_string_construction_needs_no_auxiliary_allocation() {
    assert_no_auxiliary_allocations(IndexKind::Str((0..ROWS).map(|i| format!("{i:04}")).collect()));
}

#[test]
fn decreasing_string_construction_needs_no_auxiliary_allocation() {
    assert_no_auxiliary_allocations(IndexKind::Str((0..ROWS).rev().map(|i| format!("{i:04}")).collect()));
}
