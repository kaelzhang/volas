//! An append consumes its label admission; it must not allocate lookup keys again.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{Column, DataFrame, Index, IndexKind};

#[test]
fn string_append_allocates_only_the_labels_it_keeps() {
    let mut labels = Vec::with_capacity(64);
    labels.extend(["z".to_owned(), "a".to_owned(), "m".to_owned()]);
    let mut values = Vec::with_capacity(64);
    values.extend([1., 2., 3.]);
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::f64(values)], Some(Index::new(None, IndexKind::Str(labels)).unwrap())).unwrap();
    let incoming: Vec<String> = (0..16).map(|i| format!("new-{i:02}")).collect();
    let rows = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![4.; incoming.len()])], Some(Index::new(None, IndexKind::Str(incoming)).unwrap())).unwrap();
    let (result, allocations, _) = measure(|| target.append(&rows));
    result.unwrap();
    assert!(allocations <= 18, "one allocation per retained string plus storage growth; got {allocations}");
    assert_eq!(target.height(), 19);
}

#[test]
fn replacement_and_append_do_not_materialize_a_second_label_selection() {
    let mut labels = Vec::with_capacity(64);
    labels.extend(["z".to_owned(), "a".to_owned(), "m".to_owned()]);
    let mut values = Vec::with_capacity(64);
    values.extend([1., 2., 3.]);
    let mut target = DataFrame::new(vec!["x".into()], vec![Column::f64(values)], Some(Index::new(None, IndexKind::Str(labels)).unwrap())).unwrap();
    let incoming: Vec<String> = ["m".to_owned()].into_iter().chain((0..16).map(|i| format!("new-{i:02}"))).collect();
    let rows = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![4.; incoming.len()])], Some(Index::new(None, IndexKind::Str(incoming)).unwrap())).unwrap();
    let (result, allocations, _) = measure(|| target.append(&rows));
    result.unwrap();
    assert!(allocations <= 18, "only the appended strings should allocate; got {allocations}");
    assert_eq!(target.height(), 19);
    assert_eq!(target.columns()[0].as_f64().unwrap()[2], 4.);
}
