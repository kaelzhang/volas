use std::sync::Arc;
use volas_core::{Column, CombineOp, DataFrame, FoldLayout, History, Reach};
fn reach() -> Reach { Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window } }
#[test]
fn computed_installation_publishes_one_schema_change() {
    let mut frame = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![1.])], None).unwrap();
    let old = FoldLayout::new(&frame, |_| CombineOp::Sum).unwrap();
    frame.set_computed_column("signal", Column::f64(vec![2.]), Arc::new(()), reach()).unwrap();
    assert!(!old.matches(&frame));
    assert!(frame.is_computed("signal"));
    let installed = FoldLayout::new(&frame, |_| CombineOp::Sum).unwrap();
    frame.set_computed_column("signal", Column::f64(vec![3.]), Arc::new(()), reach()).unwrap();
    assert!(installed.matches(&frame));
    let before = frame.clone();
    assert!(frame.set_computed_column("bad", Column::f64(vec![1., 2.]), Arc::new(()), reach()).is_err());
    assert!(frame.equals(&before));
    assert!(!frame.is_computed("bad"));
    frame.set_computed_column("x", Column::f64(vec![1.]), Arc::new(()), reach()).unwrap();
    assert!(!installed.matches(&frame));
}
