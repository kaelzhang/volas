use volas_core::{Column, ColumnView, CombineOp, RowGroups};

#[test]
fn immutable_groups_validate_every_position_count_and_source_height() {
    assert!(RowGroups::new(&[vec![]], 1).is_err());
    assert!(RowGroups::new(&[vec![0, 9, 0]], 1).is_err());
    for (positions, offsets) in [(vec![0], vec![]), (vec![0], vec![1]),
        (vec![0], vec![0]), (vec![0], vec![0, 2, 1]),
        (vec![0], vec![0, 0, 1]), (vec![9], vec![0, 1])] {
        assert!(RowGroups::from_offsets(&positions, &offsets, 1).is_err());
    }
    let rows = [vec![0], vec![0]];
    let nested = RowGroups::new(&rows, 1).unwrap();
    let flat = RowGroups::from_offsets(&[0, 0], &[0, 1, 2], 1).unwrap();
    for proof in [nested, flat] {
        let src = Column::i64_from_options([None]);
        let out = proof.reduce(&src, CombineOp::Replace).unwrap();
        let ColumnView::I64(_, valid) = out.view() else { panic!("expected i64") };
        assert_eq!((out.len(), valid.null_count()), (2, 2));
        assert!(valid.covers(out.len()));
        assert!(proof.reduce(&Column::i64(vec![1, 2]), CombineOp::Replace).is_err());
        assert_eq!(proof.prefix(1).reduce(&src, CombineOp::Replace).unwrap().len(), 1);
        assert_eq!(proof.prefix(99).reduce(&src, CombineOp::Replace).unwrap().len(), 2);
        assert_eq!(proof.prefix(0).reduce(&src, CombineOp::Replace).unwrap().len(), 0);
    }
    let empty: [Vec<usize>; 0] = [];
    for proof in [RowGroups::new(&empty, 0).unwrap(), RowGroups::from_offsets(&[], &[0], 0).unwrap()] {
        assert_eq!(proof.reduce(&Column::i64(vec![]), CombineOp::Replace).unwrap().len(), 0);
    }
}

#[test]
fn immutable_position_slice_reduces_after_complete_admission() {
    let src = Column::i64(vec![2, 3]);
    let mut out = Column::i64(vec![10]);
    out.check_reduction(0, CombineOp::Sum, &src, &[1, 0]).unwrap().land().unwrap();
    assert_eq!(out.as_i64().unwrap(), &[5]);
    let rows: &[usize] = &[0];
    out.check_reduction(0, CombineOp::Sum, &src, rows).unwrap().land().unwrap();
    assert_eq!(out.as_i64().unwrap(), &[2]);
}
