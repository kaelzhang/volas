//! Column transforms allocate their output, without whole-input temporaries.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{BinOp, CmpOp, Column, DType};

#[test]
fn homogeneous_float_arithmetic_allocates_only_its_output() {
    for dtype in [DType::F64, DType::F32] {
        let a = Column::f64(vec![1.; 30_000]).cast(dtype).unwrap();
        let b = Column::f64(vec![2.; 30_000]).cast(dtype).unwrap();
        for op in [Some(BinOp::Add), Some(BinOp::Sub), Some(BinOp::Mul), None] {
            let (out, calls, bytes) = measure(|| match op {
                Some(op) => a.binary(&b, op).unwrap(),
                None => a.div(&b).unwrap(),
            });
            assert_eq!(out.len(), a.len());
            assert!(calls <= 2, "{dtype} {op:?}: {calls} allocations, {bytes} bytes");
        }
    }
}

#[test]
fn same_float_select_allocates_only_its_output() {
    for dtype in [DType::F64, DType::F32] {
        let a = Column::f64(vec![1.; 30_000]).cast(dtype).unwrap();
        let b = Column::f64(vec![2.; 30_000]).cast(dtype).unwrap();
        let cond = vec![true; a.len()];
        let (out, calls, _) = measure(|| a.select(&cond, &b, dtype).unwrap());
        assert!(out.equals(&a));
        assert!(calls <= 2, "{dtype}: {calls} allocations");
    }
}

#[test]
fn float_comparison_allocates_only_its_output() {
    for dtype in [DType::F64, DType::F32] {
        let a = Column::f64(vec![1.; 30_000]).cast(dtype).unwrap();
        let b = Column::f64(vec![2.; 30_000]).cast(dtype).unwrap();
        let (out, calls, _) = measure(|| a.compare(&b, CmpOp::Lt).unwrap());
        assert!(out.as_bool().unwrap().iter().all(|&x| x));
        assert!(calls <= 2, "{dtype}: {calls} allocations");
        let (orders, calls, _) = measure(|| a.order_rows(&b).unwrap());
        assert_eq!(orders.len(), a.len());
        assert_eq!(calls, 1, "{dtype}: nullable orders allocate only the output");
    }
}

#[test]
fn float_cast_preallocates_the_output_without_a_validity_mask() {
    for (from, to) in [(DType::F64, DType::F32), (DType::F32, DType::F64)] {
        let a = Column::f64(vec![1.; 30_000]).cast(from).unwrap();
        let (out, calls, _) = measure(|| a.cast(to).unwrap());
        assert_eq!(out.dtype(), to);
        assert_eq!(out.len(), a.len());
        assert!(calls <= 2, "{from} -> {to}: {calls} allocations");
    }
}

#[test]
fn dense_fills_learn_the_buffer_fact_and_share_storage() {
    for dtype in [DType::F64, DType::F32] {
        for forward in [None, Some(true), Some(false)] {
            let c = Column::f64(vec![1.; 30_000]).cast(dtype).unwrap();
            assert_eq!(c.known_has_nan(), None);
            let (out, calls, _) = measure(|| match forward {
                None => c.fillna(2.).unwrap(),
                Some(dir) => c.fill_dir(dir),
            });
            assert_eq!(c.known_has_nan(), Some(false));
            assert!(out.equals(&c));
            assert_eq!(calls, 0);
        }
    }
}

#[test]
fn ordinary_string_gather_keeps_first_append_capacity() {
    let n = 30_000;
    let source = Column::str(vec!["bar".into(); n]);
    let idx: Vec<_> = (0..n).collect();
    let groups: Vec<Vec<_>> = (0..n).map(|i| vec![i]).collect();
    let bar = source.slice(0, 1);
    for mut out in [source.take(&idx), volas_core::RowGroups::new(&groups, source.len()).unwrap().reduce(&source, volas_core::CombineOp::Replace).unwrap()] {
        let (_, calls, bytes) = measure(|| out.append(&bar).unwrap());
        assert_eq!(out.len(), n + 1);
        assert_eq!(calls, 0, "{calls} allocations / {bytes} bytes");
    }
}
