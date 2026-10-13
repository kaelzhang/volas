//! Bit-level and error contracts of the borrowed column paths.
use std::cmp::Ordering::{Equal, Greater, Less};
use volas_core::{CmpOp, Column, DType, VolasError};

#[test]
fn homogeneous_selection_preserves_every_picked_bit() {
    let a = [
        0.0,
        -0.0,
        f64::from_bits(0x7ff8_0000_0000_0024),
        f64::NEG_INFINITY,
    ];
    let b = [
        -0.0,
        0.0,
        f64::INFINITY,
        f64::from_bits(0xfff8_0000_0000_0001),
    ];
    for cond in [[true; 4], [false; 4], [true, false, false, true]] {
        let out = Column::f64(a.to_vec())
            .select(&cond, &Column::f64(b.to_vec()), DType::F64)
            .unwrap();
        for (i, &x) in out.as_f64().unwrap().iter().enumerate() {
            assert_eq!(x.to_bits(), if cond[i] { a[i] } else { b[i] }.to_bits());
        }
        let a = [0.0f32, -0.0, f32::from_bits(0x7fc0_0024), f32::NEG_INFINITY];
        let b = [-0.0f32, 0.0, f32::INFINITY, f32::from_bits(0xffc0_0001)];
        let out = Column::f32(a.to_vec())
            .select(&cond, &Column::f32(b.to_vec()), DType::F32)
            .unwrap();
        for (i, &x) in out.to_f32_vec().iter().enumerate() {
            assert_eq!(x.to_bits(), if cond[i] { a[i] } else { b[i] }.to_bits());
        }
    }
    let a = Column::f64(vec![]);
    assert_eq!(a.select(&[], &a, DType::F64).unwrap().len(), 0);
    // Operand shape wins over condition shape and dtype errors.
    assert!(
        matches!(a.select(&[true], &Column::str(vec!["x".into()]), DType::F64), Err(VolasError::Shape(s)) if s.contains("operands"))
    );
}

#[test]
fn floating_comparisons_keep_ieee_and_nullable_order_semantics() {
    let a = [f64::NAN, -0., f64::NEG_INFINITY, 2., 3., f64::INFINITY];
    let b = [1., 0., 0., 1., f64::NAN, f64::INFINITY];
    let expected = [
        None,
        Some(Equal),
        Some(Less),
        Some(Greater),
        None,
        Some(Equal),
    ];
    for left in [
        Column::f64(a.to_vec()),
        Column::f32(a.map(|x| x as f32).to_vec()),
    ] {
        for right in [
            Column::f64(b.to_vec()),
            Column::f32(b.map(|x| x as f32).to_vec()),
        ] {
            assert_eq!(left.order_rows(&right).unwrap(), expected);
            for op in [
                CmpOp::Eq,
                CmpOp::Ne,
                CmpOp::Lt,
                CmpOp::Le,
                CmpOp::Gt,
                CmpOp::Ge,
            ] {
                let want: Vec<_> = a
                    .iter()
                    .zip(b)
                    .map(|(&x, y)| match op {
                        CmpOp::Eq => x == y,
                        CmpOp::Ne => x != y,
                        CmpOp::Lt => x < y,
                        CmpOp::Le => x <= y,
                        CmpOp::Gt => x > y,
                        CmpOp::Ge => x >= y,
                    })
                    .collect();
                let out = left.compare(&right, op).unwrap();
                assert_eq!(out.as_bool().unwrap(), want);
                assert_eq!(out.null_count(), 0);
            }
        }
    }
}

#[test]
fn float_casts_normalize_missing_and_keep_signed_zero_and_range_errors() {
    let xs = [
        -0.,
        0.,
        f64::from_bits(0xfff8_0000_0000_0024),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f32::MAX as f64,
        f32::from_bits(1) as f64,
    ];
    let c = Column::f64(xs.to_vec());
    let out = c.cast(DType::F32).unwrap();
    for (&x, &y) in xs.iter().zip(&out.to_f32_vec()) {
        assert_eq!(
            y.to_bits(),
            if x.is_nan() { f32::NAN } else { x as f32 }.to_bits()
        );
    }
    let xs = [
        -0.,
        0.,
        f32::from_bits(0xffc0_0024),
        f32::INFINITY,
        f32::from_bits(1),
    ];
    let out = Column::f32(xs.to_vec()).cast(DType::F64).unwrap();
    for (&x, &y) in xs.iter().zip(out.as_f64().unwrap()) {
        assert_eq!(
            y.to_bits(),
            if x.is_nan() { f64::NAN } else { x as f64 }.to_bits()
        );
    }
    let c = Column::f64(vec![f64::NAN, 1e300, -1e300]);
    let before: Vec<_> = c.as_f64().unwrap().iter().map(|x| x.to_bits()).collect();
    assert!(matches!(c.cast(DType::F32), Err(VolasError::Overflow(s)) if !s.starts_with('-')));
    assert_eq!(
        c.as_f64()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        before
    );
    for dtype in [DType::F64, DType::F32] {
        let c = Column::i64_from_options([None, Some((1 << 24) + 1)]);
        if dtype == DType::F32 {
            assert!(matches!(c.cast(dtype), Err(VolasError::DType(_))));
        } else {
            assert!(c.cast(dtype).unwrap().get_f64(0).is_nan());
        }
        let b = Column::bool_from_options([None, Some(true), Some(false)])
            .cast(dtype)
            .unwrap();
        assert!(b.get_f64(0).is_nan());
        assert_eq!((b.get_f64(1), b.get_f64(2)), (1., 0.));
    }
}
