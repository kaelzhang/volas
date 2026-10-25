//! Compare the slice loops with the original checked-kernel arithmetic.
use super::*;

#[test]
fn homogeneous_arithmetic_keeps_every_float_bit() {
    let special = [0, 1, 1 << 63, 0x7ff0000000000000, 0xfff0000000000000,
        0x7ff8000000000123, 0xfff8000000004567, 0x7ff0000000000001, u64::MAX];
    let mut pairs: Vec<_> = special.iter().flat_map(|&a| special.iter().map(move |&b| (a, b))).collect();
    let mut seed = 173u64;
    for _ in 0..100_000 {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        let a = seed;
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        pairs.push((a, seed));
    }
    let a: Vec<_> = pairs.iter().map(|p| f64::from_bits(p.0)).collect();
    let b: Vec<_> = pairs.iter().map(|p| f64::from_bits(p.1)).collect();
    let special32 = [0, 1, 0x80000000, 0x7f800000, 0xff800000,
        0x7fc00123, 0xffc04567, 0x7f800001, 0xff800001, u32::MAX];
    let mut pairs32: Vec<_> = special32.iter().flat_map(|&a| special32.iter().map(move |&b| (a, b))).collect();
    pairs32.extend(pairs.iter().map(|&(a, b)| (a as u32, b as u32)));
    let af: Vec<_> = pairs32.iter().map(|p| f32::from_bits(p.0)).collect();
    let bf: Vec<_> = pairs32.iter().map(|p| f32::from_bits(p.1)).collect();
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul] {
        let want = binary_kernel(&a, &b, op, &Validity::dense()).unwrap();
        let out = Column::f64(a.clone()).binary(&Column::f64(b.clone()), op).unwrap();
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(out.as_f64().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        let want = binary_kernel(&af, &bf, op, &Validity::dense()).unwrap();
        let out = Column::f32(af.clone()).binary(&Column::f32(bf.clone()), op).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(match &out { Column { data: ColumnData::F32(v) } => v.iter(), _ => unreachable!() }.map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    for (a, b) in [(Column::f64(a), Column::f64(b)), (Column::f32(af), Column::f32(bf))] {
        let av = a.to_f64_vec(); let bv = b.to_f64_vec();
        let want: Vec<_> = av.iter().zip(&bv).map(|(&x, &y)| (x / y).to_bits()).collect();
        let out = a.div(&b).unwrap();
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(out.as_f64().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want);
    }
    for a in [Column::f64(vec![]), Column::f32(vec![])] {
        for op in [BinOp::Add, BinOp::Sub, BinOp::Mul] {
            assert_eq!(a.binary(&a, op).unwrap().dtype(), a.dtype());
        }
        assert_eq!(a.div(&a).unwrap().len(), 0);
        assert!(matches!(a.binary(&Column::str(vec!["x".into()]), BinOp::Add), Err(VolasError::Shape(_))));
    }
}
