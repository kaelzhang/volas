//! C-order export owns one destination and preserves the conversion contract.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{Column, DataFrame};

#[test]
fn row_major_export_allocates_only_the_destination() {
    let df = DataFrame::new(
        vec!["a".into(), "b".into()],
        vec![
            Column::f64(vec![1.; 30_000]),
            Column::f64(vec![-0.; 30_000]),
        ],
        None,
    )
    .unwrap();
    let ((out, h, w), calls, bytes) = measure(|| df.to_row_major_f64());
    assert_eq!((h, w), (30_000, 2));
    assert_eq!(out[1].to_bits(), (-0f64).to_bits());
    assert_eq!((calls, bytes), (1, out.len() * 8));
}

#[test]
fn row_major_export_keeps_layout_missing_and_float_bits() {
    let cols = vec![
        Column::f64(vec![-0., f64::from_bits(0xfff8_0000_0000_0024), 1.]),
        Column::f32(vec![-0., f32::NAN, f32::from_bits(1)]),
        Column::i64_from_options([Some((1 << 53) + 1), None, Some(-2)]),
        Column::bool_from_options([None, Some(false), Some(true)]),
        Column::datetime(vec![1, i64::MIN, 3]),
        Column::str_from_options([Some("text".into()), None, Some("".into())]),
    ];
    let names = (0..cols.len()).map(|i| format!("c{i}")).collect();
    let df = DataFrame::new(names, cols.clone(), None).unwrap();
    let (mut out, h, w) = df.to_row_major_f64();
    assert_eq!((h, w), (3, 6));
    for (j, c) in cols.iter().enumerate() {
        for (i, x) in c.to_f64_vec().iter().enumerate() {
            assert_eq!(out[i * w + j].to_bits(), x.to_bits());
        }
    }
    out[0] = 24.;
    assert_eq!(df.columns()[0].get_f64(0).to_bits(), (-0f64).to_bits());
}
