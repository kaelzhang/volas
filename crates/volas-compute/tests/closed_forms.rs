//! The closed-form sums over a window's positions hold for every period: the
//! linear-regression `Σx²` once overflowed `usize` from period 2,097,153.

use volas_compute::indicators as ind;

#[test]
fn a_huge_period_regression_keeps_its_slope() {
    let period = 2_100_000;
    let y: Vec<f64> = (0..period + 3).map(|i| i as f64).collect();
    let want = ind::linearreg_slope(&y[..10], 5, 0)[9];
    let got = ind::linearreg_slope(&y, period, 0);
    assert!(got[period - 2].is_nan());
    for v in &got[period - 1..] {
        assert!((v - want).abs() < 1e-6, "{v} vs {want}");
    }
    // the weighted average's weight sum too: a WMA of a constant is the constant
    let c = vec![3.25; 70_000];
    assert_eq!(ind::wma(&c, 65_536, 0)[69_999], 3.25);
}
