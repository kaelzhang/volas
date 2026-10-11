//! A multi-input kernel that reads its inputs unchecked never reads past the shorter
//! one: inputs of unequal length give an empty result.

use volas_compute::indicators as ind;

fn series(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + (i as f64 * 0.37).sin()).collect()
}

#[test]
fn unequal_inputs_give_an_empty_result() {
    let (long, short) = (series(64), series(40));
    for (a, b) in [(&long, &short), (&short, &long)] {
        assert!(ind::willr(a, b, a, 5).is_empty());
        assert!(ind::willr(a, a, b, 5).is_empty());
        assert!(ind::aroonosc(a, b, 5).is_empty());
        assert!(ind::mfi(a, a, a, b, 5, 0).is_empty());
        assert!(ind::mfi(b, a, a, a, 5, 0).is_empty());
        assert!(ind::correl(a, b, 5, 0).is_empty());
    }
    // equal lengths still compute
    assert_eq!(ind::willr(&long, &long, &long, 5).len(), 64);
    assert_eq!(ind::aroonosc(&long, &long, 5).len(), 64);
    assert_eq!(ind::mfi(&long, &long, &long, &long, 5, 0).len(), 64);
    assert_eq!(ind::correl(&long, &long, 5, 0).len(), 64);
}
