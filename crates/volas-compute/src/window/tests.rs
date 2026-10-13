use super::*;

/// The pre-slide evaluation: gather the present rows and compute their mean
/// directly before reducing central powers in source row order.
fn direct_moment(data: &[f64], p: usize, min: usize, kurtosis: bool) -> Vec<f64> {
    per_window(data, p, min, |b| {
        let n = b.len() as f64;
        if b.len() < if kurtosis { 4 } else { 3 } {
            return f64::NAN;
        }
        if is_constant(b) {
            return if kurtosis { -3.0 } else { 0.0 };
        }
        let m = window_mean(b);
        let m2 = b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let higher = b
            .iter()
            .map(|x| (x - m).powi(if kurtosis { 4 } else { 3 }))
            .sum::<f64>()
            / n;
        if m2 <= 0.0 {
            return f64::NAN;
        }
        if kurtosis {
            ((n + 1.0) * higher / (m2 * m2) - 3.0 * (n - 1.0)) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
        } else {
            (n * (n - 1.0)).sqrt() / (n - 2.0) * higher / m2.powf(1.5)
        }
    })
}

#[test]
fn sliding_moments_match_direct_windows_bit_for_bit() {
    let normal: Vec<_> = (0..137).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
    let mut missing = normal.clone();
    missing[0] = f64::NAN;
    missing[7] = f64::from_bits(0x7ff8000000000028);
    missing[81] = f64::NAN;
    let mut infinite = normal.clone();
    infinite[7] = f64::INFINITY;
    infinite[81] = f64::NEG_INFINITY;
    let fixtures = [
        normal,
        missing,
        infinite,
        vec![12345.67; 137],
        (0..137)
            .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
            .collect(),
        (0..137)
            .map(|i| 12345.67 + (i as f64 * 0.1).sin() * 1e-6)
            .collect(),
        (0..137)
            .map(|i| (i as f64 * 0.33).sin() * 2.0f64.powi((i * 53 % 1800) - 900))
            .collect(),
        (0..137).map(|i| (i as f64 - 68.0) * 1e-310).collect(),
        vec![1e308; 137],
        vec![f64::NAN; 137],
        Vec::new(),
    ];
    for data in fixtures {
        for p in [0, 1, 2, 3, 4, 20, 200, usize::MAX] {
            for min in [0, 1, 3, 20] {
                for kurtosis in [false, true] {
                    let got = if kurtosis {
                        kurt(&data, p, min)
                    } else {
                        skew(&data, p, min)
                    };
                    let want = direct_moment(&data, p, min, kurtosis);
                    for (i, (got, want)) in got.iter().zip(want).enumerate() {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "p={p}, min={min}, kurt={kurtosis}, row={i}"
                        );
                    }
                }
            }
        }
    }
}
