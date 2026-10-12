//! The finite-window kernels apply the missing-value rules (`volas_compute::missing`):
//! a per-row term that reads a missing input is missing, and a window holding a missing
//! row is missing — an extremum, a position, a tracker, a count or a clamp never steps
//! over it — while the windows clear of it are exactly what clean data gives. The resume
//! and per-row forms agree with the full computations.

use volas_compute::raw::indicators as ind;

const NAN: f64 = f64::NAN;

fn ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.37).sin() * 4.0 + i as f64 * 0.05).collect();
    let high = close.iter().enumerate().map(|(i, c)| c + 1.0 + (i as f64 * 0.21).cos().abs()).collect();
    let low = close.iter().enumerate().map(|(i, c)| c - 1.0 - (i as f64 * 0.17).sin().abs()).collect();
    (high, low, close)
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// Rows `[k, k + span)` are NaN, every other row equals `clean`.
fn missing_for(got: &[f64], clean: &[f64], k: usize, span: usize) {
    for (i, (g, c)) in got.iter().zip(clean).enumerate() {
        if (k..k + span).contains(&i) {
            assert!(g.is_nan(), "row {i} reads the missing row: {g}");
        } else {
            assert!(g.to_bits() == c.to_bits() || (g.is_nan() && c.is_nan()), "row {i}: {g} != {c}");
        }
    }
}

#[test]
fn a_position_window_holding_a_missing_row_is_missing_on_every_path() {
    let (_, _, mut close) = ohlc(60);
    let clean = close.clone();
    close[20] = NAN;
    for (full, state, resume) in [
        (ind::maxindex as fn(&[f64], usize, usize) -> Vec<f64>,
         ind::maxindex_final_state as fn(&[f64], usize, usize) -> Option<Vec<f64>>,
         ind::maxindex_resume as fn(&[f64], usize, usize, usize, &[f64]) -> Option<(Vec<f64>, Vec<f64>)>),
        (ind::minindex, ind::minindex_final_state, ind::minindex_resume),
    ] {
        let got = full(&close, 5, 7);
        missing_for(&got, &full(&clean, 5, 7), 20, 5);
        // a resume over rows whose windows hold the missing row agrees with the full one
        for from in [18, 21, 24, 30] {
            let st = state(&close[..from], 5, 7).unwrap();
            let (tail, _) = resume(&close, 5, from, 7, &st).unwrap();
            assert!(same(&tail, &got[from..]), "resume from {from}");
        }
    }
}

/// Every extremum path resolves a signed-zero tie the same way (total order: `+0 > −0`),
/// so a missing value later in the series — which may select another path — never
/// changes an earlier extremum's bits.
#[test]
fn a_future_missing_value_never_changes_an_earlier_extremum() {
    use ndarray::ArrayView1;
    use volas_compute::raw::kernels;
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for x in [vec![1.0, -0.0, 0.0], vec![-1.0, 0.0, -0.0], vec![-0.0, 0.0, -0.0], vec![0.0, 0.0, -0.0]] {
        let mut later = x.clone();
        later.push(NAN);
        for f in [kernels::rolling_min as fn(ArrayView1<f64>, usize) -> ndarray::Array1<f64>, kernels::rolling_max] {
            let a = f(ArrayView1::from(&x), 3).to_vec();
            let b = f(ArrayView1::from(&later), 3).to_vec();
            assert_eq!(bits(&a), bits(&b[..3]), "{x:?}");
        }
        let (hh, ll) = kernels::rolling_max_min(&later, &later, 3);
        assert_eq!(bits(&hh[..3]), bits(&kernels::rolling_max(ArrayView1::from(&x), 3).to_vec()));
        assert_eq!(bits(&ll[..3]), bits(&kernels::rolling_min(ArrayView1::from(&x), 3).to_vec()));
    }
    assert_eq!(kernels::rolling_min(ArrayView1::from(&[1.0, -0.0, 0.0]), 3)[2].to_bits(), (-0.0f64).to_bits());
    assert_eq!(kernels::rolling_max(ArrayView1::from(&[-1.0, -0.0, 0.0]), 3)[2].to_bits(), 0.0f64.to_bits());
    // the hhv:10 tracker and the stochastic trackers agree with the sweep
    let mut z = vec![0.0; 12];
    z[3] = -0.0;
    z[7] = -0.0;
    let mut zn = z.clone();
    zn.push(NAN);
    assert_eq!(bits(&ind::hhv(&z, 10)), bits(&ind::hhv(&zn, 10)[..12]));
    assert_eq!(bits(&ind::llv(&z, 10)), bits(&ind::llv(&zn, 10)[..12]));
    let k = ind::stoch_fastk(&z, &z, &z, 5);
    let kn = ind::stoch_fastk(&zn, &zn, &zn, 5);
    assert_eq!(bits(&k), bits(&kn[..12]));
}

#[test]
fn a_tracker_window_holding_a_missing_row_is_missing() {
    let (high, low, close) = ohlc(60);
    for col in 0..3 {
        let mut x = [high.clone(), low.clone(), close.clone()];
        x[col][30] = NAN;
        let span = if col == 2 { 1 } else { 5 }; // willr reads the close at the row only
        missing_for(&ind::willr(&x[0], &x[1], &x[2], 5), &ind::willr(&high, &low, &close, 5), 30, span);
    }
    let mut h = high.clone();
    h[30] = NAN;
    missing_for(&ind::aroon_up(&h, &low, 5), &ind::aroon_up(&high, &low, 5), 30, 6);
    missing_for(&ind::aroonosc(&h, &low, 5), &ind::aroonosc(&high, &low, 5), 30, 6);
    let mut l = low.clone();
    l[30] = NAN;
    missing_for(&ind::aroon_down(&high, &l, 5), &ind::aroon_down(&high, &low, 5), 30, 6);
    missing_for(&ind::aroonosc(&high, &l, 5), &ind::aroonosc(&high, &low, 5), 30, 6);
}

/// The per-row %K reduces exactly what the rolling kernels do, missing rows included.
#[test]
fn a_per_row_stochastic_k_equals_the_full_one() {
    let (mut high, low, mut close) = ohlc(40);
    high[10] = NAN;
    close[25] = NAN;
    let full = ind::stoch_fastk(&high, &low, &close, 5);
    for (i, v) in full.iter().enumerate() {
        assert!(same(&[ind::stoch_fastk_at(&high, &low, &close, 5, i)], &[*v]), "row {i}");
    }
    assert!(full[10..15].iter().all(|v| v.is_nan()) && full[25].is_nan() && full[15].is_finite());
    // a flat range reads no close — but a missing close is still missing
    let flat = [1.0; 6];
    let k = ind::stoch_fastk(&flat, &flat, &[1.0, 1.0, 1.0, NAN, 1.0, 1.0], 2);
    assert!(same(&k, &[NAN, 0.0, 0.0, NAN, 0.0, 0.0]));
}

#[test]
fn the_true_range_term_is_missing_when_an_input_it_reads_is() {
    let (h, l, c) = ([3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [2.0, NAN, 4.0]);
    assert_eq!(ind::tr_at(&h, &l, &c, 1), 2.0); // reads the prior close, not its own
    assert!(ind::tr_at(&h, &l, &c, 2).is_nan());
    assert!(same(&ind::tr(&h, &l, &c), &[NAN, 2.0, NAN]));
    // an infinity is a value
    assert_eq!(ind::tr_at(&[0.0, f64::INFINITY], &[0.0, 0.0], &[0.0, 0.0], 1), f64::INFINITY);
}

#[test]
fn the_momentum_family_applies_its_zero_base_to_present_values_only() {
    use ind::Momentum;
    let data = [0.0, NAN, 1.0, 2.0, 0.0, 3.0];
    for (name, want) in [
        ("mom", [NAN, NAN, NAN, 1.0, -2.0, 3.0]),
        ("roc", [NAN, NAN, NAN, 100.0, -100.0, 0.0]),
        ("rocp", [NAN, NAN, NAN, 1.0, -1.0, 0.0]),
        ("rocr", [NAN, NAN, NAN, 2.0, 0.0, 0.0]),
        ("rocr100", [NAN, NAN, NAN, 200.0, 0.0, 0.0]),
    ] {
        let kind = Momentum::of(name).unwrap();
        let rows: Vec<f64> = (0..data.len()).map(|i| kind.at(&data, 1, i)).collect();
        assert!(same(&rows, &want), "{name}: {rows:?}");
    }
    assert_eq!(Momentum::of("rsi"), None);
    assert_eq!(Momentum::Roc.value(5.0, 0.0), 0.0); // TA-Lib's zero-base guard
    assert!(same(&ind::roc(&data, 1), &[NAN, NAN, NAN, 100.0, -100.0, 0.0]));
    assert!(same(&ind::rocr(&data, 1), &[NAN, NAN, NAN, 2.0, 0.0, 0.0]));
}

#[test]
fn a_missing_mavp_period_selects_no_average() {
    assert_eq!(ind::mavp_period(7.9, 2, 30), Some(7));
    assert_eq!(ind::mavp_period(0.0, 2, 30), Some(2));
    assert_eq!(ind::mavp_period(99.0, 2, 30), Some(30));
    assert_eq!(ind::mavp_period(NAN, 2, 30), None);
}

/// `rsv` is missing before its first full window and while a window holds a missing
/// value; KDJ's RSV covers the bars available before its first full window (通达信).
#[test]
fn rsv_and_kdj_share_one_rsv_with_their_own_warm_up() {
    let (high, low, close) = ohlc(30);
    let rsv = ind::rsv(&high, &low, &close, 9);
    assert!(rsv[..8].iter().all(|v| v.is_nan()) && rsv[8..].iter().all(|v| v.is_finite()));
    let partial = |i: usize| {
        let lo = low[..=i].iter().copied().fold(f64::INFINITY, f64::min);
        let hi = high[..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (close[i] - lo) / (hi - lo) * 100.0
    };
    let k = ind::kdj_k(&high, &low, &close, 9, 3, 50.0);
    let mut want = 50.0;
    for (i, &got) in k.iter().enumerate() {
        let r = if i < 8 { partial(i) } else { rsv[i] };
        want = (1.0 - 1.0 / 3.0) * want + (1.0 / 3.0) * r;
        assert!((got - want).abs() < 1e-9, "row {i}: {got} vs {want}");
    }
    let mut h = high.clone();
    h[3] = NAN;
    let k = ind::kdj_k(&h, &low, &close, 9, 3, 50.0);
    assert!(k[..3].iter().all(|v| v.is_finite()) && k[3..].iter().all(|v| v.is_nan()));
    let r = ind::rsv(&h, &low, &close, 4);
    assert!(r[3..7].iter().all(|v| v.is_nan()) && r[7].is_finite());
}
