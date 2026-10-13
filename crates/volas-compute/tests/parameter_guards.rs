use volas_compute::window::{self as w, WindowError as E};

#[test]
fn ewm_rejects_alpha_outside_its_domain() {
    for a in [0., -1., 2., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for data in [&[][..], &[1., 3.][..]] {
            assert_eq!(w::ewm_mean(data, a, false, false, 0), Err(E::Alpha));
            assert_eq!(w::ewm_sum(data, a, false, 0), Err(E::Alpha));
            assert_eq!(w::ewm_var(data, a, true, true, false, 0), Err(E::Alpha));
            assert_eq!(w::ewm_cov(data, data, a, true, true, false, 0), Err(E::Alpha));
            assert_eq!(w::ewm_corr(data, data, a, true, true, 0), Err(E::Alpha));
        }
    }
    assert_eq!(w::ewm_mean(&[1., 3.], 1., false, false, 0).unwrap(), [1., 3.]);
}

#[test]
fn quantile_rejects_infinite_q_without_panicking() {
    for q in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -1., 2.] {
        for data in [&[][..], &[1., 3.][..]] {
            assert_eq!(w::quantile(data, 2, 1, q, "linear"), Err(E::Quantile));
        }
    }
}

#[test]
fn quantile_rejects_unknown_interpolation() {
    assert_eq!(w::quantile(&[1., 3.], 2, 1, 0.5, "bogus"), Err(E::Interpolation("bogus".into())));
    assert_eq!(w::quantile(&[], 2, 1, 0.5, "bogus"), Err(E::Interpolation("bogus".into())));
    for mode in ["linear", "lower", "higher", "nearest", "midpoint"] {
        assert_eq!(w::quantile(&[1., 3.], 2, 1, 0., mode).unwrap(), [1., 1.]);
    }
}

#[test]
fn every_rolling_entry_rejects_zero_and_impossible_min_periods() {
    for (p, min, err) in [(0, 0, E::Window), (1, 2, E::MinPeriods)] {
        for data in [&[][..], &[1., 3.][..]] {
            for result in [w::count(data, p, min, true), w::sum(data, p, min), w::mean(data, p, min),
                w::min(data, p, min), w::max(data, p, min), w::median(data, p, min),
                w::var(data, p, min, 1), w::std(data, p, min, 1), w::sem(data, p, min, 1),
                w::dev(data, p, min), w::mode(data, p, min), w::skew(data, p, min),
                w::kurt(data, p, min), w::nunique(data, p, min), w::quantile(data, p, min, 0.5, "linear"),
                w::rank(data, p, min, "average", true, false),
                w::rank_padded(data, p, min, "average", true, false, data.len()),
                w::cov(data, data, p, min, 1), w::corr(data, data, p, min)] {
                assert_eq!(result, Err(err.clone()));
            }
            assert_eq!(w::edge_positions(&[], p, min, false), Err(err.clone()));
        }
    }
}

#[test]
fn pairwise_entries_reject_unequal_lengths() {
    for (x,y) in [(&[1.][..], &[][..]), (&[][..], &[1.][..])] {
        assert_eq!(w::cov(x,y,2,1,1), Err(E::Shape));
        assert_eq!(w::corr(x,y,2,1), Err(E::Shape));
        assert_eq!(w::ewm_cov(x,y,0.5,true,false,false,1), Err(E::Shape));
        assert_eq!(w::ewm_corr(x,y,0.5,true,false,1), Err(E::Shape));
    }
}

#[test]
fn rank_rejects_unknown_methods_and_invalid_padding() {
    assert_eq!(w::rank(&[],2,1,"bogus",true,false), Err(E::Rank("bogus".into())));
    assert_eq!(w::rank_padded(&[1.],2,1,"min",true,false,0), Err(E::Padding));
    assert_eq!(w::rank_padded(&[],2,1,"max",true,false,1), Err(E::Padding));
    assert!(w::rank_padded(&[],2,1,"average",true,false,0).unwrap().is_empty());
}

#[test]
fn errors_are_typed_and_display_the_public_contract() {
    for err in [E::Window, E::MinPeriods, E::Alpha, E::Quantile, E::Interpolation("x".into()), E::Rank("x".into()), E::Shape, E::Padding] {
        let typed: &dyn std::error::Error = &err;
        assert!(!typed.to_string().is_empty());
    }
}

#[test]
fn a_large_valid_window_does_not_overflow_its_deque_position() {
    assert_eq!(w::min(&[4.,3.,2.,1.], usize::MAX - 1, 1).unwrap(), [4.,3.,2.,1.]);
    assert_eq!(w::max(&[1.,2.,3.,4.], usize::MAX - 1, 1).unwrap(), [1.,2.,3.,4.]);
}
