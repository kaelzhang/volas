use super::*;

fn r(x: f64, k: i32) -> f64 {
    round(x, k).unwrap()
}

/// The step test compares `k·log2 5` with integers: its margin holds for every k.
#[test]
fn log2_5_multiples_stay_off_integers() {
    for k in 1..=MAX_PLACES {
        let t = k as f64 * LOG2_5;
        assert!((t - t.round()).abs() > 1e-4, "k {k}");
    }
}

#[test]
fn places_are_python_round() {
    assert_eq!(r(2.675, 2), 2.67); // 2.675 is 2.67499999…
    assert_eq!(r(0.125, 2), 0.12); // an exact half: to even
    assert_eq!(r(0.375, 2), 0.38);
    assert_eq!(r(2.5, 0), 2.0);
    assert_eq!(r(-2.5, 0), -2.0);
    assert_eq!(r(1e300, 10), 1e300); // whole at that precision
    assert_eq!(r(1.25, 400), 1.25);
    assert_eq!(r(1.25, i32::MAX), 1.25);
    assert_eq!(r(2f64.powi(-24), 23), 5.960464477539062e-08); // the denser floats below
    assert_eq!(r(5e-324, 323), 0.0);
    assert!(r(-1e-30, 5).is_sign_negative() && r(-1e-30, 5) == 0.0);
    // the scaled product lands on a half the exact value is short of, or past
    assert_eq!(r(85062.45, 1), 85062.4);
    assert_eq!(r(511821.65, 1), 511821.7);
    // an exact half past 2^52 once scaled: the product already rounded it to even
    assert_eq!(r(450359962737049.75, 1), 450359962737049.8);
    assert_eq!(r(450359962737050.25, 1), 450359962737050.2);
    assert_eq!(r(4503599627370495.5, 0), 4503599627370496.0);
    // the scaled product falls short of a half its exact value is not
    assert_eq!(r(1.005, 2), 1.0); // 1.00499999999999989…
    assert_eq!(r(1.0050000000000001, 2), 1.01);
    // past 22 places: exact digits
    assert_eq!(r(1.234_567_890_123_456_8e-10, 25), 1.234567890123457e-10);
    for x in [f64::NAN, f64::INFINITY, -f64::INFINITY, 0.0, -0.0] {
        assert_eq!(round(x, 3).unwrap().to_bits(), x.to_bits());
    }
}

#[test]
fn tens_are_python_round() {
    assert_eq!(r(15.0, -1), 20.0);
    assert_eq!(r(25.0, -1), 20.0); // a whole half: to even
    assert_eq!(r(5.5, -1), 10.0); // past the half by the fraction
    assert_eq!(r(4.5, -1), 0.0);
    assert!(r(-3.0, -1).is_sign_negative());
    assert_eq!(r(0.3, -1), 0.0);
    assert_eq!(r(1234567.875, -3), 1235000.0);
    assert_eq!(r(1e20, -16), 1e20);
    assert_eq!(r(123456789012345678.0, -38), 0.0);
    assert_eq!(r(1.5e38, -38), 2e38);
    assert_eq!(r(6e37, -38), 1e38); // fewer digits than places, past the half
    assert_eq!(r(1.8e19, -19), 2e19); // u64's multiple would pass 2^64
    assert_eq!(r(-9.2e18, -18), -9e18);
    assert_eq!(r(-2.5e38, -38), -2e38);
    assert_eq!(r(1e300, -299), 1e300);
    assert_eq!(r(1.23456e300, -299), 1.2e300);
    assert_eq!(r(9.9e300, -301), 1e301); // carried into a new digit
    assert_eq!(r(4e300, -301), 0.0);
    assert_eq!(r(7e300, -301), 1e301);
    assert_eq!(r(1e300, -302), 0.0); // fewer digits than places
    assert_eq!(r(1.7e308, -309), 0.0);
    assert_eq!(round(1.7e308, -308), None); // 2e308
    assert_eq!(r(-1.7e308, -307), -1.7e308);
    assert_eq!(r(2.5e15 + 0.5, -15), 3e15); // past the half by the fraction
    assert_eq!(r(2.5e15 + 0.5, -16), 0.0);
}

#[test]
fn float32_is_the_nearest_float32_of_the_decimal() {
    assert_eq!(round(2.675f32, 2), Some(2.67f32)); // 2.675f32 is 2.67499995…
    assert_eq!(round(0.125f32, 2), Some(0.12f32));
    assert_eq!(round(1.5e-5f32, 15), Some(1.5e-5f32));
    assert_eq!(round(1.2345678e-7f32, 12), Some(1.23457e-7f32)); // 10^12 is not a float32
    assert_eq!(round(3.4e38f32, -38), Some(3e38f32));
    assert_eq!(round(f32::MAX, -35), None);
    assert_eq!(round(1e-45f32, 50), Some(1e-45f32));
}

#[test]
fn scaled_decimals_are_the_nearest_float() {
    assert_eq!(scaled_to_f64(2675, 3), Some(2.675));
    assert_eq!(scaled_to_f64(-15, -2), Some(-1500.0));
    assert_eq!(scaled_to_f64(1, 400), Some(0.0));
    assert_eq!(scaled_to_f64(i128::MAX, 0), Some(i128::MAX as f64));
    assert_eq!(scaled_to_f64(1, -400), None);
    assert_eq!(
        scaled_text_to_f64("12345678901234567890123456789012345678901234567890", 49),
        Some(1.2345678901234568)
    );
    assert_eq!(scaled_text_to_f64("-1", -309), None);
}

#[test]
fn integer_rounding_never_overflows_inside() {
    assert_eq!(round_i64(i64::MAX, -38), Some(0));
    assert_eq!(round_i64(i64::MAX, -39), Some(0));
    assert_eq!(round_i64(i64::MAX, -1), None);
    assert_eq!(round_i64(25, -1), Some(20));
    assert_eq!(round_i64(-35, -1), Some(-40));
    assert_eq!(round_i64(7, 3), Some(7));
    let mut d = b"99".to_vec();
    increment(&mut d);
    assert_eq!(d, b"100");
}
