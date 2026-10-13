//! Tracked writes preserve facts about untouched cells.
use volas_core::Buffer;

#[test]
fn writes_that_keep_an_old_nan_keep_its_fact() {
    let mut b = Buffer::from_vec(vec![1., f64::NAN, 2.]);
    assert!(b.has_nan());
    b.set(2, 3.);
    assert_eq!(b.known_has_nan(), Some(true));
    b.scatter(&[0, 0], |k| [f64::NAN, 4.][k]);
    assert_eq!(b.known_has_nan(), Some(true));
    b.scatter(&[], |_| unreachable!());
    assert_eq!(b.known_has_nan(), Some(true));
    b.set(1, 5.);
    assert_eq!(b.known_has_nan(), None);
    assert!(!b.has_nan());
}

#[test]
fn untouched_nan_fact_survives_copy_on_write_for_float32() {
    let original = Buffer::from_vec(vec![f32::NAN, 1., -0.]);
    assert!(original.has_nan());
    let mut copy = original.clone();
    copy.scatter(&[1, 1, 2], |k| [f32::NAN, 4., 5.][k]);
    assert_eq!(copy.known_has_nan(), Some(true));
    assert_eq!(original.known_has_nan(), Some(true));
    assert_eq!(original[1], 1.);
    assert_eq!(original[2].to_bits(), (-0f32).to_bits());
    copy.set(0, 6.);
    assert_eq!(copy.known_has_nan(), None);
    assert!(!copy.has_nan());
    assert!(original.has_nan());
}
