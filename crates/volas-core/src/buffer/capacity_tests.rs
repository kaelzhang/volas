use super::*;

#[test]
fn owned_capacity_fact_keeps_the_existing_storage_size() {
    assert_eq!(std::mem::size_of::<Data<f64>>(), 4 * std::mem::size_of::<usize>());
    assert_eq!(std::mem::size_of::<Data<i64>>(), 4 * std::mem::size_of::<usize>());
}

#[test]
fn admitted_unique_capacity_is_inherited_and_retired_by_aliases() {
    let mut values = Vec::with_capacity(8);
    values.push(1.0);
    let mut buffer = Buffer::from_vec(values);
    assert!(buffer.has_owned_capacity(7));
    assert!(!buffer.has_owned_capacity(8));
    let pointer = buffer.as_ptr();
    buffer.try_reserve(1).unwrap();
    assert_eq!(buffer.as_ptr(), pointer);
    let alias = buffer.clone();
    assert!(!buffer.has_owned_capacity(1));
    assert!(!alias.has_owned_capacity(0));
    buffer.try_reserve(1).unwrap();
    assert_ne!(buffer.as_ptr(), alias.as_ptr());
    assert!(buffer.has_owned_capacity(1));
    buffer.append(&[2.]);
    assert_eq!(alias.as_slice(), &[1.]);
    assert_eq!(buffer.as_slice(), &[1., 2.]);
}

#[test]
fn exported_weak_owners_cannot_upgrade_after_capacity_admission() {
    let mut values = Vec::with_capacity(8);
    values.push(1.0);
    let mut buffer = Buffer::from_vec(values);
    let exported = buffer.keepalive().downcast::<Vec<f64>>().unwrap();
    let weak = Arc::downgrade(&exported);
    drop(exported);
    assert!(!buffer.has_owned_capacity(1));
    buffer.try_reserve(1).unwrap();
    assert!(weak.upgrade().is_none());
    assert!(buffer.has_owned_capacity(1));
    buffer.append(&[2.]);
    assert_eq!(buffer.as_slice(), &[1., 2.]);
}

#[test]
fn borrowed_capacity_is_prepared_before_it_becomes_unique() {
    let values = Arc::new(vec![1i64, 2, 3]);
    let mut buffer = Buffer::from_shared_range(values.clone(), 1..3).unwrap();
    assert!(!buffer.has_owned_capacity(0));
    buffer.try_reserve(1).unwrap();
    assert!(buffer.has_owned_capacity(1));
    buffer.make_mut().push(4);
    assert_eq!(values.as_slice(), &[1, 2, 3]);
    assert_eq!(buffer.as_slice(), &[2, 3, 4]);
}

#[test]
fn shared_and_borrowed_growth_keeps_amortized_spare() {
    let shared = Buffer::from_vec(vec![7i64; 128]);
    let snapshot = shared.clone();
    let borrowed = Buffer::from_shared_range(Arc::new(vec![7i64; 128]), 0..128).unwrap();
    for mut buffer in [shared, borrowed] {
        buffer.try_reserve(1).unwrap();
        buffer.make_mut().push(8);
        assert!(buffer.has_owned_capacity(1), "COW growth must retain spare after its first append");
        let pointer = buffer.as_ptr();
        buffer.try_reserve(1).unwrap();
        assert_eq!(buffer.as_ptr(), pointer);
    }
    assert_eq!(snapshot.as_slice(), &[7; 128]);
}
