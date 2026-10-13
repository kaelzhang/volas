//! Byte-level fixtures exercise the reader independently of a host database.
use super::*;

fn header(version: u8, times: usize, types: usize, chars: usize, leaps: usize, std: usize, ut: usize) -> Vec<u8> {
    let mut out = b"TZif".to_vec();
    out.push(version);
    out.extend([0; 15]);
    for n in [ut, std, leaps, times, types, chars] { out.extend((n as i32).to_be_bytes()); }
    out
}

fn block(width: usize, times: &[(i64, u8)], types: &[(i32, bool, u8)], chars: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(t, _) in times { out.extend(&t.to_be_bytes()[8 - width..]); }
    out.extend(times.iter().map(|t| t.1));
    for &(off, dst, abbr) in types {
        out.extend(off.to_be_bytes());
        out.extend([u8::from(dst), abbr]);
    }
    out.extend(chars);
    out
}

pub(super) fn fixture(version: u8, times: &[(i64, u8)], types: &[(i32, bool, u8)], chars: &[u8], footer: &str) -> Vec<u8> {
    let mut out = header(version, times.len(), types.len(), chars.len(), 0, 0, 0);
    out.extend(block(4, times, types, chars));
    if version != 0 {
        out.extend(header(version, times.len(), types.len(), chars.len(), 0, 0, 0));
        out.extend(block(8, times, types, chars));
        out.extend(format!("\n{footer}\n").as_bytes());
    }
    out
}

fn zone(bytes: Vec<u8>) -> Result<Tz> {
    Tz::from_tzif("Test/Rules", bytes, "bytes".into(), None)
}

#[test]
fn every_version_and_exact_transition_boundaries() {
    for version in [0, b'2', b'3', b'4'] {
        let bytes = fixture(version, &[(-10000, 0), (0, 1), (10000, 0)], &[(0, false, 0), (3600, true, 4)], b"STD\0DST\0", "");
        let tz = zone(bytes.clone()).unwrap();
        assert_eq!(tz.offset_secs_at(-1), 0);
        assert_eq!(tz.offset_secs_at(0), 3600);
        assert_eq!(tz.offset_secs_at(10_000_000_000_000 - 1), 3600);
        assert_eq!(tz.offset_secs_at(10_000_000_000_000), 0);
        assert_eq!(tz.dst_secs_at(0), 3600);
        assert_eq!(tz.zone_name_at(0), "DST");
        assert!(tz.localize(Civil::from_epoch_ns(1)).unwrap_err().to_string().contains("gap"));
        assert!(tz.localize(Civil::from_epoch_seconds(10000, 1)).unwrap_err().to_string().contains("fold"));
        assert_eq!(tz.localize_earliest(Civil::from_epoch_ns(1)).unwrap(), 0);
        assert_eq!(tz.localize_earliest(Civil::from_epoch_seconds(10000, 1)).unwrap(), 6_400_000_000_001);
        let Tz::Named(rules) = &tz else { panic!() };
        assert_eq!(rules.key(), "Test/Rules");
        assert_eq!(rules.bytes(), bytes);
        assert_eq!(rules.source(), "bytes");
        assert_eq!(rules.version(), None);
        assert!(!rules.is_canonical_utc());
        assert_eq!(tz, zone(bytes).unwrap());
        assert!(format!("{rules:?}").contains("Test/Rules"));
    }
}

#[test]
fn footer_without_transitions_covers_the_whole_axis() {
    let tz = zone(fixture(b'3', &[], &[(0, false, 0)], b"UTC\0", "EST5EDT,M3.2.0,M11.1.0")).unwrap();
    for year in [1678, 1800, 2024, 2115, 2261] {
        let t = datetime::parse_ns(&format!("{year}-07-01 00:00:00")).unwrap();
        assert_eq!(tz.offset_secs_at(t), -14400);
        assert_eq!(tz.zone_name_at(t), "EDT");
    }
    let fixed = zone(fixture(b'2', &[], &[(0, false, 0)], b"UTC\0", "<+0330>-3:30")).unwrap();
    assert_eq!(fixed.offset_secs_at(i64::MIN + 1), 12600);
    assert_eq!(fixed.offset_secs_at(i64::MAX), 12600);
    let after_axis = zone(fixture(b'2', &[(i64::MAX, 0)], &[(0, false, 0)], b"UTC\0", "EST5EDT,M3.2.0,M11.1.0")).unwrap();
    assert_eq!(after_axis.offset_secs_at(i64::MAX), 0);
    let permanent = zone(fixture(b'3', &[], &[(0, false, 0)], b"UTC\0", "EST5EDT,0/0,J365/25")).unwrap();
    for text in ["2150-01-01 05:00:00", "2150-07-01 00:00:00", "2151-01-01 05:00:00"] {
        assert_eq!(permanent.offset_secs_at(datetime::parse_ns(text).unwrap()), -14400);
    }
}

#[test]
fn cpython_initial_final_and_dst_derivation() {
    let chars = b"DST\0STD\0OTHER\0";
    let types = [(3600, true, 0), (0, false, 4), (7200, true, 8)];
    let tz = zone(fixture(0, &[(0, 0), (10000, 1), (20000, 2), (30000, 0), (40000, 1)], &types, chars, "")).unwrap();
    assert_eq!(tz.offset_secs_at(-1), 0); // first standard type
    assert_eq!(tz.dst_secs_at(0), 3600); // following standard comparison
    assert_eq!(tz.dst_secs_at(20_000_000_000_000), 7200);
    assert_eq!(tz.offset_secs_at(i64::MAX), 0);
    let no_standard = zone(fixture(0, &[(0, 1)], &[(1800, true, 0), (3600, true, 0)], b"DST\0", "")).unwrap();
    assert_eq!(no_standard.offset_secs_at(-1), 3600);
    let no_transitions = zone(fixture(0, &[], &types, chars, "")).unwrap();
    assert_eq!(no_transitions.offset_secs_at(0), 7200); // CPython fallback = final type
    let overlapping = zone(fixture(0, &[], &[(0, false, 1)], b"XLMT\0", "")).unwrap();
    assert_eq!(overlapping.zone_name_at(0), "LMT");
    let unresolved = zone(fixture(0, &[(0, 1), (10_000, 0)], &[(3600, true, 0), (7200, true, 0), (0, false, 4)], b"DST\0STD\0", "")).unwrap();
    assert_eq!(unresolved.dst_secs_at(10_000_000_000_000), 3600);
    assert!(tz.localize_fields([2024, 13, 1, 0, 0, 0], 0).is_err());
    assert!(tz.localize_fields([i64::MAX, 1, 1, 0, 0, 0], 0).is_err());
}

#[test]
fn explicit_byte_loads_are_independent_and_clones_share_rules() {
    let bytes = fixture(0, &[], &[(0, false, 0)], b"UTC\0", "");
    let a = zone(bytes.clone()).unwrap();
    let b = zone(bytes.clone()).unwrap();
    assert_eq!(a, b);
    let (Tz::Named(ar), Tz::Named(br)) = (&a, &b) else { panic!() };
    assert!(!Arc::ptr_eq(ar, br));
    let cloned = a.clone();
    let (Tz::Named(ar), Tz::Named(br)) = (&a, &cloned) else { panic!() };
    assert!(Arc::ptr_eq(ar, br));
    let c = zone(fixture(0, &[], &[(3600, false, 0)], b"NEW\0", "")).unwrap();
    assert_ne!(a, c);
    assert_eq!(a.offset_secs_at(0), 0);
    assert!(Tz::from_tzif("../invalid", bytes, "memory".into(), None).is_err());
}

#[test]
fn canonical_utc_checks_reachable_metadata_over_the_full_axis() {
    let canonical = |data| {
        let Tz::Named(rules) = zone(data).unwrap() else { panic!() };
        rules.is_canonical_utc()
    };
    assert!(canonical(fixture(0, &[], &[(0, false, 0)], b"UTC\0", "")));
    // An unused type's offset, DST flag and name cannot change the answer.
    assert!(canonical(fixture(0, &[], &[(3600, true, 4), (0, false, 0)], b"UTC\0BAD\0", "")));
    for (offset, dst, abbr) in [(1, false, 0), (0, true, 0), (0, false, 4)] {
        assert!(!canonical(fixture(0, &[], &[(offset, dst, abbr)], b"UTC\0GMT\0", "")));
    }
    assert!(!canonical(fixture(0, &[(100, 1), (101, 0)], &[(0, false, 0), (1, false, 0)], b"UTC\0", "")));

    let first = i64::MIN.div_euclid(1_000_000_000);
    let last = i64::MAX.div_euclid(1_000_000_000);
    let types = [(1, false, 4), (0, false, 0)];
    // Neither preceding nor following intervals intersect valid nanoseconds.
    assert!(canonical(fixture(b'3', &[(first, 1), (last + 1, 0)], &types, b"UTC\0BAD\0", "")));
    // FIRST itself cannot be multiplied into i64 nanoseconds, but its final
    // fraction is valid. Skipping it would silently normalize a non-UTC zone.
    let partial = zone(fixture(b'3', &[(first + 1, 1)], &types, b"UTC\0BAD\0", "")).unwrap();
    assert_eq!(partial.offset_secs_at(i64::MIN + 1), 1);
    let Tz::Named(rules) = partial else { panic!() };
    assert!(!rules.is_canonical_utc());
    assert!(!canonical(fixture(b'3', &[(first, 1), (last, 0)], &types, b"UTC\0BAD\0", "")));
    // With no explicit transitions, the footer governs the entire axis.
    assert!(canonical(fixture(b'3', &[], &types[..1], b"UTC\0BAD\0", "UTC0")));
    assert!(!canonical(fixture(b'3', &[], &[(0, false, 0)], b"UTC\0", "GMT0")));
}

#[test]
fn cursors_agree_with_scalar_queries_in_both_directions() {
    let tz = Tz::parse("America/New_York").unwrap();
    let mut values: Vec<i64> = (1700..2260).step_by(5).flat_map(|y| [1, 7].map(move |m| {
        datetime::parse_ns(&format!("{y}-{m:02}-15 12:00:00.123456789")).unwrap()
    })).collect();
    let mut cursor = tz.cursor();
    for &ns in &values {
        assert_eq!(cursor.offset_secs_at(ns), tz.offset_secs_at(ns));
        assert_eq!(cursor.civil(ns), tz.civil(ns));
    }
    values.reverse();
    values.push(datetime::NAT);
    let got = tz.localize_values(&values, &Tz::Naive).unwrap();
    for (&ns, result) in values.iter().zip(got) {
        if ns == datetime::NAT { assert_eq!(result, ns); }
        else { assert_eq!(result, tz.localize_from(ns, &Tz::Naive).unwrap()); }
    }
    assert!(tz.floor_local_grid(0, 0).is_err());
    assert_eq!(Tz::Offset(3600).localize_values(&[0], &Tz::Utc).unwrap(), [-3_600_000_000_000]);
    assert!(Tz::fixed(i32::MIN).is_err());
}

#[test]
fn closely_spaced_transitions_share_the_same_period_and_local_indexes() {
    // A one-second DST interval can skip a grid point and then fall back
    // before the next grid point. The skipped point remains a real boundary.
    let tz = zone(fixture(0, &[(100, 1), (101, 0)], &[(0, false, 0), (3600, true, 4)], b"STD\0DST\0", "")).unwrap();
    let hour = 3_600_000_000_000;
    assert_eq!(tz.floor_local_grid(500_000_000_000, hour).unwrap(), 100_000_000_000);
    assert_eq!(tz.floor_local_grid(99_000_000_000, hour).unwrap(), 0);
    assert_eq!(tz.offset_secs_at(100_000_000_000), 3600);
    assert_eq!(tz.offset_secs_at(101_000_000_000), 0);
    let wall = Civil::from_epoch_seconds(3700, 0);
    assert!(tz.localize(wall).unwrap_err().to_string().contains("fold"));
    assert_eq!(tz.localize_earliest(wall).unwrap(), 100_000_000_000);
}

#[test]
fn malformed_and_truncated_files_are_errors_not_panics() {
    let valid = fixture(b'2', &[(0, 0)], &[(0, false, 0)], b"UTC\0", "UTC0");
    for n in 0..valid.len() { assert!(zone(valid[..n].to_vec()).is_err(), "length {n}"); }
    for (index, byte) in [(0, b'X'), (4, b'5'), (36, 255), (40, 255)] {
        let mut bad = valid.clone(); bad[index] = byte;
        assert!(zone(bad).is_err());
    }
    let bad = [
        fixture(0, &[(10, 0), (0, 0)], &[(0, false, 0)], b"UTC\0", ""),
        fixture(0, &[(0, 1)], &[(0, false, 0)], b"UTC\0", ""),
        fixture(0, &[], &[(i32::MIN, false, 0)], b"UTC\0", ""),
        fixture(0, &[], &[(0, false, 20)], b"UTC\0", ""),
        fixture(0, &[], &[(0, false, 0)], b"UTC", ""),
        fixture(0, &[], &[(0, false, 0)], b"\xff\0", ""),
        fixture(b'2', &[], &[(0, false, 0)], b"UTC\0", "garbage"),
        fixture(b'2', &[], &[(0, false, 0)], b"UTC\0", "UTC0\0"),
    ];
    for bytes in bad { assert!(zone(bytes).is_err()); }
    for bytes in [header(0, 0, 0, 4, 0, 0, 0), header(0, 0, 257, 4, 0, 0, 0),
        header(0, 0, 1, 0, 0, 0, 0), header(0, 0, 1, 4, 0, 2, 0), header(0, 0, 1, 4, 0, 0, 2)] {
        assert!(zone(bytes).is_err());
    }
    let mut bad = fixture(0, &[], &[(0, false, 0)], b"UTC\0", "");
    bad[48] = 2; assert!(zone(bad).is_err());
}

#[test]
fn leap_records_and_transition_indicators_are_consumed() {
    for version in [0, b'2', b'3', b'4'] {
        let mut bytes = header(version, 0, 1, 4, 2, 1, 1);
        let mut add = |width: usize| {
            bytes.extend(block(width, &[], &[(0, false, 0)], b"UTC\0"));
            for (t, correction) in [(100i64, 1i32), (200, if version == b'4' { 1 } else { 2 })] {
                bytes.extend(&t.to_be_bytes()[8-width..]);
                bytes.extend(correction.to_be_bytes());
            }
            bytes.extend([1, 1]);
        };
        add(4);
        if version != 0 {
            bytes.extend(header(version, 0, 1, 4, 2, 1, 1));
            bytes.extend(block(8, &[], &[(0, false, 0)], b"UTC\0"));
            for (t, correction) in [(100i64, 1i32), (200, if version == b'4' { 1 } else { 2 })] {
                bytes.extend(t.to_be_bytes()); bytes.extend(correction.to_be_bytes());
            }
            bytes.extend([1, 1]);
            bytes.extend(b"\nUTC0\n");
        }
        assert_eq!(zone(bytes).unwrap().offset_secs_at(0), 0);
    }
    let mut bad = header(0, 0, 1, 4, 1, 0, 0);
    bad.extend(block(4, &[], &[(0, false, 0)], b"UTC\0"));
    bad.extend([0; 8]);
    assert!(zone(bad).is_err());
    let mut bad = header(0, 0, 1, 4, 0, 1, 1);
    bad.extend(block(4, &[], &[(0, false, 0)], b"UTC\0"));
    bad.extend([0, 1]);
    assert!(zone(bad).is_err());
}

// Dublin may be distributed in rearguard form by an OS. Use explicit standard
// TZif footer bytes to exercise negative SAVE independently of that choice.
pub(super) fn negative_dst_zone() -> Tz {
    zone(fixture(b'3', &[], &[(3600, false, 0)], b"IST\0", "IST-1GMT0,M10.5.0,M3.5.0/1")).unwrap()
}
