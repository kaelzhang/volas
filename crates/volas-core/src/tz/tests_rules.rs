//! Full-axis named rules: historical transitions and generated future rules
//! use the same compiled interval indexes.

use super::*;

fn at(s: &str) -> i64 { crate::datetime::parse_ns(s).unwrap() }
fn wall(s: &str) -> Civil { Civil::from_epoch_ns(at(s)) }

#[test]
fn far_future_offsets_dst_and_names_follow_the_rules() {
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(ny.offset_secs_at(at("2115-07-01 12:00:00")), -4 * 3600);
    assert_eq!(ny.zone_name_at(at("2115-07-01 12:00:00")), "EDT");
    assert_eq!(ny.offset_secs_at(at("2115-12-01 12:00:00")), -5 * 3600);
    assert_eq!(ny.civil_parts(at("2115-07-01 12:00:00")), (2115, 7, 1, 8, 0, 0));
    let dublin = super::tests_tzif::negative_dst_zone();
    assert_eq!(dublin.dst_secs_at(at("2150-01-15 12:00:00")), -3600);
    assert_eq!(dublin.zone_name_at(at("2150-01-15 12:00:00")), "GMT");
    let lord_howe = Tz::parse("Australia/Lord_Howe").unwrap();
    assert_eq!(lord_howe.offset_secs_at(at("2150-01-15 12:00:00")), 11 * 3600);
    assert_eq!(lord_howe.dst_secs_at(at("2150-01-15 12:00:00")), 1800);
    // a zone without letters and without DST keeps its numeric name
    let tehran = Tz::parse("Asia/Tehran").unwrap();
    assert_eq!(tehran.zone_name_at(at("2200-01-15 12:00:00")), "+0330");
}

#[test]
fn far_future_wall_clocks_resolve_gaps_and_folds() {
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(ny.localize(wall("2115-07-01 08:00:00")).unwrap(), at("2115-07-01 12:00:00"));
    let gap = ny.localize(wall("2115-03-10 02:30:00")).unwrap_err();
    assert!(gap.to_string().contains("DST gap"), "{gap}");
    let fold = ny.localize(wall("2115-11-03 01:30:00")).unwrap_err();
    assert!(fold.to_string().contains("DST fold"), "{fold}");
    // a period start takes a fold's first occurrence and a gap's end
    assert_eq!(ny.localize_earliest(wall("2115-11-03 01:30:00")).unwrap(), at("2115-11-03 05:30:00"));
    assert_eq!(ny.localize_earliest(wall("2115-03-10 02:30:00")).unwrap(), at("2115-03-10 07:00:00"));
    // Santiago springs forward at midnight: the day starts at 01:00 -03
    let santiago = Tz::parse("America/Santiago").unwrap();
    assert_eq!(santiago.day_start(2150, 9, 6).unwrap(), at("2150-09-06 04:00:00"));
    // the wall clocks straddling the table's end resolve like any other
    assert_eq!(ny.localize(wall("2099-12-31 19:00:00")).unwrap(), at("2100-01-01 00:00:00"));
    // Civil dates beyond the instant axis still report an overflow.
    let end = Civil::new(3000, 12, 31, 23, 0, 0, 0).unwrap();
    assert!(matches!(ny.localize(end), Err(VolasError::Overflow(_))));
    assert!(matches!(ny.localize(Civil::new(2300, 7, 1, 12, 0, 0, 0).unwrap()), Err(VolasError::Overflow(_))));
    assert!(matches!(ny.day_start(2300, 3, 10), Err(VolasError::Overflow(_))));
}
