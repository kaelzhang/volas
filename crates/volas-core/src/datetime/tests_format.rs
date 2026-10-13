//! Native text-owner coverage of calendar, clock, offset, and format validation.

use super::*;

#[test]
fn native_strftime_covers_calendar_vocabulary_and_modifiers() {
    let ns = parse_ns("2024-01-07 15:04:05.123456789").unwrap();
    let f = |fmt| strftime(ns, &Tz::Utc, fmt).unwrap();
    assert_eq!(f("%Y|%C|%y|%G|%g|%m|%d|%e|%H|%k|%I|%l|%M|%S|%j|%q|%u|%w|%U|%V|%W|%a|%A|%b|%h|%B|%p|%P"), "2024|20|24|2024|24|01|07| 7|15|15|03| 3|04|05|007|1|7|0|01|01|01|Sun|Sunday|Jan|Jan|January|PM|pm");
    assert_eq!(f("%D|%x|%F|%v|%R|%T|%X|%r|%c"), "01/07/24|01/07/24|2024-01-07| 7-Jan-2024|15:04|15:04:05|15:04:05|03:04:05 PM|Sun Jan  7 15:04:05 2024");
    assert_eq!(f("%+"), "2024-01-07T15:04:05.123456789+00:00");
    assert_eq!(
        f("%-d|%_d|%0e|%f|%.f|%3f|%6f|%9f|%.3f|%.6f|%.9f"),
        "7| 7|07|123456|.123456789|123|123456|123456789|.123|.123456|.123456789"
    );
    assert_eq!(f("日期 %Y%n%t%%"), "日期 2024\n\t%");
    assert_eq!(strftime(0, &Tz::Utc, "%p %P %.f").unwrap(), "AM am ");
    assert_eq!(strftime(NAT, &Tz::Naive, "%Y").unwrap(), "NaT");
    assert_eq!(strftime(0, &Tz::Offset(-19_270), "%:::z").unwrap(), "-05");
    for fmt in [
        "%", "%Q", "%#z", "%#Y", "%::::z", "%:Y", "%.3Y", "%3Y", "%_A", "%2f", "%.2f",
    ] {
        assert!(strftime(ns, &Tz::Utc, fmt).is_err(), "{fmt}");
    }
}

#[test]
fn native_explicit_parse_resolves_dates_and_checks_redundant_fields() {
    for (text, fmt, expected) in [
        ("20240107", "%Y%m%d", "2024-01-07"),
        ("01/07/24", "%D", "2024-01-07"),
        (" 7-Jan-2024", "%v", "2024-01-07"),
        ("Sun Jan  7 15:04:05 2024", "%c", "2024-01-07 15:04:05"),
        (
            "sunday january 7 2024 03:04:05 pm",
            "%A %B %d %Y %r",
            "2024-01-07 15:04:05",
        ),
        ("2024 060", "%Y %j", "2024-02-29"),
        ("2020-W53-5", "%G-W%V-%u", "2021-01-01"),
        ("20-W53-5", "%g-W%V-%u", "2021-01-01"),
        ("99-W52-6", "%g-W%V-%u", "2000-01-01"),
        ("2024 01 0", "%Y %U %w", "2024-01-07"),
        ("2024 01 1", "%Y %W %u", "2024-01-01"),
        ("19 99-12-31", "%C %y-%m-%d", "1999-12-31"),
        ("70-01-01", "%y-%m-%d", "1970-01-01"),
        ("2024-01-07 15 03 PM", "%F %H %I %p", "2024-01-07 15:00"),
        (
            "2024-01-07 00 123456789",
            "%F %S %f",
            "2024-01-07 00:00:00.123456789",
        ),
        (
            "2024-01-07 00.123456789987",
            "%F %S%.f",
            "2024-01-07 00:00:00.123456789",
        ),
        ("2024-01-07 00.123", "%F %S%.3f", "2024-01-07 00:00:00.123"),
        ("2024-01-07 00 123456", "%F %S %6f", "2024-01-07 00:00:00.123456"),
        ("2024-01-07", "%F%.f", "2024-01-07"),
        ("2024-01-07+08", "%F%#z", "2024-01-06 16:00"),
        ("2024-01-07+08", "%F%:::z", "2024-01-06 16:00"),
        ("2024-01-07Z", "%F%#z", "2024-01-07"),
        ("2024-01-07+05:21:10", "%F%::z", "2024-01-06 18:38:50"),
        ("0+0800 08", "%s%z %H", "1970-01-01"),
        ("2024%01%07", "%Y%%%m%%%d", "2024-01-07"),
        ("2024  01\n07", "%Y%t%m%n%d", "2024-01-07"),
        (
            "2024-01-07 1 007 2024 01 7",
            "%F %q %j %G %V %u",
            "2024-01-07",
        ),
    ] {
        assert_eq!(
            parse_ns_format(text, fmt).unwrap(),
            parse_ns(expected).unwrap(),
            "{text} / {fmt}"
        );
    }
    for (text, fmt) in [
        ("2023 366", "%Y %j"),
        ("2021-W53-1", "%G-W%V-%u"),
        ("2024-01-07 2", "%F %q"),
        ("2024-01-07 8", "%F %u"),
        ("2024-01-07 7", "%F %w"),
        ("2024-01-07 8", "%F %d"),
        ("2024-01-07 AM", "%F %p"),
        ("2024-01-07 03", "%F %I"),
        ("2024-01-07 15 03 AM", "%F %H %I %p"),
        ("2024-01-07 24", "%F %H"),
        ("2024-01-07 61", "%F %S"),
        ("2024-01-07 123", "%F %6f"),
        ("2024-01-07.", "%F%.f"),
        ("2024-01-07+08", "%F%z"),
        ("2024-01-07+25:00", "%F%:z"),
        ("2024-01-07 2025", "%F %Y"),
        ("2024-01-07 2025", "%F %G"),
        ("2024-01-07 8", "%F %j"),
        ("2024-01-07 2", "%F %V"),
        ("2024-01-07 2", "%F %W"),
        ("2024-01-07 2", "%F %U"),
        ("2024-01-07 Mon", "%F %a"),
        ("2024-01-07 Feb", "%F %b"),
        ("2024-01-07 noon", "%F %p"),
        ("2024-01-07 trailing", "%F"),
        ("2024-01-07", "%Q"),
        ("2024", "%Y"),
        ("2024-02-30", "%F"),
        ("9223372036854775807+0100", "%s%z"),
    ] {
        assert!(parse_ns_format(text, fmt).is_err(), "{text} / {fmt}");
    }
}

#[test]
fn native_unformatted_parse_rejects_bad_calendar_and_clock_fields() {
    for text in ["2024/2/29 1:02:03.4", "2024-2-29T1:02:03.4"] {
        assert_eq!(
            parse_ns(text).unwrap(),
            parse_ns("2024-02-29 01:02:03.400").unwrap()
        );
    }
    for text in [
        "2024-13-01",
        "2024-01-00",
        "2023-02-29",
        "2024-01-01 24:00",
        "2024-01-01 12:60",
        "2024-01-01 12:00:61",
        "2024-01-01 12:00:00:",
        "2024-01-01 12:00:00.",
        "2024-01-01 12:00:x",
        "2024-01-01 12:",
        "2024-01-01 :00",
        "2024-1/1",
        "2024-01-01 12:00:00.日",
        "12:00:00.5",
        "99999999999999999999-01-01",
    ] {
        assert!(parse_ns(text).is_err(), "{text}");
    }
    assert!(parse_ns("09:30:15").is_ok());
}

#[test]
fn native_format_preserves_fraction_modifiers_and_rfc3339() {
    assert_eq!(
        strftime(1, &Tz::Utc, "%-f|%_f|%0f|%f").unwrap(),
        "1|        1|000000001|000000"
    );
    assert_eq!(
        strftime(0, &Tz::Naive, "%+").unwrap(),
        "1970-01-01T00:00:00+00:00"
    );
    for text in [
        "2024-1-7T12:34:56Z",
        "+2024-1-7T12:34:56Z",
        "2024-1-7t12:34:56utc",
        "2024 -1-7 12 :34:56 UTC",
        "2024-1-7T12:34:56+0000",
    ] {
        assert_eq!(
            parse_ns_format(text, "%+").unwrap(),
            parse_ns("2024-01-07 12:34:56").unwrap()
        );
    }
    assert_eq!(
        parse_ns_format("2024-1-7T12:34:56Z!", "%+!").unwrap(),
        parse_ns("2024-01-07 12:34:56").unwrap()
    );
    for text in [
        "2024-1-7Q12:34:56Z",
        "2024-1-7",
        "2024-1-7T12:34:56MST",
        "2024-1-7T12:34:56+",
    ] {
        assert!(parse_ns_format(text, "%+").is_err());
    }
}

#[test]
fn text_ingress_preserves_normalization_of_leap_second_notation() {
    let next = parse_ns("2017-01-01 00:00:00.123456789").unwrap();
    assert_eq!(parse_ns("2016-12-31 23:59:60.123456789").unwrap(), next);
    assert_eq!(
        parse_ns_format("2016-12-31 23:59:60.123456789", "%F %T%.f").unwrap(),
        next
    );
    assert_eq!(
        parse_ns_format("2016-12-31T23:59:60.123456789Z", "%+").unwrap(),
        next
    );
}

#[test]
fn text_leap_carry_uses_the_original_wall_interval_and_final_range_gate() {
    let first = "1677-09-21 00:12:60.145224193";
    let last = "2262-04-11 23:47:60.854775807";
    assert_eq!(parse_ns(&format!("{first}+00:00:17")).unwrap(), NAT + 1);
    assert_eq!(parse_ns_in_tz(first, &Tz::Offset(17)).unwrap(), NAT + 1);
    assert_eq!(parse_ns_in_tz(last, &Tz::Offset(44)).unwrap(), i64::MAX);
    assert!(parse_ns_in_tz("2262-04-11 23:47:60.854775808", &Tz::Offset(44)).is_err());
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(parse_ns_in_tz("2024-03-10 01:59:60", &ny).unwrap(), parse_ns("2024-03-10T07:00:00Z").unwrap());
    assert_eq!(parse_ns_in_tz("2024-11-03 00:59:60", &ny).unwrap(), parse_ns("2024-11-03T05:00:00Z").unwrap());
}

#[test]
fn native_text_helpers_keep_fraction_groups_nat_and_zone_parts() {
    assert_eq!(format_ns(123_000_000), "1970-01-01 00:00:00.123");
    assert_eq!(format_ns(123_456_000), "1970-01-01 00:00:00.123456");
    assert_eq!(format_ns(NAT), "NaT");
    assert_eq!(civil_parts_tz(0, &Tz::Offset(-3600)), (1969, 12, 31, 23, 0, 0));
    // Even a valid time of day cannot carry an i64-second civil clock past its end.
    let end = Civil::from_epoch_seconds(i64::MAX, 0);
    let text = format!("+{}-{:02}-{:02} {:02}:{:02}:{:02}", end.year(), end.month(), end.day(), end.hour(), end.minute(), end.second() + 1);
    assert!(parse_ns_format(&text, "%F %T").is_err());
    for (text, fmt) in [("-1", "%s"), ("2024-01-07 123456", "%F %6f"), ("2024-01-07 13:45:00.001", "%F %T%.6f")] {
        assert!(parse_ns_format(text, fmt).is_err());
    }
}
