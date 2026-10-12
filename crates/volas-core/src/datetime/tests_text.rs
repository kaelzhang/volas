//! The instant <-> text boundary: offset grammar, offset-aware parse, strftime,
//! offset-carrying output and the one float-epoch rule.

use super::*;
use crate::tz::{format_offset, parse_offset_seconds};

#[test]
fn one_strict_offset_grammar() {
    let ok = [("+8", 28_800), ("-08", -28_800), ("+0530", 19_800), ("+052110", 19_270),
        ("+05:30", 19_800), ("-05:21:10", -19_270), ("+8:30", 30_600), ("+23:59:59", 86_399)];
    for (s, secs) in ok {
        assert_eq!(parse_offset_seconds(s), Some(secs), "{s}");
    }
    for s in ["8", "+", "+-1:00", "+01:-30", "+24", "+05:60", "+05:00:60", "+1:5", "+123",
        "+1日", "+ 5", "+05:", "+05:30:00:00", "+0530 "] {
        assert_eq!(parse_offset_seconds(s), None, "{s}");
    }
    assert_eq!(Tz::parse("+05:21:10").unwrap().name(), "+05:21:10");
    assert!(Tz::parse("+-600000:00").unwrap_err().to_string().contains("invalid UTC offset"));
    assert_eq!(Tz::fixed(0).unwrap(), Tz::Utc);
    assert!(Tz::fixed(86_400).is_err());
    assert_eq!(format_offset(-19_270, ""), "-052110");
    assert_eq!(format_offset(28_800, ":"), "+08:00");
}

#[test]
fn offset_aware_text_reads_every_offset_form() {
    let at = |s: &str| parse_ns(s).unwrap();
    let base = at("2024-01-01 00:00:00");
    assert_eq!(at("2024-01-01T08:00:00+08:00"), base);
    assert_eq!(at("2024-01-01 08:00:00 +0800"), base);
    assert_eq!(at("2024-01-01 08:00+08"), base);
    assert_eq!(at("2024-01-01T00:00:00Z"), base);
    assert_eq!(at("2024-01-01 05:21:10+05:21:10"), base);
    assert_eq!(offset_suffix_secs("2024-01-01 05:21:10+052110"), Some(19_270));
    assert_eq!(offset_suffix_secs("2024-01-01"), None);
    assert!(parse_ns("2024-01-01 08:00+25:00").is_err());
}

#[test]
fn strftime_keeps_every_field_of_the_instant() {
    let ns = parse_ns("2024-01-01 04:00:00.123456789").unwrap(); // 12:00 at +08:00
    let p8 = Tz::parse("+08:00").unwrap();
    let f = |tz, fmt| strftime(ns, tz, fmt).unwrap();
    assert_eq!(f(p8, "%H:%M:%S.%f %z %:z %::z %:::z %Z"), "12:00:00.123456 +0800 +08:00 +08:00:00 +08 UTC+08:00");
    assert_eq!(f(p8, "%s %%f"), "1704081600 %f");
    assert_eq!(f(Tz::Naive, "%H %z%Z|"), "04 |");
    assert_eq!(f(Tz::Utc, "%Z %z"), "UTC +0000");
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(f(ny, "%H %Z %z"), "23 EST -0500");
    let lmt = parse_ns("1900-01-01 00:00:00").unwrap();
    let kolkata = Tz::parse("Asia/Kolkata").unwrap();
    assert_eq!(strftime(lmt, kolkata, "%z %::z").unwrap(), "+052110 +05:21:10");
    // a numeric tzdata abbreviation is spelled from the offset
    let sao_paulo = Tz::parse("America/Sao_Paulo").unwrap();
    assert_eq!(f(sao_paulo, "%Z"), "-03");
    let kathmandu = Tz::parse("Asia/Kathmandu").unwrap();
    assert_eq!(f(kathmandu, "%Z"), "+0545");
    // an invalid or unformattable spec is an error, never a panic
    assert!(strftime(ns, p8, "%Q").is_err());
    assert!(strftime(ns, p8, "%#z").is_err());
    assert!(strftime(ns, p8, "100%").is_err()); // a dangling %
}

#[test]
fn csv_text_carries_the_offset_of_an_aware_instant() {
    let ny = Tz::parse("America/New_York").unwrap();
    let first = parse_ns("2024-11-03 05:30:00").unwrap();
    let second = first + 3_600 * NS_PER_SEC;
    assert_eq!(format_ns_offset(first, ny), "2024-11-03 01:30:00-04:00");
    assert_eq!(format_ns_offset(second, ny), "2024-11-03 01:30:00-05:00");
    assert_eq!(parse_ns(&format_ns_offset(second, ny)).unwrap(), second);
    assert_eq!(format_ns_offset(first, Tz::Naive), "2024-11-03 05:30:00");
    assert_eq!(format_ns_offset(NAT, ny), "NaT");
    assert_eq!(format_ns_offset(first, Tz::Utc), "2024-11-03 05:30:00+00:00");
}

#[test]
fn one_exact_float_epoch_rule() {
    assert_eq!(epoch_to_ns_f64(1_704_067_200.000_001, "s").unwrap(), 1_704_067_200_000_000_954);
    assert_eq!(epoch_to_ns_f64(1_577_836_800.25, "s").unwrap(), 1_577_836_800_250_000_000);
    assert_eq!(epoch_to_ns_f64(-1.5, "s").unwrap(), -1_500_000_000);
    // ties go to even
    assert_eq!(epoch_to_ns_f64(0.5, "ns").unwrap(), 0);
    assert_eq!(epoch_to_ns_f64(1.5, "ns").unwrap(), 2);
    assert_eq!(epoch_to_ns_f64(-2.5, "ns").unwrap(), -2);
    assert_eq!(epoch_to_ns_f64(-0.0, "s").unwrap(), 0);
    assert_eq!(epoch_to_ns_f64(f64::MIN_POSITIVE / 4.0, "s").unwrap(), 0); // subnormal
    assert_eq!(epoch_to_ns_f64(1e-300, "s").unwrap(), 0);
    assert_eq!(epoch_to_ns_f64(4.0e12, "us").unwrap(), 4 * 10_i64.pow(15));
    assert!(epoch_to_ns_f64(1e300, "s").is_err());
    assert!(epoch_to_ns_f64(9.3e9, "s").is_err());
}
