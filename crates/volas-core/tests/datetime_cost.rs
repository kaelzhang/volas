//! Numeric fields write directly into the final text allocation.
#[path = "support/allocation.rs"]
mod allocation;
use allocation::measure;
use volas_core::{datetime, Tz};

#[test]
fn numeric_formatting_has_no_field_allocations() {
    let (out, calls, bytes) = measure(|| datetime::strftime(0, &Tz::Utc, "%F %T").unwrap());
    assert_eq!(out, "1970-01-01 00:00:00");
    assert!(calls <= 4, "{calls} allocations, {bytes} bytes");
}

#[test]
fn compiled_formats_allocate_only_final_text_and_parsing_allocates_nothing() {
    let formatter = datetime::Strftime::new("%F %T");
    let parser = datetime::Strptime::new("%F %T");
    for ns in [0, 1_234_000_000, i64::MAX, i64::MIN + 1_000_000_000] {
        let (text, calls, _) = measure(|| formatter.format(ns, &Tz::Utc).unwrap());
        assert_eq!(calls, 1, "{text}");
        let (got, calls, _) = measure(|| parser.parse(&text).unwrap());
        assert_eq!(got, datetime::parse_ns_format(&text, "%F %T").unwrap());
        assert_eq!(calls, 0);
    }
}

#[test]
fn zone_fields_write_into_the_final_text_buffer() {
    let fmt = datetime::Strftime::new("%Z|%z|%:z|%::z|%:::z");
    for tz in [Tz::Naive, Tz::Utc, Tz::Offset(-19_270), Tz::parse("America/New_York").unwrap()] {
        let (_, calls, _) = measure(|| fmt.format(0, &tz).unwrap());
        // The longest output may grow once; neither zone field allocates a String.
        assert!(calls <= 2, "{tz:?}: {calls}");
    }
    let fmt = datetime::Strftime::new("%Z");
    for tz in [Tz::Utc, Tz::Offset(-19_270), Tz::parse("America/New_York").unwrap()] {
        let (_, calls, _) = measure(|| fmt.format(0, &tz).unwrap());
        assert_eq!(calls, 1);
    }
}
