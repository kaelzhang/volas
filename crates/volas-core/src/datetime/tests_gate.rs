//! The instant-axis range gate and the unit / rounding arithmetic behind it.

use super::*;

const MAX: i64 = i64::MAX;
const MIN: i64 = i64::MIN + 1; // the first instant; i64::MIN is NaT
const DAY: i64 = 86_400 * NS_PER_SEC;

fn overflow<T: std::fmt::Debug>(r: Result<T>) -> bool {
    matches!(r, Err(VolasError::Overflow(_)))
}

#[test]
fn gate_rejects_overflow_and_the_sentinel() {
    assert_eq!(instant(Some(MIN)).unwrap(), MIN);
    assert!(overflow(instant(None)));
    assert!(overflow(instant(Some(NAT))));
    assert!(overflow(instant_wide(MAX as i128 + 1)));
    assert_eq!(shift(MAX - 1, 1).unwrap(), MAX);
    assert!(overflow(shift(MAX, 1)));
    assert!(overflow(shift(MIN, -1))); // lands on NaT
    assert_eq!(span_ns(15, 60 * NS_PER_SEC).unwrap(), 900 * NS_PER_SEC);
    assert!(overflow(span_ns(106_752, DAY)));
}

#[test]
fn rounding_is_exact_at_the_range_edges() {
    use Rounding::*;
    assert_eq!(round_wall(MAX, 0, DAY, Floor).unwrap(), MAX - MAX % DAY);
    assert!(overflow(round_wall(MAX, 0, DAY, Ceil)));
    assert!(overflow(round_wall(MIN, 0, DAY, Floor)));
    assert_eq!(round_wall(DAY, 0, DAY, Ceil).unwrap(), DAY); // on the grid
    assert_eq!(round_wall(6 * 10_i64.pow(18), 0, 100_000 * DAY, HalfEven).unwrap(), 100_000 * DAY);
    // ties go to the even multiple of the WALL-clock grid
    assert_eq!(round_wall(5, 0, 10, HalfEven).unwrap(), 0);
    assert_eq!(round_wall(15, 0, 10, HalfEven).unwrap(), 20);
    assert_eq!(round_wall(14, 0, 10, HalfEven).unwrap(), 10);
    // +01:00: 00:30 local rounds to 00:00 local (23:00Z the day before)
    let h = 3_600 * NS_PER_SEC;
    assert_eq!(round_wall(-h / 2, 3_600, h, HalfEven).unwrap(), -h);
}

#[test]
fn numpy_units_scale_checked_and_exactly() {
    assert_eq!(datetime64_to_ns(2, "D", 1).unwrap(), 2 * DAY);
    assert_eq!(datetime64_to_ns(3, "s", 5).unwrap(), 15 * NS_PER_SEC);
    assert_eq!(datetime64_to_ns(2_000, "ps", 1).unwrap(), 2);
    assert!(matches!(datetime64_to_ns(1_001, "ps", 1), Err(VolasError::Value(_))));
    assert!(matches!(datetime64_to_ns(1, "eon", 1), Err(VolasError::Value(_))));
    assert!(overflow(datetime64_to_ns(200_000, "D", 1)));
    assert_eq!(datetime64_to_ns(54, "Y", 1).unwrap(), parse_ns("2024-01-01").unwrap());
    assert_eq!(datetime64_to_ns(-1, "M", 1).unwrap(), parse_ns("1969-12-01").unwrap());
    assert!(overflow(datetime64_to_ns(400, "Y", 1)));
    assert!(overflow(datetime64_to_ns(i64::MAX, "Y", 1))); // the year leaves i32
    assert_eq!(datetime64_values_to_ns(&[1, NAT], "ns", 1).unwrap(), vec![1, NAT]);
    assert_eq!(datetime64_values_to_ns(&[1, NAT], "us", 1).unwrap(), vec![1_000, NAT]);
    assert_eq!(timedelta64_to_ns(-3, "us", 1).unwrap(), -3_000);
    assert!(matches!(timedelta64_to_ns(NAT, "s", 1), Err(VolasError::Value(_))));
    assert!(matches!(timedelta64_to_ns(1, "M", 1), Err(VolasError::Value(_))));
    assert!(overflow(timedelta64_to_ns(106_752, "D", 1)));
}

#[test]
fn epochs_pass_the_gate() {
    assert!(overflow(epoch_to_ns(i64::MIN, "ns")));
    assert!(overflow(epoch_to_ns(10_000_000_000, "s")));
    assert!(matches!(epoch_to_ns(1, "D"), Err(VolasError::Value(_))));
    assert!(overflow(epoch_to_ns_f64(-(2f64.powi(63)), "ns")));
    assert!(overflow(epoch_to_ns_f64(2f64.powi(63), "ns")));
    assert!(matches!(epoch_to_ns_f64(f64::INFINITY, "s"), Err(VolasError::Value(_))));
}

#[test]
fn parse_is_exact_at_the_range_edges() {
    // the first instant round-trips through its own text (the whole second alone
    // is below the range; the fraction brings it back in)
    assert_eq!(parse_ns(&format_ns(MIN)).unwrap(), MIN);
    assert_eq!(parse_ns(&format_ns(MAX)).unwrap(), MAX);
    assert!(overflow(parse_ns("1677-09-21T00:12:43.145224192Z")));
    assert!(overflow(parse_ns("2262-04-11 23:47:16.854775808")));
    let minus8 = Tz::parse("-08:00").unwrap();
    assert!(overflow(parse_ns_in_tz("2262-04-11 23:00", minus8)));
    // a wall clock past the range whose instant is inside it
    let plus8 = Tz::parse("+08:00").unwrap();
    assert_eq!(parse_ns_in_tz("2262-04-12 07:47:16.854775807", plus8).unwrap(), MAX);
    let fmt = "%Y-%m-%d %H:%M%z";
    assert!(parse_ns_format("2262-04-12 07:00+0800", fmt).is_ok());
    assert!(overflow(parse_ns_format("2262-04-11 23:00-0800", fmt)));
    assert!(parse_ns("09:30").is_ok());
    assert!(matches!(parse_ns("nope"), Err(VolasError::Value(_))));
}

#[test]
fn civil_parts_never_wrap_at_the_edges() {
    let plus8 = Tz::parse("+08:00").unwrap();
    assert_eq!(plus8.civil_parts(MAX), (2262, 4, 12, 7, 47, 16));
    let minus8 = Tz::parse("-08:00").unwrap();
    assert_eq!(minus8.civil_parts(MIN).0, 1677);
}
