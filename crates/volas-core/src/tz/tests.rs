use super::*;

#[test]
fn parse_forms() {
    assert_eq!(Tz::Naive.zone_name_at(0), "");
    assert!(Tz::parse("").is_err()); // F47: empty spec is an error, never silent
    assert_eq!(Tz::parse("UTC").unwrap(), Tz::Utc);
    assert_eq!(Tz::parse("+00:00").unwrap(), Tz::Utc);
    assert_eq!(Tz::parse("+08:00").unwrap(), Tz::Offset(28800));
    assert_eq!(Tz::parse("+0800").unwrap(), Tz::Offset(28800));
    assert_eq!(Tz::parse("+8").unwrap(), Tz::Offset(28800));
    assert_eq!(Tz::parse("-05:00").unwrap(), Tz::Offset(-18000));
    assert!(matches!(
        Tz::parse("America/New_York").unwrap(),
        Tz::Named(_)
    ));
    assert!(Tz::parse("Not/AZone").is_err());
}

#[test]
fn offset_round_trip() {
    // 2020-01-01 08:00:00+08:00 == 2020-01-01 00:00:00 UTC
    let tz = Tz::parse("+08:00").unwrap();
    let utc_ns = tz.wall_to_utc_ns(2020, 1, 1, 8, 0, 0).unwrap();
    assert_eq!(crate::datetime::format_ns(utc_ns), "2020-01-01 00:00:00");
    // and rendering that instant back in +08:00 shows the wall clock
    assert_eq!(tz.civil_parts(utc_ns), (2020, 1, 1, 8, 0, 0));
    assert_eq!(tz.name(), "+08:00");
}

#[test]
fn named_zone_dst() {
    let ny = Tz::parse("America/New_York").unwrap();
    // Winter: EST = UTC-5. 2021-01-01 12:00 NY -> 17:00 UTC.
    let w = ny.wall_to_utc_ns(2021, 1, 1, 12, 0, 0).unwrap();
    assert_eq!(crate::datetime::format_ns(w), "2021-01-01 17:00:00");
    // Summer: EDT = UTC-4. 2021-07-01 12:00 NY -> 16:00 UTC.
    let s = ny.wall_to_utc_ns(2021, 7, 1, 12, 0, 0).unwrap();
    assert_eq!(crate::datetime::format_ns(s), "2021-07-01 16:00:00");
    // round-trip rendering
    assert_eq!(ny.civil_parts(w), (2021, 1, 1, 12, 0, 0));
    assert_eq!(ny.civil_parts(s), (2021, 7, 1, 12, 0, 0));
}

#[test]
fn offset_and_dst_at_instant() {
    // fixed offsets: constant offset, zero DST; named zones answer per instant.
    let fixed = Tz::parse("+08:00").unwrap();
    assert_eq!(fixed.offset_secs_at(0), 28800);
    assert_eq!(fixed.dst_secs_at(0), 0);
    assert_eq!(Tz::Utc.offset_secs_at(0), 0);
    let ny = Tz::parse("America/New_York").unwrap();
    let summer = ny.wall_to_utc_ns(2021, 7, 1, 12, 0, 0).unwrap();
    assert_eq!(ny.offset_secs_at(summer), -4 * 3600);   // EDT
    assert_eq!(ny.dst_secs_at(summer), 3600);           // one DST hour
    // from the transition data: a southern summer, a 30-minute DST, Dublin's
    // negative winter DST
    let jan = |zone: &str| {
        let tz = Tz::parse(zone).unwrap();
        tz.dst_secs_at(tz.wall_to_utc_ns(2024, 1, 15, 12, 0, 0).unwrap())
    };
    assert_eq!(jan("Australia/Sydney"), 3600);
    assert_eq!(jan("Australia/Lord_Howe"), 1800);
    let dublin = super::tests_tzif::negative_dst_zone();
    assert_eq!(dublin.dst_secs_at(crate::datetime::parse_ns("2024-01-15 12:00:00").unwrap()), -3600);
}

#[test]
fn period_start_in_a_gap_is_the_transition_instant() {
    let at = |s: &str| crate::datetime::parse_ns(s).unwrap();
    let wall = |s: &str| crate::datetime::Civil::from_epoch_ns(crate::datetime::parse_ns(s).unwrap());
    // Monrovia: -00:44:30 -> UTC at 1972-01-07 00:00 local (a 44:30 gap)
    let monrovia = Tz::parse("Africa/Monrovia").unwrap();
    assert_eq!(monrovia.localize_earliest(wall("1972-01-07 00:00:00")).unwrap(), at("1972-01-07 00:44:30"));
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(ny.localize_earliest(wall("2024-03-10 02:30:00")).unwrap(), at("2024-03-10 07:00:00"));
    // a fold takes the first occurrence; a fixed zone is plain localization
    assert_eq!(ny.localize_earliest(wall("2024-11-03 01:30:00")).unwrap(), at("2024-11-03 05:30:00"));
    assert_eq!(Tz::Utc.localize_earliest(wall("2024-11-03 01:30:00")).unwrap(), at("2024-11-03 01:30:00"));
    // a start before the instant range is an Overflow error, not a panic
    assert!(matches!(Tz::Utc.day_start(1677, 9, 21), Err(VolasError::Overflow(_))));
    // the day start of a gap midnight (São Paulo 2018-11-04 sprang 00:00 -> 01:00)
    let sp = Tz::parse("America/Sao_Paulo").unwrap();
    assert_eq!(sp.day_start(2018, 11, 4).unwrap(), at("2018-11-04 03:00:00"));
    for (y, m, d) in [(2024, 2, 30), (2024, 13, 1), (2024, -1, 1), (1 << 40, 1, 1)] {
        assert!(matches!(Tz::Utc.day_start(y, m, d), Err(VolasError::Value(_))), "{y}-{m}-{d}");
    }
}

/// Sub-day periods follow real instants on the local grid: a fall-back
/// repeated hour starts two periods, a skipped grid point is replaced by the
/// gap's end, and a fold that is not on the grid stays inside one period.
#[test]
fn floor_local_grid_follows_real_instants() {
    let at = |s: &str| crate::datetime::parse_ns(s).unwrap();
    const MIN: i64 = 60_000_000_000;
    let ny = Tz::parse("America/New_York").unwrap();
    // 2026-11-01: 01:00-02:00 repeats (EDT 05:00Z-06:00Z, then EST 06:00Z-07:00Z)
    let h = |s: &str| ny.floor_local_grid(at(s), 60 * MIN).unwrap();
    assert_eq!(h("2026-11-01 04:59:59"), at("2026-11-01 04:00:00")); // 00:00 EDT
    assert_eq!(h("2026-11-01 05:30:00"), at("2026-11-01 05:00:00")); // first 01:00
    assert_eq!(h("2026-11-01 06:30:00"), at("2026-11-01 06:00:00")); // second 01:00
    assert_eq!(h("2026-11-01 07:00:00"), at("2026-11-01 07:00:00")); // 02:00 EST
    // a 2h grid has no point inside the fold: 00:00-02:00 is one 3-hour period
    let h2 = |s: &str| ny.floor_local_grid(at(s), 120 * MIN).unwrap();
    assert_eq!(h2("2026-11-01 06:30:00"), at("2026-11-01 04:00:00"));
    // 2026-03-08: 02:00 EST springs to 03:00 EDT at 07:00Z
    assert_eq!(h("2026-03-08 06:59:59"), at("2026-03-08 06:00:00")); // 01:00 EST
    assert_eq!(h("2026-03-08 07:30:00"), at("2026-03-08 07:00:00")); // 03:00 EDT
    // the 2h grid point 02:00 is skipped: its period starts where the gap ends
    assert_eq!(h2("2026-03-08 07:30:00"), at("2026-03-08 07:00:00"));
    // Lord Howe (30-minute DST). 2025-10-05: 02:00 +10:30 springs to 02:30 +11
    // (15:30Z): the 1h period of 02:00 starts at the gap's end
    let lh = Tz::parse("Australia/Lord_Howe").unwrap();
    let lh1 = |s: &str| lh.floor_local_grid(at(s), 60 * MIN).unwrap();
    assert_eq!(lh1("2025-10-04 15:40:00"), at("2025-10-04 15:30:00"));
    // 2025-04-06: 02:00 +11 falls back to 01:30 +10:30 (15:00Z): 01:00 is not
    // repeated, so the repeated half hour stays in the period of 01:00 +11
    assert_eq!(lh1("2025-04-05 15:10:00"), at("2025-04-05 14:00:00"));
    assert_eq!(lh1("2025-04-05 15:30:00"), at("2025-04-05 15:30:00")); // 02:00 +10:30
    // fixed zones are arithmetic; a start before the range is an Overflow error
    let sh = Tz::parse("+08:00").unwrap();
    assert_eq!(sh.floor_local_grid(at("2024-05-22 03:47:00"), 15 * MIN).unwrap(), at("2024-05-22 03:45:00"));
    assert!(matches!(
        Tz::Utc.floor_local_grid(i64::MIN + 1, 1_000_000_000),
        Err(VolasError::Overflow(_))
    ));
    assert!(matches!(
        ny.floor_local_grid(i64::MIN + 1, 60 * MIN),
        Err(VolasError::Overflow(_))
    ));
}

#[test]
fn localize_keeps_the_fraction_and_names_dst_anomalies() {
    let ny = Tz::parse("America/New_York").unwrap();
    let wall = |s: &str| crate::datetime::parse_ns(s).unwrap();
    // a naive axis value re-read as a NY wall clock keeps its nanoseconds
    let ns = ny.localize_from(wall("2021-01-04 09:30:00.123456789"), &Tz::Naive).unwrap();
    assert_eq!(ns, wall("2021-01-04 14:30:00.123456789"));
    let gap = ny.localize_from(wall("2020-03-08 02:30:00"), &Tz::Naive).unwrap_err();
    assert!(gap.to_string().contains("DST gap"));
    let fold = ny.localize_from(wall("2020-11-01 01:30:00"), &Tz::Naive).unwrap_err();
    assert!(fold.to_string().contains("DST fold"));
}

#[test]
fn replace_civil_checks_every_field() {
    let utc = Tz::Utc;
    let t = crate::datetime::parse_ns("2024-01-31 10:00:00.5").unwrap();
    let none = [None; 6];
    let mut f = none;
    f[3] = Some(11);
    assert_eq!(
        utc.replace_civil(t, f).unwrap(),
        crate::datetime::parse_ns("2024-01-31 11:00:00.5").unwrap()
    );
    let mut f = none;
    f[0] = Some(2024 + (1 << 32));
    assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Value(_))));
    let mut f = none;
    f[1] = Some(-1);
    assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Value(_))));
    let mut f = none;
    f[1] = Some(2); // Feb 31
    assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Value(_))));
    let mut f = none;
    f[0] = Some(2262);
    f[1] = Some(4);
    f[2] = Some(12);
    assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Overflow(_))));
}

#[test]
fn tz_edge_branches() {
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(ny.fixed_offset_secs(), None); // a named zone has no fixed offset
    assert_eq!(Tz::Naive.fixed_offset_secs(), Some(0));
    assert_eq!(Tz::parse("-05:00").unwrap().fixed_offset_secs(), Some(-18_000));
    assert_eq!(Tz::Utc.name(), "UTC");
    // DST fall-back (01:30 on 2020-11-01) is ambiguous -> rejected (F33,
    // symmetric with the gap); the spring-forward gap (02:30 on 2020-03-08)
    // does not exist -> None.
    assert!(ny.wall_to_utc_ns(2020, 11, 1, 1, 30, 0).is_none());
    assert!(ny.wall_to_utc_ns(2020, 3, 8, 2, 30, 0).is_none());
    // the two states: naive is unanchored, UTC is anchored.
    assert!(!Tz::Naive.is_aware());
    assert!(Tz::Utc.is_aware());
    assert_eq!(Tz::Naive.name(), "naive");
    // parse rejects 3-char rests and out-of-range hours.
    assert!(Tz::parse("+123").is_err());
    assert!(Tz::parse("+25:00").is_err());
}

/// A partial date text names its period; anything else names no period.
#[test]
fn a_partial_date_text_names_its_period() {
    let ns = |s: &str| crate::datetime::parse_ns(s).unwrap();
    let period = |t: &str| Tz::Utc.period_of_text(t);
    assert_eq!(period("2024"), Some((ns("2024-01-01 00:00:00"), ns("2025-01-01 00:00:00"), 1)));
    assert_eq!(period("2024-12"), Some((ns("2024-12-01 00:00:00"), ns("2025-01-01 00:00:00"), 2)));
    assert_eq!(period("2024-02-29"), Some((ns("2024-02-29 00:00:00"), ns("2024-03-01 00:00:00"), 3)));
    assert_eq!(period("2024-02-29T23"), Some((ns("2024-02-29 23:00:00"), ns("2024-03-01 00:00:00"), 4)));
    assert_eq!(period("2024-02-29 23:59"), Some((ns("2024-02-29 23:59:00"), ns("2024-03-01 00:00:00"), 5)));
    assert_eq!(period("2024-02-29 23:59:59"), Some((ns("2024-02-29 23:59:59"), ns("2024-03-01 00:00:00"), 6)));
    for t in ["2024-02 10", "2024-01-02 10:00:00:00", "2024-13", "24-01", "2024-01-02 10:00:00.5", "x"] {
        assert_eq!(period(t), None, "{t}");
    }
    // a wall clock in a DST gap or fold names no instant, so no period; a day does
    let ny = Tz::parse("America/New_York").unwrap();
    assert_eq!(ny.period_of_text("2024-03-10 02:30"), None);
    assert_eq!(ny.period_of_text("2024-11-03 01:30"), None);
    assert!(ny.period_of_text("2024-03-10").is_some());
}
