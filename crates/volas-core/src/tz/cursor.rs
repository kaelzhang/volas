//! Borrowed cursors over the compiled rules for array and period queries.

use super::Tz;
use crate::datetime::{self, Civil};
use crate::error::{Result, VolasError};

pub struct TzCursor<'a> {
    zone: &'a Tz,
    utc: usize,
    local: usize,
}

impl<'a> TzCursor<'a> {
    pub(super) fn new(zone: &'a Tz) -> Self { Self { zone, utc: usize::MAX, local: usize::MAX } }

    /// Ordered instants reuse the current UTC interval; arbitrary order is
    /// also supported. A cursor borrows its rules without cloning their Arc.
    #[inline]
    pub fn offset_secs_at(&mut self, ns: i64) -> i32 {
        match self.zone {
            Tz::Naive | Tz::Utc => 0,
            Tz::Offset(off) => *off,
            Tz::Named(rules) => rules.offset_cursor(ns.div_euclid(1_000_000_000), &mut self.utc),
        }
    }

    pub fn civil(&mut self, ns: i64) -> Civil {
        Civil::from_epoch_seconds(ns.div_euclid(1_000_000_000) + self.offset_secs_at(ns) as i64,
            ns.rem_euclid(1_000_000_000) as u32)
    }

    pub(super) fn localize_wall_ns(&mut self, wall: i128) -> Result<i64> {
        match self.zone {
            Tz::Named(rules) => {
                let seconds = wall.div_euclid(1_000_000_000) as i64;
                self.zone.resolve_local(wall, rules.local_cursor(seconds, &mut self.local), false)
            }
            _ => datetime::instant_wide(wall - self.zone.fixed_offset_secs().unwrap() as i128 * 1_000_000_000),
        }
    }

    pub fn floor_local_grid(&mut self, ns: i64, step_ns: i64) -> Result<i64> {
        if step_ns <= 0 { return Err(VolasError::Value("a local grid step must be positive".into())); }
        let step = step_ns as i128;
        let mut at = ns;
        loop {
            let off = self.offset_secs_at(at);
            let delta = off as i128 * 1_000_000_000;
            let start = (at as i128 + delta).div_euclid(step) * step - delta;
            let Tz::Named(rules) = self.zone else { return datetime::instant_wide(start) };
            let transition = rules.interval_start(self.utc) as i128 * 1_000_000_000;
            if start >= transition { return datetime::instant_wide(start); }
            let Some(prev) = rules.previous_offset(self.utc) else { return datetime::instant_wide(start) };
            let before = prev as i128 * 1_000_000_000;
            if delta > before && (transition + delta).div_euclid(step) * step >= transition + before {
                return datetime::instant_wide(transition);
            }
            at = datetime::instant_wide(transition - 1)?;
        }
    }
}
