//! One time-grid admission shared by the downstream period operations.
use volas_core::{AdmittedBar, IndexKind, Result, Tz, VolasError};
use crate::TimeFrame;

/// A schema-admitted bar whose instant and period key were computed once by the
/// time owner. Neither its instant nor its grid can be supplied independently.
pub struct TimedBar<'b, 'a, 'p, 'z> {
    bar: &'b AdmittedBar<'a, 'p>,
    tf: TimeFrame,
    tz: &'z Tz,
    timestamp: i64,
    key: i64,
}
impl<'b, 'a, 'p, 'z> TimedBar<'b, 'a, 'p, 'z> {
    /// Admit the bar's immutable timestamp on this grid.
    pub fn new(bar: &'b AdmittedBar<'a, 'p>, tf: TimeFrame, tz: &'z Tz) -> Result<Self> {
        let IndexKind::Datetime(ts, zone) = bar.rows().index().kind() else {
            return Err(VolasError::Value("a timed bar requires a DatetimeIndex".into()));
        };
        if zone.is_aware() != tz.is_aware() { return Err(VolasError::Value("timed bar timezone awareness differs from its grid".into())); }
        let timestamp = ts[0];
        let key = tf.period_start(timestamp, tz)?;
        Ok(Self { bar, tf, tz, timestamp, key })
    }
    /// Reduce and append the first bar using its existing schema and time proof.
    pub fn start(&self, target: &mut volas_core::DataFrame) -> Result<volas_core::DataFrame> {
        let row = self.reduced_named(self.bar.rows().index().name().map(str::to_owned))?;
        row.append_new(target)?;
        Ok(row.into_frame())
    }

    pub(crate) fn reduced_named(&self, name: Option<String>) -> Result<AdmittedBar<'static, '_>> {
        let index = volas_core::Index::new(name, IndexKind::Datetime(vec![self.key], self.tz.clone()))?;
        self.bar.reduced(index)
    }

    /// The one computed period key.
    pub fn key(&self) -> i64 { self.key }
    pub(crate) fn bar(&self) -> &'b AdmittedBar<'a, 'p> { self.bar }
    pub(crate) fn timestamp(&self) -> i64 { self.timestamp }
    pub(crate) fn time_frame(&self) -> TimeFrame { self.tf }
    pub(crate) fn zone(&self) -> &'z Tz { self.tz }
}
