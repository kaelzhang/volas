//! Single-bar candlestick patterns.

use super::{candle_average, color, lowershadow, realbody, uppershadow, BODY_DOJI, BODY_LONG, SHADOW_VERY_SHORT};

/// Doji (TA-Lib CDLDOJI): a real body no larger than the recent doji-body threshold —
/// open ≈ close. Always non-negative (uncertainty, not direction): `100` or `0`.
/// Lookback `avg_period(BodyDoji)` = 10.
pub fn cdl_doji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let lb = BODY_DOJI.avg_period;
    for i in lb..n {
        out[i] = if realbody(open, close, i) <= candle_average(BODY_DOJI, open, high, low, close, i)
        {
            100.0
        } else {
            0.0
        };
    }
    out
}

/// Marubozu (TA-Lib CDLMARUBOZU): a long real body with negligible shadows on both
/// ends. `color·100` (bullish white / bearish black) or `0`. Lookback
/// `max(avg_period(BodyLong), avg_period(ShadowVeryShort))` = 10.
pub fn cdl_marubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let lb = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    for i in lb..n {
        let very_short = candle_average(SHADOW_VERY_SHORT, open, high, low, close, i);
        out[i] = if realbody(open, close, i) > candle_average(BODY_LONG, open, high, low, close, i)
            && uppershadow(open, high, close, i) < very_short
            && lowershadow(open, low, close, i) < very_short
        {
            color(open, close, i) * 100.0
        } else {
            0.0
        };
    }
    out
}
