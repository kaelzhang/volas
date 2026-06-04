// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------

/// On Balance Volume (TA-Lib OBV): a running total that adds the bar's volume when
/// `real` rises vs the prior bar, subtracts it when `real` falls, and is unchanged
/// when flat. Seeded with `volume[0]`; lookback 0 (no warm-up).
pub fn obv(real: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = real.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let mut obv = volume[0];
    let mut prev = real[0];
    out[0] = obv;
    for i in 1..n {
        if real[i] > prev {
            obv += volume[i];
        } else if real[i] < prev {
            obv -= volume[i];
        }
        out[i] = obv;
        prev = real[i];
    }
    out
}

/// Money flow volume for one bar: `((close-low) - (high-close)) / (high-low) ·
/// volume`, with a zero high-low range contributing nothing (TA-Lib's guard).
#[inline]
fn money_flow_volume(high: f64, low: f64, close: f64, volume: f64) -> f64 {
    let range = high - low;
    if range > 0.0 {
        ((close - low) - (high - close)) / range * volume
    } else {
        0.0
    }
}

/// Chaikin Accumulation/Distribution line (TA-Lib AD): the running total of each
/// bar's money flow volume. Lookback 0 (no warm-up).
pub fn ad(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let mut ad = 0.0;
    for i in 0..n {
        ad += money_flow_volume(high[i], low[i], close[i], volume[i]);
        out[i] = ad;
    }
    out
}

/// Chaikin A/D Oscillator (TA-Lib ADOSC): `fastEMA - slowEMA` of the A/D line.
/// Unlike a standalone EMA, both EMAs are seeded with the first A/D value and run
/// from index 0 (TA-Lib's exact scheme); the output is masked until index
/// `max(fast, slow) - 1`.
pub fn adosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
) -> Vec<f64> {
    let line = ad(high, low, close, volume);
    let n = line.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let lookback = fast.max(slow).saturating_sub(1);
    let fastk = 2.0 / (fast as f64 + 1.0);
    let slowk = 2.0 / (slow as f64 + 1.0);
    let mut fast_ema = line[0];
    let mut slow_ema = line[0];
    if lookback == 0 {
        out[0] = fast_ema - slow_ema;
    }
    for i in 1..n {
        fast_ema = fastk * line[i] + (1.0 - fastk) * fast_ema;
        slow_ema = slowk * line[i] + (1.0 - slowk) * slow_ema;
        if i >= lookback {
            out[i] = fast_ema - slow_ema;
        }
    }
    out
}
