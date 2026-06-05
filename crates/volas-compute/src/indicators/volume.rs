// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------

/// On Balance Volume (TA-Lib OBV): a running total that adds the bar's volume when
/// `real` rises vs the prior bar, subtracts it when `real` falls, and is unchanged
/// when flat. Seeded with `volume[0]`; lookback 0 (no warm-up).
pub fn obv(real: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = real.len();
    if n == 0 {
        return Vec::new();
    }
    // Every bar is written, so `vec![NaN; n]` would be fully overwritten — build into a
    // reserved buffer with `push` instead (each slot written once).
    let mut out = Vec::with_capacity(n);
    let mut obv = volume[0];
    let mut prev = real[0];
    out.push(obv);
    for i in 1..n {
        // Branchless direction sign (+1 up / -1 down / 0 flat): the up/down branch is
        // unpredictable on real price data, so a misprediction-free form is faster.
        // Bit-identical: `+1.0·v == +v`, `-1.0·v == -v`, `0·v == 0`.
        let dir = ((real[i] > prev) as i8 - (real[i] < prev) as i8) as f64;
        obv += dir * volume[i];
        out.push(obv);
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
    // Every bar is written — push into a reserved buffer (one write per slot) rather
    // than allocate-and-fill `vec![NaN; n]` only to overwrite all of it.
    let mut out = Vec::with_capacity(n);
    let mut ad = 0.0;
    for i in 0..n {
        ad += money_flow_volume(high[i], low[i], close[i], volume[i]);
        out.push(ad);
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
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let lookback = fast.max(slow).saturating_sub(1);
    let fastk = 2.0 / (fast as f64 + 1.0);
    let slowk = 2.0 / (slow as f64 + 1.0);
    // Fuse the Chaikin A/D line into the oscillator's single pass: accumulate the A/D
    // value on the fly and feed both EMAs, so the intermediate A/D array is never
    // materialised (TA-Lib computes ADOSC in one pass). The EMAs use TA-Lib's
    // `(price − prev)·k + prev` form, fused to a single rounding off the recurrence's
    // critical path; the two independent chains interleave (ILP). EMA is contractive
    // (k < 1), so the ~1e-16 divergence decays — within the parity tolerance.
    let mut ad_line = money_flow_volume(high[0], low[0], close[0], volume[0]);
    let mut fast_ema = ad_line;
    let mut slow_ema = ad_line;
    if lookback == 0 {
        out[0] = fast_ema - slow_ema;
    }
    for i in 1..n {
        ad_line += money_flow_volume(high[i], low[i], close[i], volume[i]);
        fast_ema = (ad_line - fast_ema).mul_add(fastk, fast_ema);
        slow_ema = (ad_line - slow_ema).mul_add(slowk, slow_ema);
        if i >= lookback {
            out[i] = fast_ema - slow_ema;
        }
    }
    out
}
