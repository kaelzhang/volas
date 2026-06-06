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

/// Final OBV state `[running_obv, prev_real]` after a full [`obv`] compute over
/// `real`/`volume` — the seed an [`obv_resume`] needs to continue at row `n`.
/// Mirrors [`obv`]'s recurrence exactly (seed `obv = volume[0]`, `prev = real[0]`).
/// `None` for an empty series (no state to carry).
pub fn obv_final_state(real: &[f64], volume: &[f64]) -> Option<Vec<f64>> {
    let n = real.len();
    if n == 0 {
        return None;
    }
    let mut obv = volume[0];
    let mut prev = real[0];
    for i in 1..n {
        let dir = ((real[i] > prev) as i8 - (real[i] < prev) as i8) as f64;
        obv += dir * volume[i];
        prev = real[i];
    }
    Some(vec![obv, prev])
}

/// Final AD state `[running_ad]` after a full [`ad`] compute. `None` for an empty
/// series.
pub fn ad_final_state(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let mut ad = 0.0;
    for i in 0..n {
        ad += money_flow_volume(high[i], low[i], close[i], volume[i]);
    }
    Some(vec![ad])
}

/// Final ADOSC state `[ad_line, fast_ema, slow_ema]` after a full [`adosc`]
/// compute. Mirrors [`adosc`]'s exact fused EMA recurrence so an [`adosc_resume`]
/// continues on identical bits. `None` for an empty series.
pub fn adosc_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let fastk = 2.0 / (fast as f64 + 1.0);
    let slowk = 2.0 / (slow as f64 + 1.0);
    let mut ad_line = money_flow_volume(high[0], low[0], close[0], volume[0]);
    let mut fast_ema = ad_line;
    let mut slow_ema = ad_line;
    for i in 1..n {
        ad_line += money_flow_volume(high[i], low[i], close[i], volume[i]);
        fast_ema = (ad_line - fast_ema).mul_add(fastk, fast_ema);
        slow_ema = (ad_line - slow_ema).mul_add(slowk, slow_ema);
    }
    Some(vec![ad_line, fast_ema, slow_ema])
}

/// Resume OBV from a carried state over the new rows `[from, n)`, bit-identical to
/// a full [`obv`] recompute. `state = [running_obv, prev_real]` is the internal
/// state as of row `from - 1` (the last valid row). Returns the values for rows
/// `[from, n)` and the new `[running_obv, prev_real]` state as of row `n - 1`.
///
/// `from >= 1` always (a fresh compute, `from == 0`, uses [`obv`] directly), and
/// the recurrence reads only `real[from..]`/`volume[from..]` plus the carried
/// `prev_real`, so it never reaches before `from` — sound after a head-dropping
/// slice that retained none of the pre-`from` rows.
pub fn obv_resume(real: &[f64], volume: &[f64], from: usize, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = real.len();
    let mut obv = state[0];
    let mut prev = state[1];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        let dir = ((real[i] > prev) as i8 - (real[i] < prev) as i8) as f64;
        obv += dir * volume[i];
        out.push(obv);
        prev = real[i];
    }
    (out, vec![obv, prev])
}

/// Resume AD (Chaikin A/D line) from a carried state over the new rows `[from, n)`,
/// bit-identical to a full [`ad`] recompute. `state = [running_ad]` as of row
/// `from - 1`. AD's per-bar money-flow term is independent of prior bars, so only
/// the running total is carried. Returns the new-row values and the updated
/// `[running_ad]`.
pub fn ad_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut ad = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        ad += money_flow_volume(high[i], low[i], close[i], volume[i]);
        out.push(ad);
    }
    (out, vec![ad])
}

/// Resume ADOSC (Chaikin A/D Oscillator) from a carried state over the new rows
/// `[from, n)`, bit-identical to a full [`adosc`] recompute.
/// `state = [ad_line, fast_ema, slow_ema]` as of row `from - 1`. The two EMAs are
/// advanced with TA-Lib's exact `(price − prev)·k + prev` (fused) recurrence — the
/// same instruction sequence as [`adosc`], so the carried-state continuation lands
/// on identical bits. `from > lookback` on every resume (the cached tail starts at
/// `valid_rows == lookback + 1` at the earliest), so every emitted row is past the
/// warm-up mask and finite. Returns the new-row values and the updated state.
pub fn adosc_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let fastk = 2.0 / (fast as f64 + 1.0);
    let slowk = 2.0 / (slow as f64 + 1.0);
    let mut ad_line = state[0];
    let mut fast_ema = state[1];
    let mut slow_ema = state[2];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        ad_line += money_flow_volume(high[i], low[i], close[i], volume[i]);
        fast_ema = (ad_line - fast_ema).mul_add(fastk, fast_ema);
        slow_ema = (ad_line - slow_ema).mul_add(slowk, slow_ema);
        out.push(fast_ema - slow_ema);
    }
    (out, vec![ad_line, fast_ema, slow_ema])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::test_support::*;

    /// Empty-series guards: every `*_final_state` declines to carry state for `n == 0`
    /// (the `return None` arms — no first bar to seed from).
    #[test]
    fn final_state_declines_on_empty() {
        assert!(obv_final_state(&[], &[]).is_none()); // volume.rs:38
        assert!(ad_final_state(&[], &[], &[], &[]).is_none()); // volume.rs:55
        assert!(adosc_final_state(&[], &[], &[], &[], 3, 10).is_none()); // volume.rs:77
    }

    /// Each resume, fed the carried state of a full compute over the head, reproduces
    /// the tail of a full compute over the whole input — bit-for-bit.
    #[test]
    fn resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(120);
        // A volume track that rises and falls (so OBV's up/down/flat sign all fire).
        let vol: Vec<f64> = (0..120)
            .map(|i| 1000.0 + 300.0 * ((i as f64) * 0.27).sin())
            .collect();
        let (fast, slow) = (3usize, 10usize);

        let obv_full = obv(&close, &vol);
        let ad_full = ad(&high, &low, &close, &vol);
        let adosc_full = adosc(&high, &low, &close, &vol, fast, slow);

        // OBV / AD carry no warm-up (lookback 0), so any `from >= 1` matches the full.
        for &from in &[1usize, 2, 30, 60, 119] {
            let head = &close[..from];

            let st = obv_final_state(head, &vol[..from]).unwrap();
            let (tail, _) = obv_resume(&close, &vol, from, &st);
            assert_bits(&tail, &obv_full[from..], "obv");

            let st = ad_final_state(&high[..from], &low[..from], head, &vol[..from]).unwrap();
            let (tail, _) = ad_resume(&high, &low, &close, &vol, from, &st);
            assert_bits(&tail, &ad_full[from..], "ad");
        }

        // ADOSC resumes only past its `max(fast,slow)-1` warm-up (the cached tail starts
        // at `valid_rows == lookback + 1`), where the full output is already finite.
        let lookback = fast.max(slow) - 1;
        for &from in &[lookback + 1, lookback + 5, 60, 119] {
            let st =
                adosc_final_state(&high[..from], &low[..from], &close[..from], &vol[..from], fast, slow)
                    .unwrap();
            let (tail, _) = adosc_resume(&high, &low, &close, &vol, fast, slow, from, &st);
            assert_bits(&tail, &adosc_full[from..], "adosc");
        }
    }

    /// A zero high-low range bar contributes nothing to the A/D line (the `range > 0`
    /// guard's `else` arm), so a flat bar leaves the running total unchanged.
    #[test]
    fn flat_bar_adds_zero_money_flow() {
        let high = [10.0, 10.0];
        let low = [10.0, 10.0];
        let close = [10.0, 10.0];
        let vol = [5.0, 5.0];
        assert_eq!(ad(&high, &low, &close, &vol), vec![0.0, 0.0]);
    }
}
