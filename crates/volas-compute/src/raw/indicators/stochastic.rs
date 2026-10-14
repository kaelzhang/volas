//! Stochastic oscillator batch fast paths.

use super::oscillators::rsi;
use crate::accum::StreamMean;
use crate::missing;
use crate::raw::indicators::MaStage;
use crate::raw::kernels::{self, canonical_zero};

/// TA-Lib stochastic raw %K: `100·(close − LL) / (HH − LL)` over `period` (HH/LL the
/// highest high / lowest low). A flat range yields 0, but — unlike [`rsv`] — the
/// warm-up is NaN, so the smoothing MAs in `stoch`/`stochf` begin at the right row.
/// Lookback `period-1`.
pub fn stoch_fastk(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    // The TA-Lib-style tracker wins on short and medium default stochastic windows
    // by avoiding van-Herk scratch buffers. On long arrays the O(n) van-Herk path has
    // better streaming behavior, so keep this as a fixed-cost fast path only.
    if close.len() <= 4096 && period > 0 && period <= 16 && period <= close.len() {
        if let Some(out) = stoch_fastk_small_window(high, low, close, period) {
            return out;
        }
    }
    // One fused pass for max(high) and min(low) instead of two separate van Herk sweeps
    // (NaN while a window holds a missing high / low).
    let (hh, ll) = kernels::rolling_max_min(high, low, period);
    (0..close.len()).map(|i| fastk_value(close[i], hh[i], ll[i])).collect()
}

/// Raw %K from a close and its window's extremes `hh` / `ll`: [`fastk`], missing when any
/// of the three is (the per-row term rule, [`missing::term`]): a flat range would
/// otherwise read no close at all.
#[inline(always)]
fn fastk_value(close: f64, hh: f64, ll: f64) -> f64 {
    missing::term(&[close, hh, ll], fastk(close, hh, ll))
}

/// The raw %K formula `(close − ll) / ((hh − ll) / 100)` (TA-Lib's order), `0` for a flat
/// range — the one owner of it for every stochastic path. The extremes' zeros are made `+0` first
/// ([`canonical_zero`]), so a path that kept the other signed zero gives the same bits.
#[inline(always)]
fn fastk(close: f64, hh: f64, ll: f64) -> f64 {
    let (hh, ll) = (canonical_zero(hh), canonical_zero(ll));
    // `ta_STOCHF.c`'s order: the range in percent points first, then the quotient
    let diff = (hh - ll) / 100.0;
    if diff != 0.0 { (close - ll) / diff } else { 0.0 }
}

fn stoch_fastk_small_window(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    let n = close.len();
    if high.len() != n || low.len() != n || period == 0 || period > n {
        return None;
    }
    let lookback = period - 1;
    // NaN-prefilled buffer: the warm-up stays NaN and the loop overwrites the
    // valid region (D2 2026-06-12 — replaces the with_capacity + set_len pattern;
    // the prefill is a vectorized splat, measured at parity by make perf-ab).
    let mut out = vec![f64::NAN; n];

    let high_ptr = high.as_ptr();
    let low_ptr = low.as_ptr();
    let close_ptr = close.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut today = lookback;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut lowest_idx = usize::MAX;
    let mut highest = 0.0;
    let mut lowest = 0.0;

    while today < n {
        // For the NaN-free OHLC hot path this mirrors TA-Lib's small-window tracker:
        // keep current extrema and rescan only when they fall out of the window.
        // If any input contains NaN, decline to the generic rolling kernel so the
        // existing NaN semantics stay centralized there.
        let low_today = unsafe { *low_ptr.add(today) };
        if missing::is_missing(low_today) {
            return None;
        }
        if lowest_idx == usize::MAX || lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = unsafe { *low_ptr.add(trailing) };
            if missing::is_missing(lowest) {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *low_ptr.add(idx) };
                if missing::is_missing(value) {
                    return None;
                }
                if value < lowest {
                    lowest_idx = idx;
                    lowest = value;
                }
                idx += 1;
            }
        } else if low_today <= lowest {
            lowest_idx = today;
            lowest = low_today;
        }

        let high_today = unsafe { *high_ptr.add(today) };
        if missing::is_missing(high_today) {
            return None;
        }
        if highest_idx == usize::MAX || highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *high_ptr.add(trailing) };
            if missing::is_missing(highest) {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *high_ptr.add(idx) };
                if missing::is_missing(value) {
                    return None;
                }
                if value > highest {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else if high_today >= highest {
            highest_idx = today;
            highest = high_today;
        }

        let c = unsafe { *close_ptr.add(today) };
        if missing::is_missing(c) {
            return None;
        }
        unsafe {
            *out_ptr.add(today) = fastk(c, highest, lowest);
        }
        trailing += 1;
        today += 1;
    }
    Some(out)
}

/// Default STOCHF `.d` (`fastk=5, fastd=3, SMA`) in one pass. This keeps the
/// TA-Lib-style 5-bar extremum tracker and maintains the 3-bar D sum as raw `%K`
/// values are produced, avoiding the full `fastk` buffer plus a second SMA pass.
/// Returns `None` on NaN or shape mismatch so callers can preserve the generic path.
pub fn stochf_d_default_sma(high: &[f64], low: &[f64], close: &[f64], _origin: usize) -> Option<Vec<f64>> {
    stoch_d_sma_fused::<1>(high, low, close)
}

/// Default STOCH `.d` (`fastk=5, slowk=3 SMA, slowd=3 SMA`) in one pass. The raw
/// `%K`, slowK sum, and slowD sum advance together; only the requested slowD line is
/// materialized. Falls back through `None` for NaN-bearing inputs.
pub fn stoch_d_default_sma(high: &[f64], low: &[f64], close: &[f64], _origin: usize) -> Option<Vec<f64>> {
    stoch_d_sma_fused::<2>(high, low, close)
}

/// The fused default stochastic smoothing: each
/// 3-bar SMA is `kernels::sma`'s fast path over the line it smooths ([`StreamMean`]).
fn stoch_d_sma_fused<const STAGES: usize>(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Option<Vec<f64>> {
    let n = close.len();
    if high.len() != n || low.len() != n || n < 5 {
        return None;
    }
    // NaN-prefilled buffer: the warm-up stays NaN and the loop overwrites the
    // valid region (D2 2026-06-12 — replaces the with_capacity + set_len pattern;
    // the prefill is a vectorized splat, measured at parity by make perf-ab).
    let mut out = vec![f64::NAN; n];

    let high_ptr = high.as_ptr();
    let low_ptr = low.as_ptr();
    let close_ptr = close.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut today = 4usize;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut lowest_idx = usize::MAX;
    let mut highest = 0.0;
    let mut lowest = 0.0;
    // raw %K starts at row 4, slowK (STAGES 2) at row 6
    let mut k_mean = StreamMean::<3>::new();
    let mut d_mean = StreamMean::<3>::new();

    while today < n {
        let low_today = unsafe { *low_ptr.add(today) };
        if missing::is_missing(low_today) {
            return None;
        }
        if lowest_idx == usize::MAX || lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = unsafe { *low_ptr.add(trailing) };
            if missing::is_missing(lowest) {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *low_ptr.add(idx) };
                if missing::is_missing(value) {
                    return None;
                }
                if value < lowest {
                    lowest_idx = idx;
                    lowest = value;
                }
                idx += 1;
            }
        } else if low_today <= lowest {
            lowest_idx = today;
            lowest = low_today;
        }

        let high_today = unsafe { *high_ptr.add(today) };
        if missing::is_missing(high_today) {
            return None;
        }
        if highest_idx == usize::MAX || highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *high_ptr.add(trailing) };
            if missing::is_missing(highest) {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *high_ptr.add(idx) };
                if missing::is_missing(value) {
                    return None;
                }
                if value > highest {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else if high_today >= highest {
            highest_idx = today;
            highest = high_today;
        }

        let c = unsafe { *close_ptr.add(today) };
        if missing::is_missing(c) {
            return None;
        }
        let fastk = fastk(c, highest, lowest);

        if let Some(slowk) = k_mean.push(fastk) {
            if STAGES == 1 {
                unsafe {
                    *out_ptr.add(today) = slowk;
                }
            } else if let Some(slowd) = d_mean.push(slowk) {
                unsafe {
                    *out_ptr.add(today) = slowd;
                }
            }
        }

        trailing += 1;
        today += 1;
    }
    // An infinity poisons the running sums; the generic path is window-local.
    if !(k_mean.finite() && d_mean.finite()) {
        return None;
    }
    Some(out)
}

fn stoch_fastk_same_series_small_window(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let mut today = period - 1;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut lowest_idx = usize::MAX;
    let mut highest = 0.0;
    let mut lowest = 0.0;
    while today < n {
        let x = data[today];
        if missing::is_missing(x) {
            return None;
        }
        if lowest_idx == usize::MAX
            || highest_idx == usize::MAX
            || lowest_idx < trailing
            || highest_idx < trailing
        {
            lowest_idx = trailing;
            lowest = data[trailing];
            highest_idx = trailing;
            highest = lowest;
            if missing::is_missing(lowest) {
                return None;
            }
            for (off, &value) in data[(trailing + 1)..=today].iter().enumerate() {
                if missing::is_missing(value) {
                    return None;
                }
                let idx = trailing + 1 + off;
                if value <= lowest {
                    lowest_idx = idx;
                    lowest = value;
                }
                if value >= highest {
                    highest_idx = idx;
                    highest = value;
                }
            }
        } else {
            if x <= lowest {
                lowest_idx = today;
                lowest = x;
            }
            if x >= highest {
                highest_idx = today;
                highest = x;
            }
        }

        out[today] = fastk(x, highest, lowest);
        trailing += 1;
        today += 1;
    }
    Some(out)
}

/// TA-Lib StochRSI raw %K: the stochastic %K of the RSI line — `stoch_fastk` applied
/// to `rsi(close, rsi_period)` over `fastk_period`. Because RSI itself warms up, the
/// %K is masked to NaN until a full `fastk_period` window of finite RSI is available
/// (index `rsi_period + fastk_period - 1`). The `fastd` line is then `ma_typed` of this.
pub fn stochrsi_fastk(close: &[f64], rsi_period: usize, fastk_period: usize) -> Vec<f64> {
    let rsi = rsi(close, rsi_period);
    stochrsi_from_rsi(&rsi, fastk_period)
}

/// StochRSI's raw K from the already computed RSI line, shared with state capture.
pub fn stochrsi_from_rsi(rsi: &[f64], fastk_period: usize) -> Vec<f64> {
    let n = rsi.len();
    let mut fk = vec![f64::NAN; n];
    // RSI warms up with a leading-NaN prefix, so a rolling min/max over the whole line
    // fails the NaN-free check and drops to the slow monotonic-deque path — StochRSI's
    // hot spot. Run the stochastic %K on RSI's *finite tail* instead (NaN-free → the
    // van Herk fast path) and place it back at `start`. The first valid value then
    // lands at the first fully-finite RSI window — exactly the rows the old explicit
    // mask kept — and the result is bit-identical (min/max over a finite window is
    // order-independent and exact), so the separate masking pass is no longer needed.
    let start = rsi.iter().position(|&x| !missing::is_missing(x)).unwrap_or(n);
    let tail = &rsi[start..];
    if tail.len() >= fastk_period {
        // This StochRSI-specific call passes the same RSI buffer as high/low/close.
        // Track one rolling window here so ordinary STOCH/STOCHF keep the lean generic
        // `stoch_fastk` path with no extra same-buffer branch.
        let fk_tail = if fastk_period > 0 && fastk_period <= 8 {
            stoch_fastk_same_series_small_window(tail, fastk_period)
                .unwrap_or_else(|| stoch_fastk(tail, tail, tail, fastk_period))
        } else {
            stoch_fastk(tail, tail, tail, fastk_period)
        };
        fk[start..].copy_from_slice(&fk_tail);
    }
    fk
}

/// Default STOCHRSI `.d` (`rsi=14, fastk=5, fastd=3, SMA`) with the stochastic
/// and D-smoothing stages fused over the finite RSI tail. RSI itself remains the
/// single recursive source of truth; this only removes the intermediate fastK Vec.
pub fn stochrsi_d_default_sma(close: &[f64], _origin: usize) -> Option<Vec<f64>> {
    let rsi = rsi(close, 14);
    let n = rsi.len();
    let start = rsi.iter().position(|&x| !missing::is_missing(x)).unwrap_or(n);
    let tail = &rsi[start..];
    if tail.len() < 7 {
        return Some(vec![f64::NAN; n]);
    }

    // NaN-prefilled buffer: the warm-up stays NaN and the loop overwrites the
    // valid region (D2 2026-06-12 — replaces the with_capacity + set_len pattern;
    // the prefill is a vectorized splat, measured at parity by make perf-ab).
    let mut out = vec![f64::NAN; n];
    let out_ptr = out.as_mut_ptr();
    let src = tail.as_ptr();
    let mut today = 4usize;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut lowest_idx = usize::MAX;
    let mut highest = 0.0;
    let mut lowest = 0.0;
    // %K starts at (complete) row `start + 4`: `kernels::sma`'s fast path over it
    let mut k_mean = StreamMean::<3>::new();

    while today < tail.len() {
        let x = unsafe { *src.add(today) };
        if missing::is_missing(x) {
            return None;
        }
        if lowest_idx == usize::MAX
            || highest_idx == usize::MAX
            || lowest_idx < trailing
            || highest_idx < trailing
        {
            lowest_idx = trailing;
            lowest = unsafe { *src.add(trailing) };
            highest_idx = trailing;
            highest = lowest;
            if missing::is_missing(lowest) {
                return None; // LCOV_EXCL_LINE
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *src.add(idx) };
                if missing::is_missing(value) {
                    return None; // LCOV_EXCL_LINE
                }
                if value <= lowest {
                    lowest_idx = idx;
                    lowest = value;
                }
                if value >= highest {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else {
            if x <= lowest {
                lowest_idx = today;
                lowest = x;
            }
            if x >= highest {
                highest_idx = today;
                highest = x;
            }
        }

        let fastk = fastk(x, highest, lowest);
        if let Some(d) = k_mean.push(fastk) {
            unsafe {
                *out_ptr.add(start + today) = d;
            }
        }

        trailing += 1;
        today += 1;
    }
    Some(out)
}


/// Raw %K at row `i` alone (see [`stoch_fastk`]): the extremes of the window — exactly
/// what the rolling kernels reduce, NaN while it holds a missing high / low — so it
/// equals `stoch_fastk(..)[i]`.
pub fn stoch_fastk_at(high: &[f64], low: &[f64], close: &[f64], period: usize, i: usize) -> f64 {
    if period == 0 || i + 1 < period {
        return f64::NAN;
    }
    let window = i + 1 - period..=i;
    let (hw, lw) = (&high[window.clone()], &low[window]);
    if missing::holds_missing(hw) || missing::holds_missing(lw) {
        return f64::NAN;
    }
    let hh = hw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ll = lw.iter().copied().fold(f64::INFINITY, f64::min);
    fastk_value(close[i], hh, ll)
}

/// The smoothing stages of a stochastic line over raw %K: `[slowK]` / `[fastD]`, or
/// `[slowK, slowD]` — each a generic MA stage `(period, matype)`.
fn stoch_stages(stages: &[(usize, usize)]) -> Option<(MaStage, Option<MaStage>)> {
    match *stages {
        [(p, mt)] => Some((MaStage::new(p, mt)?, None)),
        [(p1, mt1), (p2, mt2)] => Some((MaStage::new(p1, mt1)?, Some(MaStage::new(p2, mt2)?))),
        _ => None,
    }
}

/// The length of a smoothed stochastic line's state: each stage's, plus for a second
/// stage the last values of the first it reads.
fn stoch_state_len((first, second): (MaStage, Option<MaStage>)) -> usize {
    first.state_len() + second.map_or(0, |s| s.chained_len())
}

/// A smoothed stochastic line's state after the rows (`stages` as in [`stoch_step`]),
/// row 0 at absolute position `origin`.
pub fn stoch_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    stages: &[(usize, usize)],
    origin: usize,
) -> Option<Vec<f64>> {
    let (first, second) = stoch_stages(stages)?;
    let mut init = first.initial(origin);
    if let Some(second) = second {
        init.extend(second.chained_initial(origin));
    }
    crate::guard::state_after(init, close.len(), |row, prev, next| {
        stoch_step(high, low, close, fastk_period, stages, row, prev, next)
    })
}

/// One row of a smoothed stochastic line at `row`: raw %K over `fastk_period`, then one
/// or two generic MA stages `stages` (`(period, matype)`: `[slowK]` for `stoch.k`,
/// `[slowK, slowD]` for `stoch.d`, `[fastD]` for `stochf.d`), from `state` into `next`.
#[allow(clippy::too_many_arguments)]
pub fn stoch_step(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    stages: &[(usize, usize)],
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let ma = stoch_stages(stages)?;
    if row >= close.len() {
        return None;
    }
    let len = stoch_state_len(ma);
    if state.len() != len || next.len() != len {
        return None;
    }
    let fastk = |j: usize| stoch_fastk_at(high, low, close, fastk_period, j);
    let l1 = ma.0.state_len();
    let (n1, n_rest) = next.split_at_mut(l1);
    let first = ma.0.step(row, fastk, &state[..l1], n1)?;
    let Some(second) = ma.1 else { return Some(first) };
    let value = second.step_chained(row, first, &state[l1..], n_rest)?;
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = close.iter().map(|v| v + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|v| v - 2.0).collect();
        (high, low, close)
    }

    #[test]
    fn stochastic_small_window_paths_cover_nan_declines() {
        let (high, low, close) = ohlc(8);
        let out = stoch_fastk(&high, &low, &close, 5);
        assert!(out[3].is_nan());
        assert!(out[4].is_finite());

        let flat = vec![3.0; 8];
        let out = stoch_fastk_small_window(&flat, &flat, &flat, 5).unwrap();
        assert_eq!(out[4], 0.0);

        assert!(stoch_fastk_small_window(&high[..7], &low, &close, 5).is_none());

        let mut x = low.clone();
        x[4] = f64::NAN;
        assert!(stoch_fastk_small_window(&high, &x, &close, 5).is_none());
        let mut x = low.clone();
        x[0] = f64::NAN;
        assert!(stoch_fastk_small_window(&high, &x, &close, 5).is_none());
        let mut x = low.clone();
        x[2] = f64::NAN;
        assert!(stoch_fastk_small_window(&high, &x, &close, 5).is_none());

        let mut x = high.clone();
        x[4] = f64::NAN;
        assert!(stoch_fastk_small_window(&x, &low, &close, 5).is_none());
        let mut x = high.clone();
        x[0] = f64::NAN;
        assert!(stoch_fastk_small_window(&x, &low, &close, 5).is_none());
        let mut x = high.clone();
        x[2] = f64::NAN;
        assert!(stoch_fastk_small_window(&x, &low, &close, 5).is_none());

        let mut x = close.clone();
        x[4] = f64::NAN;
        assert!(stoch_fastk_small_window(&high, &low, &x, 5).is_none());
    }

    #[test]
    fn stochastic_fused_default_paths_cover_nan_declines() {
        let (high, low, close) = ohlc(12);
        assert!(stoch_d_sma_fused::<1>(&high[..4], &low[..4], &close[..4]).is_none());

        let out = stochf_d_default_sma(&high, &low, &close, 0).unwrap();
        assert!(out[5].is_nan());
        assert!(out[6].is_finite());
        let out = stoch_d_default_sma(&high, &low, &close, 0).unwrap();
        assert!(out[7].is_nan());
        assert!(out[8].is_finite());

        let flat = vec![3.0; 12];
        let out = stoch_d_sma_fused::<2>(&flat, &flat, &flat).unwrap();
        assert_eq!(out[8], 0.0);

        let mut x = low.clone();
        x[4] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&high, &x, &close).is_none());
        let mut x = low.clone();
        x[0] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&high, &x, &close).is_none());
        let mut x = low.clone();
        x[2] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&high, &x, &close).is_none());

        let mut x = high.clone();
        x[4] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&x, &low, &close).is_none());
        let mut x = high.clone();
        x[0] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&x, &low, &close).is_none());
        let mut x = high.clone();
        x[2] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&x, &low, &close).is_none());

        let mut x = close.clone();
        x[4] = f64::NAN;
        assert!(stoch_d_sma_fused::<2>(&high, &low, &x).is_none());
    }

    #[test]
    fn same_series_and_stochrsi_paths_cover_guard_branches() {
        let flat = vec![7.0; 8];
        let out = stoch_fastk_same_series_small_window(&flat, 5).unwrap();
        assert_eq!(out[4], 0.0);

        let mut x = (0..8).map(|i| 10.0 + i as f64).collect::<Vec<_>>();
        x[4] = f64::NAN;
        assert!(stoch_fastk_same_series_small_window(&x, 5).is_none());
        let mut x = (0..8).map(|i| 10.0 + i as f64).collect::<Vec<_>>();
        x[0] = f64::NAN;
        assert!(stoch_fastk_same_series_small_window(&x, 5).is_none());
        let mut x = (0..8).map(|i| 10.0 + i as f64).collect::<Vec<_>>();
        x[2] = f64::NAN;
        assert!(stoch_fastk_same_series_small_window(&x, 5).is_none());

        let short = (0..20).map(|i| 100.0 + i as f64).collect::<Vec<_>>();
        let out = stochrsi_d_default_sma(&short, 0).unwrap();
        assert!(out.iter().all(|v| v.is_nan()));

        let long = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.17).sin() * 4.0 + i as f64 * 0.1)
            .collect::<Vec<_>>();
        let out = stochrsi_d_default_sma(&long, 0).unwrap();
        assert!(out.iter().any(|v| v.is_finite()));

        let mut with_nan = long;
        with_nan[20] = f64::NAN;
        assert!(stochrsi_d_default_sma(&with_nan, 0).is_none());
    }
}
