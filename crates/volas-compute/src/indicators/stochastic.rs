//! Stochastic oscillator batch fast paths.

use super::oscillators::rsi;
use crate::kernels;

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
    // One fused pass for max(high) and min(low) instead of two separate van Herk sweeps.
    let (hh, ll) = kernels::rolling_max_min(high, low, period);
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if !hh[i].is_nan() {
            let diff = hh[i] - ll[i];
            out[i] = if diff != 0.0 {
                100.0 * (close[i] - ll[i]) / diff
            } else {
                0.0
            };
        }
    }
    out
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
    let mut out = Vec::with_capacity(n);
    unsafe {
        out.set_len(n);
    }
    out[..lookback].fill(f64::NAN);

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
        if low_today.is_nan() {
            return None;
        }
        if lowest_idx == usize::MAX || lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = unsafe { *low_ptr.add(trailing) };
            if lowest.is_nan() {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *low_ptr.add(idx) };
                if value.is_nan() {
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
        if high_today.is_nan() {
            return None;
        }
        if highest_idx == usize::MAX || highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *high_ptr.add(trailing) };
            if highest.is_nan() {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *high_ptr.add(idx) };
                if value.is_nan() {
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
        if c.is_nan() {
            return None;
        }
        let diff = highest - lowest;
        unsafe {
            *out_ptr.add(today) = if diff != 0.0 {
                100.0 * (c - lowest) / diff
            } else {
                0.0
            };
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
pub fn stochf_d_default_sma(high: &[f64], low: &[f64], close: &[f64]) -> Option<Vec<f64>> {
    stoch_d_sma_fused::<1>(high, low, close)
}

/// Default STOCH `.d` (`fastk=5, slowk=3 SMA, slowd=3 SMA`) in one pass. The raw
/// `%K`, slowK sum, and slowD sum advance together; only the requested slowD line is
/// materialized. Falls back through `None` for NaN-bearing inputs.
pub fn stoch_d_default_sma(high: &[f64], low: &[f64], close: &[f64]) -> Option<Vec<f64>> {
    stoch_d_sma_fused::<2>(high, low, close)
}

fn stoch_d_sma_fused<const STAGES: usize>(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Option<Vec<f64>> {
    let n = close.len();
    if high.len() != n || low.len() != n || n < 5 {
        return None;
    }
    let lookback = 4 + STAGES * 2;
    let mut out = Vec::with_capacity(n);
    unsafe {
        out.set_len(n);
    }
    out[..lookback.min(n)].fill(f64::NAN);

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
    let mut k_window = [0.0f64; 3];
    let mut d_window = [0.0f64; 3];
    let mut k_sum = 0.0;
    let mut d_sum = 0.0;
    let mut k_count = 0usize;
    let mut d_count = 0usize;

    while today < n {
        let low_today = unsafe { *low_ptr.add(today) };
        if low_today.is_nan() {
            return None;
        }
        if lowest_idx == usize::MAX || lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = unsafe { *low_ptr.add(trailing) };
            if lowest.is_nan() {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *low_ptr.add(idx) };
                if value.is_nan() {
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
        if high_today.is_nan() {
            return None;
        }
        if highest_idx == usize::MAX || highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *high_ptr.add(trailing) };
            if highest.is_nan() {
                return None;
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *high_ptr.add(idx) };
                if value.is_nan() {
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
        if c.is_nan() {
            return None;
        }
        let diff = highest - lowest;
        let fastk = if diff != 0.0 {
            100.0 * (c - lowest) / diff
        } else {
            0.0
        };

        let k_slot = k_count % 3;
        k_sum += fastk;
        if k_count >= 3 {
            k_sum -= k_window[k_slot];
        }
        k_window[k_slot] = fastk;
        k_count += 1;

        if k_count >= 3 {
            let slowk = k_sum / 3.0;
            if STAGES == 1 {
                unsafe {
                    *out_ptr.add(today) = slowk;
                }
            } else {
                let d_slot = d_count % 3;
                d_sum += slowk;
                if d_count >= 3 {
                    d_sum -= d_window[d_slot];
                }
                d_window[d_slot] = slowk;
                d_count += 1;
                if d_count >= 3 {
                    unsafe {
                        *out_ptr.add(today) = d_sum / 3.0;
                    }
                }
            }
        }

        trailing += 1;
        today += 1;
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
        if x.is_nan() {
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
            if lowest.is_nan() {
                return None;
            }
            for (off, &value) in data[(trailing + 1)..=today].iter().enumerate() {
                if value.is_nan() {
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

        let diff = highest - lowest;
        out[today] = if diff != 0.0 {
            100.0 * (x - lowest) / diff
        } else {
            0.0
        };
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
    let n = rsi.len();
    let mut fk = vec![f64::NAN; n];
    // RSI warms up with a leading-NaN prefix, so a rolling min/max over the whole line
    // fails the NaN-free check and drops to the slow monotonic-deque path — StochRSI's
    // hot spot. Run the stochastic %K on RSI's *finite tail* instead (NaN-free → the
    // van Herk fast path) and place it back at `start`. The first valid value then
    // lands at the first fully-finite RSI window — exactly the rows the old explicit
    // mask kept — and the result is bit-identical (min/max over a finite window is
    // order-independent and exact), so the separate masking pass is no longer needed.
    let start = rsi.iter().position(|x| !x.is_nan()).unwrap_or(n);
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
pub fn stochrsi_d_default_sma(close: &[f64]) -> Option<Vec<f64>> {
    let rsi = rsi(close, 14);
    let n = rsi.len();
    let start = rsi.iter().position(|x| !x.is_nan()).unwrap_or(n);
    let tail = &rsi[start..];
    if tail.len() < 7 {
        return Some(vec![f64::NAN; n]);
    }

    let mut out = Vec::with_capacity(n);
    unsafe {
        out.set_len(n);
    }
    out[..(start + 6).min(n)].fill(f64::NAN);
    let out_ptr = out.as_mut_ptr();
    let src = tail.as_ptr();
    let mut today = 4usize;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut lowest_idx = usize::MAX;
    let mut highest = 0.0;
    let mut lowest = 0.0;
    let mut k_window = [0.0f64; 3];
    let mut k_sum = 0.0;
    let mut k_count = 0usize;

    while today < tail.len() {
        let x = unsafe { *src.add(today) };
        if x.is_nan() {
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
            if lowest.is_nan() {
                return None; // LCOV_EXCL_LINE
            }
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *src.add(idx) };
                if value.is_nan() {
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

        let diff = highest - lowest;
        let fastk = if diff != 0.0 {
            100.0 * (x - lowest) / diff
        } else {
            0.0
        };
        let slot = k_count % 3;
        k_sum += fastk;
        if k_count >= 3 {
            k_sum -= k_window[slot];
        }
        k_window[slot] = fastk;
        k_count += 1;
        if k_count >= 3 {
            unsafe {
                *out_ptr.add(start + today) = k_sum / 3.0;
            }
        }

        trailing += 1;
        today += 1;
    }
    Some(out)
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

        let out = stochf_d_default_sma(&high, &low, &close).unwrap();
        assert!(out[5].is_nan());
        assert!(out[6].is_finite());
        let out = stoch_d_default_sma(&high, &low, &close).unwrap();
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
        let out = stochrsi_d_default_sma(&short).unwrap();
        assert!(out.iter().all(|v| v.is_nan()));

        let long = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.17).sin() * 4.0 + i as f64 * 0.1)
            .collect::<Vec<_>>();
        let out = stochrsi_d_default_sma(&long).unwrap();
        assert!(out.iter().any(|v| v.is_finite()));

        let mut with_nan = long;
        with_nan[20] = f64::NAN;
        assert!(stochrsi_d_default_sma(&with_nan).is_none());
    }
}
