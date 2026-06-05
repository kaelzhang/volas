use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Trend-following
// ---------------------------------------------------------------------------

/// Simple moving average.
pub fn ma(close: &[f64], period: usize) -> Vec<f64> {
    // Move the kernel's owned buffer out (no copy) rather than `to_vec`.
    kernels::sma(av(close), period).into_raw_vec_and_offset().0
}

/// Exponential moving average (TA-Lib: SMA-seeded, `k = 2/(period+1)`).
pub fn ema(close: &[f64], period: usize) -> Vec<f64> {
    kernels::ema_seeded(av(close), period)
        .into_raw_vec_and_offset()
        .0
}

/// Smoothed moving average (Wilder's RMA: SMA-seeded, `alpha = 1/period`).
pub fn smma(close: &[f64], period: usize) -> Vec<f64> {
    kernels::wilder(av(close), period).to_vec()
}

/// Weighted moving average — linearly increasing weights `1..=period`, the newest
/// bar weighted heaviest (TA-Lib WMA). O(n) via a running sum + running weighted
/// sum. Lookback `period-1`.
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (derived inputs — e.g. the stochastic %K line —
    // warm up with NaN): compute from the first finite value onward, mirroring sma/ema.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = wma(&data[start..], period);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    let denom = (period * (period + 1) / 2) as f64; // sum of weights 1..=period
    let pf = period as f64;
    // Seed the first full window directly.
    let mut sum = 0.0; // plain window sum
    let mut wsum = 0.0; // weighted sum: newest bar * period ... oldest * 1
    for j in 0..period {
        sum += data[j];
        wsum += data[j] * (j + 1) as f64;
    }
    out[period - 1] = wsum / denom;
    // Slide: dropping the oldest (weight 1) raises every retained weight by one.
    for i in period..n {
        wsum += pf * data[i] - sum;
        sum += data[i] - data[i - period];
        out[i] = wsum / denom;
    }
    out
}

/// Staggered warm-up for a cascade of `S` SMA-seeded EMAs (`k = 2/(period+1)`): each
/// stage SMA-seeds over the first `period` finite values of its predecessor, so stage
/// `j` seeds at `j·(period-1)`. Returns `(lookback, stage values at lookback)` where
/// `lookback = S·(period-1)` is the first fully-valid bar, or `None` if it never seeds.
/// Shared by DEMA/TEMA/T3 — their combine and steady-state recurrence differ, but the
/// warm-up is identical (and bit-identical to chaining `ema_seeded` `S` times).
fn cascade_warmup<const S: usize>(data: &[f64], period: usize, k: f64) -> Option<(usize, [f64; S])> {
    let n = data.len();
    if period == 0 {
        return None;
    }
    // Account for a leading-NaN warm-up prefix (derived inputs — e.g. the stochastic
    // %K line — warm up with NaN). The cascade seeds over the first finite values, so
    // the last stage is ready at `start + S*(period-1)`, not from index 0; computing
    // the lookback from index 0 returns unseeded stages (garbage) too early.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    let lookback = start + S * (period - 1);
    if lookback >= n {
        return None;
    }
    let mut e = [0.0f64; S];
    let mut acc = [0.0f64; S];
    let mut cnt = [0usize; S];
    let mut seeded = [false; S];
    for &raw in &data[..=lookback] {
        let mut x = raw;
        for s in 0..S {
            if seeded[s] {
                e[s] = (x - e[s]).mul_add(k, e[s]);
                x = e[s];
            } else if !x.is_nan() {
                acc[s] += x;
                cnt[s] += 1;
                if cnt[s] == period {
                    e[s] = acc[s] / period as f64;
                    seeded[s] = true;
                    x = e[s];
                } else {
                    x = f64::NAN;
                }
            } else {
                x = f64::NAN;
            }
        }
    }
    Some((lookback, e))
}

/// Double EMA: `2*EMA - EMA(EMA)` (TA-Lib DEMA). Lookback `2*(period-1)`. Single-pass
/// lattice over the two cascaded EMAs (vs two `ema_seeded` passes + a combine);
/// bit-identical to `2·e1 − e2`.
pub fn dema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let k = 2.0 / (period as f64 + 1.0);
    let Some((lookback, e)) = cascade_warmup::<2>(data, period, k) else {
        return out;
    };
    out[lookback] = 2.0 * e[0] - e[1];
    let (mut e0, mut e1) = (e[0], e[1]);
    for i in (lookback + 1)..n {
        e0 = (data[i] - e0).mul_add(k, e0);
        e1 = (e0 - e1).mul_add(k, e1);
        out[i] = 2.0 * e0 - e1;
    }
    out
}

/// Triple EMA: `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` (TA-Lib TEMA). Lookback
/// `3*(period-1)`. Single-pass lattice over the three cascaded EMAs; bit-identical to
/// `3·e1 − 3·e2 + e3`.
pub fn tema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let k = 2.0 / (period as f64 + 1.0);
    let Some((lookback, e)) = cascade_warmup::<3>(data, period, k) else {
        return out;
    };
    out[lookback] = 3.0 * e[0] - 3.0 * e[1] + e[2];
    let (mut e0, mut e1, mut e2) = (e[0], e[1], e[2]);
    for i in (lookback + 1)..n {
        e0 = (data[i] - e0).mul_add(k, e0);
        e1 = (e0 - e1).mul_add(k, e1);
        e2 = (e1 - e2).mul_add(k, e2);
        out[i] = 3.0 * e0 - 3.0 * e1 + e2;
    }
    out
}

/// Triangular moving average (TA-Lib TRIMA): a double SMA whose net weights rise to
/// the window centre then fall. Since convolution commutes, the two SMA windows are
/// `((n+1)/2, (n+1)/2)` for odd `n` and `(n/2, n/2+1)` for even `n` (either order).
/// Lookback `period-1`.
pub fn trima(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (see `wma`).
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = trima(&data[start..], period);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    let (a, b) = if period % 2 == 1 {
        let m = (period + 1) / 2;
        (m, m)
    } else {
        let m = period / 2;
        (m, m + 1)
    };
    // Fuse the two SMA convolutions into one pass (no intermediate `inner` array): a
    // running `sum1` slides the inner a-window, each inner mean feeds `sum2` — the
    // outer b-window sum over a ring of the last `b` inner means. Both windows use SMA's
    // exact add-then-subtract order, and the outer slide begins at the first finite
    // inner mean (index a-1, mirroring the leading-NaN skip), so this is bit-identical
    // to `sma(sma(data, a), b)`.
    let af = a as f64;
    let bf = b as f64;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut ring = vec![0.0; b];
    let mut slot = 0usize;
    let mut inner_cnt = 0usize;
    for i in 0..n {
        sum1 += data[i];
        if i >= a {
            sum1 -= data[i - a];
        }
        if i + 1 >= a {
            let inner = sum1 / af;
            sum2 += inner;
            if inner_cnt >= b {
                sum2 -= ring[slot]; // the inner mean leaving the outer window
            }
            ring[slot] = inner;
            slot += 1;
            if slot == b {
                slot = 0;
            }
            inner_cnt += 1;
            if inner_cnt >= b {
                out[i] = sum2 / bf;
            }
        }
    }
    out
}

/// Tillson T3 (TA-Lib T3): `c1·e6 + c2·e5 + c3·e4 + c4·e3` over six cascaded
/// SMA-seeded EMAs, with `vfactor`-derived coefficients `c1=-v³`, `c2=3(v²−c1)`,
/// `c3=-6v²−3(v−c1)`, `c4=1+3v−c1+3v²` (computed in TA-Lib's exact float order).
/// Default period 5, vfactor 0.7; lookback `6·(period-1)`.
pub fn t3(data: &[f64], period: usize, vfactor: f64) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let v2 = vfactor * vfactor;
    let c1 = -(v2 * vfactor);
    let c2 = 3.0 * (v2 - c1);
    let c3 = -6.0 * v2 - 3.0 * (vfactor - c1);
    let c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * v2;
    let k = 2.0 / (period as f64 + 1.0);
    // Single-pass lattice instead of six sequential `ema_seeded` passes: the six EMAs
    // cascade (stage j consumes stage j-1's *current* output) so they share one
    // staggered-warmup traversal ([`cascade_warmup`]); the steady state then advances a
    // six-deep FMA chain per bar (TA-Lib's lattice) and emits the combine in place.
    // Bit-identical to the six-call form.
    let Some((lookback, e)) = cascade_warmup::<6>(data, period, k) else {
        return out;
    };
    out[lookback] = c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2];
    // Steady state: all six stages seeded. The cascade is inherently sequential (a
    // six-deep FMA dependency chain per bar, exactly as in TA-Lib's lattice), so we
    // unroll it into registers and emit the combine in the same pass.
    let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5) =
        (e[0], e[1], e[2], e[3], e[4], e[5]);
    for i in (lookback + 1)..n {
        e0 = (data[i] - e0).mul_add(k, e0);
        e1 = (e0 - e1).mul_add(k, e1);
        e2 = (e1 - e2).mul_add(k, e2);
        e3 = (e2 - e3).mul_add(k, e3);
        e4 = (e3 - e4).mul_add(k, e4);
        e5 = (e4 - e5).mul_add(k, e5);
        out[i] = c1 * e5 + c2 * e4 + c3 * e3 + c4 * e2;
    }
    out
}

/// Kaufman Adaptive Moving Average (TA-Lib KAMA). The smoothing constant adapts each
/// bar via the efficiency ratio `ER = |price[i]-price[i-period]| / Σ|1-bar changes|`:
/// `SC = (ER·(fast−slow) + slow)²` with fast `=2/3`, slow `=2/31`; then
/// `KAMA[i] = KAMA[i-1] + SC·(price[i] − KAMA[i-1])`, seeded `KAMA[period-1] =
/// price[period-1]`. First value at index `period` (lookback = period). Faithful port
/// of TA-Lib's sliding-sum recurrence.
pub fn kama(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period + 1 > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (see `wma`).
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = kama(&data[start..], period);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    const CONST_MAX: f64 = 2.0 / (30.0 + 1.0); // slow smoothing constant
    let const_diff = 2.0 / (2.0 + 1.0) - CONST_MAX; // fast − slow

    let efficiency_sc = |period_roc: f64, sum_roc1: f64| {
        let er = if sum_roc1 <= period_roc || sum_roc1.abs() < 1e-14 {
            1.0
        } else {
            (period_roc / sum_roc1).abs()
        };
        let sc = er.mul_add(const_diff, CONST_MAX);
        sc * sc
    };

    // sumROC1 = Σ_{j=0}^{period-1} |price[j] − price[j+1]| (the first window).
    let mut sum_roc1 = 0.0;
    for j in 0..period {
        sum_roc1 += (data[j] - data[j + 1]).abs();
    }
    let mut trailing_idx = 0usize;
    let mut today = period;
    // First KAMA (index `period`): the prior KAMA is seeded with yesterday's price.
    let mut prev_kama = data[today - 1];
    let mut trailing_value = data[trailing_idx];
    let period_roc = data[today] - trailing_value;
    trailing_idx += 1;
    let sc = efficiency_sc(period_roc, sum_roc1);
    // `(price − prev)·sc + prev` fused: one rounding off the recurrence's critical
    // path. KAMA is contractive (sc ≤ (2/3)² < 1), so the ~1e-16 divergence decays —
    // within the 1e-9 TA-Lib parity tolerance.
    prev_kama = (data[today] - prev_kama).mul_add(sc, prev_kama);
    out[today] = prev_kama;
    today += 1;

    while today < n {
        let tr2 = data[trailing_idx];
        trailing_idx += 1;
        let period_roc = data[today] - tr2;
        sum_roc1 -= (trailing_value - tr2).abs(); // drop the oldest 1-bar change
        sum_roc1 += (data[today] - data[today - 1]).abs(); // add the newest
        trailing_value = tr2;
        let sc = efficiency_sc(period_roc, sum_roc1);
        prev_kama = (data[today] - prev_kama).mul_add(sc, prev_kama);
        out[today] = prev_kama;
        today += 1;
    }
    out
}

/// Parabolic SAR (TA-Lib SAR). Faithful port of TA-Lib's recurrence: initial trend is
/// chosen from the first bar's −DM1; each step trails the stop by `af·(ep − sar)`,
/// `af` ramping by `acceleration` (capped at `maximum`) on every new extreme, resetting
/// on a reversal; the SAR is clamped within the prior two bars' range. Default
/// acceleration 0.02, maximum 0.2; lookback 1. (TA-Lib applies no rounding.)
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 {
        return out;
    }
    let af_init = acceleration.min(maximum); // TA-Lib clamps the step to the cap
                                             // Initial direction from the one-period −DM at bar 1: a positive −DM ⇒ short.
    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m {
        diff_m
    } else {
        0.0
    };
    let mut is_long = !(minus_dm1 > 0.0);

    let mut af = af_init;
    let (mut ep, mut sar) = if is_long {
        (high[1], low[0])
    } else {
        (low[1], high[0])
    };
    // "Cheat" the first iteration: prime new high/low with bar 1 (as TA-Lib does).
    let mut new_low = low[1];
    let mut new_high = high[1];

    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                // Reverse to short: stop becomes the extreme point, clamped up.
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                out[today] = sar;
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out[today] = sar;
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            // Reverse to long: stop becomes the extreme point, clamped down.
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            out[today] = sar;
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out[today] = sar;
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    out
}

/// Parabolic SAR Extended (TA-Lib SAREXT). As [`sar`], but with separate long/short
/// acceleration (init/step/max), an optional start value (`>0` forces an initial long at
/// that level, `<0` an initial short at `|start|`, `0` = SAR's directional bootstrap), an
/// `offset_on_reverse` that nudges the stop on each reversal, and a **signed** output —
/// negative while short, positive while long — so reversals are visible. Lookback 1.
#[allow(clippy::too_many_arguments)]
pub fn sarext(
    high: &[f64],
    low: &[f64],
    start_value: f64,
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 {
        return out;
    }
    // TA-Lib clamps the init/step factors to their caps.
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);

    // Initial direction: forced by a non-zero start value, else SAR's -DM1 bootstrap.
    let mut is_long = if start_value == 0.0 {
        let diff_p = high[1] - high[0];
        let diff_m = low[0] - low[1];
        let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m { diff_m } else { 0.0 };
        !(minus_dm1 > 0.0)
    } else {
        start_value > 0.0
    };

    let (mut ep, mut sar) = if start_value == 0.0 {
        if is_long { (high[1], low[0]) } else { (low[1], high[0]) }
    } else if start_value > 0.0 {
        (high[1], start_value)
    } else {
        (low[1], start_value.abs())
    };

    let (mut af_long, mut af_short) = (af_long_init, af_short_init);
    let mut new_low = low[1];
    let mut new_high = high[1];

    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                out[today] = -sar;
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out[today] = sar;
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            out[today] = sar;
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out[today] = -sar;
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    out
}

fn macd_line(close: &[f64], fast: usize, slow: usize) -> Array1<f64> {
    // TA-Lib MACD line = fast EMA - slow EMA (SMA-seeded EMAs). Best practice: the
    // line is emitted from its natural start (the slow EMA's first valid row), not
    // delayed to the signal line's start as TA-Lib's aligned 3-output form does.
    // `ema_diff_seeded` fuses both EMAs into one interleaved pass (ILP over the two
    // independent recurrences) and emits the difference directly.
    kernels::ema_diff_seeded(av(close), fast, slow)
}

/// MACD line (DIF).
pub fn macd(close: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    macd_line(close, fast, slow).to_vec()
}

/// MACD signal line (DEA) — SMA-seeded EMA of the MACD line.
pub fn macd_signal(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let line = macd_line(close, fast, slow);
    kernels::ema_seeded(line.view(), signal).to_vec()
}

/// MACD histogram — TA-Lib convention `MACD - signal` (not the stock-pandas `2x`).
pub fn macd_histogram(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let line = macd_line(close, fast, slow);
    let sig = kernels::ema_seeded(line.view(), signal);
    (&line - &sig).to_vec()
}

/// Bull and Bear Index (`mean of ma:a, ma:b, ma:c, ma:d`).
pub fn bbi(close: &[f64], a: usize, b: usize, c: usize, d: usize) -> Vec<f64> {
    let data = av(close);
    let ma_a = kernels::sma(data, a);
    let ma_b = kernels::sma(data, b);
    let ma_c = kernels::sma(data, c);
    let ma_d = kernels::sma(data, d);
    ((&ma_a + &ma_b + &ma_c + &ma_d) / 4.0).to_vec()
}

/// True Range.
pub fn tr(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![f64::NAN; n];
    // TA-Lib TRANGE: index 0 has no prior close, so it has no TR (NaN); TR is
    // defined from index 1 onward.
    for i in 1..n {
        let prev_close = close[i - 1];
        let hl = high[i] - low[i];
        let hc = (high[i] - prev_close).abs();
        let lc = (low[i] - prev_close).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    tr
}

/// Average True Range — TA-Lib semantics: SMA-seeded Wilder smoothing of TR (the
/// first ATR, at index `period`, is the SMA of the first `period` TRs; thereafter
/// `ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period`).
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let tr = tr(high, low, close);
    kernels::wilder(av(&tr), period).to_vec()
}

/// Normalized Average True Range — `ATR/close · 100` (TA-Lib NATR), expressing ATR
/// as a percentage of price. Lookback = period (same as ATR).
pub fn natr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let atr = atr(high, low, close, period);
    (0..close.len())
        .map(|i| atr[i] / close[i] * 100.0)
        .collect()
}
