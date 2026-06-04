use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Momentum — change relative to the price `period` bars earlier
// ---------------------------------------------------------------------------

/// Momentum: `data[i] - data[i-period]` (TA-Lib MOM). NaN during warm-up.
pub fn mom(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        out[i] = data[i] - data[i - period];
    }
    out
}

/// Shared shape for the rate-of-change ratios (ROC/ROCP/ROCR/ROCR100): relate
/// each row to the price `period` bars earlier via `f(current, prior)`, NaN
/// during warm-up. A prior price of exactly zero yields `0.0`, matching TA-Lib's
/// divide-by-zero guard (purely theoretical for a positive price series).
fn roc_ratio(data: &[f64], period: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        let prior = data[i - period];
        out[i] = if prior == 0.0 { 0.0 } else { f(data[i], prior) };
    }
    out
}

/// Rate of change: `100 * (data/data[period ago] - 1)` (TA-Lib ROC).
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| (cur / prior - 1.0) * 100.0)
}

/// Rate of change percentage: `data/data[period ago] - 1` (TA-Lib ROCP).
pub fn rocp(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior - 1.0)
}

/// Rate of change ratio: `data/data[period ago]` (TA-Lib ROCR).
pub fn rocr(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior)
}

/// Rate of change ratio ×100: `100 * data/data[period ago]` (TA-Lib ROCR100).
pub fn rocr100(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior * 100.0)
}

/// Williams %R: `-100 * (HH - close) / (HH - LL)` over `period`, where HH/LL are
/// the highest high / lowest low (TA-Lib WILLR). A flat range (HH == LL) yields 0.
/// Lookback `period-1`. The operation order mirrors TA-Lib bit-for-bit.
pub fn willr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(high), period);
    let ll = kernels::rolling_min(av(low), period);
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let diff = (hh[i] - ll[i]) / -100.0; // NaN during warm-up -> stays NaN below
        if diff != 0.0 {
            out[i] = (hh[i] - close[i]) / diff;
        } else if !hh[i].is_nan() {
            out[i] = 0.0; // finite flat range
        }
    }
    out
}

/// Balance of Power: `(close − open)/(high − low)` (TA-Lib BOP). A bar with no
/// range (`high − low < ε`) yields 0. Lookback 0.
pub fn bop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| {
            let range = high[i] - low[i];
            if range < 1e-14 {
                0.0
            } else {
                (close[i] - open[i]) / range
            }
        })
        .collect()
}

/// Intraday Momentum Index (TA-Lib IMI): `100·upSum/(upSum+downSum)` over `period`,
/// where each up bar (`close>open`) adds `close-open` to upSum and each down bar adds
/// `open-close` to downSum. Lookback `period-1`. (An all-doji window yields NaN, as in
/// TA-Lib — no divide-by-zero guard. O(n·period), matching TA-Lib.)
pub fn imi(open: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let lookback = period - 1;
    for today in lookback..n {
        let (mut up_sum, mut down_sum) = (0.0, 0.0);
        for i in (today - lookback)..=today {
            let (c, o) = (close[i], open[i]);
            if c > o {
                up_sum += c - o;
            } else {
                down_sum += o - c;
            }
        }
        out[today] = 100.0 * (up_sum / (up_sum + down_sum));
    }
    out
}

/// Commodity Channel Index (TA-Lib CCI): `(tp − SMA(tp)) / (0.015 · meanDev)` over
/// `period`, where `tp = (high+low+close)/3` and `meanDev` is the mean absolute
/// deviation of `tp` from its average. A zero numerator or zero deviation yields 0
/// (TA-Lib's guard). Lookback `period-1`. O(n·period), as in TA-Lib.
pub fn cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let p = period as f64;
    let tp: Vec<f64> = (0..n).map(|i| (high[i] + low[i] + close[i]) / 3.0).collect();
    for i in (period - 1)..n {
        let window = &tp[i + 1 - period..=i];
        let avg = window.iter().sum::<f64>() / p;
        let sum_dev: f64 = window.iter().map(|x| (x - avg).abs()).sum();
        let num = tp[i] - avg;
        out[i] = if num != 0.0 && sum_dev != 0.0 {
            num / (0.015 * (sum_dev / p))
        } else {
            0.0
        };
    }
    out
}

/// TRIX (TA-Lib): the 1-period percent rate-of-change of a triple SMA-seeded EMA of
/// `close`. Lookback `3·(period-1) + 1`. (Reuses the verified `roc` and `ema_seeded`.)
pub fn trix(close: &[f64], period: usize) -> Vec<f64> {
    let e1 = kernels::ema_seeded(av(close), period);
    let e2 = kernels::ema_seeded(e1.view(), period);
    let e3 = kernels::ema_seeded(e2.view(), period);
    roc(&e3.to_vec(), 1)
}

/// Aroon up/down over a `period+1`-bar window (TA-Lib AROON): for each row the
/// most-recent highest high / lowest low in `[i-period, i]` gives "days since the
/// extreme", and up/down = `(100/period)·(period − daysSince)`. Both NaN until index
/// `period`; ties resolve to the most recent bar, matching TA-Lib's tracker.
fn aroon_up_down(high: &[f64], low: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut up = vec![f64::NAN; n];
    let mut down = vec![f64::NAN; n];
    if period == 0 {
        return (up, down);
    }
    let factor = 100.0 / period as f64;
    let pf = period as f64;
    for i in period..n {
        let lo = i - period;
        let (mut hi_idx, mut hi) = (lo, high[lo]);
        let (mut lo_idx, mut lo_v) = (lo, low[lo]);
        for j in (lo + 1)..=i {
            if high[j] >= hi {
                hi = high[j];
                hi_idx = j;
            }
            if low[j] <= lo_v {
                lo_v = low[j];
                lo_idx = j;
            }
        }
        up[i] = factor * (pf - (i - hi_idx) as f64);
        down[i] = factor * (pf - (i - lo_idx) as f64);
    }
    (up, down)
}

/// Aroon Up (TA-Lib AROON, up output). Lookback `period`.
pub fn aroon_up(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    aroon_up_down(high, low, period).0
}

/// Aroon Down (TA-Lib AROON, down output). Lookback `period`.
pub fn aroon_down(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    aroon_up_down(high, low, period).1
}

/// Aroon Oscillator: `aroonUp − aroonDown` (TA-Lib AROONOSC). Lookback `period`.
pub fn aroonosc(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let (up, down) = aroon_up_down(high, low, period);
    up.iter().zip(&down).map(|(u, d)| u - d).collect()
}

/// Money Flow Index (TA-Lib MFI): `100·posMF/(posMF+negMF)` over `period`, where each
/// bar's money flow `tp·volume` (tp = (h+l+c)/3) is signed by `tp`'s direction vs the
/// prior bar. A total flow below 1.0 yields 0 (TA-Lib's guard). First value at index
/// `period` (lookback = period). The sliding sums recompute the leaving bar's flow,
/// which is bit-identical to TA-Lib's stored value.
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period + 1 > n {
        return out;
    }
    let tp = |i: usize| (high[i] + low[i] + close[i]) / 3.0;
    // Per-bar (positive, negative) money flow, signed by tp's change vs the prior bar.
    let mf = |i: usize| -> (f64, f64) {
        let diff = tp(i) - tp(i - 1);
        let flow = tp(i) * volume[i];
        if diff < 0.0 {
            (0.0, flow)
        } else if diff > 0.0 {
            (flow, 0.0)
        } else {
            (0.0, 0.0)
        }
    };
    let mfi_val = |pos: f64, neg: f64| {
        let total = pos + neg;
        if total < 1.0 {
            0.0
        } else {
            100.0 * pos / total
        }
    };
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    for i in 1..=period {
        let (p, ng) = mf(i);
        pos_sum += p;
        neg_sum += ng;
    }
    out[period] = mfi_val(pos_sum, neg_sum);
    for i in (period + 1)..n {
        let (dp, dn) = mf(i - period); // bar leaving the window
        pos_sum -= dp;
        neg_sum -= dn;
        let (ap, an) = mf(i); // bar entering
        pos_sum += ap;
        neg_sum += an;
        out[i] = mfi_val(pos_sum, neg_sum);
    }
    out
}

/// Ultimate Oscillator (TA-Lib ULTOSC): a weighted blend of buying-pressure / true-
/// range ratios over three periods. Per bar, BP = `close − min(low, prevClose)` and
/// TR = the true range; with sliding sums `aₖ/bₖ` over `pₖ` bars, the value is
/// `100·(4·a₁/b₁ + 2·a₂/b₂ + a₃/b₃)/7` (a term is dropped if its TR-sum is ~0). The
/// 4/2/1 weights follow argument position, not magnitude. Default 7/14/28; lookback
/// = max period.
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let max_p = p1.max(p2).max(p3);
    if p1 == 0 || p2 == 0 || p3 == 0 || max_p >= n {
        return out;
    }
    // (buying pressure, true range) for bar i (needs the prior close, so i >= 1).
    let term = |i: usize| -> (f64, f64) {
        let prev_close = close[i - 1];
        let true_low = low[i].min(prev_close);
        let bp = close[i] - true_low;
        let mut tr = high[i] - low[i];
        tr = tr.max((prev_close - high[i]).abs());
        tr = tr.max((prev_close - low[i]).abs());
        (bp, tr)
    };
    let start = max_p; // first output index
    let (mut a1, mut b1, mut a2, mut b2, mut a3, mut b3) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in (start - p1 + 1)..start {
        let (a, b) = term(i);
        a1 += a;
        b1 += b;
    }
    for i in (start - p2 + 1)..start {
        let (a, b) = term(i);
        a2 += a;
        b2 += b;
    }
    for i in (start - p3 + 1)..start {
        let (a, b) = term(i);
        a3 += a;
        b3 += b;
    }
    let (mut t1, mut t2, mut t3) = (start - p1 + 1, start - p2 + 1, start - p3 + 1);
    for today in start..n {
        let (a, b) = term(today);
        a1 += a;
        a2 += a;
        a3 += a;
        b1 += b;
        b2 += b;
        b3 += b;
        let mut output = 0.0;
        if b1.abs() >= 1e-14 {
            output += 4.0 * (a1 / b1);
        }
        if b2.abs() >= 1e-14 {
            output += 2.0 * (a2 / b2);
        }
        if b3.abs() >= 1e-14 {
            output += a3 / b3;
        }
        let (at, bt) = term(t1);
        a1 -= at;
        b1 -= bt;
        t1 += 1;
        let (at, bt) = term(t2);
        a2 -= at;
        b2 -= bt;
        t2 += 1;
        let (at, bt) = term(t3);
        a3 -= at;
        b3 -= bt;
        t3 += 1;
        out[today] = 100.0 * (output / 7.0);
    }
    out
}
