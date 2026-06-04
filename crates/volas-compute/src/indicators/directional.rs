// ---------------------------------------------------------------------------
// Directional movement — Wilder's +DM/-DM/TR, ±DI, DX, ADX, ADXR
// ---------------------------------------------------------------------------
//
// All seven derive from one-period directional movement and true range, smoothed by
// Wilder's *sum* recurrence (not the SMA-seeded average used by ATR): the seed is the
// sum of the first `period-1` one-period values, then `s = s - s/period + term`.
// Faithful port of TA-Lib (which performs no integer rounding).

/// One-period +DM / -DM at bar `i` (needs the prior bar). Per Wilder: the larger of
/// the up-move (`high-prevHigh`) and down-move (`prevLow-low`) wins, the other is 0;
/// ties and non-positive moves yield 0 for both.
#[inline]
fn dm1(high: &[f64], low: &[f64], i: usize) -> (f64, f64) {
    let diff_p = high[i] - high[i - 1];
    let diff_m = low[i - 1] - low[i];
    let plus = if diff_p > 0.0 && diff_p > diff_m { diff_p } else { 0.0 };
    let minus = if diff_m > 0.0 && diff_p < diff_m { diff_m } else { 0.0 };
    (plus, minus)
}

/// One-period true range at bar `i` (needs the prior close).
#[inline]
fn tr1(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    let hl = high[i] - low[i];
    let hc = (high[i] - close[i - 1]).abs();
    let lc = (low[i] - close[i - 1]).abs();
    hl.max(hc).max(lc)
}

/// Wilder "sum" smoothing of a one-period `term`: seed `= Σ term(i)` over bars
/// `[1, period-1]`, placed at index `period-1`; then `s = s - s/period + term(i)`.
/// NaN before index `period-1`.
fn wilder_sum(n: usize, period: usize, term: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    if period == 0 || period >= n {
        return out;
    }
    let mut s = 0.0;
    for i in 1..period {
        s += term(i);
    }
    out[period - 1] = s;
    // `s - s/period + term` == `s*(1 - 1/period) + term`: precompute the factor and
    // fuse (mul_add), taking the per-element division off the recurrence's critical
    // path. Wilder smoothing is contractive, so the ~1e-16 reassociation decays —
    // within the 1e-9 parity tolerance. Speeds every directional indicator.
    let a = 1.0 - 1.0 / period as f64;
    for i in period..n {
        s = s.mul_add(a, term(i));
        out[i] = s;
    }
    out
}

/// Wilder-smoothed +DM (TA-Lib PLUS_DM). Lookback `period-1`.
pub fn plus_dm(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    wilder_sum(high.len(), period, |i| dm1(high, low, i).0)
}

/// Wilder-smoothed -DM (TA-Lib MINUS_DM). Lookback `period-1`.
pub fn minus_dm(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    wilder_sum(high.len(), period, |i| dm1(high, low, i).1)
}

/// `100 · smoothedDM / smoothedTR`, emitted from index `period` (one bar after the DM
/// seed), with a ~0 TR yielding 0. Shared by ±DI.
fn di(dm_sm: &[f64], tr_sm: &[f64], period: usize) -> Vec<f64> {
    let n = dm_sm.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        let t = tr_sm[i];
        out[i] = if t.abs() < 1e-14 {
            0.0
        } else {
            100.0 * dm_sm[i] / t
        };
    }
    out
}

/// +DI (TA-Lib PLUS_DI). Lookback `period`.
pub fn plus_di(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sp = wilder_sum(high.len(), period, |i| dm1(high, low, i).0);
    let st = wilder_sum(high.len(), period, |i| tr1(high, low, close, i));
    di(&sp, &st, period)
}

/// -DI (TA-Lib MINUS_DI). Lookback `period`.
pub fn minus_di(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sm = wilder_sum(high.len(), period, |i| dm1(high, low, i).1);
    let st = wilder_sum(high.len(), period, |i| tr1(high, low, close, i));
    di(&sm, &st, period)
}

/// Directional Movement Index `DX = 100·|+DI − −DI| / (+DI + −DI)` (TA-Lib DX); a ~0
/// TR or ~0 DI-sum yields 0. Lookback `period`.
pub fn dx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sp = wilder_sum(high.len(), period, |i| dm1(high, low, i).0);
    let sm = wilder_sum(high.len(), period, |i| dm1(high, low, i).1);
    let st = wilder_sum(high.len(), period, |i| tr1(high, low, close, i));
    let n = sp.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        let t = st[i];
        out[i] = if t.abs() < 1e-14 {
            0.0
        } else {
            let plus_di = 100.0 * sp[i] / t;
            let minus_di = 100.0 * sm[i] / t;
            let sum = plus_di + minus_di;
            if sum.abs() < 1e-14 {
                0.0
            } else {
                100.0 * (minus_di - plus_di).abs() / sum
            }
        };
    }
    out
}

/// Average Directional Movement Index (TA-Lib ADX): SMA-seeded Wilder *average* of DX
/// (`ADX[2p-1] = mean(DX[p..2p-1])`, then `ADX = (ADX_prev·(p-1) + DX)/p`).
/// Lookback `2·period-1`.
pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let dxv = dx(high, low, close, period);
    let n = dxv.len();
    let mut out = vec![f64::NAN; n];
    let first = 2 * period - 1;
    if period == 0 || first >= n {
        return out;
    }
    let pf = period as f64;
    let mut sum = 0.0;
    for v in dxv.iter().take(2 * period).skip(period) {
        sum += v;
    }
    let mut prev = sum / pf;
    out[first] = prev;
    for i in (2 * period)..n {
        prev = (prev * (pf - 1.0) + dxv[i]) / pf;
        out[i] = prev;
    }
    out
}

/// Average Directional Movement Index Rating: `(ADX[i] + ADX[i−(period−1)]) / 2`
/// (TA-Lib ADXR). Lookback `3·period-2`.
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let a = adx(high, low, close, period);
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 {
        return out;
    }
    let off = period - 1;
    for i in (3 * period - 2)..n {
        out[i] = (a[i] + a[i - off]) / 2.0;
    }
    out
}
