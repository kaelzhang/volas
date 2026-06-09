//! Group B indicators (gap report 2026-06-07, §9): convention-sensitive market
//! indicators. Each kernel pins ONE authoritative convention, cited inline and matched
//! bit-for-shape by the source-pinned oracle (`test/oracle_reference.py`). Finite-memory
//! members refresh on `append` through the engine's windowed fast-path (no state-carry);
//! recursive members carry a `*_final_state` / `*_resume` pair like Group A.

use crate::indicators::av;
use crate::kernels;

/// Rolling `n`-window sum that skips a leading-NaN prefix — so a series with an undefined
/// first bar (`tr`, prev-close diffs) sums from its first finite value — via `sma * n`.
fn rolling_sum(x: &[f64], n: usize) -> Vec<f64> {
    (kernels::sma(av(x), n) * n as f64).to_vec()
}

/// Vortex Indicator (+VI / −VI): `+VM = |high − prev low|`, `−VM = |low − prev high|`;
/// `+VI = Σₙ(+VM) / Σₙ(TR)`, `−VI = Σₙ(−VM) / Σₙ(TR)`. `plus` selects the +VI line.
/// Source: StockCharts ChartSchool / Wikipedia — Vortex Indicator.
pub fn vortex(high: &[f64], low: &[f64], close: &[f64], n: usize, plus: bool) -> Vec<f64> {
    let len = high.len();
    let tr = super::tr(high, low, close); // NaN at bar 0
    let vm: Vec<f64> = (0..len)
        .map(|i| {
            if i == 0 {
                f64::NAN
            } else if plus {
                (high[i] - low[i - 1]).abs()
            } else {
                (low[i] - high[i - 1]).abs()
            }
        })
        .collect();
    let sum_vm = rolling_sum(&vm, n);
    let sum_tr = rolling_sum(&tr, n);
    (0..len).map(|i| sum_vm[i] / sum_tr[i]).collect()
}

/// BRAR — AR (人气指标) = `Σₙ(H − O) / Σₙ(O − L) × 100`. `H − O` and `O − L` are always ≥ 0.
/// Source: 通达信 / MBA智库 — 人气意愿指标 (BRAR).
pub fn brar_ar(open: &[f64], high: &[f64], low: &[f64], n: usize) -> Vec<f64> {
    let len = high.len();
    let ho: Vec<f64> = (0..len).map(|i| high[i] - open[i]).collect();
    let ol: Vec<f64> = (0..len).map(|i| open[i] - low[i]).collect();
    let (s_ho, s_ol) = (rolling_sum(&ho, n), rolling_sum(&ol, n));
    (0..len).map(|i| s_ho[i] / s_ol[i] * 100.0).collect()
}

/// BRAR — BR (意愿指标) = `Σₙ max(0, H − Cᵧ) / Σₙ max(0, Cᵧ − L) × 100`, `Cᵧ` = prior close.
/// The `max(0, …)` clamp is the 通达信 convention (a high below — or low above — the prior
/// close contributes nothing). Source: 通达信 / MBA智库 — 人气意愿指标 (BRAR).
pub fn brar_br(high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let len = high.len();
    let up: Vec<f64> = (0..len)
        .map(|i| if i == 0 { f64::NAN } else { (high[i] - close[i - 1]).max(0.0) })
        .collect();
    let dn: Vec<f64> = (0..len)
        .map(|i| if i == 0 { f64::NAN } else { (close[i - 1] - low[i]).max(0.0) })
        .collect();
    let (s_up, s_dn) = (rolling_sum(&up, n), rolling_sum(&dn, n));
    (0..len).map(|i| s_up[i] / s_dn[i] * 100.0).collect()
}

/// VR 成交量比率 = `(UVS + ½·PVS) / (DVS + ½·PVS) × 100` over `n` bars, where UVS / DVS / PVS
/// are the summed volumes of up- / down- / flat-close days (vs the prior close).
/// Source: MBA智库 — 成交量比率 (VR).
pub fn vr(close: &[f64], volume: &[f64], n: usize) -> Vec<f64> {
    let len = close.len();
    let (mut uv, mut dv, mut pv) = (vec![f64::NAN; len], vec![f64::NAN; len], vec![f64::NAN; len]);
    for i in 1..len {
        let (mut u, mut d, mut p) = (0.0, 0.0, 0.0);
        if close[i] > close[i - 1] {
            u = volume[i];
        } else if close[i] < close[i - 1] {
            d = volume[i];
        } else {
            p = volume[i];
        }
        uv[i] = u;
        dv[i] = d;
        pv[i] = p;
    }
    let (suv, sdv, spv) = (rolling_sum(&uv, n), rolling_sum(&dv, n), rolling_sum(&pv, n));
    (0..len)
        .map(|i| (suv[i] + 0.5 * spv[i]) / (sdv[i] + 0.5 * spv[i]) * 100.0)
        .collect()
}

/// Coppock Curve = `WMA_w(ROC_long + ROC_short)`, `ROC_p = (C / C₋ₚ − 1) × 100`.
/// Source: StockCharts ChartSchool / Wikipedia — Coppock Curve.
pub fn coppock(close: &[f64], w: usize, roc_long: usize, roc_short: usize) -> Vec<f64> {
    let len = close.len();
    let roc = |p: usize| -> Vec<f64> {
        (0..len)
            .map(|i| if i >= p { (close[i] / close[i - p] - 1.0) * 100.0 } else { f64::NAN })
            .collect::<Vec<_>>()
    };
    let (rl, rs) = (roc(roc_long), roc(roc_short));
    let sum: Vec<f64> = (0..len).map(|i| rl[i] + rs[i]).collect();
    super::wma(&sum, w)
}

/// 4-bar symmetric weighted MA, weights `[1, 2, 2, 1] / 6` (newest first); NaN until 3 prior
/// bars exist. The RVI numerator / denominator / signal smoother.
fn swma4(x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i >= 3 {
                (x[i] + 2.0 * x[i - 1] + 2.0 * x[i - 2] + x[i - 3]) / 6.0
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Relative Vigor Index = `SMAₙ(swma4(C − O)) / SMAₙ(swma4(H − L))`.
/// Source: Fidelity / MetaTrader — Relative Vigor Index.
pub fn relative_vigor(open: &[f64], high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let len = close.len();
    let co: Vec<f64> = (0..len).map(|i| close[i] - open[i]).collect();
    let hl: Vec<f64> = (0..len).map(|i| high[i] - low[i]).collect();
    let num = kernels::sma(av(&swma4(&co)), n);
    let den = kernels::sma(av(&swma4(&hl)), n);
    (0..len).map(|i| num[i] / den[i]).collect()
}

/// RVI signal line = `swma4(RVI)`. Source: Fidelity / MetaTrader — Relative Vigor Index.
pub fn relative_vigor_signal(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    n: usize,
) -> Vec<f64> {
    swma4(&relative_vigor(open, high, low, close, n))
}

/// DKX 多空线 = `WMA(MID, 20)`, `MID = (3·C + L + O + H) / 6`. The fixed 20-period linear
/// weights 20…1 are exactly TA-Lib WMA's. Source: 百度百科 / 东方财富 — 多空线 (DKX).
pub fn dkx(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mid: Vec<f64> = (0..close.len())
        .map(|i| (3.0 * close[i] + low[i] + open[i] + high[i]) / 6.0)
        .collect();
    super::wma(&mid, 20)
}

/// MADKX = `SMA_m(DKX)`, the DKX signal line. Source: 百度百科 / 东方财富 — 多空线 (DKX).
pub fn dkx_ma(open: &[f64], high: &[f64], low: &[f64], close: &[f64], m: usize) -> Vec<f64> {
    super::ma(&dkx(open, high, low, close), m)
}

/// WVAD 威廉变异离散量 = `Σₙ( (C − O) / (H − L) · V )`. Source: 通达信 / MBA智库 — WVAD.
pub fn wvad(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    n: usize,
) -> Vec<f64> {
    let w: Vec<f64> = (0..close.len())
        .map(|i| (close[i] - open[i]) / (high[i] - low[i]) * volume[i])
        .collect();
    rolling_sum(&w, n)
}

/// Which CDP support/resistance line to emit.
#[derive(Clone, Copy)]
pub enum CdpLine {
    Cdp,
    Ah,
    Nh,
    Nl,
    Al,
}

/// CDP 逆势操作: from the PRIOR bar, `CDP = (H + L + 2C) / 4`, then `AH = CDP + (H − L)`,
/// `NH = 2·CDP − L`, `NL = 2·CDP − H`, `AL = CDP − (H − L)` — five intraday levels.
/// Source: 百度百科 / 维基 — 逆势操作 (CDP).
pub fn cdp(high: &[f64], low: &[f64], close: &[f64], line: CdpLine) -> Vec<f64> {
    (0..high.len())
        .map(|i| {
            if i == 0 {
                return f64::NAN;
            }
            let (h, l, c) = (high[i - 1], low[i - 1], close[i - 1]);
            let cdp = (h + l + 2.0 * c) / 4.0;
            match line {
                CdpLine::Cdp => cdp,
                CdpLine::Ah => cdp + (h - l),
                CdpLine::Nh => 2.0 * cdp - l,
                CdpLine::Nl => 2.0 * cdp - h,
                CdpLine::Al => cdp - (h - l),
            }
        })
        .collect()
}

/// Which MIKE support / resistance line to emit.
#[derive(Clone, Copy)]
pub enum MikeLine {
    WeakR,
    MidR,
    StrongR,
    WeakS,
    MidS,
    StrongS,
}

/// MIKE 麦克支撑压力: `TYP = (H+L+C)/3`, `HH`/`LL` = n-day high-of-high / low-of-low.
/// Resistance WEKR = TYP+(TYP−LL), MIDR = TYP+(HH−LL), STOR = 2·HH−LL; support mirrors them:
/// WEKS = TYP−(HH−TYP), MIDS = TYP−(HH−LL), STOS = 2·LL−HH. Source: 百度百科 / MBA智库 — 麦克指标.
pub fn mike(high: &[f64], low: &[f64], close: &[f64], n: usize, line: MikeLine) -> Vec<f64> {
    let len = high.len();
    let hh = super::hhv(high, n);
    let ll = super::llv(low, n);
    (0..len)
        .map(|i| {
            let typ = (high[i] + low[i] + close[i]) / 3.0;
            let (hi, lo) = (hh[i], ll[i]);
            match line {
                MikeLine::WeakR => typ + (typ - lo),
                MikeLine::MidR => typ + (hi - lo),
                MikeLine::StrongR => 2.0 * hi - lo,
                MikeLine::WeakS => typ - (hi - typ),
                MikeLine::MidS => typ - (hi - lo),
                MikeLine::StrongS => 2.0 * lo - hi,
            }
        })
        .collect()
}

/// Keltner Channel band (modern convention): `EMA(close, ema_period) ± mult · ATR(atr_period)`.
/// `upper` selects the `+` band. The middle line is just `EMA(close)`, so callers emit that
/// via `ema` directly. Source: StockCharts ChartSchool — Keltner Channels.
pub fn keltner_band(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    ema_period: usize,
    atr_period: usize,
    mult: f64,
    upper: bool,
) -> Vec<f64> {
    let ema = super::ema(close, ema_period);
    let atr = super::atr(high, low, close, atr_period);
    let sign = if upper { 1.0 } else { -1.0 };
    (0..close.len()).map(|i| ema[i] + sign * mult * atr[i]).collect()
}

/// Keltner band state `[ema, atr]` after a full compute, or `None` before both seed. The
/// middle line reuses `ema_final_state`. Source: StockCharts ChartSchool — Keltner Channels.
pub fn keltner_band_final_state(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    ema_period: usize,
    atr_period: usize,
) -> Option<Vec<f64>> {
    let e = super::ema_final_state(close, ema_period)?;
    let a = super::atr_final_state(high, low, close, atr_period)?;
    Some(vec![e[0], a[0]])
}

/// Resume a Keltner band over `[from, n)` by composing the EMA and ATR resumes — bit-identical
/// to a full recompute. `None` at `from == 0` (the ATR resume declines).
pub fn keltner_band_resume(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    ema_period: usize,
    atr_period: usize,
    mult: f64,
    upper: bool,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let (e_vals, e_state) = super::ema_resume(close, ema_period, from, &state[0..1]);
    let (a_vals, a_state) = super::atr_resume(high, low, close, atr_period, from, &state[1..2])?;
    let sign = if upper { 1.0 } else { -1.0 };
    let vals: Vec<f64> = (0..e_vals.len()).map(|i| e_vals[i] + sign * mult * a_vals[i]).collect();
    Some((vals, vec![e_state[0], a_state[0]]))
}

/// Stochastic Momentum Index (Blau / LazyBear): `HH`/`LL` = k-day high/low; `D = C − (HH+LL)/2`,
/// `Ds = EMA_d(EMA_d(D))`, `Dhl = EMA_d(EMA_d(HH−LL))`; `SMI = Ds / (Dhl/2) × 100`. The short
/// double-EMA smoothing keeps it effectively finite-memory. Source: William Blau / LazyBear — SMI.
pub fn stoch_momentum(high: &[f64], low: &[f64], close: &[f64], k: usize, d: usize) -> Vec<f64> {
    let len = high.len();
    let hh = super::hhv(high, k);
    let ll = super::llv(low, k);
    let rdiff: Vec<f64> = (0..len).map(|i| close[i] - (hh[i] + ll[i]) / 2.0).collect();
    let diff: Vec<f64> = (0..len).map(|i| hh[i] - ll[i]).collect();
    let ds = super::ema(&super::ema(&rdiff, d), d);
    let dhl = super::ema(&super::ema(&diff, d), d);
    (0..len).map(|i| ds[i] / (dhl[i] * 0.5) * 100.0).collect()
}

/// SMI signal line = `EMA_signal(SMI)`. Source: William Blau / LazyBear — SMI.
pub fn stoch_momentum_signal(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k: usize,
    d: usize,
    signal: usize,
) -> Vec<f64> {
    super::ema(&stoch_momentum(high, low, close, k, d), signal)
}

/// TTM Squeeze momentum = `linregₙ( C − ((HHₙ + LLₙ)/2 + SMAₙ(C)) / 2 )` (John Carter /
/// thinkorswim). Source: John Carter / StockCharts — TTM Squeeze.
pub fn ttm_squeeze_momentum(high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let len = high.len();
    let hh = super::hhv(high, n);
    let ll = super::llv(low, n);
    let sma = super::ma(close, n);
    let delta: Vec<f64> = (0..len)
        .map(|i| close[i] - ((hh[i] + ll[i]) / 2.0 + sma[i]) / 2.0)
        .collect();
    // `delta` warms up with NaN; linearreg's running sum would propagate it, so run the
    // regression over the finite tail and place it back at the matching offset.
    let start = delta.iter().position(|x| !x.is_nan()).unwrap_or(len);
    let mut out = vec![f64::NAN; len];
    if start < len {
        let sub = super::linearreg(&delta[start..], n);
        out[start..].copy_from_slice(&sub);
    }
    out
}

/// TTM Squeeze on/off: `1.0` when the Bollinger Bands (n, `bb_mult`·σ) sit inside the Keltner
/// Channels (n, `kc_mult`·SMAₙ(TR)), else `0.0`; NaN until both warm up. Source: John Carter /
/// StockCharts — TTM Squeeze (Carter's original SMA-of-range Keltner).
pub fn ttm_squeeze_on(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    n: usize,
    bb_mult: f64,
    kc_mult: f64,
) -> Vec<f64> {
    let len = high.len();
    let sma = super::ma(close, n);
    let sd = super::stddev(close, n, 1.0);
    let tr = super::tr(high, low, close);
    let atr = kernels::sma(av(&tr), n).to_vec(); // SMA of TR (skips the leading-NaN tr[0])
    (0..len)
        .map(|i| {
            let bb_u = sma[i] + bb_mult * sd[i];
            let bb_l = sma[i] - bb_mult * sd[i];
            let kc_u = sma[i] + kc_mult * atr[i];
            let kc_l = sma[i] - kc_mult * atr[i];
            if bb_u.is_nan() || kc_u.is_nan() {
                f64::NAN
            } else if bb_l > kc_l && bb_u < kc_u {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}
