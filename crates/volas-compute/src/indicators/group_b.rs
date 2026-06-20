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

/// WAD Williams Accumulation/Distribution (威廉多空力度线), cumulative: on an up close
/// `+= C − min(prev C, L)`, on a down close `+= C − max(prev C, H)`, otherwise unchanged.
/// Distinct from TA-Lib's AD. Source: Larry Williams / Tulip Indicators — Williams A/D.
pub fn wad(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; close.len()];
    let mut acc = 0.0;
    for i in 1..close.len() {
        acc += wad_step(high[i], low[i], close[i], close[i - 1]);
        out[i] = acc;
    }
    out
}

/// One WAD increment vs the prior close.
fn wad_step(h: f64, l: f64, c: f64, prev_c: f64) -> f64 {
    if c > prev_c {
        c - prev_c.min(l)
    } else if c < prev_c {
        c - prev_c.max(h)
    } else {
        0.0
    }
}

/// `[wad, prev_close]` after a full [`wad`] — `None` for an empty series.
pub fn wad_final_state(high: &[f64], low: &[f64], close: &[f64]) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let mut acc = 0.0;
    for i in 1..n {
        acc += wad_step(high[i], low[i], close[i], close[i - 1]);
    }
    Some(vec![acc, close[n - 1]])
}

/// Resume [`wad`] from `[wad_{from-1}, close_{from-1}]` over `[from, n)`, bit-identical to a
/// full recompute.
pub fn wad_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut acc = state[0];
    let mut prev = state[1];
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    for i in from..close.len() {
        acc += wad_step(high[i], low[i], close[i], prev);
        out.push(acc);
        prev = close[i];
    }
    (out, vec![acc, prev])
}

/// One Wilder Swing Index increment. `op` / `cp` are the prior bar's open / close; `t` is the
/// limit-move scaling. `SI = 50·(N/R)·(K/T)` with Wilder's R selected by the largest of
/// `|H−Cp|`, `|L−Cp|`, `|H−L|`.
fn asi_si(o: f64, h: f64, l: f64, c: f64, op: f64, cp: f64, t: f64) -> f64 {
    let n = (c - cp) + 0.5 * (c - o) + 0.25 * (cp - op);
    let (tr1, tr2, tr3) = ((h - cp).abs(), (l - cp).abs(), (h - l).abs());
    let move_term = 0.25 * (cp - op).abs();
    let r = if tr1 >= tr2 && tr1 >= tr3 {
        tr1 - 0.5 * tr2 + move_term
    } else if tr2 >= tr1 && tr2 >= tr3 {
        tr2 - 0.5 * tr1 + move_term
    } else {
        tr3 + move_term
    };
    let k = tr1.max(tr2);
    50.0 * (n / r) * (k / t)
}

/// ASI Accumulative Swing Index (Wilder, 振动升降指标), cumulative: `ASI = Σ SI`, with the
/// Swing Index per [`asi_si`]. `t` is Wilder's limit-move parameter. Source: J. Welles Wilder,
/// *New Concepts in Technical Trading Systems*.
pub fn asi(open: &[f64], high: &[f64], low: &[f64], close: &[f64], t: f64) -> Vec<f64> {
    let mut out = vec![0.0; close.len()];
    let mut acc = 0.0;
    for i in 1..close.len() {
        acc += asi_si(open[i], high[i], low[i], close[i], open[i - 1], close[i - 1], t);
        out[i] = acc;
    }
    out
}

/// `[asi, prev_close, prev_open]` after a full [`asi`] — `None` for an empty series.
pub fn asi_final_state(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    t: f64,
) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let mut acc = 0.0;
    for i in 1..n {
        acc += asi_si(open[i], high[i], low[i], close[i], open[i - 1], close[i - 1], t);
    }
    Some(vec![acc, close[n - 1], open[n - 1]])
}

/// Resume [`asi`] from `[asi_{from-1}, close_{from-1}, open_{from-1}]` over `[from, n)`.
pub fn asi_resume(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    t: f64,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (mut acc, mut pc, mut po) = (state[0], state[1], state[2]);
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    for i in from..close.len() {
        acc += asi_si(open[i], high[i], low[i], close[i], po, pc, t);
        out.push(acc);
        pc = close[i];
        po = open[i];
    }
    (out, vec![acc, pc, po])
}

/// Which standard pivot-point level to emit.
#[derive(Clone, Copy)]
pub enum PivotLine {
    Pp,
    R1,
    S1,
    R2,
    S2,
    R3,
    S3,
}

/// Standard floor Pivot Points, from the PRIOR bar: `PP = (H + L + C) / 3`; `R1 = 2·PP − L`,
/// `S1 = 2·PP − H`, `R2 = PP + (H − L)`, `S2 = PP − (H − L)`, `R3 = H + 2·(PP − L)`,
/// `S3 = L − 2·(H − PP)`. Source: Investopedia / floor-trader standard — Pivot Points.
pub fn pivot_points(high: &[f64], low: &[f64], close: &[f64], line: PivotLine) -> Vec<f64> {
    (0..high.len())
        .map(|i| {
            if i == 0 {
                return f64::NAN;
            }
            let (h, l, c) = (high[i - 1], low[i - 1], close[i - 1]);
            let pp = (h + l + c) / 3.0;
            match line {
                PivotLine::Pp => pp,
                PivotLine::R1 => 2.0 * pp - l,
                PivotLine::S1 => 2.0 * pp - h,
                PivotLine::R2 => pp + (h - l),
                PivotLine::S2 => pp - (h - l),
                PivotLine::R3 => h + 2.0 * (pp - l),
                PivotLine::S3 => l - 2.0 * (h - pp),
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

/// Keltner Channel Width (TradingView `ta.kcw`): the channel span normalized by
/// the basis, `(upper - lower) / middle = 2·mult·ATR(atr_period) / EMA(ema_period)`.
/// This is the width of volas's OWN Keltner Channel (EMA basis + ATR range), so
/// `kcw` always equals `(keltner.upper - keltner.lower) / keltner` for the same
/// parameters. Pine's `ta.kcw` uses an EMA-of-range basis instead of ATR — a
/// documented divergence that keeps volas's Keltner family internally consistent.
pub fn kcw(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    ema_period: usize,
    atr_period: usize,
    mult: f64,
) -> Vec<f64> {
    let ema = super::ema(close, ema_period);
    let atr = super::atr(high, low, close, atr_period);
    (0..close.len())
        .map(|i| {
            if ema[i] != 0.0 {
                2.0 * mult * atr[i] / ema[i]
            } else {
                f64::NAN
            }
        })
        .collect()
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
// One parameter per series/state the resume reads; a struct would only repackage them.
#[allow(clippy::too_many_arguments)]
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

/// Shift a series forward by `k` bars: `out[i] = x[i-k]` (NaN for the first `k`). Used to
/// displace Ichimoku's leading spans to the bar where they are read (a causal, leading-NaN
/// line — the future projection beyond the data is a charting concern, not a value one).
fn shift_forward(x: &[f64], k: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if i >= k { x[i - k] } else { f64::NAN }).collect()
}

/// Which Ichimoku line to emit.
#[derive(Clone, Copy)]
pub enum IchimokuLine {
    Tenkan,
    Kijun,
    SenkouA,
    SenkouB,
    Chikou,
}

/// Ichimoku Cloud. Tenkan = `(HH_t + LL_t)/2`, Kijun = `(HH_k + LL_k)/2`; Senkou A =
/// `(Tenkan+Kijun)/2` and Senkou B = `(HH_sb + LL_sb)/2`, each displaced forward `kijun`
/// bars to the bar where they apply; Chikou = the close (the lagging span — plotted `kijun`
/// bars back, returned causally so it never depends on a future bar). Source: StockCharts /
/// Fidelity — Ichimoku Cloud.
pub fn ichimoku(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    tenkan: usize,
    kijun: usize,
    senkou_b: usize,
    line: IchimokuLine,
) -> Vec<f64> {
    let midpoint = |period: usize| -> Vec<f64> {
        let hh = super::hhv(high, period);
        let ll = super::llv(low, period);
        (0..high.len()).map(|i| (hh[i] + ll[i]) / 2.0).collect()
    };
    match line {
        IchimokuLine::Tenkan => midpoint(tenkan),
        IchimokuLine::Kijun => midpoint(kijun),
        IchimokuLine::SenkouA => {
            let (t, k) = (midpoint(tenkan), midpoint(kijun));
            let raw: Vec<f64> = (0..high.len()).map(|i| (t[i] + k[i]) / 2.0).collect();
            shift_forward(&raw, kijun)
        }
        IchimokuLine::SenkouB => shift_forward(&midpoint(senkou_b), kijun),
        IchimokuLine::Chikou => close.to_vec(),
    }
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

/// One Supertrend recurrence step: tighten the bands against the prior bar, then flip the
/// trend (`+1` up / `−1` down) on a close crossing the relevant band. Returns the new
/// `(trend, final_upper, final_lower)`. Source: TradingView — Supertrend.
// One parameter per band/state input the step folds; a struct would only repackage them.
#[allow(clippy::too_many_arguments)]
fn supertrend_step(
    hl2: f64,
    atr: f64,
    close: f64,
    mult: f64,
    prev_trend: f64,
    prev_fu: f64,
    prev_fl: f64,
    prev_close: f64,
) -> (f64, f64, f64) {
    let (bu, bl) = (hl2 + mult * atr, hl2 - mult * atr);
    let fu = if bu < prev_fu || prev_close > prev_fu { bu } else { prev_fu };
    let fl = if bl > prev_fl || prev_close < prev_fl { bl } else { prev_fl };
    let trend = if prev_trend < 0.0 {
        if close > fu { 1.0 } else { -1.0 }
    } else if close < fl {
        -1.0
    } else {
        1.0
    };
    (trend, fu, fl)
}

/// The Supertrend output at a bar given its `(trend, final_upper, final_lower)`: the line
/// (lower band in an up-trend, upper band in a down-trend) or the trend direction.
fn supertrend_out(trend: f64, fu: f64, fl: f64, want_direction: bool) -> f64 {
    if want_direction {
        trend
    } else if trend > 0.0 {
        fl
    } else {
        fu
    }
}

/// Supertrend (TradingView): `hl2 ± mult·ATR` bands, recursively tightened and flipped into a
/// single trailing line. `want_direction` returns the `+1`/`−1` trend instead of the line.
/// Source: TradingView — Supertrend.
pub fn supertrend(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    want_direction: bool,
) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    let atr = super::atr(high, low, close, period);
    if period >= len {
        return out; // ATR never seeds
    }
    let hl2 = |i: usize| (high[i] + low[i]) / 2.0;
    // Seed at the first ATR-valid bar: trend starts down (the upper band), per TradingView.
    let (mut trend, mut fu, mut fl, mut pc) = (
        -1.0_f64,
        hl2(period) + mult * atr[period],
        hl2(period) - mult * atr[period],
        close[period],
    );
    out[period] = supertrend_out(trend, fu, fl, want_direction);
    for i in (period + 1)..len {
        let step = supertrend_step(hl2(i), atr[i], close[i], mult, trend, fu, fl, pc);
        (trend, fu, fl) = step;
        out[i] = supertrend_out(trend, fu, fl, want_direction);
        pc = close[i];
    }
    out
}

/// `[trend, final_upper, final_lower, prev_close, atr]` after a full [`supertrend`], or `None`
/// before the ATR seeds. The shared state for both the line and direction sub-commands.
pub fn supertrend_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
) -> Option<Vec<f64>> {
    let len = close.len();
    // `atr_final_state` already declines (returns None) when `period >= len`, so reaching here
    // guarantees `period < len` — the seed index is in range.
    let atr_end = super::atr_final_state(high, low, close, period)?;
    let atr = super::atr(high, low, close, period);
    let hl2 = |i: usize| (high[i] + low[i]) / 2.0;
    let (mut trend, mut fu, mut fl, mut pc) = (
        -1.0_f64,
        hl2(period) + mult * atr[period],
        hl2(period) - mult * atr[period],
        close[period],
    );
    for i in (period + 1)..len {
        let step = supertrend_step(hl2(i), atr[i], close[i], mult, trend, fu, fl, pc);
        (trend, fu, fl) = step;
        pc = close[i];
    }
    Some(vec![trend, fu, fl, close[len - 1], atr_end[0]])
}

/// Resume [`supertrend`] over `[from, n)` by advancing the carried trend / bands with the ATR
/// resume — bit-identical to a full recompute. `None` at `from == 0` (the ATR resume declines).
// One parameter per series/state the resume reads; a struct would only repackage them.
#[allow(clippy::too_many_arguments)]
pub fn supertrend_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    want_direction: bool,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let (mut trend, mut fu, mut fl, mut pc) = (state[0], state[1], state[2], state[3]);
    let (atr_tail, atr_new) = super::atr_resume(high, low, close, period, from, &state[4..5])?;
    let hl2 = |i: usize| (high[i] + low[i]) / 2.0;
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    for (j, i) in (from..close.len()).enumerate() {
        let step = supertrend_step(hl2(i), atr_tail[j], close[i], mult, trend, fu, fl, pc);
        (trend, fu, fl) = step;
        out.push(supertrend_out(trend, fu, fl, want_direction));
        pc = close[i];
    }
    Some((out, vec![trend, fu, fl, pc, atr_new[0]]))
}


/// Scalar single-row twin of [`wad_resume`]: the cumulative Williams A/D at `row`
/// continued from `state = [wad_{row-1}, close_{row-1}]`, value only — no tail/state
/// `Vec`. Mirrors `wad_resume`'s single step `acc += wad_step(high[i], low[i], close[i],
/// prev)` bit-for-bit (the same shared `wad_step`). `None` at `row == 0`, short `state`,
/// or `row` out of range.
pub fn wad_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 2 || row >= close.len() {
        return None;
    }
    Some(state[0] + wad_step(high[row], low[row], close[row], state[1]))
}

/// Scalar single-row twin of [`keltner_band_resume`]: the Keltner band at `row` =
/// `EMA(close)_{row} + sign·mult·ATR_{row}`, composed from the EMA and ATR scalar
/// twins — value only, no `Vec`. `upper` selects the `+` band, `!upper` the `−` band.
/// State is `[ema_{row-1}, atr_{row-1}]`. Bit-identical to `keltner_band_resume` (the
/// same `ema_resume_one` / `atr_resume_one` steps and the same `ema + sign·mult·atr`
/// combine). `None` at `row == 0` (the ATR twin declines), short `state`, or out of range.
#[allow(clippy::too_many_arguments)]
pub fn keltner_band_resume_one(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    ema_period: usize,
    atr_period: usize,
    mult: f64,
    upper: bool,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if state.len() < 2 {
        return None;
    }
    let e = super::ema_resume_one(close, ema_period, row, &state[0..1])?;
    let a = super::atr_resume_one(high, low, close, atr_period, row, &state[1..2])?;
    let sign = if upper { 1.0 } else { -1.0 };
    Some(e + sign * mult * a)
}

/// Scalar single-row twin of [`supertrend_resume`]: the Supertrend output at `row`
/// (the trailing line, or the `+1`/`−1` trend when `want_direction`) continued from
/// `state = [trend, final_upper, final_lower, prev_close, atr]_{row-1}`, value only —
/// no tail/state `Vec`. Advances the ATR via `atr_resume_one`, runs ONE `supertrend_step`,
/// then emits via `supertrend_out` — bit-identical to `supertrend_resume`'s single
/// iteration. `None` at `row == 0` (the ATR twin declines), short `state`, or out of range.
#[allow(clippy::too_many_arguments)]
pub fn supertrend_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    want_direction: bool,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if state.len() < 5 {
        return None;
    }
    let (trend, fu, fl, pc) = (state[0], state[1], state[2], state[3]);
    let atr = super::atr_resume_one(high, low, close, period, row, &state[4..5])?;
    let hl2 = (high[row] + low[row]) / 2.0;
    let (trend, fu, fl) = supertrend_step(hl2, atr, close[row], mult, trend, fu, fl, pc);
    Some(supertrend_out(trend, fu, fl, want_direction))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::test_support::*;

    /// The cumulative `*_final_state` guards decline on an empty series (no first bar to seed).
    #[test]
    fn cumulative_final_state_declines_on_empty() {
        assert!(wad_final_state(&[], &[], &[]).is_none());
        assert!(asi_final_state(&[], &[], &[], &[], 3.0).is_none());
    }

    /// wad / asi resume, fed the carried state of a full compute over the head `[0, from)`,
    /// reproduces the tail of a full compute over the whole input — bit-for-bit.
    #[test]
    fn cumulative_resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(160);
        let open: Vec<f64> = close.iter().map(|c| c - 0.3).collect();
        let wad_full = wad(&high, &low, &close);
        let asi_full = asi(&open, &high, &low, &close, 3.0);
        for &from in &[1usize, 2, 40, 80, 159] {
            let st = wad_final_state(&high[..from], &low[..from], &close[..from]).unwrap();
            assert_bits(&wad_resume(&high, &low, &close, from, &st).0, &wad_full[from..], "wad");

            let st =
                asi_final_state(&open[..from], &high[..from], &low[..from], &close[..from], 3.0)
                    .unwrap();
            assert_bits(
                &asi_resume(&open, &high, &low, &close, 3.0, from, &st).0,
                &asi_full[from..],
                "asi",
            );
        }
    }

    /// Supertrend blanks before its ATR seeds, and its resume reproduces a full recompute
    /// bit-for-bit — including the recursive band-tightening and trend flips.
    #[test]
    fn supertrend_resume_and_guards() {
        let (high, low, close) = ohlc(160);
        // Too-short frame: the ATR never seeds → an all-NaN line and no carried state.
        assert!(supertrend(&high[..5], &low[..5], &close[..5], 10, 3.0, false)
            .iter()
            .all(|x| x.is_nan()));
        assert!(supertrend_final_state(&high[..5], &low[..5], &close[..5], 10, 3.0).is_none());

        let line_full = supertrend(&high, &low, &close, 10, 3.0, false);
        let dir_full = supertrend(&high, &low, &close, 10, 3.0, true);
        for &from in &[11usize, 20, 80, 159] {
            let st = supertrend_final_state(&high[..from], &low[..from], &close[..from], 10, 3.0)
                .unwrap();
            let (line, _) =
                supertrend_resume(&high, &low, &close, 10, 3.0, false, from, &st).unwrap();
            assert_bits(&line, &line_full[from..], "supertrend");
            let (dir, _) =
                supertrend_resume(&high, &low, &close, 10, 3.0, true, from, &st).unwrap();
            assert_bits(&dir, &dir_full[from..], "supertrend.direction");
        }
    }
}
