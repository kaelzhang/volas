//! Intraday support and resistance levels from the prior bar.
use crate::missing;

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
    cdp_with(high, low, close, line, false)
}

/// Batch levels using the caller's all-finite fact (`true` requires finite inputs).
pub fn cdp_with(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    line: CdpLine,
    complete: bool,
) -> Vec<f64> {
    let base = |h, l, c| (h + l + 2.0 * c) / 4.0;
    // Choose the fixed line once, so each numerical loop has one formula.
    match line {
        CdpLine::Cdp => prior_levels(high, low, close, complete, base),
        CdpLine::Ah => prior_levels(high, low, close, complete, |h, l, c| {
            base(h, l, c) + (h - l)
        }),
        CdpLine::Nh => prior_levels(high, low, close, complete, |h, l, c| {
            2.0 * base(h, l, c) - l
        }),
        CdpLine::Nl => prior_levels(high, low, close, complete, |h, l, c| {
            2.0 * base(h, l, c) - h
        }),
        CdpLine::Al => prior_levels(high, low, close, complete, |h, l, c| {
            base(h, l, c) - (h - l)
        }),
    }
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
    pivot_points_with(high, low, close, line, false)
}

/// Batch levels using the caller's all-finite fact (`true` requires finite inputs).
pub fn pivot_points_with(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    line: PivotLine,
    complete: bool,
) -> Vec<f64> {
    let base = |h, l, c| (h + l + c) / 3.0;
    match line {
        PivotLine::Pp => prior_levels(high, low, close, complete, base),
        PivotLine::R1 => prior_levels(high, low, close, complete, |h, l, c| {
            2.0 * base(h, l, c) - l
        }),
        PivotLine::S1 => prior_levels(high, low, close, complete, |h, l, c| {
            2.0 * base(h, l, c) - h
        }),
        PivotLine::R2 => prior_levels(high, low, close, complete, |h, l, c| {
            base(h, l, c) + (h - l)
        }),
        PivotLine::S2 => prior_levels(high, low, close, complete, |h, l, c| {
            base(h, l, c) - (h - l)
        }),
        PivotLine::R3 => prior_levels(high, low, close, complete, |h, l, c| {
            h + 2.0 * (base(h, l, c) - l)
        }),
        PivotLine::S3 => prior_levels(high, low, close, complete, |h, l, c| {
            l - 2.0 * (h - base(h, l, c))
        }),
    }
}

fn prior_levels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    complete: bool,
    value: impl Fn(f64, f64, f64) -> f64,
) -> Vec<f64> {
    missing::map_terms(
        high.len(),
        1,
        complete,
        |i| [high[i - 1], low[i - 1], close[i - 1]],
        |_, [h, l, c]| value(h, l, c),
    )
}
