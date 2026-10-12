//! The resumable finite-window commands: those whose full computation carries a running
//! accumulator across the series (a sliding sum, sum of squares, weighted sum, …), and
//! the formulas of a row's own inputs (price transforms, TR, the momentum family). Each
//! resumes from the kernel's own state — its running sums in its exact operation order,
//! or nothing — one row at a time, bit-identical to recomputing the column; any other
//! finite-window command recomputes its last window instead (exact for it).
//!
//! The states and steps live next to their kernels in `volas_compute::raw`; this module
//! binds a command's arguments and inputs to them. The commands composed of moving
//! averages here (`apo`, `ppo`, `bias`, `dma`, `macdext`, `mavp`, `ma`) compose
//! [`FiniteMa`] stages, as their dispatch arms compose `ma_typed`.

use crate::exec::{arg_f64, arg_usize, hv_annualization, percent_oscillator};
use crate::operands::Operands;
use crate::types::ArgValue;
use volas_compute::raw::indicators::{self as ind, BollLine, FiniteMa, LinregLine};

/// The resumable state of the finite-window command `name` / `sub` over the rows `ops`
/// reads, or `None` when it carries none (a pure per-window command, or a moving-average
/// type that is recursive).
pub(crate) fn state(ops: &Operands, name: &str, sub: Option<&str>, args: &[ArgValue]) -> Option<Vec<f64>> {
    let p = |i| arg_usize(args, i);
    let close = || ops.f64(0, "close").ok();
    let hl = || Some((ops.f64(0, "high").ok()?, ops.f64(1, "low").ok()?));
    let hlc = || Some((ops.f64(0, "high").ok()?, ops.f64(1, "low").ok()?, ops.f64(2, "close").ok()?));
    let ohlc = || {
        Some((
            ops.f64(0, "open").ok()?,
            ops.f64(1, "high").ok()?,
            ops.f64(2, "low").ok()?,
            ops.f64(3, "close").ok()?,
        ))
    };
    match (name, sub) {
        // a formula of its own row (and a fixed prior one) carries nothing
        ("avgprice" | "medprice" | "typprice" | "wclprice" | "tr" | "bop", _)
        | ("mom" | "roc" | "rocp" | "rocr" | "rocr100", _) => Some(Vec::new()),
        ("ma", _) => ind::ma_state(&close()?, p(0), p(1), ops.origin()),
        ("wma", _) => ind::ma_state(&close()?, p(0), 2, ops.origin()),
        ("trima", _) => ind::trima_state(&close()?, p(0), ops.origin()),
        ("boll", None) => ind::ma_state(&close()?, p(0), 0, ops.origin()),
        ("boll", Some(_)) | ("bbw", _) => ind::boll_band_state(&close()?, p(0), ops.origin()),
        ("accbands", None) => ind::ma_state(&close()?, p(0), 0, ops.origin()),
        ("accbands", Some(band)) => {
            let (h, l) = hl()?;
            ind::accbands_band_state(&h, &l, p(0), band == "upper", ops.origin())
        }
        ("bbi", _) => ind::bbi_state(&close()?, p(0), p(1), p(2), p(3), ops.origin()),
        ("hma", _) => ind::hma_state(&close()?, p(0), ops.origin()),
        ("vwma", _) => ind::vwma_state(&close()?, &ops.f64(1, "volume").ok()?, p(0), ops.origin()),
        ("hv", _) => ind::hv_state(&close()?, p(0), ops.origin()),
        ("var" | "stddev", _) => ind::var_state(&close()?, p(0), ops.origin()),
        ("sum", _) => ind::sum_state(&close()?, p(0), ops.origin()),
        (
            "linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf",
            _,
        ) => ind::linreg_state(&close()?, p(0), ops.origin()),
        ("correl", _) => ind::correl_state(&close()?, &ops.f64_required(1).ok()?, p(0), ops.origin()),
        ("beta", _) => ind::beta_state(&close()?, &ops.f64_required(1).ok()?, p(0), ops.origin()),
        ("cci", _) => {
            let (h, l, c) = hlc()?;
            ind::cci_state(&h, &l, &c, p(0), ops.origin())
        }
        ("mfi", _) => {
            let (h, l, c) = hlc()?;
            ind::mfi_state(&h, &l, &c, &ops.f64(3, "volume").ok()?, p(0), ops.origin())
        }
        ("cog", _) => ind::cog_state(&close()?, p(0), ops.origin()),
        ("imi", _) => ind::imi_state(&ops.f64(0, "open").ok()?, &ops.f64(1, "close").ok()?, p(0), ops.origin()),
        ("ultosc", _) => {
            let (h, l, c) = hlc()?;
            ind::ultosc_state(&h, &l, &c, p(0), p(1), p(2), ops.origin())
        }
        ("cmf", _) => {
            let (h, l, c) = hlc()?;
            ind::cmf_state(&h, &l, &c, &ops.f64(3, "volume").ok()?, p(0), ops.origin())
        }
        ("dpo", _) => ind::dpo_state(&close()?, p(0), ops.origin()),
        ("emv", _) => {
            let (h, l) = hl()?;
            ind::emv_state(&h, &l, &ops.f64(2, "volume").ok()?, p(0), ops.origin())
        }
        ("kst", _) => ind::kst_state(&close()?, ops.origin()),
        ("coppock", _) => ind::coppock_state(&close()?, p(0), p(1), p(2), ops.origin()),
        ("relative_vigor", sub) => {
            let (o, h, l, c) = ohlc()?;
            ind::relative_vigor_state(&o, &h, &l, &c, p(0), sub == Some("signal"), ops.origin())
        }
        ("dkx", sub) => {
            let (o, h, l, c) = ohlc()?;
            ind::dkx_state(&o, &h, &l, &c, (sub == Some("ma")).then(|| p(0)), ops.origin())
        }
        ("wvad", _) => {
            let (o, h, l, c) = ohlc()?;
            ind::wvad_state(&o, &h, &l, &c, &ops.f64(4, "volume").ok()?, p(0), ops.origin())
        }
        ("vr", _) => ind::vr_state(&close()?, &ops.f64(1, "volume").ok()?, p(0), ops.origin()),
        ("stoch", Some("k" | "d")) | ("stochf", Some("d")) => {
            let (h, l, c) = hlc()?;
            let (stages, k) = stoch_stages(name, sub, args);
            ind::stoch_state(&h, &l, &c, p(0), &stages[..k], ops.origin())
        }
        ("apo" | "ppo" | "bias" | "dma" | "macdext" | "mavp", _) => {
            let mut prev = composite_initial(name, sub, args, ops.origin())?;
            let mut next = prev.clone();
            for row in 0..close()?.len() {
                composite_step(ops, name, sub, args, &prev, row, &mut next)?;
                std::mem::swap(&mut prev, &mut next);
            }
            Some(prev)
        }
        _ => None,
    }
}

/// One row of the finite-window command `name` / `sub` at `row`, from `prev` into `next`
/// (see [`state`]). `None` when it has no step.
pub(crate) fn step(
    ops: &Operands,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    prev: &[f64],
    row: usize,
    next: &mut [f64],
) -> Option<f64> {
    let p = |i| arg_usize(args, i);
    let close = || ops.f64(0, "close").ok();
    let hl = || Some((ops.f64(0, "high").ok()?, ops.f64(1, "low").ok()?));
    let hlc = || Some((ops.f64(0, "high").ok()?, ops.f64(1, "low").ok()?, ops.f64(2, "close").ok()?));
    let ohlc = || {
        Some((
            ops.f64(0, "open").ok()?,
            ops.f64(1, "high").ok()?,
            ops.f64(2, "low").ok()?,
            ops.f64(3, "close").ok()?,
        ))
    };
    if row >= ops.len() {
        return None;
    }
    match (name, sub) {
        // a formula of its own row (and a fixed prior one)
        ("avgprice", _) => {
            let (o, h, l, c) = ohlc()?;
            Some((o[row] + h[row] + l[row] + c[row]) / 4.0)
        }
        ("medprice", _) => {
            let (h, l) = hl()?;
            Some((h[row] + l[row]) / 2.0)
        }
        ("bop", _) => {
            let (o, h, l, c) = ohlc()?;
            let range = h[row] - l[row];
            Some(if range < 1e-14 { 0.0 } else { (c[row] - o[row]) / range })
        }
        ("typprice", _) => {
            let (h, l, c) = hlc()?;
            Some((h[row] + l[row] + c[row]) / 3.0)
        }
        ("wclprice", _) => {
            let (h, l, c) = hlc()?;
            Some((h[row] + l[row] + 2.0 * c[row]) / 4.0)
        }
        ("tr", _) => {
            let (h, l, c) = hlc()?;
            Some(if row == 0 { f64::NAN } else { ind::tr_at(&h, &l, &c, row) })
        }
        ("mom" | "roc" | "rocp" | "rocr" | "rocr100", _) => Some(ind::Momentum::of(name)?.at(&close()?, p(0), row)),
        ("ma", _) => ind::ma_step(&close()?, p(0), p(1), row, prev, next),
        ("wma", _) => ind::ma_step(&close()?, p(0), 2, row, prev, next),
        ("trima", _) => ind::trima_step(&close()?, p(0), row, prev, next),
        ("boll", None) => ind::ma_step(&close()?, p(0), 0, row, prev, next),
        ("boll", Some(band)) => {
            let line = if band == "upper" { BollLine::Upper } else { BollLine::Lower };
            ind::boll_band_step(&close()?, p(0), arg_f64(args, 1), line, row, prev, next)
        }
        ("bbw", _) => ind::boll_band_step(&close()?, p(0), 0.0, BollLine::Width, row, prev, next),
        ("accbands", None) => ind::ma_step(&close()?, p(0), 0, row, prev, next),
        ("accbands", Some(band)) => {
            let (h, l) = hl()?;
            ind::accbands_band_step(&h, &l, p(0), band == "upper", row, prev, next)
        }
        ("bbi", _) => ind::bbi_step(&close()?, p(0), p(1), p(2), p(3), row, prev, next),
        ("hma", _) => ind::hma_step(&close()?, p(0), row, prev, next),
        ("vwma", _) => ind::vwma_step(&close()?, &ops.f64(1, "volume").ok()?, p(0), row, prev, next),
        ("hv", _) => ind::hv_step(&close()?, p(0), hv_annualization(args), row, prev, next),
        ("var", _) => ind::var_step(&close()?, p(0), None, row, prev, next),
        ("stddev", _) => ind::var_step(&close()?, p(0), Some(arg_f64(args, 1)), row, prev, next),
        ("sum", _) => ind::sum_step(&close()?, p(0), row, prev, next),
        ("linearreg", _) => ind::linreg_step(&close()?, p(0), LinregLine::Value, row, prev, next),
        ("linearreg_slope", _) => ind::linreg_step(&close()?, p(0), LinregLine::Slope, row, prev, next),
        ("linearreg_intercept", _) => {
            ind::linreg_step(&close()?, p(0), LinregLine::Intercept, row, prev, next)
        }
        ("linearreg_angle", _) => ind::linreg_step(&close()?, p(0), LinregLine::Angle, row, prev, next),
        ("tsf", _) => ind::linreg_step(&close()?, p(0), LinregLine::Forecast, row, prev, next),
        ("correl", _) => {
            ind::correl_step(&close()?, &ops.f64_required(1).ok()?, p(0), row, prev, next)
        }
        ("beta", _) => ind::beta_step(&close()?, &ops.f64_required(1).ok()?, p(0), row, prev, next),
        ("cci", _) => {
            let (h, l, c) = hlc()?;
            ind::cci_step(&h, &l, &c, p(0), row, prev, next)
        }
        ("mfi", _) => {
            let (h, l, c) = hlc()?;
            ind::mfi_step(&h, &l, &c, &ops.f64(3, "volume").ok()?, p(0), row, prev, next)
        }
        ("cog", _) => ind::cog_step(&close()?, p(0), row, prev, next),
        ("imi", _) => ind::imi_step(&ops.f64(0, "open").ok()?, &ops.f64(1, "close").ok()?, p(0), row, prev, next),
        ("ultosc", _) => {
            let (h, l, c) = hlc()?;
            ind::ultosc_step(&h, &l, &c, p(0), p(1), p(2), row, prev, next)
        }
        ("cmf", _) => {
            let (h, l, c) = hlc()?;
            ind::cmf_step(&h, &l, &c, &ops.f64(3, "volume").ok()?, p(0), row, prev, next)
        }
        ("dpo", _) => ind::dpo_step(&close()?, p(0), row, prev, next),
        ("emv", _) => {
            let (h, l) = hl()?;
            ind::emv_step(&h, &l, &ops.f64(2, "volume").ok()?, p(0), row, prev, next)
        }
        ("kst", _) => ind::kst_step(&close()?, row, prev, next),
        ("coppock", _) => ind::coppock_step(&close()?, p(0), p(1), p(2), row, prev, next),
        ("relative_vigor", sub) => {
            let (o, h, l, c) = ohlc()?;
            ind::relative_vigor_step(&o, &h, &l, &c, p(0), sub == Some("signal"), row, prev, next)
        }
        ("dkx", sub) => {
            let (o, h, l, c) = ohlc()?;
            ind::dkx_step(&o, &h, &l, &c, (sub == Some("ma")).then(|| p(0)), row, prev, next)
        }
        ("wvad", _) => {
            let (o, h, l, c) = ohlc()?;
            ind::wvad_step(&o, &h, &l, &c, &ops.f64(4, "volume").ok()?, p(0), row, prev, next)
        }
        ("vr", _) => ind::vr_step(&close()?, &ops.f64(1, "volume").ok()?, p(0), row, prev, next),
        ("stoch", Some("k" | "d")) | ("stochf", Some("d")) => {
            let (h, l, c) = hlc()?;
            let (stages, k) = stoch_stages(name, sub, args);
            ind::stoch_step(&h, &l, &c, p(0), &stages[..k], row, prev, next)
        }
        ("apo" | "ppo" | "bias" | "dma" | "macdext" | "mavp", _) => {
            composite_step(ops, name, sub, args, prev, row, next)
        }
        _ => None,
    }
}

/// A stochastic line's smoothing stages `(period, matype)` (see `ind::stoch_step`) and
/// how many of them there are.
fn stoch_stages(name: &str, sub: Option<&str>, args: &[ArgValue]) -> ([(usize, usize); 2], usize) {
    let p = |i| arg_usize(args, i);
    match (name, sub) {
        ("stoch", Some("d")) => ([(p(1), p(2)), (p(3), p(4))], 2),
        _ => ([(p(1), p(2)), (0, 0)], 1),
    }
}

/// The averages a composite command is made of: the line's two `(fast, slow)` and, for
/// a smoothed line, the smoother — as their dispatch arms build them.
struct Averages {
    fast: FiniteMa,
    slow: FiniteMa,
    smooth: Option<FiniteMa>,
}

fn averages(name: &str, sub: Option<&str>, args: &[ArgValue]) -> Option<Averages> {
    let p = |i| arg_usize(args, i);
    let ma = FiniteMa::new;
    Some(match (name, sub) {
        ("bias", _) => Averages { fast: ma(1, 0)?, slow: ma(p(0), 0)?, smooth: None },
        ("dma", sub) => Averages {
            fast: ma(p(0), 0)?,
            slow: ma(p(1), 0)?,
            smooth: if sub.is_some() { Some(ma(p(2), 0)?) } else { None },
        },
        ("macdext", sub) => {
            let ((fp, fm), (sp, sm)) = ind::oscillator_order((p(0), p(1)), (p(2), p(3)));
            Averages {
                fast: ma(fp, fm)?,
                slow: ma(sp, sm)?,
                smooth: if sub.is_some() { Some(ma(p(4), p(5))?) } else { None },
            }
        }
        // apo / ppo, in TA-Lib's order
        _ => {
            let ((fp, fm), (sp, sm)) = ind::oscillator_order((p(0), p(2)), (p(1), p(2)));
            Averages { fast: ma(fp, fm)?, slow: ma(sp, sm)?, smooth: None }
        }
    })
}

/// The length of a composite command's state (`mavp`: one average per period).
fn composite_len(name: &str, sub: Option<&str>, args: &[ArgValue]) -> Option<usize> {
    if name == "mavp" {
        let (lo, hi, mt) = (arg_usize(args, 0), arg_usize(args, 1), arg_usize(args, 2));
        let stages: Option<usize> = (lo..=hi).map(|q| Some(FiniteMa::new(q, mt)?.state_len())).sum();
        return Some(1 + stages?);
    }
    let a = averages(name, sub, args)?;
    let smooth = a.smooth.map_or(0, |s| s.chained_len());
    Some(a.fast.state_len() + a.slow.state_len() + smooth)
}

/// A composite command's state before its first row, at absolute position `origin`:
/// each average's own (a `mavp` average starts at its own first row, see
/// [`mavp_step`]).
fn composite_initial(name: &str, sub: Option<&str>, args: &[ArgValue], origin: usize) -> Option<Vec<f64>> {
    if name == "mavp" {
        let (lo, hi, mt) = (arg_usize(args, 0), arg_usize(args, 1), arg_usize(args, 2));
        let lb = ind::generic_ma_lookback(hi, mt);
        let mut v = vec![0.0];
        for q in lo..=hi {
            v.extend(FiniteMa::new(q, mt)?.initial(origin + lb - ind::generic_ma_lookback(q, mt)));
        }
        return Some(v);
    }
    let a = averages(name, sub, args)?;
    let mut v = a.fast.initial(origin);
    v.extend(a.slow.initial(origin));
    if let Some(smooth) = a.smooth {
        v.extend(smooth.chained_initial(origin));
    }
    Some(v)
}

/// One row of a composite command at `row`, from `prev` into `next`.
fn composite_step(
    ops: &Operands,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    prev: &[f64],
    row: usize,
    next: &mut [f64],
) -> Option<f64> {
    if prev.len() != composite_len(name, sub, args)? || next.len() != prev.len() {
        return None;
    }
    let data = ops.f64(0, "close").ok()?;
    if row >= data.len() {
        return None;
    }
    if name == "mavp" {
        return mavp_step(&data, &ops.f64_required(1).ok()?, args, prev, row, next);
    }
    let a = averages(name, sub, args)?;
    let term = |j: usize| data[j];
    let (lf, ls) = (a.fast.state_len(), a.slow.state_len());
    let (n_fast, rest) = next.split_at_mut(lf);
    let (n_slow, n_rest) = rest.split_at_mut(ls);
    let f = a.fast.step(row, term, &prev[..lf], n_fast)?;
    let s = a.slow.step(row, term, &prev[lf..lf + ls], n_slow)?;
    Some(match (name, a.smooth) {
        ("ppo" | "bias", _) => percent_oscillator(f, s),
        (_, None) => f - s,
        (_, Some(smooth)) => {
            let line = f - s;
            let signal = smooth.step_chained(row, line, &prev[lf + ls..], n_rest)?;
            match (name, sub) {
                ("macdext", Some("histogram")) => line - signal,
                _ => signal,
            }
        }
    })
}

/// One row of `mavp` at `row`: every period's average advances (each started where its
/// first value lands at the output start, as the dispatch arm does), and the row reads
/// the one its `periods` value selects ([`ind::mavp_period`]; none for a missing one).
/// `prev[0]` counts the rows seen.
fn mavp_step(
    real: &[f64],
    periods: &[f64],
    args: &[ArgValue],
    prev: &[f64],
    row: usize,
    next: &mut [f64],
) -> Option<f64> {
    let (lo, hi, mt) = (arg_usize(args, 0), arg_usize(args, 1), arg_usize(args, 2));
    let lb = ind::generic_ma_lookback(hi, mt);
    let seen = prev[0] as usize;
    next[0] = (seen + 1) as f64;
    let mut at = 1;
    let mut value = f64::NAN;
    let want = ind::mavp_period(periods[row], lo, hi);
    for q in lo..=hi {
        let stage = FiniteMa::new(q, mt)?;
        let len = stage.state_len();
        let (s, n) = (&prev[at..at + len], &mut next[at..at + len]);
        at += len;
        let start = lb - ind::generic_ma_lookback(q, mt);
        if seen < start {
            n.copy_from_slice(s);
            continue;
        }
        // the stage's own row `seen - start` is this frame's `row`
        let off = seen - start;
        let v = stage.step(off, |j| real[row + j - off], s, n)?;
        if Some(q) == want {
            value = v;
        }
    }
    Some(if seen >= lb { value } else { f64::NAN })
}
