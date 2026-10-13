//! Binding a batch kernel that publishes its own terminal continuation states.
use crate::{complete::BoundCommand, exec::{arg_f64, arg_usize}, operands::Operands};
use volas_compute::{pass::KernelPass, raw::{indicators as ind, kernels}};

/// Run recursive batch kernels once, over exactly the recorded complete inputs.
pub(super) fn recursive_pass(df: &volas_core::DataFrame, cmd: &BoundCommand) -> Option<super::states::WithStates> {
    use std::borrow::Cow;
    use volas_compute::missing::{CompleteRows, Input};
    let ops = Operands::new(df, cmd.series);
    ops.probe();
    super::initial_state::state_of(&ops, cmd.name, cmd.sub, cmd.args);
    if ops.failed() { return None; }
    let reads: Vec<_> = ops.reads().collect();
    let inputs: Vec<_> = reads.iter().map(|(_, r)| Input::known(r.view(), r.has_nan())).collect();
    let rows = CompleteRows::of(&inputs);
    let mut views: Vec<Option<Cow<'_, [f64]>>> = vec![None; 8];
    for (slot, r) in &reads { views[*slot] = Some(rows.gather(r.view())); }
    let complete = Operands::over(df, cmd.series, 0, views);
    if complete.len() < 2 { return None; }
    let pass = compute(&complete, cmd)?;
    let n = df.height();
    let before = if reads.iter().any(|(_, r)| r.view()[n - 1].is_nan()) { pass.after.clone() } else { pass.before };
    let after = super::state::capture_read(&ops, cmd, n, pass.after, true)?;
    let before = super::state::capture_read(&ops, cmd, n - 1, before, true)?;
    Some((volas_core::Column::f64(rows.scatter(n, pass.values)), Some(after), Some(before)))
}

pub(super) fn compute(ops: &Operands, cmd: &BoundCommand) -> Option<KernelPass> {
    let p = |i| arg_usize(cmd.args, i);
    let close = || ops.f64(0, "close").ok();
    let origin = ops.origin();
    match (cmd.name, cmd.sub) {
        ("ema" | "dema" | "tema" | "kama" | "t3" | "mama" | "rsi", _) => {
            let x = close()?;
            KernelPass::run(|s| match cmd.name {
                "ema" => ind::ema_pass(&x, p(0), s),
                "dema" => ind::dema_pass(&x, p(0), s),
                "tema" => ind::tema_pass(&x, p(0), s),
                "kama" => ind::kama_pass(&x, p(0), origin, s),
                "t3" => ind::t3_pass(&x, p(0), arg_f64(cmd.args, 1), s),
                "mama" => ind::mama_pass(&x, arg_f64(cmd.args, 0), arg_f64(cmd.args, 1), cmd.sub.is_some(), s),
                _ => ind::rsi_pass(&x, p(0), s),
            })
        }
        ("crsi", _) => {
            let x = close()?;
            KernelPass::run(|s| ind::crsi_pass(&x, p(0), p(1), p(2), s))
        }
        ("stochrsi", _) => stochrsi(ops, cmd),
        ("stoch_momentum", _) => {
            let h = ops.f64(0, "high").ok()?;
            let l = ops.f64(1, "low").ok()?;
            let c = ops.f64(2, "close").ok()?;
            KernelPass::run(|s| ind::stoch_momentum_pass(&h, &l, &c, p(0), p(1), cmd.sub.map(|_| p(2)), s))
        }
        ("ma", _) if ind::generic_ma_type(p(0), p(1)) == 0 => {
            let x = close()?;
            KernelPass::run(|s| kernels::sma_pass((&*x).into(), p(0), origin, s).into_raw_vec_and_offset().0)
        }
        ("ma", _) | ("wma" | "trima" | "sum", _) => {
            let x = close()?;
            let kind = if cmd.name == "ma" { p(1) } else if cmd.name == "wma" { 2 } else { 5 };
            KernelPass::run(|s| {
                if cmd.name == "sum" { ind::sum_pass(&x, p(0), origin, s) }
                else if kind == 2 { ind::wma_pass(&x, p(0), origin, s) }
                else { ind::trima_pass(&x, p(0), origin, s) }
            })
        }
        ("cci", _) => {
            let h = ops.f64(0, "high").ok()?;
            let l = ops.f64(1, "low").ok()?;
            let c = ops.f64(2, "close").ok()?;
            KernelPass::run(|s| ind::cci_pass(&h, &l, &c, p(0), origin, s))
        }
        ("boll" | "accbands", None) => {
            let x = close()?;
            KernelPass::run(|s| kernels::sma_pass((&*x).into(), p(0), origin, s).into_raw_vec_and_offset().0)
        }
        ("boll", Some(_)) | ("bbw", _) => {
            let x = close()?;
            KernelPass::run(|s| {
                let (m, d) = kernels::rolling_mean_std_pass((&*x).into(), p(0), 0, origin, s);
                m.iter().zip(&d).map(|(m, d)| match cmd.sub {
                    Some("upper") => m + arg_f64(cmd.args, 1) * d,
                    Some(_) => m - arg_f64(cmd.args, 1) * d,
                    None => 4.0 * d / m,
                }).collect()
            })
        }
        ("vwma", _) => {
            let x = close()?;
            let v = ops.f64(1, "volume").ok()?;
            KernelPass::run(|s| ind::vwma_pass(&x, &v, p(0), origin, s))
        }
        ("var" | "stddev", _) => {
            let x = close()?;
            KernelPass::run(|s| {
                let mut v = ind::var_pass(&x, p(0), origin, s);
                if cmd.name == "stddev" {
                    for x in &mut v { *x = if x.is_nan() { f64::NAN } else if *x > 0.0 { x.sqrt() * arg_f64(cmd.args, 1) } else { 0.0 }; }
                }
                v
            })
        }
        ("linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf", _) => {
            let x = close()?;
            let line = match cmd.name {
                "linearreg" => ind::LinregLine::Value,
                "linearreg_slope" => ind::LinregLine::Slope,
                "linearreg_intercept" => ind::LinregLine::Intercept,
                "linearreg_angle" => ind::LinregLine::Angle,
                _ => ind::LinregLine::Forecast,
            };
            KernelPass::run(|s| ind::linreg_pass(&x, p(0), line, origin, s))
        }
        ("mfi", _) => {
            let h = ops.f64(0, "high").ok()?;
            let l = ops.f64(1, "low").ok()?;
            let c = ops.f64(2, "close").ok()?;
            let v = ops.f64(3, "volume").ok()?;
            KernelPass::run(|s| ind::mfi_pass(&h, &l, &c, &v, p(0), origin, s))
        }
        ("accbands", Some(band)) => {
            let h = ops.f64(0, "high").ok()?;
            let l = ops.f64(1, "low").ok()?;
            KernelPass::run(|s| if band == "upper" { ind::accbands_upper_pass(&h, &l, p(0), origin, s) } else { ind::accbands_lower_pass(&h, &l, p(0), origin, s) })
        }
        ("hv" | "hma" | "bbi" | "dpo" | "kst" | "coppock" | "cog", _) => {
            let x = close()?;
            KernelPass::run(|s| match cmd.name {
                "hv" => ind::hv_pass(&x, p(0), crate::exec::hv_annualization(cmd.args), origin, s),
                "hma" => ind::hma_pass(&x, p(0), origin, s),
                "bbi" => ind::bbi_pass(&x, p(0), p(1), p(2), p(3), origin, s),
                "dpo" => ind::dpo_pass(&x, p(0), origin, s),
                "kst" => ind::kst_pass(&x, origin, s),
                "coppock" => ind::coppock_pass(&x, p(0), p(1), p(2), origin, s),
                _ => ind::cog_pass(&x, p(0), origin, s),
            })
        }
        ("correl" | "beta" | "vr", _) => {
            let x = close()?;
            let y = if cmd.name == "vr" { ops.f64(1, "volume").ok()? } else { ops.f64_required(1).ok()? };
            KernelPass::run(|s| match cmd.name {
                "correl" => ind::correl_pass(&x, &y, p(0), origin, s),
                "beta" => ind::beta_pass(&x, &y, p(0), origin, s),
                _ => ind::vr_pass(&x, &y, p(0), origin, s),
            })
        }
        ("imi", _) => {
            let o = ops.f64(0, "open").ok()?;
            let c = ops.f64(1, "close").ok()?;
            KernelPass::run(|s| ind::imi_pass(&o, &c, p(0), origin, s))
        }
        ("emv", _) => {
            let h = ops.f64(0, "high").ok()?;
            let l = ops.f64(1, "low").ok()?;
            let v = ops.f64(2, "volume").ok()?;
            KernelPass::run(|s| ind::emv_pass(&h, &l, &v, p(0), origin, s))
        }
        ("cmf" | "ultosc", _) => {
            let h = ops.f64(0, "high").ok()?;
            let l = ops.f64(1, "low").ok()?;
            let c = ops.f64(2, "close").ok()?;
            if cmd.name == "ultosc" {
                KernelPass::run(|s| ind::ultosc_pass(&h, &l, &c, p(0), p(1), p(2), origin, s))
            } else {
                let v = ops.f64(3, "volume").ok()?;
                KernelPass::run(|s| ind::cmf_pass(&h, &l, &c, &v, p(0), origin, s))
            }
        }
        ("relative_vigor" | "dkx" | "wvad", sub) => {
            let o = ops.f64(0, "open").ok()?;
            let h = ops.f64(1, "high").ok()?;
            let l = ops.f64(2, "low").ok()?;
            let c = ops.f64(3, "close").ok()?;
            if cmd.name == "wvad" {
                let v = ops.f64(4, "volume").ok()?;
                KernelPass::run(|s| ind::wvad_pass(&o, &h, &l, &c, &v, p(0), origin, s))
            } else {
                KernelPass::run(|s| match (cmd.name, sub) {
                    ("relative_vigor", None) => ind::relative_vigor_pass(&o, &h, &l, &c, p(0), origin, s),
                    ("relative_vigor", _) => ind::relative_vigor_signal_pass(&o, &h, &l, &c, p(0), origin, s),
                    ("dkx", None) => ind::dkx_pass(&o, &h, &l, &c, origin, s),
                    _ => ind::dkx_ma_pass(&o, &h, &l, &c, p(0), origin, s),
                })
            }
        }
        ("apo" | "ppo" | "bias" | "dma" | "macdext", _) => averages(ops, cmd),
        ("mavp", _) => mavp(ops, cmd),
        ("stoch" | "stochf", _) => stochastic(ops, cmd),
        _ => None,
    }
}

fn stage(stage: ind::MaStage, data: &[f64], origin: usize) -> Option<KernelPass> {
    let mut states = volas_compute::pass::States::default();
    let values = stage.pass(data, origin, &mut states)?;
    Some(KernelPass { values, after: states.after?, before: states.before? })
}

fn averages(ops: &Operands, cmd: &BoundCommand) -> Option<KernelPass> {
    let data = ops.f64(0, "close").ok()?;
    let a = super::window::averages(cmd.name, cmd.sub, cmd.args)?;
    let mut fast = stage(a.fast, &data, ops.origin())?;
    let slow = stage(a.slow, &data, ops.origin())?;
    fast.after.extend(slow.after);
    fast.before.extend(slow.before);
    for (f, s) in fast.values.iter_mut().zip(slow.values) {
        *f = if matches!(cmd.name, "ppo" | "bias") { crate::exec::percent_oscillator(*f, s) } else { *f - s };
    }
    if let Some(smooth) = a.smooth {
        let mut signal = stage(smooth, &fast.values, ops.origin())?;
        let n = data.len();
        signal.after.extend_from_slice(&fast.values[n - smooth.reach()..]);
        signal.before.extend_from_slice(&fast.values[n - 1 - smooth.reach()..n - 1]);
        fast.after.extend(signal.after);
        fast.before.extend(signal.before);
        if cmd.sub == Some("histogram") {
            for (f, s) in fast.values.iter_mut().zip(signal.values) { *f -= s; }
        } else { fast.values = signal.values; }
    }
    Some(fast)
}

fn mavp(ops: &Operands, cmd: &BoundCommand) -> Option<KernelPass> {
    let data = ops.f64(0, "close").ok()?;
    let periods = ops.f64_required(1).ok()?;
    let p = |i| arg_usize(cmd.args, i);
    let (lo, hi, mt) = (p(0), p(1), p(2));
    let lb = ind::generic_ma_lookback(hi, mt);
    let n = data.len();
    let mut result = KernelPass { values: vec![f64::NAN; n], after: vec![n as f64], before: vec![(n - 1) as f64] };
    // Classify the output rows once. Each stage writes only its selected rows,
    // rather than re-reading and clamping every period for every stage.
    let mut selected = vec![Vec::new(); hi - lo + 1];
    for (i, &period) in periods.iter().enumerate().skip(lb) {
        if let Some(q) = ind::mavp_period(period, lo, hi) { selected[q - lo].push(i); }
    }
    for q in lo..=hi {
        let start = lb - ind::generic_ma_lookback(q, mt);
        let pass = stage(ind::MaStage::new(q, mt)?, &data[start..], ops.origin() + start)?;
        result.after.extend(pass.after);
        result.before.extend(pass.before);
        for &i in &selected[q - lo] { result.values[i] = pass.values[i - start]; }
    }
    Some(result)
}

fn stochastic(ops: &Operands, cmd: &BoundCommand) -> Option<KernelPass> {
    let h = ops.f64(0, "high").ok()?;
    let l = ops.f64(1, "low").ok()?;
    let c = ops.f64(2, "close").ok()?;
    let raw = ind::stoch_fastk(&h, &l, &c, arg_usize(cmd.args, 0));
    let (stages, count) = super::window::stoch_stages(cmd.name, cmd.sub, cmd.args);
    let mut result = stage(ind::MaStage::new(stages[0].0, stages[0].1)?, &raw, ops.origin())?;
    if count == 2 {
        let smooth = ind::MaStage::new(stages[1].0, stages[1].1)?;
        let pass = stage(smooth, &result.values, ops.origin())?;
        result.after.extend(pass.after);
        result.before.extend(pass.before);
        let n = raw.len();
        result.after.extend_from_slice(&result.values[n - smooth.reach()..]);
        result.before.extend_from_slice(&result.values[n - 1 - smooth.reach()..n - 1]);
        result.values = pass.values;
    }
    Some(result)
}

fn stochrsi(ops: &Operands, cmd: &BoundCommand) -> Option<KernelPass> {
    let x = ops.f64(0, "close").ok()?;
    let p = |i| arg_usize(cmd.args, i);
    let mut rsi = KernelPass::run(|s| ind::rsi_pass(&x, p(0), s))?;
    let n = x.len();
    let c = p(1) - 1;
    rsi.after.extend_from_slice(&rsi.values[n - c..]);
    rsi.before.extend_from_slice(&rsi.values[n - 1 - c..n - 1]);
    let raw = ind::stochrsi_from_rsi(&rsi.values, p(1));
    if cmd.sub == Some("d") {
        let ma = ind::MaStage::new(p(2), p(3))?;
        let pass = stage(ma, &raw, ops.origin())?;
        rsi.after.extend(pass.after);
        rsi.before.extend(pass.before);
        rsi.after.extend_from_slice(&raw[n - ma.reach()..]);
        rsi.before.extend_from_slice(&raw[n - 1 - ma.reach()..n - 1]);
        rsi.values = pass.values;
    } else { rsi.values = raw; }
    Some(rsi)
}
