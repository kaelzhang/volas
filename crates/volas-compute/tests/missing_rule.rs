//! The public recursive functions apply the missing-row rule.
//!
//! For every recursive function of `volas_compute::{indicators, kernels}` — full
//! computations, carried-state captures, resumes and single-row resumes — over inputs
//! with a leading run of missing rows (staggered across the inputs) and interior
//! missing rows (in every input, and in one input only):
//!
//! - a full computation equals the raw function over the series with every row missing
//!   any of its inputs deleted, NaN put back at the deleted positions;
//! - a state capture equals the raw capture over the deleted series;
//! - a resume from a state equals the raw resume over the deleted series from the
//!   matching row, its missing new rows NaN;
//! - a single-row resume equals the raw one over the deleted series, and at a missing
//!   row is NaN with the state not advanced.
//!
//! All bit-exact. Generated from the declarations in `src/indicators.rs`.

use ndarray::{Array1, ArrayView1};
use volas_compute::indicators::{self as ind, KdjLine};
use volas_compute::kernels as kern;
use volas_compute::raw::{indicators as raw, kernels as raw_kern};

const N: usize = 400;
const FROM: usize = 300;
/// A complete row right after a row missing every input.
const ROW: usize = 304;
const MISSING_ROW: usize = 303;
/// A new row missing one input (`close`).
const MISSING_CLOSE: usize = 310;

struct Data {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

fn data() -> Data {
    let t = |i: usize| i as f64;
    let close: Vec<f64> = (0..N)
        .map(|i| 100.0 + 0.05 * t(i) + 6.0 * (t(i) * 0.3).sin() + 2.0 * (t(i) * 0.11).cos())
        .collect();
    let open: Vec<f64> = (0..N).map(|i| close[i] - 0.5 * (t(i) * 0.7).sin()).collect();
    let high = (0..N).map(|i| close[i].max(open[i]) + 1.0 + (t(i) * 0.37).sin().abs()).collect();
    let low = (0..N).map(|i| close[i].min(open[i]) - 1.0 - (t(i) * 0.23).cos().abs()).collect();
    let volume = (0..N).map(|i| 1000.0 + 300.0 * (t(i) * 0.17).sin().abs()).collect();
    let mut d = Data { open, high, low, close, volume };
    let nan = f64::NAN;
    for c in [&mut d.open, &mut d.high, &mut d.low, &mut d.close, &mut d.volume] {
        for i in [0, 1, 2, 100, 230, MISSING_ROW] {
            c[i] = nan; // a leading run, an interior row and a new row, in every input
        }
    }
    d.close[3] = nan; // the leading run, staggered
    d.high[4] = nan;
    d.close[150] = nan; // interior rows of one input
    d.close[151] = nan;
    d.high[200] = nan;
    d.low[201] = nan;
    d.volume[250] = nan;
    d.open[260] = nan;
    d.close[310] = nan; // a new row of one input
    d
}

/// The rows where every input is present.
fn kept(x: &[&[f64]]) -> Vec<usize> {
    (0..x[0].len()).filter(|&i| x.iter().all(|c| !c[i].is_nan())).collect()
}

fn deleted(x: &[&[f64]], rows: &[usize]) -> Vec<Vec<f64>> {
    x.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect()
}

fn refs(x: &[Vec<f64>]) -> Vec<&[f64]> {
    x.iter().map(|c| c.as_slice()).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| if x.is_nan() { u64::MAX } else { x.to_bits() }).collect()
}

/// Outputs as a list of series.
trait Flat {
    fn flat(self) -> Vec<Vec<f64>>;
}
impl Flat for Vec<f64> {
    fn flat(self) -> Vec<Vec<f64>> {
        vec![self]
    }
}
impl Flat for (Vec<f64>, Vec<f64>) {
    fn flat(self) -> Vec<Vec<f64>> {
        vec![self.0, self.1]
    }
}
impl Flat for Option<Vec<f64>> {
    fn flat(self) -> Vec<Vec<f64>> {
        self.into_iter().collect()
    }
}
impl Flat for Array1<f64> {
    fn flat(self) -> Vec<Vec<f64>> {
        vec![self.to_vec()]
    }
}

/// A resume's output, fallible or not.
trait IntoOption {
    fn into_option(self) -> Option<(Vec<f64>, Vec<f64>)>;
}
impl IntoOption for (Vec<f64>, Vec<f64>) {
    fn into_option(self) -> Option<(Vec<f64>, Vec<f64>)> {
        Some(self)
    }
}
impl IntoOption for Option<(Vec<f64>, Vec<f64>)> {
    fn into_option(self) -> Option<(Vec<f64>, Vec<f64>)> {
        self
    }
}

type Series<'a> = &'a dyn Fn(&[&[f64]]) -> Vec<Vec<f64>>;

fn series(name: &str, x: &[&[f64]], got: Series, raw: Series) {
    let rows = kept(x);
    let got = got(x);
    let want = raw(&refs(&deleted(x, &rows)));
    assert_eq!(got.len(), want.len(), "{name}");
    for (g, w) in got.iter().zip(want) {
        let mut full = vec![f64::NAN; x[0].len()];
        for (&i, v) in rows.iter().zip(w) {
            full[i] = v;
        }
        assert_eq!(bits(g), bits(&full), "{name}");
    }
}

fn state(name: &str, x: &[&[f64]], got: &dyn Fn(&[&[f64]]) -> String, raw: &dyn Fn(&[&[f64]]) -> String) {
    let rows = kept(x);
    let want = raw(&refs(&deleted(x, &rows)));
    assert!(!want.contains("None"), "{name}: no state to compare");
    assert_eq!(got(x), want, "{name}");
}

type StateOf<'a> = &'a dyn Fn(&[&[f64]]) -> Option<Vec<f64>>;
type Resume<'a> = &'a dyn Fn(&[&[f64]], usize, &[f64]) -> Option<(Vec<f64>, Vec<f64>)>;

fn resume(name: &str, x: &[&[f64]], state_of: StateOf, got: Resume, raw: Resume) {
    let rows = kept(x);
    let at = rows.partition_point(|&i| i < FROM);
    let compact = deleted(x, &rows);
    let head: Vec<&[f64]> = compact.iter().map(|c| &c[..at]).collect();
    let st = state_of(&head).unwrap_or_else(|| panic!("{name}: no state"));
    let (out, next) = raw(&refs(&compact), at, &st).unwrap_or_else(|| panic!("{name}: raw declined"));
    let mut full = vec![f64::NAN; N - FROM];
    for (k, v) in out.into_iter().enumerate() {
        full[rows[at + k] - FROM] = v;
    }
    let got = got(x, FROM, &st).map(|(g, s)| (bits(&g), bits(&s)));
    assert_eq!(got, Some((bits(&full), bits(&next))), "{name}");
}

type Step<'a> = &'a dyn Fn(&[&[f64]], usize, &[f64]) -> String;

fn step(name: &str, x: &[&[f64]], state_of: StateOf, got: Step, raw: Step, missing: &dyn Fn(&[f64]) -> String) {
    let rows = kept(x);
    let at = rows.partition_point(|&i| i < ROW);
    let compact = deleted(x, &rows);
    let head: Vec<&[f64]> = compact.iter().map(|c| &c[..at]).collect();
    let st = state_of(&head).unwrap_or_else(|| panic!("{name}: no state"));
    let upto: Vec<&[f64]> = compact.iter().map(|c| &c[..=at]).collect();
    assert_eq!(rows[at], ROW, "{name}: the row must be complete");
    assert_eq!(got(x, ROW, &st), raw(&upto, at, &st), "{name}");
    assert!(!got(x, ROW, &st).contains("NaN"), "{name}: a value to compare");
    assert_eq!(got(x, MISSING_ROW, &st), missing(&st), "{name}: a missing row");
    if x.iter().any(|c| c[MISSING_CLOSE].is_nan()) {
        assert_eq!(got(x, MISSING_CLOSE, &st), missing(&st), "{name}: a row missing one input");
    }
    assert_eq!(got(x, N, &st), raw(x, N, &st), "{name}: past the end");
}

#[test]
fn a_full_computation_is_the_raw_one_over_the_complete_rows() {
    let d = data();
    series("ad", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x| ind::ad(x[0],
        x[1], x[2], x[3]).flat(), &|x| raw::ad(x[0], x[1], x[2], x[3]).flat());
    series("adosc", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x|
        ind::adosc(x[0], x[1], x[2], x[3], 3, 6).flat(), &|x| raw::adosc(x[0], x[1], x[2],
        x[3], 3, 6).flat());
    series("adx", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::adx(x[0], x[1], x[2],
        5).flat(), &|x| raw::adx(x[0], x[1], x[2], 5).flat());
    series("adxr", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::adxr(x[0], x[1],
        x[2], 5).flat(), &|x| raw::adxr(x[0], x[1], x[2], 5).flat());
    series("asi", &[&d.open[..], &d.high[..], &d.low[..], &d.close[..]], &|x| ind::asi(x[0],
        x[1], x[2], x[3], 300.0).flat(), &|x| raw::asi(x[0], x[1], x[2], x[3],
        300.0).flat());
    series("atr", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::atr(x[0], x[1], x[2],
        5).flat(), &|x| raw::atr(x[0], x[1], x[2], 5).flat());
    series("cmo", &[&d.close[..]], &|x| ind::cmo(x[0], 5).flat(), &|x| raw::cmo(x[0], 5).flat());
    series("crsi", &[&d.close[..]], &|x| ind::crsi(x[0], 3, 2, 20).flat(), &|x|
        raw::crsi(x[0], 3, 2, 20).flat());
    series("dema", &[&d.close[..]], &|x| ind::dema(x[0], 5).flat(), &|x| raw::dema(x[0], 5).flat());
    series("dx", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::dx(x[0], x[1], x[2],
        5).flat(), &|x| raw::dx(x[0], x[1], x[2], 5).flat());
    series("efi", &[&d.close[..], &d.volume[..]], &|x| ind::efi(x[0], x[1], 5).flat(), &|x|
        raw::efi(x[0], x[1], 5).flat());
    series("ema", &[&d.close[..]], &|x| ind::ema(x[0], 5).flat(), &|x| raw::ema(x[0], 5).flat());
    series("ht_dcperiod", &[&d.close[..]], &|x| ind::ht_dcperiod(x[0]).flat(), &|x|
        raw::ht_dcperiod(x[0]).flat());
    series("ht_dcphase", &[&d.close[..]], &|x| ind::ht_dcphase(x[0]).flat(), &|x|
        raw::ht_dcphase(x[0]).flat());
    series("ht_phasor", &[&d.close[..]], &|x| ind::ht_phasor(x[0]).flat(), &|x|
        raw::ht_phasor(x[0]).flat());
    series("ht_phasor_line", &[&d.close[..]], &|x| ind::ht_phasor_line(x[0], true).flat(),
        &|x| raw::ht_phasor_line(x[0], true).flat());
    series("ht_sine", &[&d.close[..]], &|x| ind::ht_sine(x[0]).flat(), &|x|
        raw::ht_sine(x[0]).flat());
    series("ht_trendline", &[&d.close[..]], &|x| ind::ht_trendline(x[0]).flat(), &|x|
        raw::ht_trendline(x[0]).flat());
    series("ht_trendmode", &[&d.close[..]], &|x| ind::ht_trendmode(x[0]).flat(), &|x|
        raw::ht_trendmode(x[0]).flat());
    series("kama", &[&d.close[..]], &|x| ind::kama(x[0], 5).flat(), &|x| raw::kama(x[0], 5).flat());
    series("kcw", &[&d.close[..], &d.high[..], &d.low[..]], &|x| ind::kcw(x[0], x[1], x[2],
        5, 5, 2.0).flat(), &|x| raw::kcw(x[0], x[1], x[2], 5, 5, 2.0).flat());
    series("kdj_d", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::kdj_d(x[0], x[1],
        x[2], 5, 3, 3, 50.0).flat(), &|x| raw::kdj_d(x[0], x[1], x[2], 5, 3, 3,
        50.0).flat());
    series("kdj_j", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::kdj_j(x[0], x[1],
        x[2], 5, 3, 3, 50.0).flat(), &|x| raw::kdj_j(x[0], x[1], x[2], 5, 3, 3,
        50.0).flat());
    series("kdj_k", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::kdj_k(x[0], x[1],
        x[2], 5, 3, 50.0).flat(), &|x| raw::kdj_k(x[0], x[1], x[2], 5, 3, 50.0).flat());
    series("keltner_band", &[&d.close[..], &d.high[..], &d.low[..]], &|x|
        ind::keltner_band(x[0], x[1], x[2], 5, 5, 2.0, true).flat(), &|x|
        raw::keltner_band(x[0], x[1], x[2], 5, 5, 2.0, true).flat());
    series("macd", &[&d.close[..]], &|x| ind::macd(x[0], 3, 6).flat(), &|x| raw::macd(x[0],
        3, 6).flat());
    series("macd_histogram", &[&d.close[..]], &|x| ind::macd_histogram(x[0], 3, 6,
        4).flat(), &|x| raw::macd_histogram(x[0], 3, 6, 4).flat());
    series("macd_signal", &[&d.close[..]], &|x| ind::macd_signal(x[0], 3, 6, 4).flat(), &|x|
        raw::macd_signal(x[0], 3, 6, 4).flat());
    series("mama", &[&d.close[..]], &|x| ind::mama(x[0], 0.5, 0.05).flat(), &|x|
        raw::mama(x[0], 0.5, 0.05).flat());
    series("mama_line", &[&d.close[..]], &|x| ind::mama_line(x[0], 0.5, 0.05, true).flat(),
        &|x| raw::mama_line(x[0], 0.5, 0.05, true).flat());
    series("mass_index", &[&d.high[..], &d.low[..]], &|x| ind::mass_index(x[0], x[1],
        5).flat(), &|x| raw::mass_index(x[0], x[1], 5).flat());
    series("minus_di", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::minus_di(x[0],
        x[1], x[2], 5).flat(), &|x| raw::minus_di(x[0], x[1], x[2], 5).flat());
    series("minus_dm", &[&d.high[..], &d.low[..]], &|x| ind::minus_dm(x[0], x[1], 5).flat(),
        &|x| raw::minus_dm(x[0], x[1], 5).flat());
    series("natr", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::natr(x[0], x[1],
        x[2], 5).flat(), &|x| raw::natr(x[0], x[1], x[2], 5).flat());
    series("nvi", &[&d.close[..], &d.volume[..]], &|x| ind::nvi(x[0], x[1]).flat(), &|x|
        raw::nvi(x[0], x[1]).flat());
    series("obv", &[&d.close[..], &d.volume[..]], &|x| ind::obv(x[0], x[1]).flat(), &|x|
        raw::obv(x[0], x[1]).flat());
    series("plus_di", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::plus_di(x[0],
        x[1], x[2], 5).flat(), &|x| raw::plus_di(x[0], x[1], x[2], 5).flat());
    series("plus_dm", &[&d.high[..], &d.low[..]], &|x| ind::plus_dm(x[0], x[1], 5).flat(),
        &|x| raw::plus_dm(x[0], x[1], 5).flat());
    series("pvi", &[&d.close[..], &d.volume[..]], &|x| ind::pvi(x[0], x[1]).flat(), &|x|
        raw::pvi(x[0], x[1]).flat());
    series("pvt", &[&d.close[..], &d.volume[..]], &|x| ind::pvt(x[0], x[1]).flat(), &|x|
        raw::pvt(x[0], x[1]).flat());
    series("rsi", &[&d.close[..]], &|x| ind::rsi(x[0], 5).flat(), &|x| raw::rsi(x[0], 5).flat());
    series("sar", &[&d.high[..], &d.low[..]], &|x| ind::sar(x[0], x[1], 0.02, 0.2).flat(),
        &|x| raw::sar(x[0], x[1], 0.02, 0.2).flat());
    series("sarext", &[&d.high[..], &d.low[..]], &|x| ind::sarext(x[0], x[1], 0.0, 0.0,
        0.02, 0.02, 0.2, 0.02, 0.02, 0.2).flat(), &|x| raw::sarext(x[0], x[1], 0.0, 0.0,
        0.02, 0.02, 0.2, 0.02, 0.02, 0.2).flat());
    series("smma", &[&d.close[..]], &|x| ind::smma(x[0], 5).flat(), &|x| raw::smma(x[0], 5).flat());
    series("stoch_momentum", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        ind::stoch_momentum(x[0], x[1], x[2], 5, 3).flat(), &|x| raw::stoch_momentum(x[0],
        x[1], x[2], 5, 3).flat());
    series("stoch_momentum_signal", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        ind::stoch_momentum_signal(x[0], x[1], x[2], 5, 3, 4).flat(), &|x|
        raw::stoch_momentum_signal(x[0], x[1], x[2], 5, 3, 4).flat());
    series("stochrsi_d_default_sma", &[&d.close[..]], &|x|
        ind::stochrsi_d_default_sma(x[0]).flat(), &|x|
        raw::stochrsi_d_default_sma(x[0]).flat());
    series("stochrsi_fastk", &[&d.close[..]], &|x| ind::stochrsi_fastk(x[0], 5, 3).flat(),
        &|x| raw::stochrsi_fastk(x[0], 5, 3).flat());
    series("supertrend", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        ind::supertrend(x[0], x[1], x[2], 5, 2.0, false).flat(), &|x| raw::supertrend(x[0],
        x[1], x[2], 5, 2.0, false).flat());
    series("t3", &[&d.close[..]], &|x| ind::t3(x[0], 5, 0.7).flat(), &|x| raw::t3(x[0], 5,
        0.7).flat());
    series("tema", &[&d.close[..]], &|x| ind::tema(x[0], 5).flat(), &|x| raw::tema(x[0], 5).flat());
    series("trix", &[&d.close[..]], &|x| ind::trix(x[0], 5).flat(), &|x| raw::trix(x[0], 5).flat());
    series("tsi", &[&d.close[..]], &|x| ind::tsi(x[0], 6, 3).flat(), &|x| raw::tsi(x[0], 6,
        3).flat());
    series("ttm_squeeze_momentum", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        ind::ttm_squeeze_momentum(x[0], x[1], x[2], 5).flat(), &|x|
        raw::ttm_squeeze_momentum(x[0], x[1], x[2], 5).flat());
    series("ttm_squeeze_on", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        ind::ttm_squeeze_on(x[0], x[1], x[2], 5, 2.0, 1.5).flat(), &|x|
        raw::ttm_squeeze_on(x[0], x[1], x[2], 5, 2.0, 1.5).flat());
    series("wad", &[&d.high[..], &d.low[..], &d.close[..]], &|x| ind::wad(x[0], x[1],
        x[2]).flat(), &|x| raw::wad(x[0], x[1], x[2]).flat());
}

#[test]
fn a_state_describes_the_last_complete_row() {
    let d = data();
    state("ad_final_state", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x|
        format!("{:?}", ind::ad_final_state(x[0], x[1], x[2], x[3])), &|x| format!("{:?}",
        raw::ad_final_state(x[0], x[1], x[2], x[3])));
    state("adosc_final_state", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x|
        format!("{:?}", ind::adosc_final_state(x[0], x[1], x[2], x[3], 3, 6)), &|x|
        format!("{:?}", raw::adosc_final_state(x[0], x[1], x[2], x[3], 3, 6)));
    state("adx_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x| format!("{:?}",
        ind::adx_final_state(x[0], x[1], x[2], 5)), &|x| format!("{:?}",
        raw::adx_final_state(x[0], x[1], x[2], 5)));
    state("adxr_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x| format!("{:?}",
        ind::adxr_final_state(x[0], x[1], x[2], 5)), &|x| format!("{:?}",
        raw::adxr_final_state(x[0], x[1], x[2], 5)));
    state("asi_final_state", &[&d.open[..], &d.high[..], &d.low[..], &d.close[..]], &|x|
        format!("{:?}", ind::asi_final_state(x[0], x[1], x[2], x[3], 300.0)), &|x|
        format!("{:?}", raw::asi_final_state(x[0], x[1], x[2], x[3], 300.0)));
    state("atr_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x| format!("{:?}",
        ind::atr_final_state(x[0], x[1], x[2], 5)), &|x| format!("{:?}",
        raw::atr_final_state(x[0], x[1], x[2], 5)));
    state("cmo_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::cmo_final_state(x[0], 5)), &|x| format!("{:?}", raw::cmo_final_state(x[0],
        5)));
    state("dema_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::dema_final_state(x[0], 5)), &|x| format!("{:?}", raw::dema_final_state(x[0],
        5)));
    state("dx_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x| format!("{:?}",
        ind::dx_final_state(x[0], x[1], x[2], 5)), &|x| format!("{:?}",
        raw::dx_final_state(x[0], x[1], x[2], 5)));
    state("efi_final_state", &[&d.close[..], &d.volume[..]], &|x| format!("{:?}",
        ind::efi_final_state(x[0], x[1], 5)), &|x| format!("{:?}",
        raw::efi_final_state(x[0], x[1], 5)));
    state("ema_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::ema_final_state(x[0], 5)), &|x| format!("{:?}", raw::ema_final_state(x[0],
        5)));
    state("ht_core_state", &[&d.close[..]], &|x| format!("{:?}", ind::ht_core_state(x[0])),
        &|x| format!("{:?}", raw::ht_core_state(x[0])));
    state("ht_dcphase_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::ht_dcphase_state(x[0])), &|x| format!("{:?}", raw::ht_dcphase_state(x[0])));
    state("ht_sine_state", &[&d.close[..]], &|x| format!("{:?}", ind::ht_sine_state(x[0])),
        &|x| format!("{:?}", raw::ht_sine_state(x[0])));
    state("ht_trendline_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::ht_trendline_state(x[0])), &|x| format!("{:?}",
        raw::ht_trendline_state(x[0])));
    state("ht_trendmode_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::ht_trendmode_state(x[0])), &|x| format!("{:?}",
        raw::ht_trendmode_state(x[0])));
    state("kama_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::kama_final_state(x[0], 5)), &|x| format!("{:?}", raw::kama_final_state(x[0],
        5)));
    state("kdj_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x| format!("{:?}",
        ind::kdj_final_state(x[0], x[1], x[2], 5, 3, 3, 50.0, true)), &|x| format!("{:?}",
        raw::kdj_final_state(x[0], x[1], x[2], 5, 3, 3, 50.0, true)));
    state("keltner_band_final_state", &[&d.close[..], &d.high[..], &d.low[..]], &|x|
        format!("{:?}", ind::keltner_band_final_state(x[0], x[1], x[2], 5, 5)), &|x|
        format!("{:?}", raw::keltner_band_final_state(x[0], x[1], x[2], 5, 5)));
    state("macd_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::macd_final_state(x[0], 3, 6)), &|x| format!("{:?}", raw::macd_final_state(x[0],
        3, 6)));
    state("macd_signal_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::macd_signal_final_state(x[0], 3, 6, 4)), &|x| format!("{:?}",
        raw::macd_signal_final_state(x[0], 3, 6, 4)));
    state("mama_state", &[&d.close[..]], &|x| format!("{:?}", ind::mama_state(x[0], 0.5,
        0.05)), &|x| format!("{:?}", raw::mama_state(x[0], 0.5, 0.05)));
    state("mass_index_final_state", &[&d.high[..], &d.low[..]], &|x| format!("{:?}",
        ind::mass_index_final_state(x[0], x[1], 5)), &|x| format!("{:?}",
        raw::mass_index_final_state(x[0], x[1], 5)));
    state("minus_di_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        format!("{:?}", ind::minus_di_final_state(x[0], x[1], x[2], 5)), &|x|
        format!("{:?}", raw::minus_di_final_state(x[0], x[1], x[2], 5)));
    state("minus_dm_final_state", &[&d.high[..], &d.low[..]], &|x| format!("{:?}",
        ind::minus_dm_final_state(x[0], x[1], 5)), &|x| format!("{:?}",
        raw::minus_dm_final_state(x[0], x[1], 5)));
    state("nvi_final_state", &[&d.close[..], &d.volume[..]], &|x| format!("{:?}",
        ind::nvi_final_state(x[0], x[1])), &|x| format!("{:?}", raw::nvi_final_state(x[0],
        x[1])));
    state("obv_final_state", &[&d.close[..], &d.volume[..]], &|x| format!("{:?}",
        ind::obv_final_state(x[0], x[1])), &|x| format!("{:?}", raw::obv_final_state(x[0],
        x[1])));
    state("plus_di_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        format!("{:?}", ind::plus_di_final_state(x[0], x[1], x[2], 5)), &|x| format!("{:?}",
        raw::plus_di_final_state(x[0], x[1], x[2], 5)));
    state("plus_dm_final_state", &[&d.high[..], &d.low[..]], &|x| format!("{:?}",
        ind::plus_dm_final_state(x[0], x[1], 5)), &|x| format!("{:?}",
        raw::plus_dm_final_state(x[0], x[1], 5)));
    state("pvi_final_state", &[&d.close[..], &d.volume[..]], &|x| format!("{:?}",
        ind::pvi_final_state(x[0], x[1])), &|x| format!("{:?}", raw::pvi_final_state(x[0],
        x[1])));
    state("pvt_final_state", &[&d.close[..], &d.volume[..]], &|x| format!("{:?}",
        ind::pvt_final_state(x[0], x[1])), &|x| format!("{:?}", raw::pvt_final_state(x[0],
        x[1])));
    state("rsi_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::rsi_final_state(x[0], 5)), &|x| format!("{:?}", raw::rsi_final_state(x[0],
        5)));
    state("sar_final_state", &[&d.high[..], &d.low[..]], &|x| format!("{:?}",
        ind::sar_final_state(x[0], x[1], 0.02, 0.2)), &|x| format!("{:?}",
        raw::sar_final_state(x[0], x[1], 0.02, 0.2)));
    state("sarext_final_state", &[&d.high[..], &d.low[..]], &|x| format!("{:?}",
        ind::sarext_final_state(x[0], x[1], 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2)),
        &|x| format!("{:?}", raw::sarext_final_state(x[0], x[1], 0.0, 0.0, 0.02, 0.02, 0.2,
        0.02, 0.02, 0.2)));
    state("smma_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::smma_final_state(x[0], 5)), &|x| format!("{:?}", raw::smma_final_state(x[0],
        5)));
    state("stochrsi_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::stochrsi_final_state(x[0], 5, 3, true, 3)), &|x| format!("{:?}",
        raw::stochrsi_final_state(x[0], 5, 3, true, 3)));
    state("supertrend_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        format!("{:?}", ind::supertrend_final_state(x[0], x[1], x[2], 5, 2.0)), &|x|
        format!("{:?}", raw::supertrend_final_state(x[0], x[1], x[2], 5, 2.0)));
    state("t3_final_state", &[&d.close[..]], &|x| format!("{:?}", ind::t3_final_state(x[0],
        5)), &|x| format!("{:?}", raw::t3_final_state(x[0], 5)));
    state("tema_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::tema_final_state(x[0], 5)), &|x| format!("{:?}", raw::tema_final_state(x[0],
        5)));
    state("trix_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::trix_final_state(x[0], 5)), &|x| format!("{:?}", raw::trix_final_state(x[0],
        5)));
    state("tsi_final_state", &[&d.close[..]], &|x| format!("{:?}",
        ind::tsi_final_state(x[0], 6, 3)), &|x| format!("{:?}", raw::tsi_final_state(x[0],
        6, 3)));
    state("wad_final_state", &[&d.high[..], &d.low[..], &d.close[..]], &|x| format!("{:?}",
        ind::wad_final_state(x[0], x[1], x[2])), &|x| format!("{:?}",
        raw::wad_final_state(x[0], x[1], x[2])));
}

#[test]
fn a_resume_continues_over_the_complete_rows() {
    let d = data();
    resume("ad_resume", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x|
        raw::ad_final_state(x[0], x[1], x[2], x[3]), &|x, from, st| ind::ad_resume(x[0],
        x[1], x[2], x[3], from, st).into_option(), &|x, from, st| raw::ad_resume(x[0], x[1],
        x[2], x[3], from, st).into_option());
    resume("adosc_resume", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x|
        raw::adosc_final_state(x[0], x[1], x[2], x[3], 3, 6), &|x, from, st|
        ind::adosc_resume(x[0], x[1], x[2], x[3], 3, 6, from, st).into_option(), &|x, from,
        st| raw::adosc_resume(x[0], x[1], x[2], x[3], 3, 6, from, st).into_option());
    resume("adx_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::adx_final_state(x[0], x[1], x[2], 5), &|x, from, st| ind::adx_resume(x[0],
        x[1], x[2], 5, from, st).into_option(), &|x, from, st| raw::adx_resume(x[0], x[1],
        x[2], 5, from, st).into_option());
    resume("adxr_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::adxr_final_state(x[0], x[1], x[2], 5), &|x, from, st| ind::adxr_resume(x[0],
        x[1], x[2], 5, from, st).into_option(), &|x, from, st| raw::adxr_resume(x[0], x[1],
        x[2], 5, from, st).into_option());
    resume("asi_resume", &[&d.open[..], &d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::asi_final_state(x[0], x[1], x[2], x[3], 300.0), &|x, from, st|
        ind::asi_resume(x[0], x[1], x[2], x[3], 300.0, from, st).into_option(), &|x, from,
        st| raw::asi_resume(x[0], x[1], x[2], x[3], 300.0, from, st).into_option());
    resume("atr_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::atr_final_state(x[0], x[1], x[2], 5), &|x, from, st| ind::atr_resume(x[0],
        x[1], x[2], 5, from, st).into_option(), &|x, from, st| raw::atr_resume(x[0], x[1],
        x[2], 5, from, st).into_option());
    resume("cmo_resume", &[&d.close[..]], &|x| raw::cmo_final_state(x[0], 5), &|x, from, st|
        ind::cmo_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::cmo_resume(x[0], 5, from, st).into_option());
    resume("dema_resume", &[&d.close[..]], &|x| raw::dema_final_state(x[0], 5), &|x, from,
        st| ind::dema_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::dema_resume(x[0], 5, from, st).into_option());
    resume("dx_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::dx_final_state(x[0], x[1], x[2], 5), &|x, from, st| ind::dx_resume(x[0], x[1],
        x[2], 5, from, st).into_option(), &|x, from, st| raw::dx_resume(x[0], x[1], x[2], 5,
        from, st).into_option());
    resume("efi_resume", &[&d.close[..], &d.volume[..]], &|x| raw::efi_final_state(x[0],
        x[1], 5), &|x, from, st| ind::efi_resume(x[0], x[1], 5, from, st).into_option(),
        &|x, from, st| raw::efi_resume(x[0], x[1], 5, from, st).into_option());
    resume("ema_resume", &[&d.close[..]], &|x| raw::ema_final_state(x[0], 5), &|x, from, st|
        ind::ema_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::ema_resume(x[0], 5, from, st).into_option());
    resume("ht_dcperiod_resume", &[&d.close[..]], &|x| raw::ht_core_state(x[0]), &|x, from,
        st| ind::ht_dcperiod_resume(x[0], from, st).into_option(), &|x, from, st|
        raw::ht_dcperiod_resume(x[0], from, st).into_option());
    resume("ht_dcphase_resume", &[&d.close[..]], &|x| raw::ht_dcphase_state(x[0]), &|x,
        from, st| ind::ht_dcphase_resume(x[0], from, st).into_option(), &|x, from, st|
        raw::ht_dcphase_resume(x[0], from, st).into_option());
    resume("ht_phasor_resume", &[&d.close[..]], &|x| raw::ht_core_state(x[0]), &|x, from,
        st| ind::ht_phasor_resume(x[0], true, from, st).into_option(), &|x, from, st|
        raw::ht_phasor_resume(x[0], true, from, st).into_option());
    resume("ht_sine_resume", &[&d.close[..]], &|x| raw::ht_sine_state(x[0]), &|x, from, st|
        ind::ht_sine_resume(x[0], true, from, st).into_option(), &|x, from, st|
        raw::ht_sine_resume(x[0], true, from, st).into_option());
    resume("ht_trendline_resume", &[&d.close[..]], &|x| raw::ht_trendline_state(x[0]), &|x,
        from, st| ind::ht_trendline_resume(x[0], from, st).into_option(), &|x, from, st|
        raw::ht_trendline_resume(x[0], from, st).into_option());
    resume("ht_trendmode_resume", &[&d.close[..]], &|x| raw::ht_trendmode_state(x[0]), &|x,
        from, st| ind::ht_trendmode_resume(x[0], from, st).into_option(), &|x, from, st|
        raw::ht_trendmode_resume(x[0], from, st).into_option());
    resume("kama_resume", &[&d.close[..]], &|x| raw::kama_final_state(x[0], 5), &|x, from,
        st| ind::kama_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::kama_resume(x[0], 5, from, st).into_option());
    resume("kdj_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::kdj_final_state(x[0], x[1], x[2], 5, 3, 3, 50.0, true), &|x, from, st|
        ind::kdj_resume(x[0], x[1], x[2], 5, 3, 3, KdjLine::J, from, st).into_option(), &|x,
        from, st| raw::kdj_resume(x[0], x[1], x[2], 5, 3, 3, KdjLine::J, from,
        st).into_option());
    resume("keltner_band_resume", &[&d.close[..], &d.high[..], &d.low[..]], &|x|
        raw::keltner_band_final_state(x[0], x[1], x[2], 5, 5), &|x, from, st|
        ind::keltner_band_resume(x[0], x[1], x[2], 5, 5, 2.0, true, from, st).into_option(),
        &|x, from, st| raw::keltner_band_resume(x[0], x[1], x[2], 5, 5, 2.0, true, from,
        st).into_option());
    resume("macd_resume", &[&d.close[..]], &|x| raw::macd_final_state(x[0], 3, 6), &|x,
        from, st| ind::macd_resume(x[0], 3, 6, from, st).into_option(), &|x, from, st|
        raw::macd_resume(x[0], 3, 6, from, st).into_option());
    resume("macd_signal_resume", &[&d.close[..]], &|x| raw::macd_signal_final_state(x[0], 3,
        6, 4), &|x, from, st| ind::macd_signal_resume(x[0], 3, 6, 4, true, from,
        st).into_option(), &|x, from, st| raw::macd_signal_resume(x[0], 3, 6, 4, true, from,
        st).into_option());
    resume("mama_resume", &[&d.close[..]], &|x| raw::mama_state(x[0], 0.5, 0.05), &|x, from,
        st| ind::mama_resume(x[0], 0.5, 0.05, true, from, st).into_option(), &|x, from, st|
        raw::mama_resume(x[0], 0.5, 0.05, true, from, st).into_option());
    resume("mass_index_resume", &[&d.high[..], &d.low[..]], &|x|
        raw::mass_index_final_state(x[0], x[1], 5), &|x, from, st|
        ind::mass_index_resume(x[0], x[1], 5, from, st).into_option(), &|x, from, st|
        raw::mass_index_resume(x[0], x[1], 5, from, st).into_option());
    resume("minus_di_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::minus_di_final_state(x[0], x[1], x[2], 5), &|x, from, st|
        ind::minus_di_resume(x[0], x[1], x[2], 5, from, st).into_option(), &|x, from, st|
        raw::minus_di_resume(x[0], x[1], x[2], 5, from, st).into_option());
    resume("minus_dm_resume", &[&d.high[..], &d.low[..]], &|x|
        raw::minus_dm_final_state(x[0], x[1], 5), &|x, from, st| ind::minus_dm_resume(x[0],
        x[1], 5, from, st).into_option(), &|x, from, st| raw::minus_dm_resume(x[0], x[1], 5,
        from, st).into_option());
    resume("natr_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::atr_final_state(x[0], x[1], x[2], 5), &|x, from, st| ind::natr_resume(x[0],
        x[1], x[2], 5, from, st).into_option(), &|x, from, st| raw::natr_resume(x[0], x[1],
        x[2], 5, from, st).into_option());
    resume("nvi_resume", &[&d.close[..], &d.volume[..]], &|x| raw::nvi_final_state(x[0],
        x[1]), &|x, from, st| ind::nvi_resume(x[0], x[1], from, st).into_option(), &|x,
        from, st| raw::nvi_resume(x[0], x[1], from, st).into_option());
    resume("obv_resume", &[&d.close[..], &d.volume[..]], &|x| raw::obv_final_state(x[0],
        x[1]), &|x, from, st| ind::obv_resume(x[0], x[1], from, st).into_option(), &|x,
        from, st| raw::obv_resume(x[0], x[1], from, st).into_option());
    resume("plus_di_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::plus_di_final_state(x[0], x[1], x[2], 5), &|x, from, st|
        ind::plus_di_resume(x[0], x[1], x[2], 5, from, st).into_option(), &|x, from, st|
        raw::plus_di_resume(x[0], x[1], x[2], 5, from, st).into_option());
    resume("plus_dm_resume", &[&d.high[..], &d.low[..]], &|x| raw::plus_dm_final_state(x[0],
        x[1], 5), &|x, from, st| ind::plus_dm_resume(x[0], x[1], 5, from, st).into_option(),
        &|x, from, st| raw::plus_dm_resume(x[0], x[1], 5, from, st).into_option());
    resume("pvi_resume", &[&d.close[..], &d.volume[..]], &|x| raw::pvi_final_state(x[0],
        x[1]), &|x, from, st| ind::pvi_resume(x[0], x[1], from, st).into_option(), &|x,
        from, st| raw::pvi_resume(x[0], x[1], from, st).into_option());
    resume("pvt_resume", &[&d.close[..], &d.volume[..]], &|x| raw::pvt_final_state(x[0],
        x[1]), &|x, from, st| ind::pvt_resume(x[0], x[1], from, st).into_option(), &|x,
        from, st| raw::pvt_resume(x[0], x[1], from, st).into_option());
    resume("rsi_resume", &[&d.close[..]], &|x| raw::rsi_final_state(x[0], 5), &|x, from, st|
        ind::rsi_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::rsi_resume(x[0], 5, from, st).into_option());
    resume("sar_resume", &[&d.high[..], &d.low[..]], &|x| raw::sar_final_state(x[0], x[1],
        0.02, 0.2), &|x, from, st| ind::sar_resume(x[0], x[1], 0.02, 0.2, from,
        st).into_option(), &|x, from, st| raw::sar_resume(x[0], x[1], 0.02, 0.2, from,
        st).into_option());
    resume("sarext_resume", &[&d.high[..], &d.low[..]], &|x| raw::sarext_final_state(x[0],
        x[1], 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2), &|x, from, st|
        ind::sarext_resume(x[0], x[1], 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2, from,
        st).into_option(), &|x, from, st| raw::sarext_resume(x[0], x[1], 0.0, 0.02, 0.02,
        0.2, 0.02, 0.02, 0.2, from, st).into_option());
    resume("smma_resume", &[&d.close[..]], &|x| raw::smma_final_state(x[0], 5), &|x, from,
        st| ind::smma_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::smma_resume(x[0], 5, from, st).into_option());
    resume("stochrsi_resume", &[&d.close[..]], &|x| raw::stochrsi_final_state(x[0], 5, 3,
        true, 3), &|x, from, st| ind::stochrsi_resume(x[0], 5, 3, true, 3, from,
        st).into_option(), &|x, from, st| raw::stochrsi_resume(x[0], 5, 3, true, 3, from,
        st).into_option());
    resume("supertrend_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::supertrend_final_state(x[0], x[1], x[2], 5, 2.0), &|x, from, st|
        ind::supertrend_resume(x[0], x[1], x[2], 5, 2.0, false, from, st).into_option(),
        &|x, from, st| raw::supertrend_resume(x[0], x[1], x[2], 5, 2.0, false, from,
        st).into_option());
    resume("t3_resume", &[&d.close[..]], &|x| raw::t3_final_state(x[0], 5), &|x, from, st|
        ind::t3_resume(x[0], 5, 0.7, from, st).into_option(), &|x, from, st|
        raw::t3_resume(x[0], 5, 0.7, from, st).into_option());
    resume("tema_resume", &[&d.close[..]], &|x| raw::tema_final_state(x[0], 5), &|x, from,
        st| ind::tema_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::tema_resume(x[0], 5, from, st).into_option());
    resume("trix_resume", &[&d.close[..]], &|x| raw::trix_final_state(x[0], 5), &|x, from,
        st| ind::trix_resume(x[0], 5, from, st).into_option(), &|x, from, st|
        raw::trix_resume(x[0], 5, from, st).into_option());
    resume("tsi_resume", &[&d.close[..]], &|x| raw::tsi_final_state(x[0], 6, 3), &|x, from,
        st| ind::tsi_resume(x[0], 6, 3, from, st).into_option(), &|x, from, st|
        raw::tsi_resume(x[0], 6, 3, from, st).into_option());
    resume("wad_resume", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::wad_final_state(x[0], x[1], x[2]), &|x, from, st| ind::wad_resume(x[0], x[1],
        x[2], from, st).into_option(), &|x, from, st| raw::wad_resume(x[0], x[1], x[2],
        from, st).into_option());
}

#[test]
fn a_single_row_resume_skips_a_missing_row() {
    let d = data();
    step("ad_resume_one", &[&d.high[..], &d.low[..], &d.close[..], &d.volume[..]], &|x|
        raw::ad_final_state(x[0], x[1], x[2], x[3]), &|x, row, st| format!("{:?}",
        ind::ad_resume_one(x[0], x[1], x[2], x[3], row, st)), &|x, row, st| format!("{:?}",
        raw::ad_resume_one(x[0], x[1], x[2], x[3], row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("adx_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::adx_final_state(x[0], x[1], x[2], 5), &|x, row, st| format!("{:?}",
        ind::adx_resume_one(x[0], x[1], x[2], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::adx_resume_one(x[0], x[1], x[2], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("adxr_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::adxr_final_state(x[0], x[1], x[2], 5), &|x, row, st| format!("{:?}",
        ind::adxr_resume_one(x[0], x[1], x[2], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::adxr_resume_one(x[0], x[1], x[2], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("atr_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::atr_final_state(x[0], x[1], x[2], 5), &|x, row, st| format!("{:?}",
        ind::atr_resume_one(x[0], x[1], x[2], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::atr_resume_one(x[0], x[1], x[2], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("cmo_resume_one", &[&d.close[..]], &|x| raw::cmo_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::cmo_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::cmo_resume_one(x[0], 5, row, st)), &|st| format!("{:?}",
        Some((f64::NAN, [st[0], st[1]]))));
    step("dema_resume_one", &[&d.close[..]], &|x| raw::dema_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::dema_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::dema_resume_one(x[0], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("dx_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::dx_final_state(x[0], x[1], x[2], 5), &|x, row, st| format!("{:?}",
        ind::dx_resume_one(x[0], x[1], x[2], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::dx_resume_one(x[0], x[1], x[2], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("efi_resume_one", &[&d.close[..], &d.volume[..]], &|x| raw::efi_final_state(x[0],
        x[1], 5), &|x, row, st| format!("{:?}", ind::efi_resume_one(x[0], x[1], 5, row,
        st)), &|x, row, st| format!("{:?}", raw::efi_resume_one(x[0], x[1], 5, row, st)),
        &|_st| format!("{:?}", Some(f64::NAN)));
    step("ema_resume_one", &[&d.close[..]], &|x| raw::ema_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::ema_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::ema_resume_one(x[0], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("kama_resume_one", &[&d.close[..]], &|x| raw::kama_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::kama_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::kama_resume_one(x[0], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("keltner_band_resume_one", &[&d.close[..], &d.high[..], &d.low[..]], &|x|
        raw::keltner_band_final_state(x[0], x[1], x[2], 5, 5), &|x, row, st| format!("{:?}",
        ind::keltner_band_resume_one(x[0], x[1], x[2], 5, 5, 2.0, true, row, st)), &|x, row,
        st| format!("{:?}", raw::keltner_band_resume_one(x[0], x[1], x[2], 5, 5, 2.0, true,
        row, st)), &|_st| format!("{:?}", Some(f64::NAN)));
    step("macd_resume_one", &[&d.close[..]], &|x| raw::macd_final_state(x[0], 3, 6), &|x,
        row, st| format!("{:?}", ind::macd_resume_one(x[0], 3, 6, row, st)), &|x, row, st|
        format!("{:?}", raw::macd_resume_one(x[0], 3, 6, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("macd_signal_resume_one", &[&d.close[..]], &|x| raw::macd_signal_final_state(x[0],
        3, 6, 4), &|x, row, st| format!("{:?}", ind::macd_signal_resume_one(x[0], 3, 6, 4,
        true, row, st)), &|x, row, st| format!("{:?}", raw::macd_signal_resume_one(x[0], 3,
        6, 4, true, row, st)), &|_st| format!("{:?}", Some(f64::NAN)));
    step("mass_index_resume_one", &[&d.high[..], &d.low[..]], &|x|
        raw::mass_index_final_state(x[0], x[1], 5), &|x, row, st| format!("{:?}",
        ind::mass_index_resume_one(x[0], x[1], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::mass_index_resume_one(x[0], x[1], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("minus_di_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::minus_di_final_state(x[0], x[1], x[2], 5), &|x, row, st| format!("{:?}",
        ind::minus_di_resume_one(x[0], x[1], x[2], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::minus_di_resume_one(x[0], x[1], x[2], 5, row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("minus_dm_resume_one", &[&d.high[..], &d.low[..]], &|x|
        raw::minus_dm_final_state(x[0], x[1], 5), &|x, row, st| format!("{:?}",
        ind::minus_dm_resume_one(x[0], x[1], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::minus_dm_resume_one(x[0], x[1], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("nvi_resume_one", &[&d.close[..], &d.volume[..]], &|x| raw::nvi_final_state(x[0],
        x[1]), &|x, row, st| format!("{:?}", ind::nvi_resume_one(x[0], x[1], row, st)), &|x,
        row, st| format!("{:?}", raw::nvi_resume_one(x[0], x[1], row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("obv_resume_one", &[&d.close[..], &d.volume[..]], &|x| raw::obv_final_state(x[0],
        x[1]), &|x, row, st| format!("{:?}", ind::obv_resume_one(x[0], x[1], row, st)), &|x,
        row, st| format!("{:?}", raw::obv_resume_one(x[0], x[1], row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("plus_di_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::plus_di_final_state(x[0], x[1], x[2], 5), &|x, row, st| format!("{:?}",
        ind::plus_di_resume_one(x[0], x[1], x[2], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::plus_di_resume_one(x[0], x[1], x[2], 5, row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("plus_dm_resume_one", &[&d.high[..], &d.low[..]], &|x|
        raw::plus_dm_final_state(x[0], x[1], 5), &|x, row, st| format!("{:?}",
        ind::plus_dm_resume_one(x[0], x[1], 5, row, st)), &|x, row, st| format!("{:?}",
        raw::plus_dm_resume_one(x[0], x[1], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("pvi_resume_one", &[&d.close[..], &d.volume[..]], &|x| raw::pvi_final_state(x[0],
        x[1]), &|x, row, st| format!("{:?}", ind::pvi_resume_one(x[0], x[1], row, st)), &|x,
        row, st| format!("{:?}", raw::pvi_resume_one(x[0], x[1], row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("pvt_resume_one", &[&d.close[..], &d.volume[..]], &|x| raw::pvt_final_state(x[0],
        x[1]), &|x, row, st| format!("{:?}", ind::pvt_resume_one(x[0], x[1], row, st)), &|x,
        row, st| format!("{:?}", raw::pvt_resume_one(x[0], x[1], row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("rsi_resume_one", &[&d.close[..]], &|x| raw::rsi_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::rsi_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::rsi_resume_one(x[0], 5, row, st)), &|st| format!("{:?}",
        Some((f64::NAN, [st[0], st[1]]))));
    step("sar_resume_one", &[&d.high[..], &d.low[..]], &|x| raw::sar_final_state(x[0], x[1],
        0.02, 0.2), &|x, row, st| format!("{:?}", ind::sar_resume_one(x[0], x[1], row, st)),
        &|x, row, st| format!("{:?}", raw::sar_resume_one(x[0], x[1], row, st)), &|_st|
        format!("{:?}", Some(f64::NAN)));
    step("sarext_resume_one", &[&d.high[..], &d.low[..]], &|x| raw::sarext_final_state(x[0],
        x[1], 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2), &|x, row, st| format!("{:?}",
        ind::sarext_resume_one(x[0], x[1], 0.0, row, st)), &|x, row, st| format!("{:?}",
        raw::sarext_resume_one(x[0], x[1], 0.0, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("smma_resume_one", &[&d.close[..]], &|x| raw::smma_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::smma_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::smma_resume_one(x[0], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("supertrend_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::supertrend_final_state(x[0], x[1], x[2], 5, 2.0), &|x, row, st| format!("{:?}",
        ind::supertrend_resume_one(x[0], x[1], x[2], 5, 2.0, false, row, st)), &|x, row, st|
        format!("{:?}", raw::supertrend_resume_one(x[0], x[1], x[2], 5, 2.0, false, row,
        st)), &|_st| format!("{:?}", Some(f64::NAN)));
    step("t3_resume_one", &[&d.close[..]], &|x| raw::t3_final_state(x[0], 5), &|x, row, st|
        format!("{:?}", ind::t3_resume_one(x[0], 5, 0.7, row, st)), &|x, row, st|
        format!("{:?}", raw::t3_resume_one(x[0], 5, 0.7, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("tema_resume_one", &[&d.close[..]], &|x| raw::tema_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::tema_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::tema_resume_one(x[0], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("trix_resume_one", &[&d.close[..]], &|x| raw::trix_final_state(x[0], 5), &|x, row,
        st| format!("{:?}", ind::trix_resume_one(x[0], 5, row, st)), &|x, row, st|
        format!("{:?}", raw::trix_resume_one(x[0], 5, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("tsi_resume_one", &[&d.close[..]], &|x| raw::tsi_final_state(x[0], 6, 3), &|x, row,
        st| format!("{:?}", ind::tsi_resume_one(x[0], 6, 3, row, st)), &|x, row, st|
        format!("{:?}", raw::tsi_resume_one(x[0], 6, 3, row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
    step("wad_resume_one", &[&d.high[..], &d.low[..], &d.close[..]], &|x|
        raw::wad_final_state(x[0], x[1], x[2]), &|x, row, st| format!("{:?}",
        ind::wad_resume_one(x[0], x[1], x[2], row, st)), &|x, row, st| format!("{:?}",
        raw::wad_resume_one(x[0], x[1], x[2], row, st)), &|_st| format!("{:?}",
        Some(f64::NAN)));
}

fn av<'a>(x: &[&'a [f64]]) -> ArrayView1<'a, f64> {
    ArrayView1::from(x[0])
}

#[test]
fn the_recursive_kernels_run_on_the_complete_rows() {
    let d = data();
    let x = [&d.close[..]];
    series("wilder", &x, &|x| kern::wilder(av(x), 5).flat(), &|x| raw_kern::wilder(av(x), 5).flat());
    series("ema_seeded", &x, &|x| kern::ema_seeded(av(x), 5).flat(), &|x| {
        raw_kern::ema_seeded(av(x), 5).flat()
    });
    series("ema_diff_seeded", &x, &|x| kern::ema_diff_seeded(av(x), 3, 6).flat(), &|x| {
        raw_kern::ema_diff_seeded(av(x), 3, 6).flat()
    });
    series("ewma_with_init", &x, &|x| kern::ewma_with_init(av(x), 3, 50.0).flat(), &|x| {
        raw_kern::ewma_with_init(av(x), 3, 50.0).flat()
    });
    series("ema_cascade", &x, &|x| kern::ema_cascade::<3>(x[0], 4).flat(), &|x| {
        raw_kern::ema_cascade::<3>(x[0], 4).flat()
    });
    state("ema_cascade_final", &x, &|x| format!("{:?}", kern::ema_cascade_final::<3>(x[0], 4)), &|x| {
        format!("{:?}", raw_kern::ema_cascade_final::<3>(x[0], 4))
    });
    // a strided view is read through one contiguous copy
    let both: Vec<f64> = d.close.iter().flat_map(|&v| [v, 0.0]).collect();
    let strided = ArrayView1::from(&both[..]).slice_move(ndarray::s![..;2]);
    assert_eq!(bits(&kern::wilder(strided, 5).to_vec()), bits(&kern::wilder(av(&x), 5).to_vec()));
    // a missing value stepped over leaves the cascade where it was
    let mut e = [1.0, 2.0];
    kern::ema_cascade_step(&mut e, f64::NAN, 0.5);
    assert_eq!(e, [1.0, 2.0]);
    kern::ema_cascade_step(&mut e, 3.0, 0.5);
    let mut r = [1.0, 2.0];
    raw_kern::ema_cascade_step(&mut r, 3.0, 0.5);
    assert_eq!(e, r);
}
