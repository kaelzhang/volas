//! The technical indicators, with the missing-row rule applied.
//!
//! Every function of [`crate::raw::indicators`] is here under the same name and
//! signature. A finite-window one is that very function: window locality already
//! confines a missing row to the windows holding it. A recursive one — a full
//! computation, its carried-state capture (`*_final_state` / `*_state`), a resume over
//! new rows (`*_resume`) or a single-row resume (`*_resume_one`) — runs the raw function
//! on the complete rows of its inputs (see [`crate::missing`]): a row where any input is
//! NaN outputs NaN, and the recursion continues from the last complete row as if that
//! row did not exist. So a carried state describes the last complete row; a resume's
//! `state` is the state as of the last complete row before its first new row; and a
//! missing row a single-row resume is asked for is NaN — its state does not advance
//! (keep the state passed in; for the `*_resume_one` whose value is the next state,
//! do not take a NaN as one). With no row missing, the inputs and the outputs pass
//! through uncopied.

pub use crate::raw::indicators::*;

use crate::missing::{
    on_complete_rows, resume_on_complete_rows, state_on_complete_rows, step_on_complete_rows,
    Input,
};
use crate::raw::indicators as raw;

/// Bind each named slice to the next of `rows`, then call `raw::$name`.
macro_rules! call_raw {
    ($rows:expr, $name:ident($($s:ident),+; $($arg:expr),*)) => {{
        let mut rows = $rows.iter().copied();
        $(let $s = rows.next().unwrap_or_default();)+
        raw::$name($($s,)+ $($arg),*)
    }};
}

/// Declare the rule-applying twin of each raw recursive function.
macro_rules! on_complete_rows {
    (series: $($name:ident($($s:ident),+ $(; $($p:ident: $t:ty),*)?) -> $ret:ty;)*) => {$(
        #[doc = concat!("[`raw::", stringify!($name), "`] over the complete rows of its inputs.")]
        #[allow(clippy::too_many_arguments)] // the raw function's own signature
        pub fn $name($($s: &[f64],)+ $($($p: $t),*)?) -> $ret {
            on_complete_rows(&[$(Input::new($s)),+], |rows| {
                call_raw!(rows, $name($($s),+; $($($p),*)?))
            })
        }
    )*};
    (state: $($name:ident($($s:ident),+ $(; $($p:ident: $t:ty),*)?) -> $ret:ty;)*) => {$(
        #[doc = concat!("[`raw::", stringify!($name), "`] over the complete rows of its inputs: ")]
        #[doc = "the state as of the last complete row."]
        #[allow(clippy::too_many_arguments)] // the raw function's own signature
        pub fn $name($($s: &[f64],)+ $($($p: $t),*)?) -> $ret {
            state_on_complete_rows(&[$($s),+], |rows| call_raw!(rows, $name($($s),+; $($($p),*)?)))
        }
    )*};
    (resume: $($name:ident($($s:ident),+; $($p:ident: $t:ty),*; $from:ident, $state:ident)
        -> $ret:ty $(= $total:ident)?;)*) => {$(
        #[doc = concat!("[`raw::", stringify!($name), "`] over the complete rows of its inputs: ")]
        #[doc = "`state` is the state as of the last complete row before `from`; a missing"]
        #[doc = "new row is NaN."]
        #[allow(clippy::too_many_arguments)] // the raw function's own signature
        pub fn $name($($s: &[f64],)+ $($p: $t,)* $from: usize, $state: &[f64]) -> $ret {
            let out = resume_on_complete_rows(&[$($s),+], $from, $state, |rows, $from| {
                on_complete_rows!(@wrap $($total)? call_raw!(rows, $name($($s),+; $($p,)* $from, $state)))
            });
            on_complete_rows!(@unwrap $($total)? out)
        }
    )*};
    (step: $($name:ident($($s:ident),+; $($p:ident: $t:ty),*))*) => {$(
        #[doc = concat!("[`raw::", stringify!($name), "`] over the complete rows of its inputs: ")]
        #[doc = "`state` is the state as of the last complete row before `row`, and `next`"]
        #[doc = "receives the state as of `row`; a missing `row` is NaN and does not advance"]
        #[doc = "the state (`next` is `state`)."]
        #[allow(clippy::too_many_arguments)] // the raw function's own signature
        pub fn $name(
            $($s: &[f64],)+ $($p: $t,)* row: usize, state: &[f64], next: &mut [f64],
        ) -> Option<f64> {
            let mut missing = false;
            let out = step_on_complete_rows(&[$($s),+], row, || {
                missing = true;
                Some(f64::NAN)
            }, |rows, row| call_raw!(rows, $name($($s),+; $($p,)* row, state, next)));
            if missing {
                if next.len() != state.len() {
                    return None;
                }
                next.copy_from_slice(state);
            }
            out
        }
    )*};
    (@wrap total $e:expr) => { Some($e) };
    (@wrap $e:expr) => { $e };
    (@unwrap total $e:expr) => { $e.expect("a total resume always resumes") };
    (@unwrap $e:expr) => { $e };
}

on_complete_rows! { series:
    ad(high, low, close, volume) -> Vec<f64>;
    adosc(high, low, close, volume; fast: usize, slow: usize) -> Vec<f64>;
    adx(high, low, close; period: usize) -> Vec<f64>;
    adxr(high, low, close; period: usize) -> Vec<f64>;
    asi(open, high, low, close; t: f64) -> Vec<f64>;
    atr(high, low, close; period: usize) -> Vec<f64>;
    cmo(close; period: usize) -> Vec<f64>;
    crsi(close; rsi_len: usize, streak_len: usize, rank_len: usize) -> Vec<f64>;
    dema(data; period: usize) -> Vec<f64>;
    dx(high, low, close; period: usize) -> Vec<f64>;
    efi(close, volume; n: usize) -> Vec<f64>;
    ema(close; period: usize) -> Vec<f64>;
    ht_dcperiod(price) -> Vec<f64>;
    ht_dcphase(price) -> Vec<f64>;
    ht_phasor(price) -> (Vec<f64>, Vec<f64>);
    ht_phasor_line(price; quadrature: bool) -> Vec<f64>;
    ht_sine(price) -> (Vec<f64>, Vec<f64>);
    ht_trendline(price) -> Vec<f64>;
    ht_trendmode(price) -> Vec<f64>;
    kama(data; period: usize, origin: usize) -> Vec<f64>;
    kcw(close, high, low; ema_period: usize, atr_period: usize, mult: f64) -> Vec<f64>;
    kdj_d(high, low, close; period_rsv: usize, period_k: usize, period_d: usize, init: f64) ->
        Vec<f64>;
    kdj_j(high, low, close; period_rsv: usize, period_k: usize, period_d: usize, init: f64) ->
        Vec<f64>;
    kdj_k(high, low, close; period_rsv: usize, period_k: usize, init: f64) -> Vec<f64>;
    keltner_band(close, high, low; ema_period: usize, atr_period: usize, mult: f64, upper: bool)
        -> Vec<f64>;
    macd(close; fast: usize, slow: usize) -> Vec<f64>;
    macd_histogram(close; fast: usize, slow: usize, signal: usize) -> Vec<f64>;
    macd_signal(close; fast: usize, slow: usize, signal: usize) -> Vec<f64>;
    mama(price; fast_limit: f64, slow_limit: f64) -> (Vec<f64>, Vec<f64>);
    mama_line(price; fast_limit: f64, slow_limit: f64, want_fama: bool) -> Vec<f64>;
    mass_index(high, low; n: usize, origin: usize) -> Vec<f64>;
    minus_di(high, low, close; period: usize) -> Vec<f64>;
    minus_dm(high, low; period: usize) -> Vec<f64>;
    natr(high, low, close; period: usize) -> Vec<f64>;
    nvi(close, volume) -> Vec<f64>;
    obv(real, volume) -> Vec<f64>;
    plus_di(high, low, close; period: usize) -> Vec<f64>;
    plus_dm(high, low; period: usize) -> Vec<f64>;
    pvi(close, volume) -> Vec<f64>;
    pvt(close, volume) -> Vec<f64>;
    rsi(close; period: usize) -> Vec<f64>;
    sar(high, low; acceleration: f64, maximum: f64) -> Vec<f64>;
    sarext(high, low; start_value: f64, offset_on_reverse: f64, accel_init_long: f64,
        accel_long: f64, accel_max_long: f64, accel_init_short: f64, accel_short: f64,
        accel_max_short: f64) -> Vec<f64>;
    smma(close; period: usize) -> Vec<f64>;
    stoch_momentum(high, low, close; k: usize, d: usize) -> Vec<f64>;
    stoch_momentum_signal(high, low, close; k: usize, d: usize, signal: usize) -> Vec<f64>;
    stochrsi_d_default_sma(close; origin: usize) -> Option<Vec<f64>>;
    stochrsi_d_sma(close; rsi_period: usize, fastk_period: usize, fastd_period: usize, origin:
        usize) -> Vec<f64>;
    stochrsi_fastk(close; rsi_period: usize, fastk_period: usize) -> Vec<f64>;
    supertrend(high, low, close; period: usize, mult: f64, want_direction: bool) -> Vec<f64>;
    t3(data; period: usize, vfactor: f64) -> Vec<f64>;
    tema(data; period: usize) -> Vec<f64>;
    trix(close; period: usize) -> Vec<f64>;
    tsi(close; long: usize, short: usize) -> Vec<f64>;
    ttm_squeeze_momentum(high, low, close; n: usize, origin: usize) -> Vec<f64>;
    ttm_squeeze_on(high, low, close; n: usize, bb_mult: f64, kc_mult: f64, origin: usize) ->
        Vec<f64>;
    wad(high, low, close) -> Vec<f64>;
}

on_complete_rows! { state:
    ad_final_state(high, low, close, volume) -> Option<Vec<f64>>;
    adosc_final_state(high, low, close, volume; fast: usize, slow: usize) -> Option<Vec<f64>>;
    adx_final_state(high, low, close; period: usize) -> Option<Vec<f64>>;
    adxr_final_state(high, low, close; period: usize) -> Option<Vec<f64>>;
    asi_final_state(open, high, low, close; t: f64) -> Option<Vec<f64>>;
    atr_final_state(high, low, close; period: usize) -> Option<Vec<f64>>;
    cmo_final_state(close; period: usize) -> Option<Vec<f64>>;
    dema_final_state(data; period: usize) -> Option<Vec<f64>>;
    dx_final_state(high, low, close; period: usize) -> Option<Vec<f64>>;
    efi_final_state(close, volume; n: usize) -> Option<Vec<f64>>;
    ema_final_state(data; period: usize) -> Option<Vec<f64>>;
    ht_core_state(price) -> Option<Vec<f64>>;
    ht_dcphase_state(price) -> Option<Vec<f64>>;
    ht_sine_state(price) -> Option<Vec<f64>>;
    ht_trendline_state(price) -> Option<Vec<f64>>;
    ht_trendmode_state(price) -> Option<Vec<f64>>;
    kama_final_state(data; period: usize, origin: usize) -> Option<Vec<f64>>;
    kdj_final_state(high, low, close; period_rsv: usize, period_k: usize, period_d: usize, init:
        f64, want_d: bool) -> Option<Vec<f64>>;
    keltner_band_final_state(close, high, low; ema_period: usize, atr_period: usize) ->
        Option<Vec<f64>>;
    macd_final_state(close; fast: usize, slow: usize) -> Option<Vec<f64>>;
    macd_signal_final_state(close; fast: usize, slow: usize, signal: usize) -> Option<Vec<f64>>;
    mama_state(price; fast_limit: f64, slow_limit: f64) -> Option<Vec<f64>>;
    mass_index_final_state(high, low; n: usize, origin: usize) -> Option<Vec<f64>>;
    minus_di_final_state(high, low, close; period: usize) -> Option<Vec<f64>>;
    minus_dm_final_state(high, low; period: usize) -> Option<Vec<f64>>;
    nvi_final_state(close, volume) -> Option<Vec<f64>>;
    obv_final_state(real, volume) -> Option<Vec<f64>>;
    plus_di_final_state(high, low, close; period: usize) -> Option<Vec<f64>>;
    plus_dm_final_state(high, low; period: usize) -> Option<Vec<f64>>;
    pvi_final_state(close, volume) -> Option<Vec<f64>>;
    pvt_final_state(close, volume) -> Option<Vec<f64>>;
    rsi_final_state(close; period: usize) -> Option<Vec<f64>>;
    sar_final_state(high, low; acceleration: f64, maximum: f64) -> Option<Vec<f64>>;
    sarext_final_state(high, low; start_value: f64, offset_on_reverse: f64, accel_init_long:
        f64, accel_long: f64, accel_max_long: f64, accel_init_short: f64, accel_short: f64,
        accel_max_short: f64) -> Option<Vec<f64>>;
    smma_final_state(data; period: usize) -> Option<Vec<f64>>;
    stochrsi_final_state(close; rsi_period: usize, fastk_period: usize, is_d: bool,
        fastd_period: usize, origin: usize) -> Option<Vec<f64>>;
    supertrend_final_state(high, low, close; period: usize, mult: f64) -> Option<Vec<f64>>;
    t3_final_state(data; period: usize) -> Option<Vec<f64>>;
    tema_final_state(data; period: usize) -> Option<Vec<f64>>;
    trix_final_state(close; period: usize) -> Option<Vec<f64>>;
    tsi_final_state(close; long: usize, short: usize) -> Option<Vec<f64>>;
    wad_final_state(high, low, close) -> Option<Vec<f64>>;
}

on_complete_rows! { resume:
    ad_resume(high, low, close, volume; ; from, state) -> (Vec<f64>, Vec<f64>) = total;
    adosc_resume(high, low, close, volume; fast: usize, slow: usize; from, state) -> (Vec<f64>,
        Vec<f64>) = total;
    adx_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    adxr_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    asi_resume(open, high, low, close; t: f64; from, state) -> (Vec<f64>, Vec<f64>) = total;
    atr_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    cmo_resume(close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    dema_resume(data; period: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    dx_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    efi_resume(close, volume; n: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    ema_resume(data; period: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    ht_dcperiod_resume(price; ; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    ht_dcphase_resume(price; ; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    ht_phasor_resume(price; quad: bool; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    ht_sine_resume(price; lead: bool; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    ht_trendline_resume(price; ; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    ht_trendmode_resume(price; ; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    kama_resume(data; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    kdj_resume(high, low, close; period_rsv: usize, period_k: usize, period_d: usize, line:
        KdjLine; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    keltner_band_resume(close, high, low; ema_period: usize, atr_period: usize, mult: f64,
        upper: bool; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    macd_resume(close; fast: usize, slow: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    macd_signal_resume(close; fast: usize, slow: usize, signal: usize, histogram: bool; from,
        state) -> (Vec<f64>, Vec<f64>) = total;
    mama_resume(price; fast_limit: f64, slow_limit: f64, want_fama: bool; from, state) ->
        Option<(Vec<f64>, Vec<f64>)>;
    mass_index_resume(high, low; n: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    minus_di_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    minus_dm_resume(high, low; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    natr_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    nvi_resume(close, volume; ; from, state) -> (Vec<f64>, Vec<f64>) = total;
    obv_resume(real, volume; ; from, state) -> (Vec<f64>, Vec<f64>) = total;
    plus_di_resume(high, low, close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    plus_dm_resume(high, low; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    pvi_resume(close, volume; ; from, state) -> (Vec<f64>, Vec<f64>) = total;
    pvt_resume(close, volume; ; from, state) -> (Vec<f64>, Vec<f64>) = total;
    rsi_resume(close; period: usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    sar_resume(high, low; acceleration: f64, maximum: f64; from, state) -> Option<(Vec<f64>,
        Vec<f64>)>;
    sarext_resume(high, low; offset_on_reverse: f64, accel_init_long: f64, accel_long: f64,
        accel_max_long: f64, accel_init_short: f64, accel_short: f64, accel_max_short: f64;
        from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    smma_resume(data; period: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    stochrsi_resume(close; rsi_period: usize, fastk_period: usize, is_d: bool, fastd_period:
        usize; from, state) -> Option<(Vec<f64>, Vec<f64>)>;
    supertrend_resume(high, low, close; period: usize, mult: f64, want_direction: bool; from,
        state) -> Option<(Vec<f64>, Vec<f64>)>;
    t3_resume(data; period: usize, vfactor: f64; from, state) -> (Vec<f64>, Vec<f64>) = total;
    tema_resume(data; period: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    trix_resume(close; period: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    tsi_resume(close; long: usize, short: usize; from, state) -> (Vec<f64>, Vec<f64>) = total;
    wad_resume(high, low, close; ; from, state) -> (Vec<f64>, Vec<f64>) = total;
}

on_complete_rows! { step:
    ad_resume_one(high, low, close, volume;)
    adx_resume_one(high, low, close; period: usize)
    adxr_resume_one(high, low, close; period: usize)
    atr_resume_one(high, low, close; period: usize)
    cmo_resume_one(close; period: usize)
    dema_resume_one(data; period: usize)
    dx_resume_one(high, low, close; period: usize)
    efi_resume_one(close, volume; n: usize)
    ema_resume_one(data; period: usize)
    kama_resume_one(data; period: usize)
    keltner_band_resume_one(close, high, low; ema_period: usize, atr_period: usize, mult: f64,
        upper: bool)
    macd_resume_one(close; fast: usize, slow: usize)
    macd_signal_resume_one(close; fast: usize, slow: usize, signal: usize, histogram: bool)
    mass_index_resume_one(high, low; n: usize)
    minus_di_resume_one(high, low, close; period: usize)
    minus_dm_resume_one(high, low; period: usize)
    nvi_resume_one(close, volume;)
    obv_resume_one(real, volume;)
    plus_di_resume_one(high, low, close; period: usize)
    plus_dm_resume_one(high, low; period: usize)
    pvi_resume_one(close, volume;)
    pvt_resume_one(close, volume;)
    rsi_resume_one(close; period: usize)
    sar_resume_one(high, low; acceleration: f64, maximum: f64)
    sarext_resume_one(high, low; offset_on_reverse: f64, accel_init_long: f64, accel_long: f64,
        accel_max_long: f64, accel_init_short: f64, accel_short: f64, accel_max_short: f64)
    smma_resume_one(data; period: usize)
    supertrend_resume_one(high, low, close; period: usize, mult: f64, want_direction: bool)
    t3_resume_one(data; period: usize, vfactor: f64)
    tema_resume_one(data; period: usize)
    trix_resume_one(close; period: usize)
    tsi_resume_one(close; long: usize, short: usize)
    wad_resume_one(high, low, close;)
}
