//! Candlestick input admission surrounds the original numerical kernels.
use super::*;

/// Recognize `cdl_doji` with missing-candle admission.
pub fn cdl_doji(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_doji_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("doji").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_marubozu` with missing-candle admission.
pub fn cdl_marubozu(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_marubozu_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("marubozu").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_closingmarubozu` with missing-candle admission.
pub fn cdl_closingmarubozu(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_closingmarubozu_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("closingmarubozu").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_longline` with missing-candle admission.
pub fn cdl_longline(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_longline_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("longline").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_shortline` with missing-candle admission.
pub fn cdl_shortline(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_shortline_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("shortline").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_highwave` with missing-candle admission.
pub fn cdl_highwave(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_highwave_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("highwave").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_spinningtop` with missing-candle admission.
pub fn cdl_spinningtop(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_spinningtop_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("spinningtop").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_dragonflydoji` with missing-candle admission.
pub fn cdl_dragonflydoji(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_dragonflydoji_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("dragonflydoji").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_gravestonedoji` with missing-candle admission.
pub fn cdl_gravestonedoji(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_gravestonedoji_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("gravestonedoji").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_longleggeddoji` with missing-candle admission.
pub fn cdl_longleggeddoji(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_longleggeddoji_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("longleggeddoji").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_rickshawman` with missing-candle admission.
pub fn cdl_rickshawman(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_rickshawman_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("rickshawman").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_belthold` with missing-candle admission.
pub fn cdl_belthold(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_belthold_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("belthold").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_hammer` with missing-candle admission.
pub fn cdl_hammer(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_hammer_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("hammer").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_hangingman` with missing-candle admission.
pub fn cdl_hangingman(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_hangingman_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("hangingman").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_invertedhammer` with missing-candle admission.
pub fn cdl_invertedhammer(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_invertedhammer_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("invertedhammer").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_shootingstar` with missing-candle admission.
pub fn cdl_shootingstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_shootingstar_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("shootingstar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_takuri` with missing-candle admission.
pub fn cdl_takuri(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_takuri_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("takuri").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_engulfing` with missing-candle admission.
pub fn cdl_engulfing(o: &[f64], h: &[f64], l: &[f64], c: &[f64], _origin: usize) -> Vec<f64> {
    let out = cdl_engulfing_raw(o, h, l, c, _origin);
    let reach = candle_pattern_complete("engulfing").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_harami` with missing-candle admission.
pub fn cdl_harami(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_harami_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("harami").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_haramicross` with missing-candle admission.
pub fn cdl_haramicross(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_haramicross_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("haramicross").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_piercing` with missing-candle admission.
pub fn cdl_piercing(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_piercing_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("piercing").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_darkcloudcover` with missing-candle admission.
pub fn cdl_darkcloudcover(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    penetration: f64,
    origin: usize,
) -> Vec<f64> {
    let out = cdl_darkcloudcover_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("darkcloudcover").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_dojistar` with missing-candle admission.
pub fn cdl_dojistar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_dojistar_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("dojistar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_homingpigeon` with missing-candle admission.
pub fn cdl_homingpigeon(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_homingpigeon_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("homingpigeon").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_matchinglow` with missing-candle admission.
pub fn cdl_matchinglow(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_matchinglow_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("matchinglow").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_inneck` with missing-candle admission.
pub fn cdl_inneck(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_inneck_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("inneck").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_onneck` with missing-candle admission.
pub fn cdl_onneck(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_onneck_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("onneck").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_thrusting` with missing-candle admission.
pub fn cdl_thrusting(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_thrusting_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("thrusting").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_kicking` with missing-candle admission.
pub fn cdl_kicking(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_kicking_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("kicking").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_kickingbylength` with missing-candle admission.
pub fn cdl_kickingbylength(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_kickingbylength_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("kickingbylength").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_separatinglines` with missing-candle admission.
pub fn cdl_separatinglines(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_separatinglines_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("separatinglines").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_counterattack` with missing-candle admission.
pub fn cdl_counterattack(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_counterattack_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("counterattack").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_morningstar` with missing-candle admission.
pub fn cdl_morningstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64, origin: usize) -> Vec<f64> {
    let out = cdl_morningstar_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("morningstar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_eveningstar` with missing-candle admission.
pub fn cdl_eveningstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64, origin: usize) -> Vec<f64> {
    let out = cdl_eveningstar_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("eveningstar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_3inside` with missing-candle admission.
pub fn cdl_3inside(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_3inside_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("3inside").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_3outside` with missing-candle admission.
pub fn cdl_3outside(o: &[f64], h: &[f64], l: &[f64], c: &[f64], _origin: usize) -> Vec<f64> {
    let out = cdl_3outside_raw(o, h, l, c, _origin);
    let reach = candle_pattern_complete("3outside").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_3whitesoldiers` with missing-candle admission.
pub fn cdl_3whitesoldiers(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_3whitesoldiers_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("3whitesoldiers").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_3blackcrows` with missing-candle admission.
pub fn cdl_3blackcrows(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_3blackcrows_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("3blackcrows").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_morningdojistar` with missing-candle admission.
pub fn cdl_morningdojistar(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    penetration: f64,
    origin: usize,
) -> Vec<f64> {
    let out = cdl_morningdojistar_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("morningdojistar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_eveningdojistar` with missing-candle admission.
pub fn cdl_eveningdojistar(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    penetration: f64,
    origin: usize,
) -> Vec<f64> {
    let out = cdl_eveningdojistar_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("eveningdojistar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_abandonedbaby` with missing-candle admission.
pub fn cdl_abandonedbaby(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64, origin: usize) -> Vec<f64> {
    let out = cdl_abandonedbaby_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("abandonedbaby").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_2crows` with missing-candle admission.
pub fn cdl_2crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_2crows_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("2crows").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_upsidegap2crows` with missing-candle admission.
pub fn cdl_upsidegap2crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_upsidegap2crows_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("upsidegap2crows").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_advanceblock` with missing-candle admission.
pub fn cdl_advanceblock(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_advanceblock_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("advanceblock").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_stalledpattern` with missing-candle admission.
pub fn cdl_stalledpattern(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_stalledpattern_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("stalledpattern").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_identical3crows` with missing-candle admission.
pub fn cdl_identical3crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_identical3crows_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("identical3crows").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_sticksandwich` with missing-candle admission.
pub fn cdl_sticksandwich(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_sticksandwich_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("sticksandwich").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_tristar` with missing-candle admission.
pub fn cdl_tristar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_tristar_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("tristar").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_unique3river` with missing-candle admission.
pub fn cdl_unique3river(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_unique3river_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("unique3river").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_gapsidesidewhite` with missing-candle admission.
pub fn cdl_gapsidesidewhite(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_gapsidesidewhite_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("gapsidesidewhite").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_tasukigap` with missing-candle admission.
pub fn cdl_tasukigap(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_tasukigap_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("tasukigap").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_3starsinsouth` with missing-candle admission.
pub fn cdl_3starsinsouth(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_3starsinsouth_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("3starsinsouth").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_3linestrike` with missing-candle admission.
pub fn cdl_3linestrike(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_3linestrike_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("3linestrike").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_breakaway` with missing-candle admission.
pub fn cdl_breakaway(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_breakaway_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("breakaway").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_ladderbottom` with missing-candle admission.
pub fn cdl_ladderbottom(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_ladderbottom_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("ladderbottom").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_concealbabyswall` with missing-candle admission.
pub fn cdl_concealbabyswall(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_concealbabyswall_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("concealbabyswall").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_mathold` with missing-candle admission.
pub fn cdl_mathold(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64, origin: usize) -> Vec<f64> {
    let out = cdl_mathold_raw(o, h, l, c, penetration, origin);
    let reach = candle_pattern_complete("mathold").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_risefall3methods` with missing-candle admission.
pub fn cdl_risefall3methods(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_risefall3methods_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("risefall3methods").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_xsidegap3methods` with missing-candle admission.
pub fn cdl_xsidegap3methods(o: &[f64], h: &[f64], l: &[f64], c: &[f64], _origin: usize) -> Vec<f64> {
    let out = cdl_xsidegap3methods_raw(o, h, l, c, _origin);
    let reach = candle_pattern_complete("xsidegap3methods").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_hikkake` with missing-candle admission.
pub fn cdl_hikkake(o: &[f64], h: &[f64], l: &[f64], c: &[f64], _origin: usize) -> Vec<f64> {
    let out = cdl_hikkake_raw(o, h, l, c, _origin);
    let reach = candle_pattern_complete("hikkake").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}

/// Recognize `cdl_hikkakemod` with missing-candle admission.
pub fn cdl_hikkakemod(o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let out = cdl_hikkakemod_raw(o, h, l, c, origin);
    let reach = candle_pattern_complete("hikkakemod").expect("registered candle").1;
    admit_candles(out, o, h, l, c, reach)
}
