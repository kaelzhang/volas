"""Source-pinned reference oracle for non-TA-Lib indicators.

These indicators have **no TA-Lib function**, so volas's "parity vs TA-Lib" gate
(``test_talib_parity``) cannot cover them. This module supplies an *independent*
reference for each: a small pure-numpy / pandas implementation whose formula is
**pinned to a cited authoritative source**. ``test_oracle.py`` then checks the volas
directive against this reference on the Tencent fixture.

Scope (this stage): the **Group A** roadmap indicators from
``tasks/04/research/2026-06-07-volas-indicator-gap-and-naming-report.md`` §9, plus the
already-shipped non-TA-Lib ``bbi`` which validates the harness end-to-end *today* (its
oracle test runs and passes now; the Group A cases skip until each indicator lands).

Convention pins (matching volas's kernels, so a reference will agree once the indicator
is implemented):

* ``_sma`` — rolling mean, NaN for the first ``n-1`` rows.
* ``_ema`` — EMA seeded with the SMA of the first ``n`` (TA-Lib convention, which volas's
  ``ema`` uses), not pandas's first-value seed.
* ``_rsi`` — Wilder RSI seeded with the SMA of the first ``n`` gains/losses (TA-Lib).

Caveat: a reference is only as correct as its cited formula. Cross-validating each one
against a second independent source / library (e.g. pandas-ta) is the deferred "thorough
oracle" work; until then every reference carries a source URL and a pinned convention so
it is reviewable, and each is exercised for shape/finiteness by ``test_oracle.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- shared, convention-pinned primitives ----------------------------------

def _s(x) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float))


def _sma(x, n: int) -> pd.Series:
    return _s(x).rolling(n, min_periods=n).mean()


def _ema(x, n: int) -> pd.Series:
    """EMA with TA-Lib seeding (first value = SMA of the first ``n`` valid samples, which
    volas's ``ema`` uses). Leading NaNs are skipped so chained EMAs (TSI, Mass Index) seed
    from the first real value instead of an all-NaN window."""
    vals = _s(x).to_numpy()
    out = np.full(len(vals), np.nan)
    valid = np.flatnonzero(~np.isnan(vals))
    if len(valid) >= n:
        fv = int(valid[0])
        start = fv + n - 1
        out[start] = vals[fv:fv + n].mean()
        alpha = 2.0 / (n + 1.0)
        for i in range(start + 1, len(vals)):
            out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out)


def _wma(x, n: int) -> pd.Series:
    """TA-Lib WMA: linear weights 1..n (newest heaviest), NaN for the first n-1 valid samples.
    Leading NaNs are skipped (matching volas's ``wma``), so it seeds from the first real value."""
    vals = _s(x).to_numpy()
    out = np.full(len(vals), np.nan)
    valid = np.flatnonzero(~np.isnan(vals))
    if len(valid) >= n:
        w = np.arange(1, n + 1, dtype=float)
        denom = w.sum()
        for i in range(int(valid[0]) + n - 1, len(vals)):
            window = vals[i - n + 1:i + 1]
            if not np.isnan(window).any():
                out[i] = float((window * w).sum() / denom)
    return pd.Series(out)


def _rsi(x, n: int) -> pd.Series:
    """Wilder RSI (TA-Lib seeding): SMA of the first ``n`` gains/losses, then Wilder smoothing."""
    s = _s(x)
    d = s.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = np.full(len(s), np.nan)
    al = np.full(len(s), np.nan)
    if len(s) > n:
        ag[n] = gain.iloc[1:n + 1].mean()
        al[n] = loss.iloc[1:n + 1].mean()
        for i in range(n + 1, len(s)):
            ag[i] = (ag[i - 1] * (n - 1) + gain.iat[i]) / n
            al[i] = (al[i - 1] * (n - 1) + loss.iat[i]) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = ag / al
    return pd.Series(100.0 - 100.0 / (1.0 + rs))


def _true_range(h, lo, c) -> pd.Series:
    h, lo, c = _s(h), _s(lo), _s(c)
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    # The first bar has no prior close, so TR is undefined there (TA-Lib TRANGE /
    # TradingView `ta.tr` default). pandas `.max(axis=1)` skips the NaN, so blank it.
    tr[pc.isna()] = np.nan
    return tr


def _atr(h, lo, c, n: int) -> pd.Series:
    """Wilder ATR (TA-Lib seeding): SMA of the first ``n`` true ranges (tr[0] is NaN), then
    Wilder smoothing. Matches volas's ``atr``."""
    tr = _true_range(h, lo, c).to_numpy()
    out = np.full(len(tr), np.nan)
    if len(tr) > n:
        out[n] = float(np.nanmean(tr[1:n + 1]))
        for i in range(n + 1, len(tr)):
            out[i] = out[i - 1] * (n - 1) / n + tr[i] / n
    return pd.Series(out)


def _stddev(x, n: int) -> pd.Series:
    """Population rolling standard deviation (TA-Lib STDDEV, nbdev=1)."""
    return _s(x).rolling(n, min_periods=n).std(ddof=0)


def _linreg(x, n: int) -> pd.Series:
    """TA-Lib LINEARREG endpoint = intercept + slope·(n−1), regressing on x = 0..n−1
    (oldest→newest). Skips a leading-NaN prefix (matching the volas kernel fed a finite tail)."""
    vals = _s(x).to_numpy()
    out = np.full(len(vals), np.nan)
    valid = np.flatnonzero(~np.isnan(vals))
    if len(valid) >= n:
        xs = np.arange(n, dtype=float)
        sx, sxx = xs.sum(), (xs * xs).sum()
        denom = n * sxx - sx * sx
        for i in range(int(valid[0]) + n - 1, len(vals)):
            window = vals[i - n + 1:i + 1]
            if np.isnan(window).any():
                continue
            sy, sxy = window.sum(), (xs * window).sum()
            m = (n * sxy - sx * sy) / denom
            b = (sy - m * sx) / n
            out[i] = b + m * (n - 1)
    return pd.Series(out)


# --- Group A references (each cites its pinned source) ----------------------

def bbi(o, h, lo, c, v):
    """BBI 多空指标 = mean(MA3, MA6, MA12, MA24) of close (SMA).
    Source: MBA Wiki 多空指数 (BBI). Already shipped by volas; validates the harness."""
    cs = _s(c)
    return (_sma(cs, 3) + _sma(cs, 6) + _sma(cs, 12) + _sma(cs, 24)) / 4.0


def psy(o, h, lo, c, v, n=12):
    """PSY 心理线 = 100 * (rising days in last n) / n; rising = close > prev close.
    Source: Eastmoney 心理线 <https://baike.eastmoney.com/item/心理线>."""
    up = (_s(c).diff() > 0).astype(float)
    return up.rolling(n, min_periods=n).sum() / n * 100.0


def emv(o, h, lo, c, v, n=14, scale=100_000_000.0):
    """EMV Ease of Movement (StockCharts). distance = mid - prev mid;
    box = (volume/scale)/(high-low); 1-period EMV = distance/box; EMV = SMA_n.
    Source: StockCharts ChartSchool — Ease of Movement. The volume `scale` is a
    presentation convention pinned here to StockCharts' 100,000,000."""
    h, lo, vv = _s(h), _s(lo), _s(v)
    mid = (h + lo) / 2.0
    distance = mid - mid.shift(1)
    box = (vv / scale) / (h - lo)
    return (distance / box).rolling(n, min_periods=n).mean()


def cmf(o, h, lo, c, v, n=20):
    """Chaikin Money Flow = sum_n(MFV) / sum_n(volume), MFV = ((C-L)-(H-C))/(H-L) * V.
    Source: StockCharts ChartSchool / Fidelity — Chaikin Money Flow."""
    h, lo, c, vv = _s(h), _s(lo), _s(c), _s(v)
    mfm = ((c - lo) - (h - c)) / (h - lo)
    mfv = mfm * vv
    return mfv.rolling(n, min_periods=n).sum() / vv.rolling(n, min_periods=n).sum()


def dpo(o, h, lo, c, v, n=20):
    """Detrended Price Oscillator = Price[(n/2 + 1) ago] - SMA_n.
    Source: StockCharts ChartSchool — Detrended Price Oscillator (displaced form)."""
    c = _s(c)
    shift = n // 2 + 1
    return c.shift(shift) - _sma(c, n)


def pvt(o, h, lo, c, v):
    """Price Volume Trend (cumulative): PVT_i = PVT_{i-1} + (C_i-C_{i-1})/C_{i-1} * V_i,
    PVT_0 = 0. Source: StockCharts / Investopedia — Price Volume Trend."""
    c, vv = _s(c), _s(v)
    term = (c.pct_change() * vv).fillna(0.0)
    return term.cumsum()


def nvi(o, h, lo, c, v, base=1000.0):
    """Negative Volume Index: starts at base; on a down-volume day NVI *= (1 + ROC),
    else unchanged. Source: StockCharts — Negative Volume Index."""
    c, vv = _s(c), _s(v)
    roc = c.pct_change()
    out = np.full(len(c), base)
    for i in range(1, len(c)):
        out[i] = out[i - 1] * (1.0 + roc.iat[i]) if vv.iat[i] < vv.iat[i - 1] else out[i - 1]
    return pd.Series(out)


def pvi(o, h, lo, c, v, base=1000.0):
    """Positive Volume Index: starts at base; on an up-volume day PVI *= (1 + ROC),
    else unchanged. Source: StockCharts — Positive Volume Index."""
    c, vv = _s(c), _s(v)
    roc = c.pct_change()
    out = np.full(len(c), base)
    for i in range(1, len(c)):
        out[i] = out[i - 1] * (1.0 + roc.iat[i]) if vv.iat[i] > vv.iat[i - 1] else out[i - 1]
    return pd.Series(out)


def mass_index(o, h, lo, c, v, n=25, ema_n=9):
    """Mass Index = sum_n( EMA9(H-L) / EMA9(EMA9(H-L)) ).
    Source: StockCharts ChartSchool — Mass Index."""
    rng = _s(h) - _s(lo)
    single = _ema(rng, ema_n)
    double = _ema(single, ema_n)
    return (single / double).rolling(n, min_periods=n).sum()


def efi(o, h, lo, c, v, n=13):
    """Elder Force Index = EMA_n( (C - prev C) * volume ).
    Source: StockCharts / Investopedia — Force Index."""
    raw = (_s(c).diff() * _s(v))
    return _ema(raw, n)


def tsi(o, h, lo, c, v, long=25, short=13):
    """True Strength Index = 100 * EMA_short(EMA_long(m)) / EMA_short(EMA_long(|m|)),
    m = C - prev C. Source: StockCharts ChartSchool — True Strength Index."""
    m = _s(c).diff()
    num = _ema(_ema(m, long), short)
    den = _ema(_ema(m.abs(), long), short)
    return 100.0 * num / den


def kst(o, h, lo, c, v):
    """Know Sure Thing (Pring): weighted sum of four SMA-smoothed ROCs.
    RCMA1=SMA10(ROC10), RCMA2=SMA10(ROC15), RCMA3=SMA10(ROC20), RCMA4=SMA15(ROC30);
    KST = 1*RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4. Source: StockCharts — KST."""
    c = _s(c)

    def roc(n):
        return (c / c.shift(n) - 1.0) * 100.0

    r1 = _sma(roc(10), 10)
    r2 = _sma(roc(15), 10)
    r3 = _sma(roc(20), 10)
    r4 = _sma(roc(30), 15)
    return r1 * 1.0 + r2 * 2.0 + r3 * 3.0 + r4 * 4.0


def chop(o, h, lo, c, v, n=14):
    """Choppiness Index = 100 * log10( sum_n(TR) / (maxHigh_n - minLow_n) ) / log10(n).
    Source: TradingView — Choppiness Index."""
    tr = _true_range(h, lo, c)
    num = tr.rolling(n, min_periods=n).sum()
    rng = _s(h).rolling(n, min_periods=n).max() - _s(lo).rolling(n, min_periods=n).min()
    return 100.0 * np.log10(num / rng) / np.log10(n)


def crsi(o, h, lo, c, v, rsi_len=3, streak_len=2, rank_len=100):
    """Connors RSI = mean( RSI(close, rsi_len), RSI(streak, streak_len),
    PercentRank(1-period ROC, rank_len) ). streak = signed run length of up/down closes;
    PercentRank = % of the last rank_len values strictly below the current one.
    Source: Connors Research / TradingView — Connors RSI."""
    c = _s(c)
    diff = c.diff()
    streak = np.zeros(len(c))
    for i in range(1, len(c)):
        if diff.iat[i] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1.0
        elif diff.iat[i] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1.0
        else:
            streak[i] = 0.0
    roc1 = c.pct_change() * 100.0
    prank = np.full(len(c), np.nan)
    vals = roc1.to_numpy()
    for i in range(len(c)):
        lo = i - rank_len
        if lo >= 1:  # need a full rank_len window of prior values
            window = vals[lo:i]
            prank[i] = (window < vals[i]).sum() / rank_len * 100.0
    return (_rsi(c, rsi_len) + _rsi(pd.Series(streak), streak_len) + pd.Series(prank)) / 3.0


# --- Group B references (gap report §9; each cites its pinned convention) ----

def vortex(o, h, lo, c, v, n=14, plus=True):
    """Vortex Indicator. +VM=|H-prevL|, -VM=|L-prevH|; +VI=Σn(+VM)/Σn(TR), -VI=Σn(-VM)/Σn(TR).
    Source: StockCharts ChartSchool / Wikipedia — Vortex Indicator."""
    h, lo, c = _s(h), _s(lo), _s(c)
    tr = _true_range(h, lo, c)
    vm = (h - lo.shift(1)).abs() if plus else (lo - h.shift(1)).abs()
    return vm.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum()


def brar_ar(o, h, lo, c, v, n=26):
    """BRAR AR (人气指标) = Σn(H-O) / Σn(O-L) * 100.
    Source: 通达信 / MBA智库 — 人气意愿指标 (BRAR)."""
    o, h, lo = _s(o), _s(h), _s(lo)
    return (h - o).rolling(n, min_periods=n).sum() / (o - lo).rolling(n, min_periods=n).sum() * 100.0


def brar_br(o, h, lo, c, v, n=26):
    """BRAR BR (意愿指标) = Σn max(0,H-Cy) / Σn max(0,Cy-L) * 100, Cy=prev close (通达信 clamp).
    Source: 通达信 / MBA智库 — 人气意愿指标 (BRAR)."""
    h, lo, c = _s(h), _s(lo), _s(c)
    cy = c.shift(1)
    up = (h - cy).clip(lower=0.0)
    dn = (cy - lo).clip(lower=0.0)
    return up.rolling(n, min_periods=n).sum() / dn.rolling(n, min_periods=n).sum() * 100.0


def vr(o, h, lo, c, v, n=26):
    """VR 成交量比率 = (UVS + ½PVS) / (DVS + ½PVS) * 100 over n bars (up/down/flat-close volume).
    Source: MBA智库 — 成交量比率 (VR)."""
    c, vv = _s(c), _s(v)
    dc = c.diff()
    uv = vv.where(dc > 0, 0.0)
    dv = vv.where(dc < 0, 0.0)
    pv = vv.where(dc == 0, 0.0)
    uv[dc.isna()] = np.nan
    dv[dc.isna()] = np.nan
    pv[dc.isna()] = np.nan
    suv = uv.rolling(n, min_periods=n).sum()
    sdv = dv.rolling(n, min_periods=n).sum()
    spv = pv.rolling(n, min_periods=n).sum()
    return (suv + 0.5 * spv) / (sdv + 0.5 * spv) * 100.0


def _swma4(x) -> pd.Series:
    """4-bar symmetric weighted MA, weights [1, 2, 2, 1] / 6 (newest first)."""
    x = _s(x)
    return (x + 2.0 * x.shift(1) + 2.0 * x.shift(2) + x.shift(3)) / 6.0


def coppock(o, h, lo, c, v, w=10, roc_long=14, roc_short=11):
    """Coppock Curve = WMA_w(ROC_long + ROC_short), ROC_p = (C/C_p - 1) * 100.
    Source: StockCharts ChartSchool / Wikipedia — Coppock Curve."""
    c = _s(c)

    def roc(p):
        return (c / c.shift(p) - 1.0) * 100.0

    return _wma(roc(roc_long) + roc(roc_short), w)


def relative_vigor(o, h, lo, c, v, n=10):
    """RVI = SMA_n(swma4(C-O)) / SMA_n(swma4(H-L)).
    Source: Fidelity / MetaTrader — Relative Vigor Index."""
    co = _s(c) - _s(o)
    hl = _s(h) - _s(lo)
    return _sma(_swma4(co), n) / _sma(_swma4(hl), n)


def relative_vigor_signal(o, h, lo, c, v, n=10):
    """RVI signal line = swma4(RVI). Source: Fidelity / MetaTrader — Relative Vigor Index."""
    return _swma4(relative_vigor(o, h, lo, c, v, n))


def dkx(o, h, lo, c, v):
    """DKX 多空线 = WMA(MID, 20), MID = (3C + L + O + H) / 6.
    Source: 百度百科 / 东方财富 — 多空线 (DKX)."""
    mid = (3.0 * _s(c) + _s(lo) + _s(o) + _s(h)) / 6.0
    return _wma(mid, 20)


def dkx_ma(o, h, lo, c, v, m=10):
    """MADKX = SMA_m(DKX). Source: 百度百科 / 东方财富 — 多空线 (DKX)."""
    return _sma(dkx(o, h, lo, c, v), m)


def wvad(o, h, lo, c, v, n=24):
    """WVAD 威廉变异离散量 = Σn( (C-O)/(H-L) * V ). Source: 通达信 / MBA智库 — WVAD."""
    w = (_s(c) - _s(o)) / (_s(h) - _s(lo)) * _s(v)
    return w.rolling(n, min_periods=n).sum()


def cdp(o, h, lo, c, v, line='cdp'):
    """CDP 逆势操作 (prior bar): CDP=(H+L+2C)/4; AH=CDP+(H-L), NH=2CDP-L, NL=2CDP-H, AL=CDP-(H-L).
    Source: 百度百科 / 维基 — 逆势操作 (CDP)."""
    h, lo, c = _s(h).shift(1), _s(lo).shift(1), _s(c).shift(1)
    cdp_ = (h + lo + 2.0 * c) / 4.0
    return {
        'cdp': cdp_,
        'ah': cdp_ + (h - lo),
        'nh': 2.0 * cdp_ - lo,
        'nl': 2.0 * cdp_ - h,
        'al': cdp_ - (h - lo),
    }[line]


def mike(o, h, lo, c, v, n=12, line='weakr'):
    """MIKE 麦克支撑压力. TYP=(H+L+C)/3, HH/LL = n-day max(H)/min(L). WEKR=TYP+(TYP-LL),
    MIDR=TYP+(HH-LL), STOR=2HH-LL; WEKS=TYP-(HH-TYP), MIDS=TYP-(HH-LL), STOS=2LL-HH.
    Source: 百度百科 / MBA智库 — 麦克指标 (MIKE)."""
    h, lo, c = _s(h), _s(lo), _s(c)
    typ = (h + lo + c) / 3.0
    hh = h.rolling(n, min_periods=n).max()
    ll = lo.rolling(n, min_periods=n).min()
    return {
        'weakr': typ + (typ - ll),
        'midr': typ + (hh - ll),
        'strongr': 2.0 * hh - ll,
        'weaks': typ - (hh - typ),
        'mids': typ - (hh - ll),
        'strongs': 2.0 * ll - hh,
    }[line]


def pivot_points(o, h, lo, c, v, line='pp'):
    """Standard floor Pivot Points (prior bar): PP=(H+L+C)/3; R1=2PP-L, S1=2PP-H,
    R2=PP+(H-L), S2=PP-(H-L), R3=H+2(PP-L), S3=L-2(H-PP).
    Source: Investopedia / floor-trader standard — Pivot Points."""
    h, lo, c = _s(h).shift(1), _s(lo).shift(1), _s(c).shift(1)
    pp = (h + lo + c) / 3.0
    return {
        'pp': pp,
        'r1': 2 * pp - lo, 's1': 2 * pp - h,
        'r2': pp + (h - lo), 's2': pp - (h - lo),
        'r3': h + 2 * (pp - lo), 's3': lo - 2 * (h - pp),
    }[line]


def keltner(o, h, lo, c, v, ema_period=20, atr_period=10, mult=2.0, band=None):
    """Keltner Channels (modern): middle = EMA(close, ema_period); bands = middle ± mult*ATR.
    Source: StockCharts ChartSchool — Keltner Channels."""
    mid = _ema(c, ema_period)
    if band is None:
        return mid
    atr = _atr(h, lo, c, atr_period)
    sign = 1.0 if band == 'upper' else -1.0
    return mid + sign * mult * atr


def stoch_momentum(o, h, lo, c, v, k=10, d=3, signal=3, line='smi'):
    """Stochastic Momentum Index (Blau / LazyBear): HH=max(H,k), LL=min(L,k); D=C-(HH+LL)/2;
    Ds=EMA_d(EMA_d(D)), Dhl=EMA_d(EMA_d(HH-LL)); SMI=Ds/(Dhl/2)*100; signal=EMA_signal(SMI).
    Source: William Blau / LazyBear — Stochastic Momentum Index."""
    h, lo, c = _s(h), _s(lo), _s(c)
    hh = h.rolling(k, min_periods=k).max()
    ll = lo.rolling(k, min_periods=k).min()
    rdiff = c - (hh + ll) / 2.0
    diff = hh - ll
    ds = _ema(_ema(rdiff, d), d)
    dhl = _ema(_ema(diff, d), d)
    smi = ds / (dhl * 0.5) * 100.0
    return _ema(smi, signal) if line == 'signal' else smi


def ttm_squeeze(o, h, lo, c, v, n=20, bb_mult=2.0, kc_mult=1.5, line='momentum'):
    """TTM Squeeze (John Carter / thinkorswim). momentum = linreg_n(C − ((HHn+LLn)/2 + SMAn)/2);
    on = 1 when Bollinger(n, bb_mult·σ) sits inside Keltner(n, kc_mult·SMAn(TR)), else 0.
    Source: John Carter / StockCharts — TTM Squeeze."""
    h, lo, c = _s(h), _s(lo), _s(c)
    sma = _sma(c, n)
    if line == 'on':
        sd = _stddev(c, n)
        atr = _sma(_true_range(h, lo, c), n)
        bb_u, bb_l = sma + bb_mult * sd, sma - bb_mult * sd
        kc_u, kc_l = sma + kc_mult * atr, sma - kc_mult * atr
        on = ((bb_l > kc_l) & (bb_u < kc_u)).astype(float)
        on[bb_u.isna() | kc_u.isna()] = np.nan
        return on
    hh = h.rolling(n, min_periods=n).max()
    ll = lo.rolling(n, min_periods=n).min()
    delta = c - ((hh + ll) / 2.0 + sma) / 2.0
    return _linreg(delta, n)


# --- the oracle case registry ----------------------------------------------
# (directive, reference_fn(o,h,lo,c,v) -> Series, tolerance). The directive strings are
# the proposed command interface the Group A implementation should match.

CASES: list[tuple] = [
    ("bbi", bbi, 1e-7),                         # already implemented -> runs + passes today
    ("psy:12", lambda o, h, lo, c, v: psy(o, h, lo, c, v, 12), 1e-7),
    ("emv:14", lambda o, h, lo, c, v: emv(o, h, lo, c, v, 14), 1e-6),
    ("cmf:20", lambda o, h, lo, c, v: cmf(o, h, lo, c, v, 20), 1e-7),
    ("dpo:20", lambda o, h, lo, c, v: dpo(o, h, lo, c, v, 20), 1e-7),
    ("pvt", pvt, 1e-6),
    ("nvi", nvi, 1e-6),
    ("pvi", pvi, 1e-6),
    ("mass_index:25", lambda o, h, lo, c, v: mass_index(o, h, lo, c, v, 25), 1e-6),
    ("efi:13", lambda o, h, lo, c, v: efi(o, h, lo, c, v, 13), 1e-6),
    ("tsi:25,13", lambda o, h, lo, c, v: tsi(o, h, lo, c, v, 25, 13), 1e-6),
    ("kst", kst, 1e-6),
    ("chop:14", lambda o, h, lo, c, v: chop(o, h, lo, c, v, 14), 1e-7),
    ("crsi:3,2,100", lambda o, h, lo, c, v: crsi(o, h, lo, c, v, 3, 2, 100), 1e-6),
    # Group B (gap report §9).
    ("vortex.plus:14", lambda o, h, lo, c, v: vortex(o, h, lo, c, v, 14, True), 1e-7),
    ("vortex.minus:14", lambda o, h, lo, c, v: vortex(o, h, lo, c, v, 14, False), 1e-7),
    ("brar.ar:26", lambda o, h, lo, c, v: brar_ar(o, h, lo, c, v, 26), 1e-7),
    ("brar.br:26", lambda o, h, lo, c, v: brar_br(o, h, lo, c, v, 26), 1e-7),
    ("vr:26", lambda o, h, lo, c, v: vr(o, h, lo, c, v, 26), 1e-7),
    ("coppock:10,14,11", lambda o, h, lo, c, v: coppock(o, h, lo, c, v, 10, 14, 11), 1e-6),
    ("relative_vigor:10", lambda o, h, lo, c, v: relative_vigor(o, h, lo, c, v, 10), 1e-7),
    ("relative_vigor.signal:10", lambda o, h, lo, c, v: relative_vigor_signal(o, h, lo, c, v, 10), 1e-7),
    ("dkx", dkx, 1e-6),
    ("dkx.ma:10", lambda o, h, lo, c, v: dkx_ma(o, h, lo, c, v, 10), 1e-6),
    ("wvad:24", lambda o, h, lo, c, v: wvad(o, h, lo, c, v, 24), 1e-6),
    ("cdp", lambda o, h, lo, c, v: cdp(o, h, lo, c, v, 'cdp'), 1e-7),
    ("cdp.ah", lambda o, h, lo, c, v: cdp(o, h, lo, c, v, 'ah'), 1e-7),
    ("cdp.nh", lambda o, h, lo, c, v: cdp(o, h, lo, c, v, 'nh'), 1e-7),
    ("cdp.nl", lambda o, h, lo, c, v: cdp(o, h, lo, c, v, 'nl'), 1e-7),
    ("cdp.al", lambda o, h, lo, c, v: cdp(o, h, lo, c, v, 'al'), 1e-7),
    ("mike.weakr:12", lambda o, h, lo, c, v: mike(o, h, lo, c, v, 12, 'weakr'), 1e-7),
    ("mike.midr:12", lambda o, h, lo, c, v: mike(o, h, lo, c, v, 12, 'midr'), 1e-7),
    ("mike.strongr:12", lambda o, h, lo, c, v: mike(o, h, lo, c, v, 12, 'strongr'), 1e-7),
    ("mike.weaks:12", lambda o, h, lo, c, v: mike(o, h, lo, c, v, 12, 'weaks'), 1e-7),
    ("mike.mids:12", lambda o, h, lo, c, v: mike(o, h, lo, c, v, 12, 'mids'), 1e-7),
    ("mike.strongs:12", lambda o, h, lo, c, v: mike(o, h, lo, c, v, 12, 'strongs'), 1e-7),
    ("keltner:20", lambda o, h, lo, c, v: keltner(o, h, lo, c, v, 20, 10, 2.0, None), 1e-6),
    ("keltner.upper:20,10,2", lambda o, h, lo, c, v: keltner(o, h, lo, c, v, 20, 10, 2.0, 'upper'), 1e-6),
    ("keltner.lower:20,10,2", lambda o, h, lo, c, v: keltner(o, h, lo, c, v, 20, 10, 2.0, 'lower'), 1e-6),
    ("stoch_momentum:10,3,3", lambda o, h, lo, c, v: stoch_momentum(o, h, lo, c, v, 10, 3, 3, 'smi'), 1e-6),
    ("stoch_momentum.signal:10,3,3", lambda o, h, lo, c, v: stoch_momentum(o, h, lo, c, v, 10, 3, 3, 'signal'), 1e-6),
    ("ttm_squeeze:20,2,1.5", lambda o, h, lo, c, v: ttm_squeeze(o, h, lo, c, v, 20, 2.0, 1.5, 'momentum'), 1e-6),
    ("ttm_squeeze.on:20,2,1.5", lambda o, h, lo, c, v: ttm_squeeze(o, h, lo, c, v, 20, 2.0, 1.5, 'on'), 1e-9),
    ("pivot_points", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 'pp'), 1e-9),
    ("pivot_points.r1", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 'r1'), 1e-9),
    ("pivot_points.s1", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 's1'), 1e-9),
    ("pivot_points.r2", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 'r2'), 1e-9),
    ("pivot_points.s2", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 's2'), 1e-9),
    ("pivot_points.r3", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 'r3'), 1e-9),
    ("pivot_points.s3", lambda o, h, lo, c, v: pivot_points(o, h, lo, c, v, 's3'), 1e-9),
]
