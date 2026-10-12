"""TA-Lib's smoothing recurrences transcribed from the C source (TA-Lib 0.6.4 `ta_EMA.c`,
`ta_RSI.c`, `ta_CMO.c`, `ta_ATR.c`, `ta_NATR.c`, `ta_DEMA.c`, `ta_TEMA.c`, `ta_TRIX.c`,
`ta_T3.c`, `ta_KAMA.c`, `ta_ADOSC.c`, `ta_ADX.c`, `ta_STOCHF.c`, `ta_STOCH.c`,
`ta_STOCHRSI.c`) statement by statement, in plain IEEE double arithmetic: NumPy never
fuses a multiply and an add, so this is TA-Lib as an unfused (no FMA contraction) build
computes it — the reference volas reproduces bit for bit on every platform. The volas
commands built on the same steps (SMMA, the MACD line, EFI, TSI, Mass Index) are written
here on those steps too.

volas's documented conventions where it departs from TA-Lib are kept: a recursive command
runs on the complete rows of its inputs (a row missing an input is deleted and gets NaN);
the flat-window tests of RSI / CMO / NATR are exact zero tests (TA-Lib's `TA_IS_ZERO` is an
absolute 1e-14); a per-bar term computed from a missing value is missing; the MACD line is
`EMA(fast) − EMA(slow)` from the slow EMA's seed, its signal the EMA of that line.
Outputs are full-length arrays, NaN over the warm-up.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

F = np.float64
NAN = F(np.nan)


def _complete(fn, *cols):
    """Run `fn` on the rows where every column is present; NaN at the others."""
    arrs = [np.asarray(c, dtype=float) for c in cols]
    keep = np.logical_and.reduce([~np.isnan(a) for a in arrs])
    out = np.full(len(keep), np.nan)
    if keep.any():
        with np.errstate(all='ignore'):
            out[keep] = fn(*[[F(v) for v in a[keep]] for a in arrs])
    return out


def ema_k(period):
    """`PER_TO_K`: `2.0 / (double)(period + 1)`."""
    return F(2.0) / F(period + 1)


def ema_step(prev, x, k):
    """`ta_EMA.c`: `prevMA = ((inReal[today++] - prevMA) * optInK_1) + prevMA`."""
    return ((x - prev) * k) + prev


def weighted_step(prev, x, k):
    """`ta_T3.c` / `ta_ADOSC.c`: `e = (k * x) + (one_minus_k * e)`."""
    return (k * x) + ((F(1.0) - k) * prev)


def wilder_step(prev, x, period):
    """`ta_ATR.c` / `ta_RSI.c`: `prev *= period - 1; prev += x; prev /= period`."""
    return ((prev * F(period - 1)) + x) / F(period)


def _seeded(x, period, step):
    """The SMA-seeded recurrence (`ta_EMA.c`'s default seed): the mean of the first
    `period` present values, summed in row order from 0, then `step(prev, x)` per row."""
    out = [NAN] * len(x)
    total, count = F(0.0), 0
    for i, v in enumerate(x):
        if count < period:
            if not math.isnan(v):
                total += v
                count += 1
                if count == period:
                    out[i] = total / F(period)
        else:
            out[i] = step(out[i - 1], v)
    return out


def _ema_list(x, period):
    k = ema_k(period)
    return _seeded(x, period, lambda prev, v: ema_step(prev, v, k))


def ema(close, period):
    return _complete(lambda c: _ema_list(c, period), close)


def smma(close, period):
    return _complete(lambda c: _seeded(c, period, lambda prev, v: wilder_step(prev, v, period)), close)


def dema(close, period):
    """`ta_DEMA.c`: `(2.0 * firstEMA) - secondEMA`."""
    def run(c):
        e1 = _ema_list(c, period)
        e2 = _ema_list(e1, period)
        return [(F(2.0) * a) - b for a, b in zip(e1, e2)]
    return _complete(run, close)


def tema(close, period):
    """`ta_TEMA.c`: `thirdEMA + ((3.0 * firstEMA) - (3.0 * secondEMA))`."""
    def run(c):
        e1 = _ema_list(c, period)
        e2 = _ema_list(e1, period)
        e3 = _ema_list(e2, period)
        return [z + ((F(3.0) * a) - (F(3.0) * b)) for a, b, z in zip(e1, e2, e3)]
    return _complete(run, close)


def trix(close, period):
    """`ta_TRIX.c`: the 1-period ROC `((x / prev) - 1.0) * 100.0` (0 on a zero base) of
    the triple EMA."""
    def run(c):
        e3 = _ema_list(_ema_list(_ema_list(c, period), period), period)
        out = [NAN] * len(c)
        for i in range(1, len(c)):
            prev, x = e3[i - 1], e3[i]
            if math.isnan(prev) or math.isnan(x):
                continue
            out[i] = F(0.0) if prev == 0.0 else ((x / prev) - F(1.0)) * F(100.0)
        return out
    return _complete(run, close)


def t3(close, period, vfactor=0.7):
    """`ta_T3.c`: six cascaded EMAs in the weighted form, each stage seeded with the mean
    of its first `period` inputs; `c1*e6 + c2*e5 + c3*e4 + c4*e3`."""
    def run(c):
        n = len(c)
        out = [NAN] * n
        lookback = 6 * (period - 1)
        if lookback >= n:
            return out
        k = F(2.0) / (F(period) + F(1.0))
        e = []
        today = 0
        temp = F(c[today])
        today += 1
        for _ in range(period - 1):
            temp += c[today]
            today += 1
        e.append(temp / F(period))
        for stage in range(1, 6):
            temp = e[stage - 1]
            for _ in range(period - 1):
                x = c[today]
                today += 1
                for s in range(stage):
                    e[s] = weighted_step(e[s], x, k)
                    x = e[s]
                temp += e[stage - 1]
            e.append(temp / F(period))
        v = F(vfactor)
        v2 = v * v
        c1 = -(v2 * v)
        c2 = F(3.0) * (v2 - c1)
        c3 = F(-6.0) * v2 - F(3.0) * (v - c1)
        c4 = F(1.0) + F(3.0) * v - c1 + F(3.0) * v2

        def combine():
            return c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2]

        out[lookback] = combine()
        for i in range(lookback + 1, n):
            x = c[i]
            for s in range(6):
                e[s] = weighted_step(e[s], x, k)
                x = e[s]
            out[i] = combine()
        return out
    return _complete(run, close)


def kama(close, period):
    """`ta_KAMA.c` with volas's exact window sum of 1-bar changes (correctly rounded,
    `math.fsum`): `sc = (er * constDiff) + constMax; sc *= sc; kama = ((x − kama) * sc) +
    kama`."""
    def run(c):
        n = len(c)
        out = [NAN] * n
        if period == 0 or period + 1 > n:
            return out
        const_max = F(2.0) / (F(30.0) + F(1.0))
        const_diff = F(2.0) / (F(2.0) + F(1.0)) - const_max
        prev = c[period - 1]
        for today in range(period, n):
            total = F(math.fsum(abs(c[j] - c[j - 1]) for j in range(today + 1 - period, today + 1)))
            roc = c[today] - c[today - period]
            er = F(1.0) if (total <= roc or abs(total) < 1e-14) else abs(roc / total)
            sc = (er * const_diff) + const_max
            sc = sc * sc
            prev = ((c[today] - prev) * sc) + prev
            out[today] = prev
        return out
    return _complete(run, close)


def macd(close, fast, slow, signal=None, line='macd'):
    """volas's MACD: `EMA(fast) − EMA(slow)` (periods ordered by TA-Lib's rule), the
    signal the SMA-seeded EMA of that line, the histogram `line − signal`."""
    fast, slow = min(fast, slow), max(fast, slow)

    def run(c):
        ef, es = _ema_list(c, fast), _ema_list(c, slow)
        macd_line = [a - b for a, b in zip(ef, es)]
        if line == 'macd':
            return macd_line
        sig = _ema_list(macd_line, signal)
        return sig if line == 'signal' else [a - b for a, b in zip(macd_line, sig)]
    return _complete(run, close)


def _gain_loss(c, period, value):
    """`ta_RSI.c` / `ta_CMO.c`: the seed sums the first `period` changes (a negative
    change into the loss, any other into the gain), divides each by the period, then per
    bar `prev *= period−1`, the change added to its side, `/= period`."""
    n = len(c)
    out = [NAN] * n
    if period == 0 or n <= period:
        return out
    gain, loss = F(0.0), F(0.0)
    for i in range(1, period + 1):
        d = c[i] - c[i - 1]
        if d < 0:
            loss -= d
        else:
            gain += d
    loss /= F(period)
    gain /= F(period)
    out[period] = value(gain, loss)
    for i in range(period + 1, n):
        d = c[i] - c[i - 1]
        loss *= F(period - 1)
        gain *= F(period - 1)
        if d < 0:
            loss -= d
        else:
            gain += d
        loss /= F(period)
        gain /= F(period)
        out[i] = value(gain, loss)
    return out


def rsi(close, period):
    def value(g, l):
        s = g + l
        return F(0.0) if s == 0.0 else F(100.0) * (g / s)
    return _complete(lambda c: _gain_loss(c, period, value), close)


def cmo(close, period):
    def value(g, l):
        s = g + l
        return F(0.0) if s == 0.0 else F(100.0) * ((g - l) / s)
    return _complete(lambda c: _gain_loss(c, period, value), close)


def _true_ranges(h, l, c):
    """volas's TR (`tr_at`): `max(h − l, |h − pc|, |l − pc|)`, a NaN operand skipped
    like `f64::max`; none on the first bar."""
    tr = [NAN]
    for i in range(1, len(c)):
        tr.append(F(np.fmax(np.fmax(h[i] - l[i], abs(h[i] - c[i - 1])), abs(l[i] - c[i - 1]))))
    return tr


def atr(high, low, close, period):
    def run(h, l, c):
        return _seeded(_true_ranges(h, l, c), period, lambda prev, v: wilder_step(prev, v, period))
    return _complete(run, high, low, close)


def natr(high, low, close, period):
    """`ta_NATR.c`: `(atr / close) * 100.0`, 0 on a zero close (exact)."""
    def run(h, l, c):
        a = _seeded(_true_ranges(h, l, c), period, lambda prev, v: wilder_step(prev, v, period))
        return [F(0.0) if (x == 0.0 and not math.isnan(v)) else (v / x) * F(100.0) for v, x in zip(a, c)]
    return _complete(run, high, low, close)


def adosc(high, low, close, volume, fast, slow):
    """`ta_ADOSC.c`: the A/D line `ad += ((cl − l) − (h − cl)) / (h − l) * v` (no flow
    without a positive range — missing when a value it reads is), both EMAs seeded with
    the first A/D value and stepped in the weighted form; masked until
    `max(fast, slow) − 1`."""
    def run(h, l, c, v):
        n = len(c)
        out = [NAN] * n
        kf, ks = ema_k(fast), ema_k(slow)
        ad = F(0.0)
        fe = se = None
        for i in range(n):
            loc, rng = (c[i] - l[i]) - (h[i] - c[i]), h[i] - l[i]
            if rng > 0.0:
                term = (loc / rng) * v[i]
            else:
                term = NAN if any(math.isnan(x) for x in (loc, rng, v[i])) else F(0.0)
            ad = ad + term
            if fe is None:
                fe = se = ad
            else:
                fe = weighted_step(fe, ad, kf)
                se = weighted_step(se, ad, ks)
            if i >= max(fast, slow) - 1:
                out[i] = fe - se
        return out
    return _complete(run, high, low, close, volume)


def efi(close, volume, period):
    """Elder's force index: the EMA of `(close − prev close) · volume`."""
    def run(c, v):
        raw = [NAN] + [(c[i] - c[i - 1]) * v[i] for i in range(1, len(c))]
        return _ema_list(raw, period)
    return _complete(run, close, volume)


def tsi(close, long, short):
    """True strength index: `100 · EMA_s(EMA_l(m)) / EMA_s(EMA_l(|m|))`."""
    def run(c):
        m = [NAN] + [c[i] - c[i - 1] for i in range(1, len(c))]
        am = [abs(x) for x in m]
        num = _ema_list(_ema_list(m, long), short)
        den = _ema_list(_ema_list(am, long), short)
        return [(F(100.0) * a) / b for a, b in zip(num, den)]
    return _complete(run, close)


def _fast_k_rows(h, l, c, period):
    """`ta_STOCHF.c`'s raw %K over each full window: `diff = (hh − ll) / 100.0`, then
    `(close − ll) / diff`, 0 on a flat range. A window holding a missing value is missing
    (volas's finite-window rule)."""
    n = len(c)
    out = [NAN] * n
    for i in range(period - 1, n):
        hs, ls = h[i + 1 - period:i + 1], l[i + 1 - period:i + 1]
        if any(math.isnan(v) for v in (*hs, *ls, c[i])):
            continue
        hh, ll = F(max(hs)) + F(0.0), F(min(ls)) + F(0.0)
        diff = (hh - ll) / F(100.0)
        out[i] = (c[i] - ll) / diff if diff != 0.0 else F(0.0)
    return out


def stochf_k(high, low, close, period):
    with np.errstate(all='ignore'):
        h, l, c = ([F(v) for v in np.asarray(a, dtype=float)] for a in (high, low, close))
        return np.asarray(_fast_k_rows(h, l, c, period), dtype=float)


def _exact_mean(window):
    """The correctly rounded mean of a finite window (volas's exact window sums); a
    window with a non-finite value takes its own plain mean."""
    if all(math.isfinite(v) for v in window):
        return F(float(sum(Fraction(float(v)) for v in window) / len(window)))
    return F(sum(window, F(0.0)) / F(len(window)))


def _sma_rows(x, period):
    """The SMA of a series that warms up with NaN: from the first present value, a
    window holding a missing value is missing."""
    n = len(x)
    out = [NAN] * n
    start = next((i for i, v in enumerate(x) if not math.isnan(v)), n)
    for i in range(start + period - 1, n):
        w = x[i + 1 - period:i + 1]
        if not any(math.isnan(v) for v in w):
            out[i] = _exact_mean(w)
    return out


def stoch_k(high, low, close, fastk, slowk):
    """STOCH's slow %K with an SMA: the SMA of the raw %K."""
    with np.errstate(all='ignore'):
        h, l, c = ([F(v) for v in np.asarray(a, dtype=float)] for a in (high, low, close))
        return np.asarray(_sma_rows(_fast_k_rows(h, l, c, fastk), slowk), dtype=float)


def stochrsi_k(close, rsi_period, fastk):
    """STOCHRSI's %K: the raw %K over RSI (high = low = close = RSI) — a recursive
    command, so on the complete rows (its %K window spans a deleted row)."""
    def run(c):
        def value(g, l):
            s = g + l
            return F(0.0) if s == 0.0 else F(100.0) * (g / s)
        r = _gain_loss(c, rsi_period, value)
        return _fast_k_rows(r, r, r, fastk)
    return _complete(run, close)


def adx(high, low, close, period):
    """`ta_ADX.c` over volas's TR: Wilder's sums `s − s/p + term` of ±DM and TR seeded
    over bars `[1, p)`, ±DI `100·(dm/tr)` and DX `100·(|−DI − +DI| / (−DI + +DI))` with
    TA-Lib's ~0 guards, the ADX seed the mean of the DX of bars `[p, 2p)`, then
    `((adx·(p − 1)) + dx) / p`."""
    def zero(x):
        return -1e-14 < x < 1e-14

    def run(h, l, c):
        n = len(c)
        out = [NAN] * n
        tr = _true_ranges(h, l, c)
        pm = pp = ptr = F(0.0)

        def dm(t):
            dp, dmn = h[t] - h[t - 1], l[t - 1] - l[t]
            return (dp if (dp > 0 and dp > dmn) else F(0.0)), (dmn if (dmn > 0 and dp < dmn) else F(0.0))

        for t in range(1, period):
            a, b = dm(t)
            pp, pm, ptr = pp + a, pm + b, ptr + tr[t]
        sum_dx, avg, pf = F(0.0), NAN, F(period)
        for t in range(period, n):
            a, b = dm(t)
            pp = pp - pp / pf + a
            pm = pm - pm / pf + b
            ptr = ptr - ptr / pf + tr[t]
            dx = None
            if not zero(ptr):
                pdi, mdi = F(100.0) * (pp / ptr), F(100.0) * (pm / ptr)
                if not zero(mdi + pdi):
                    dx = F(100.0) * (abs(mdi - pdi) / (mdi + pdi))
            if t < 2 * period:
                sum_dx += F(0.0) if dx is None else dx
                if t == 2 * period - 1:
                    avg = sum_dx / pf
                    out[t] = avg
            else:
                if dx is not None:
                    avg = wilder_step(avg, dx, period)
                out[t] = avg
        return out
    return _complete(run, high, low, close)


def mass_index(high, low, period):
    """Mass Index: the `period`-row sum (`mean·period`, the exact mean) of `EMA9(H − L) /
    EMA9(EMA9(H − L))`."""
    def run(h, l):
        single = _ema_list([a - b for a, b in zip(h, l)], 9)
        double = _ema_list(single, 9)
        ratio = [a / b for a, b in zip(single, double)]
        return [m * F(period) for m in _sma_rows(ratio, period)]
    return _complete(run, high, low)
