"""TA-Lib's Hilbert-transform functions transcribed from the C source (TA-Lib 0.6.4
`ta_HT_DCPERIOD.c`, `ta_HT_PHASOR.c`, `ta_MAMA.c`, `ta_HT_DCPHASE.c`, `ta_HT_SINE.c`,
`ta_HT_TRENDLINE.c`, `ta_HT_TRENDMODE.c`) statement by statement, in plain IEEE double
arithmetic: Python never fuses a multiply and an add, so this is TA-Lib as an
unfused (no FMA contraction) build computes it — the reference volas reproduces bit for
bit on every platform.

`sin` / `cos` of one argument are taken together from the platform routine an optimising
C compiler emits for TA-Lib's `std_sin(t)` / `std_cos(t)` pair (`__sincos_stret` on
Apple platforms), as volas does. Outputs are full-length arrays, NaN over the warm-up.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import sys

A, B = 0.0962, 0.5769
RAD2DEG = 45.0 / math.atan(1.0)
DEG2RAD = 1.0 / RAD2DEG
DEG2RAD_BY_360 = math.atan(1.0) * 8.0


def _sin_cos():
    if sys.platform == 'darwin':
        class SinCos(ctypes.Structure):
            _fields_ = [('sin', ctypes.c_double), ('cos', ctypes.c_double)]
        fn = ctypes.CDLL(ctypes.util.find_library('m') or None).__sincos_stret
        fn.restype, fn.argtypes = SinCos, [ctypes.c_double]
        return lambda t: (lambda r: (r.sin, r.cos))(fn(t))
    return lambda t: (math.sin(t), math.cos(t))


SIN_COS = _sin_cos()


class _Hilbert:
    """One `DO_HILBERT_EVEN` / `DO_HILBERT_ODD` channel (`ta_utility.h`)."""

    def __init__(self):
        self.odd, self.even = [0.0] * 3, [0.0] * 3
        self.prev_odd = self.prev_even = self.prev_in_odd = self.prev_in_even = 0.0

    def even_step(self, x, idx, adj):
        t = A * x
        out = -self.even[idx]
        self.even[idx] = t
        out += t
        out -= self.prev_even
        self.prev_even = B * self.prev_in_even
        out += self.prev_even
        self.prev_in_even = x
        return out * adj

    def odd_step(self, x, idx, adj):
        t = A * x
        out = -self.odd[idx]
        self.odd[idx] = t
        out += t
        out -= self.prev_odd
        self.prev_odd = B * self.prev_in_odd
        out += self.prev_odd
        self.prev_in_odd = x
        return out * adj


def _run(price, warmup_pushes, on_bar):
    """The shared loop: the 4-bar WMA primed with 3 bars and `warmup_pushes` discarded
    pushes, then per bar the Hilbert core and the period update; `on_bar(today, s)` sees
    each bar's state `s` (a dict) after the update, in TA-Lib's order."""
    n = len(price)
    today = trailing_idx = 0
    t = price[today]; today += 1
    sub = t; wsum = t
    t = price[today]; today += 1
    sub += t; wsum += t * 2.0
    t = price[today]; today += 1
    sub += t; wsum += t * 3.0
    trailing = 0.0
    for _ in range(warmup_pushes):
        t = price[today]; today += 1
        sub += t; sub -= trailing; wsum += t * 4.0
        trailing = price[trailing_idx]; trailing_idx += 1
        wsum -= sub
    det, q1c, ji, jq = _Hilbert(), _Hilbert(), _Hilbert(), _Hilbert()
    s = {'period': 0.0, 'smooth_period': 0.0, 'prev_q2': 0.0, 'prev_i2': 0.0, 're': 0.0, 'im': 0.0,
         'i1_odd3': 0.0, 'i1_even3': 0.0, 'i1_odd2': 0.0, 'i1_even2': 0.0}
    hidx = 0
    while today < n:
        adj = (0.075 * s['period']) + 0.54
        v = price[today]
        sub += v; sub -= trailing; wsum += v * 4.0
        trailing = price[trailing_idx]; trailing_idx += 1
        smoothed = wsum * 0.1
        wsum -= sub
        s['smoothed'] = smoothed
        if today % 2 == 0:
            d = det.even_step(smoothed, hidx, adj)
            q1 = q1c.even_step(d, hidx, adj)
            j_i = ji.even_step(s['i1_even3'], hidx, adj)
            j_q = jq.even_step(q1, hidx, adj)
            hidx = 0 if hidx == 2 else hidx + 1
            q2 = (0.2 * (q1 + j_i)) + (0.8 * s['prev_q2'])
            i2 = (0.2 * (s['i1_even3'] - j_q)) + (0.8 * s['prev_i2'])
            s['i1'] = s['i1_even3']
            s['i1_odd3'] = s['i1_odd2']
            s['i1_odd2'] = d
        else:
            d = det.odd_step(smoothed, hidx, adj)
            q1 = q1c.odd_step(d, hidx, adj)
            j_i = ji.odd_step(s['i1_odd3'], hidx, adj)
            j_q = jq.odd_step(q1, hidx, adj)
            q2 = (0.2 * (q1 + j_i)) + (0.8 * s['prev_q2'])
            i2 = (0.2 * (s['i1_odd3'] - j_q)) + (0.8 * s['prev_i2'])
            s['i1'] = s['i1_odd3']
            s['i1_even3'] = s['i1_even2']
            s['i1_even2'] = d
        s['q1'] = q1
        pi2, pq2 = s['prev_i2'], s['prev_q2']
        s['re'] = (0.2 * ((i2 * pi2) + (q2 * pq2))) + (0.8 * s['re'])
        s['im'] = (0.2 * ((i2 * pq2) - (q2 * pi2))) + (0.8 * s['im'])
        s['prev_q2'], s['prev_i2'] = q2, i2
        prev = s['period']
        period = prev
        if s['im'] != 0.0 and s['re'] != 0.0:
            period = 360.0 / (math.atan(s['im'] / s['re']) * RAD2DEG)
        if period > 1.5 * prev:
            period = 1.5 * prev
        if period < 0.67 * prev:
            period = 0.67 * prev
        if period < 6:
            period = 6.0
        elif period > 50:
            period = 50.0
        s['period'] = (0.2 * period) + (0.8 * prev)
        s['smooth_period'] = (0.33 * s['period']) + (0.67 * s['smooth_period'])
        on_bar(today, s)
        today += 1


def _nan(n):
    return [math.nan] * n


def ht_dcperiod(price):
    out = _nan(len(price))
    _run(price, 9, lambda t, s: out.__setitem__(t, s['smooth_period']) if t >= 32 else None)
    return out


def ht_phasor(price):
    inphase, quad = _nan(len(price)), _nan(len(price))

    def bar(t, s):
        if t >= 32:
            inphase[t], quad[t] = s['i1'], s['q1']
    _run(price, 9, bar)
    return inphase, quad


def mama(price, fast_limit=0.5, slow_limit=0.05):
    n = len(price)
    out_m, out_f = _nan(n), _nan(n)
    st = {'mama': 0.0, 'fama': 0.0, 'prev_phase': 0.0}

    # MAMA reads the phase before the period update; the update does not touch i1 / q1.
    def bar(t, s):
        phase = (math.atan(s['q1'] / s['i1']) * RAD2DEG) if s['i1'] != 0.0 else 0.0
        delta = st['prev_phase'] - phase
        st['prev_phase'] = phase
        if delta < 1.0:
            delta = 1.0
        if delta > 1.0:
            alpha = fast_limit / delta
            if alpha < slow_limit:
                alpha = slow_limit
        else:
            alpha = fast_limit
        st['mama'] = (alpha * price[t]) + ((1 - alpha) * st['mama'])
        alpha *= 0.5
        st['fama'] = (alpha * st['mama']) + ((1 - alpha) * st['fama'])
        if t >= 32:
            out_m[t], out_f[t] = st['mama'], st['fama']
    _run(price, 9, bar)
    return out_m, out_f


def _phase_family(price):
    """Per bar from bar 37: (DCPhase, smooth_period, smoothed, trendline) in TA-Lib's order."""
    rows = {}
    ring, ring_idx = [0.0] * 50, [0]
    st = {'phase': 0.0, 'it1': 0.0, 'it2': 0.0, 'it3': 0.0}

    def bar(t, s):
        ring[ring_idx[0]] = s['smoothed']
        sp = s['smooth_period']
        dc_int = int(sp + 0.5) if sp + 0.5 >= 0.0 else 0
        real = imag = 0.0
        idx = ring_idx[0]
        for i in range(dc_int):
            sn, cs = SIN_COS((i * DEG2RAD_BY_360) / dc_int)
            real += sn * ring[idx]
            imag += cs * ring[idx]
            idx = 49 if idx == 0 else idx - 1
        phase = st['phase']
        if abs(imag) > 0.0:
            phase = math.atan(real / imag) * RAD2DEG
        elif abs(imag) <= 0.01:
            if real < 0.0:
                phase -= 90.0
            elif real > 0.0:
                phase += 90.0
        phase += 90.0
        phase += 360.0 / sp
        if imag < 0.0:
            phase += 180.0
        if phase > 315.0:
            phase -= 360.0
        st['phase'] = phase
        avg = 0.0
        for j in range(dc_int):
            avg += price[t - j]
        if dc_int > 0:
            avg = avg / dc_int
        trend = (4.0 * avg + 3.0 * st['it1'] + 2.0 * st['it2'] + st['it3']) / 10.0
        st['it3'], st['it2'], st['it1'] = st['it2'], st['it1'], avg
        rows[t] = (phase, sp, s['smoothed'], trend)
        ring_idx[0] = 0 if ring_idx[0] == 49 else ring_idx[0] + 1
    _run(price, 34, bar)
    return rows


def ht_dcphase(price):
    out = _nan(len(price))
    for t, (phase, *_rest) in _phase_family(price).items():
        if t >= 63:
            out[t] = phase
    return out


def ht_sine(price):
    sine, lead = _nan(len(price)), _nan(len(price))
    for t, (phase, *_rest) in _phase_family(price).items():
        if t >= 63:
            sine[t], lead[t] = math.sin(phase * DEG2RAD), math.sin((phase + 45) * DEG2RAD)
    return sine, lead


def ht_trendline(price):
    out = _nan(len(price))
    for t, (*_rest, trend) in _phase_family(price).items():
        if t >= 63:
            out[t] = trend
    return out


def ht_trendmode(price):
    out = _nan(len(price))
    days, prev_phase, sine, lead = 0, 0.0, 0.0, 0.0
    for t, (phase, sp, smoothed, trend) in sorted(_phase_family(price).items()):
        prev_sine, prev_lead = sine, lead
        sine, lead = math.sin(phase * DEG2RAD), math.sin((phase + 45) * DEG2RAD)
        mode = 1
        if (sine > lead and prev_sine <= prev_lead) or (sine < lead and prev_sine >= prev_lead):
            days, mode = 0, 0
        days += 1
        if days < 0.5 * sp:
            mode = 0
        d = phase - prev_phase
        prev_phase = phase
        if sp != 0.0 and 0.67 * 360.0 / sp < d < 1.5 * 360.0 / sp:
            mode = 0
        if trend != 0.0 and abs((smoothed - trend) / trend) >= 0.015:
            mode = 1
        if t >= 63:
            out[t] = float(mode)
    return out


def talib_contracts(talib) -> bool:
    """Whether the installed TA-Lib was built with floating-point contraction (its
    `x*y + z` fused into FMA, as clang does on arm64): its HT_DCPERIOD then differs from
    this unfused transcription in the last bits of a fixed, non-degenerate series."""
    from test.exact import same_exact

    import numpy as np

    x = [100.0 + 0.05 * t + 6.0 * math.sin(t * 0.3) + 2.0 * math.cos(t * 0.11) for t in range(200)]
    got = np.asarray(talib.HT_DCPERIOD(np.asarray(x)), dtype=float)
    return not same_exact(got, np.asarray(ht_dcperiod(x)))
