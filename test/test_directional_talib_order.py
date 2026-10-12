"""The directional family evaluates TA-Lib's C source (0.6.4) in its own order.

Wilder's sums are `s − s/p + term` (`ta_PLUS_DM.c`, `ta_DX.c`, `ta_ADX.c`), ±DI is
`100·(dm/tr)`, DX is `100·(|−DI − +DI| / (−DI + +DI))` and the ADX average
`((adx·(p−1)) + dx) / p` — so an infinite movement poisons every later sum into NaN
exactly as TA-Lib does, and finite values agree to the last bit. ±DM / ±DI / DX contain no
`x*y + z` a compiler could fuse, so TA-Lib itself is their oracle; the ADX average does,
and a contracting TA-Lib build (clang on arm64) fuses it, so ADX / ADXR are checked
against `_adx_reference` — the C source transcribed in Python, which never fuses — unless
the installed build evaluates unfused. Every path is checked: the full computation, a
multi-row append (resume) and row-by-row appends (the single-row resume).
"""

from __future__ import annotations

import numpy as np
import pytest

import volas
from test import ht_reference as R

talib = pytest.importorskip('talib')

_CONTRACTS = R.talib_contracts(talib)


def _ohlc(n, seed):
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    o = c + rng.normal(0.0, 0.3, n)
    return np.maximum(o, c) + rng.random(n), np.minimum(o, c) - rng.random(n), c


def _true_range(h, l, prev_close):
    out = h - l
    t = abs(prev_close - h)
    if t > out:
        out = t
    t = abs(prev_close - l)
    if t > out:
        out = t
    return out


def _adx_reference(high, low, close, p, rating=False):
    """TA-Lib 0.6.4 `TA_ADX` (and `TA_ADXR` over it), statement for statement."""
    n = len(high)
    out = np.full(n, np.nan)
    pm = pp = ptr = 0.0
    for t in range(1, p):
        dp, dm = high[t] - high[t - 1], low[t - 1] - low[t]
        if dm > 0 and dp < dm:
            pm += dm
        elif dp > 0 and dp > dm:
            pp += dp
        ptr += _true_range(high[t], low[t], close[t - 1])
    sum_dx = 0.0
    adx = np.nan
    for t in range(p, n):
        dp, dm = high[t] - high[t - 1], low[t - 1] - low[t]
        pm -= pm / p
        pp -= pp / p
        if dm > 0 and dp < dm:
            pm += dm
        elif dp > 0 and dp > dm:
            pp += dp
        ptr = ptr - (ptr / p) + _true_range(high[t], low[t], close[t - 1])
        mdi = 100.0 * (pm / ptr)
        pdi = 100.0 * (pp / ptr)
        dx = 100.0 * (abs(mdi - pdi) / (mdi + pdi))
        if t < 2 * p:
            sum_dx += dx
            if t == 2 * p - 1:
                adx = sum_dx / p
                out[t] = adx
        else:
            adx = ((adx * (p - 1)) + dx) / p
            out[t] = adx
    if not rating:
        return out
    rated = np.full(n, np.nan)
    for t in range(3 * p - 2, n):
        rated[t] = (out[t] + out[t - (p - 1)]) / 2.0
    return rated


def _oracle(name, h, l, c, p):
    if name in ('adx', 'adxr') and _CONTRACTS:
        return _adx_reference(h, l, c, p, rating=name == 'adxr')
    fn = getattr(talib, name.upper())
    out = np.asarray(fn(h, l, p) if name.endswith('_dm') else fn(h, l, c, p), float)
    out[:volas.directive_lookback(f'{name}:{p}')] = np.nan  # TA-Lib writes 0 / NaN there
    return out


FAMILY = ['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx', 'adx', 'adxr']


def _frame(h, l, c):
    return volas.DataFrame({'open': c, 'high': h, 'low': l, 'close': c})


def _assert_bits(got, want, what):
    got, want = np.asarray(got, float), np.asarray(want, float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=f'{what}: NaN mask')
    m = ~np.isnan(want)
    assert np.array_equal(got[m].view(np.uint64), want[m].view(np.uint64)), \
        f'{what}: first differing row {np.flatnonzero(got[m] != want[m])[:1]}'


@pytest.mark.parametrize('period', [2, 5, 14, 30])
@pytest.mark.parametrize('name', FAMILY)
def test_full_computation_is_talib_bit_for_bit(name, period):
    for seed in range(3):
        h, l, c = _ohlc(400, seed)
        got = _frame(h, l, c).exec(f'{name}:{period}')
        _assert_bits(got, _oracle(name, h, l, c, period), f'{name}:{period} seed {seed}')


@pytest.mark.parametrize('name', FAMILY)
def test_resume_paths_are_talib_bit_for_bit(name):
    p, n, head = 14, 260, 120
    h, l, c = _ohlc(n, 7)
    want = _oracle(name, h, l, c, p)
    # multi-row appends: the Vec resume
    live = _frame(h[:head], l[:head], c[:head])
    live.exec(f'{name}:{p}')
    for i, j in ((head, head + 40), (head + 40, head + 90)):
        live.append(_frame(h[i:j], l[i:j], c[i:j]))
    # row-by-row appends: the single-row resume
    for i in range(head + 90, n):
        live.append(_frame(h[i:i + 1], l[i:i + 1], c[i:i + 1]))
    _assert_bits(live.exec(f'{name}:{p}'), want, f'{name}:{p} live')


@pytest.mark.parametrize('name', FAMILY)
def test_an_infinite_movement_poisons_the_sums_like_talib(name):
    # TA-Lib's `s − s/p` turns an infinite sum into NaN for good; a reassociated
    # `s·(1 − 1/p) + term` would keep it infinite
    p, n = 5, 120
    h, l, c = _ohlc(n, 11)
    h[40] = np.inf
    l[70] = -np.inf
    want = _oracle(name, h, l, c, p)
    assert np.isnan(want[80:]).all()
    _assert_bits(_frame(h, l, c).exec(f'{name}:{p}'), want, f'{name}:{p} full')
    live = _frame(h[:30], l[:30], c[:30])
    live.exec(f'{name}:{p}')
    for i in range(30, n):
        live.append(_frame(h[i:i + 1], l[i:i + 1], c[i:i + 1]))
    _assert_bits(live.exec(f'{name}:{p}'), want, f'{name}:{p} live')
