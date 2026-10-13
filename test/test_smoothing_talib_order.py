"""The EMA / Wilder family evaluates TA-Lib's C source (0.6.4) in its own order, unfused.

TA-Lib's smoothing steps — the EMA `((x − prev)·k) + prev` (`ta_EMA.c`), Wilder's average
`prev·(p − 1)`, `+ x`, `/ p` (`ta_RSI.c`, `ta_CMO.c`, `ta_ATR.c`, `ta_NATR.c`) and the
weighted EMA `(k·x) + ((1 − k)·prev)` (`ta_T3.c`, `ta_ADOSC.c`) — and the KAMA constant
`(er·diff) + max` are evaluated as written, never as a fused multiply-add, so every
platform computes the same bits as an unfused TA-Lib build. Each command is checked bit
for bit against `test/smoothing_reference.py` (the C source transcribed in NumPy scalars,
which never fuse) on clean, constant, step, tiny, huge, gapped (missing rows) and
infinite data, through every path: the full computation, multi-row appends (the resume)
and row-by-row appends (the single-row resume). Against the installed TA-Lib: its RSI /
CMO / ATR / NATR statements contain no `x·y + z` a compiler could contract, so they agree
bit for bit with any build; its EMA / T3 / KAMA / ADOSC expressions do, so they agree bit
for bit with an unfused build and to a stated tolerance with a contracting one (clang on
arm64).
"""

from __future__ import annotations

import numpy as np
import pytest

import volas
from test import smoothing_reference as R

from test.exact import assert_exact, same_exact

N = 160


def _ohlcv(kind, seed=3):
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, N))
    if kind == 'swing':  # crossing zero: the step's product is as large as its sum, so
        # a fused multiply-add rounds differently from the C source on most rows
        c = 100.0 * np.sin(np.arange(N) * 1.3) + rng.normal(0.0, 1.0, N)
    elif kind == 'constant':
        c = np.full(N, 42.5)
    elif kind == 'step':
        c = np.where(np.arange(N) < N // 2, 10.0, 12.5)
    elif kind == 'tiny':
        c = c * 1e-9
    elif kind == 'huge':
        c = c * 1e300
    o = c + rng.normal(0.0, 0.3, N) * (c / 100.0)
    h = np.maximum(o, c) + rng.random(N) * np.abs(c / 100.0)
    lo = np.minimum(o, c) - rng.random(N) * np.abs(c / 100.0)
    if kind in ('constant', 'step'):
        o, h, lo = c.copy(), c.copy(), c.copy()
    v = np.abs(rng.normal(1.0e6, 2.0e5, N)) + 1.0
    if kind == 'gapped':
        for col in (o, h, lo, c, v):
            col[[30, 31, 77, 120]] = np.nan
    elif kind == 'inf':
        c[60], h[60] = np.inf, np.inf
        c[100], lo[100] = -np.inf, -np.inf
    return {'open': o, 'high': h, 'low': lo, 'close': c, 'volume': v}


KINDS = ['clean', 'swing', 'constant', 'step', 'tiny', 'huge', 'gapped', 'inf']

# directive -> reference over the bar columns
CASES = {
    'ema:10': lambda d: R.ema(d['close'], 10),
    'ema:2': lambda d: R.ema(d['close'], 2),
    'smma:7': lambda d: R.smma(d['close'], 7),
    'ma:10,1': lambda d: R.ema(d['close'], 10),
    'dema:6': lambda d: R.dema(d['close'], 6),
    'tema:5': lambda d: R.tema(d['close'], 5),
    'trix:6': lambda d: R.trix(d['close'], 6),
    't3:4,0.7': lambda d: R.t3(d['close'], 4, 0.7),
    'ma:4,8': lambda d: R.t3(d['close'], 4, 0.7),
    'kama:10': lambda d: R.kama(d['close'], 10),
    'macd:12,26': lambda d: R.macd(d['close'], 12, 26),
    'macd.signal:5,13,4': lambda d: R.macd(d['close'], 5, 13, 4, 'signal'),
    'macd.histogram:13,5,4': lambda d: R.macd(d['close'], 13, 5, 4, 'histogram'),
    'rsi:14': lambda d: R.rsi(d['close'], 14),
    'rsi:3': lambda d: R.rsi(d['close'], 3),
    'cmo:9': lambda d: R.cmo(d['close'], 9),
    'atr:14': lambda d: R.atr(d['high'], d['low'], d['close'], 14),
    'natr:5': lambda d: R.natr(d['high'], d['low'], d['close'], 5),
    'adosc:3,10': lambda d: R.adosc(d['high'], d['low'], d['close'], d['volume'], 3, 10),
    'efi:13': lambda d: R.efi(d['close'], d['volume'], 13),
    'tsi:25,13': lambda d: R.tsi(d['close'], 25, 13),
    'stochf.k:5': lambda d: R.stochf_k(d['high'], d['low'], d['close'], 5),
    'stoch.k:5,3': lambda d: R.stoch_k(d['high'], d['low'], d['close'], 5, 3),
    'stochrsi.k:14,5': lambda d: R.stochrsi_k(d['close'], 14, 5),
    'adx:14': lambda d: R.adx(d['high'], d['low'], d['close'], 14),
    'mass_index:25': lambda d: R.mass_index(d['high'], d['low'], 25),
}


def _frame(data, i=0, j=N):
    return volas.DataFrame({k: v[i:j] for k, v in data.items()})


def _assert_bits(got, want, what):
    assert_exact(got, want, what)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('directive', list(CASES))
def test_every_path_is_the_c_source_order_bit_for_bit(directive, kind):
    data = _ohlcv(kind)
    want = CASES[directive](data)
    _assert_bits(_frame(data).exec(directive), want, f'{directive} {kind} full')
    head = 70
    live = _frame(data, 0, head)
    live[directive]  # cached: every append resumes it
    for i, j in ((head, head + 25), (head + 25, head + 60)):
        live.append(_frame(data, i, j))
    for i in range(head + 60, N):
        live.append(_frame(data, i, i + 1))
    _assert_bits(live[directive], want, f'{directive} {kind} resume')


talib = pytest.importorskip('talib')

# TA-Lib functions whose smoothing statements a contracting compiler cannot fuse: bit
# for bit against any build (TA-Lib writes 0 / NaN over the warm-up; volas NaN).
_UNFUSABLE = {
    'rsi:14': lambda d: talib.RSI(d['close'], 14),
    'cmo:9': lambda d: talib.CMO(d['close'], 9),
    'atr:14': lambda d: talib.ATR(d['high'], d['low'], d['close'], 14),
    'natr:5': lambda d: talib.NATR(d['high'], d['low'], d['close'], 5),
    # the raw %K `(close − ll) / ((hh − ll) / 100)` (`ta_STOCHF.c`), over RSI for StochRSI
    'stochf.k:5': lambda d: talib.STOCHF(d['high'], d['low'], d['close'], 5, 3, 0)[0],
    'stochrsi.k:14,5': lambda d: talib.STOCHRSI(d['close'], 14, 5, 3, 0)[0],
}

# TA-Lib functions with an `x·y + z` expression on the recurrence: bit for bit against an
# unfused build, to `_FUSED_RTOL` against a contracting one.
_FUSABLE = {
    'ema:10': lambda d: talib.EMA(d['close'], 10),
    'dema:6': lambda d: talib.DEMA(d['close'], 6),
    'tema:5': lambda d: talib.TEMA(d['close'], 5),
    'trix:6': lambda d: talib.TRIX(d['close'], 6),
    't3:4,0.7': lambda d: talib.T3(d['close'], 4, 0.7),
    'kama:10': lambda d: talib.KAMA(d['close'], 10),
    'adosc:3,10': lambda d: talib.ADOSC(d['high'], d['low'], d['close'], d['volume'], 3, 10),
}

# the agreement with a contracting build: rounding-level drift of the recurrences
_FUSED_RTOL = 1e-12


def _swing(n):
    """A series crossing zero: the EMA step's product is as large as its sum, so a fused
    multiply-add and the C source round differently on most rows."""
    return 100.0 * np.sin(np.arange(n) * 1.3) + np.cos(np.arange(n) * 0.7)


def _talib_contracts_ema():
    """Whether the installed TA-Lib's EMA step is FMA-contracted: on a zero-crossing
    series its EMA then differs from the unfused C transcription in the last bits."""
    x = _swing(300)
    return not same_exact(talib.EMA(x, 9), R.ema(x, 9))


_CONTRACTS = _talib_contracts_ema()


def _talib_frame(seed):
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 400)) if seed else _swing(400) + 300.0
    o = c + rng.normal(0.0, 0.3, 400)
    h = np.maximum(o, c) + rng.random(400)
    lo = np.minimum(o, c) - rng.random(400)
    v = np.abs(rng.normal(1.0e6, 2.0e5, 400)) + 1.0
    return {'open': o, 'high': h, 'low': lo, 'close': c, 'volume': v}


def _published(directive, got, want):
    """volas and TA-Lib from the first row both publish: volas NaN over its own warm-up;
    TA-Lib writes 0 / NaN over its warm-up, which for a multi-output function (STOCHF)
    is aligned to its longest output."""
    lb = volas.directive_lookback(directive)
    got = np.asarray(got, float)
    want = np.asarray(want, float)
    assert np.isnan(got[:lb]).all()
    start = max(lb, int(np.flatnonzero(~np.isnan(want))[0]))
    return got[start:], want[start:]


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('directive', list(_UNFUSABLE))
def test_unfusable_talib_functions_agree_bit_for_bit(directive, seed):
    d = _talib_frame(seed)
    got, want = _published(directive, volas.DataFrame(d).exec(directive), _UNFUSABLE[directive](d))
    _assert_bits(got, want, f'{directive} vs TA-Lib')


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('directive', list(_FUSABLE))
def test_fusable_talib_functions_agree_with_the_installed_build(directive, seed):
    d = _talib_frame(seed)
    got, want = _published(directive, volas.DataFrame(d).exec(directive), _FUSABLE[directive](d))
    if _CONTRACTS:
        scale = np.nanmax(np.abs(want))
        np.testing.assert_allclose(got, want, rtol=_FUSED_RTOL, atol=_FUSED_RTOL * scale, equal_nan=True)
    else:
        _assert_bits(got, want, f'{directive} vs unfused TA-Lib')


@pytest.mark.skipif(not _CONTRACTS, reason='the installed TA-Lib is unfused: checked bit for bit')
def test_a_contracting_talib_build_is_what_the_reference_is_not():
    # the probe is meaningful: on this build TA-Lib's EMA is the fused step, and volas
    # (unfused) is the C source order, not the build's contraction of it
    x = _swing(300)
    got = np.asarray(talib.EMA(x, 9), float)
    want = R.ema(x, 9)
    assert (got != want)[9:].sum() > 100  # the build's contraction differs on most rows
    _assert_bits(volas.DataFrame({'close': x}).exec('ema:9'), want, 'ema:9')
