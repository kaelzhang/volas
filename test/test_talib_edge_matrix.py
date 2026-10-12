"""TA-Lib parity on edge datasets — constant, step, zero-range (H == L), ×1e-10, ×1e10,
missing (NaN) and infinite inputs — with an exact NaN-mask comparison.

The oracle is TA-Lib itself, applied under volas's documented missing-value rules
(INDICATORS.md "Missing and non-finite inputs"):

* a recursive command skips a missing row and continues as if the bar did not exist,
  so its oracle is TA-Lib over the complete rows, scattered back (NaN on the missing
  rows); an infinity is a value, so TA-Lib reads it as given;
* a finite-window command's value depends on its own window only, so its oracle is
  TA-Lib over exactly that window (NaN when the window holds a missing value).

The Hilbert family is compared bit for bit; everything else to 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

import volas
from test import ht_reference as R

talib = pytest.importorskip('talib')

# The Hilbert family's oracle: TA-Lib itself when the installed build evaluates without
# FMA contraction, else TA-Lib's C source transcribed unfused (`test/ht_reference.py`) —
# a contracted build rounds differently in the last bits (see test_talib_parity).
_CONTRACTS = R.talib_contracts(talib)


def _ht(talib_fn, ref_fn):
    if _CONTRACTS:
        return lambda c: np.asarray(ref_fn([float(x) for x in c]), dtype=float)
    return talib_fn

N = 220


def _base(seed=3):
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, N))
    o = c + rng.normal(0.0, 0.3, N)
    return {'open': o, 'high': np.maximum(o, c) + rng.random(N), 'low': np.minimum(o, c) - rng.random(N),
            'close': c, 'volume': rng.random(N) * 1e6 + 1.0}


def _datasets():
    b = _base()
    flat = {k: np.full(N, 10.0) for k in ('open', 'high', 'low', 'close')}
    step_c = np.r_[np.full(N // 2, 10.0), np.full(N - N // 2, 12.0)]
    zero_range = {k: b['close'].copy() for k in ('open', 'high', 'low', 'close')}
    missing = {k: v.copy() for k, v in b.items()}
    missing['close'][50] = missing['high'][120] = missing['low'][121] = missing['volume'][130] = np.nan
    infinite = {k: v.copy() for k, v in b.items()}
    infinite['high'][60], infinite['low'][61], infinite['volume'][70] = np.inf, -np.inf, np.inf
    return {
        'constant': {**flat, 'volume': np.full(N, 1e3)},
        'step': {'open': step_c, 'high': step_c + 0.5, 'low': step_c - 0.5, 'close': step_c,
                 'volume': np.full(N, 1e3)},
        'zero_range': {**zero_range, 'volume': b['volume']},
        'tiny': {**{k: b[k] * 1e-10 for k in ('open', 'high', 'low', 'close')}, 'volume': b['volume']},
        'huge': {**{k: b[k] * 1e10 for k in ('open', 'high', 'low', 'close')}, 'volume': b['volume']},
        'missing': missing,
        'infinite': infinite,
    }


DATASETS = _datasets()
H, L, C, V = 'high', 'low', 'close', 'volume'

# directive -> (inputs, TA-Lib function of those inputs, recursive?, exact?)
MATRIX = {
    'rsi:14': ((C,), lambda c: talib.RSI(c, 14), True, False),
    'cmo:14': ((C,), lambda c: talib.CMO(c, 14), True, False),
    'natr:14': ((H, L, C), lambda h, l, c: talib.NATR(h, l, c, 14), True, False),
    'plus_dm:1': ((H, L), lambda h, l: talib.PLUS_DM(h, l, 1), True, False),
    'plus_dm:14': ((H, L), lambda h, l: talib.PLUS_DM(h, l, 14), True, False),
    'minus_dm:1': ((H, L), lambda h, l: talib.MINUS_DM(h, l, 1), True, False),
    'minus_dm:14': ((H, L), lambda h, l: talib.MINUS_DM(h, l, 14), True, False),
    'obv': ((C, V), lambda c, v: talib.OBV(c, v), True, False),
    'ad': ((H, L, C, V), lambda h, l, c, v: talib.AD(h, l, c, v), True, False),
    'adosc:3,10': ((H, L, C, V), lambda h, l, c, v: talib.ADOSC(h, l, c, v, 3, 10), True, False),
    'apo:10,4,1': ((C,), lambda c: talib.APO(c, 10, 4, 1), True, False),
    'ppo:10,4,0': ((C,), lambda c: talib.PPO(c, 10, 4, 0), False, False),
    'ma:1,0': ((C,), lambda c: talib.MA(c, 1, 0), False, True),
    'ma:1,6': ((C,), lambda c: talib.MA(c, 1, 6), False, True),
    'ma:1,7': ((C,), lambda c: talib.MA(c, 1, 7), False, True),
    'ultosc:7,14,28': ((H, L, C), lambda h, l, c: talib.ULTOSC(h, l, c, 7, 14, 28), False, False),
    'ht_dcperiod': ((C,), _ht(talib.HT_DCPERIOD, R.ht_dcperiod), True, True),
    'ht_phasor': ((C,), _ht(lambda c: talib.HT_PHASOR(c)[0], lambda c: R.ht_phasor(c)[0]), True, True),
    'ht_phasor.quadrature': ((C,), _ht(lambda c: talib.HT_PHASOR(c)[1], lambda c: R.ht_phasor(c)[1]), True, True),
    'ht_dcphase': ((C,), _ht(talib.HT_DCPHASE, R.ht_dcphase), True, True),
    'ht_sine': ((C,), _ht(lambda c: talib.HT_SINE(c)[0], lambda c: R.ht_sine(c)[0]), True, True),
    'ht_sine.leadsine': ((C,), _ht(lambda c: talib.HT_SINE(c)[1], lambda c: R.ht_sine(c)[1]), True, True),
    'ht_trendline': ((C,), _ht(talib.HT_TRENDLINE, R.ht_trendline), True, True),
    'ht_trendmode': ((C,), _ht(lambda c: talib.HT_TRENDMODE(c).astype(float), R.ht_trendmode), True, True),
    'mama': ((C,), _ht(lambda c: talib.MAMA(c, 0.5, 0.05)[0], lambda c: R.mama(c)[0]), True, True),
}


# Finite-window commands whose first window row is read only as the prior close.
PRIOR_CLOSE = {'ultosc:7,14,28'}

# The Wilder-sum smoothing of ±DM (period >= 2) folds `s·(1 − 1/p) + term` where TA-Lib
# writes `s − s/p + term`: after an infinite movement TA-Lib's sum is NaN for good, volas's
# inf. Both are poisoned; the evaluation order of the directional family is its own item.
OPEN = {('plus_dm:14', 'infinite'), ('minus_dm:14', 'infinite')}


def _oracle(directive, data):
    inputs, fn, recursive, _ = MATRIX[directive]
    cols = [np.asarray(data[k], float) for k in inputs]
    lookback = volas.directive_lookback(directive)
    out = np.full(N, np.nan)
    if recursive:
        keep = ~np.any([np.isnan(x) for x in cols], axis=0)
        rows = np.flatnonzero(keep)
        if len(rows):
            got = np.asarray(fn(*[x[keep] for x in cols]), float)
            got[:lookback] = np.nan  # volas warms up with NaN where TA-Lib writes 0
            out[rows] = got
        return out
    for i in range(lookback, N):
        window = [x[i - lookback:i + 1].copy() for x in cols]
        if directive in PRIOR_CLOSE:
            # the window's first row only lends its close to the next bar's terms; TA-Lib
            # still reads that row's high / low, so give them the close (no term uses them)
            for k, name in enumerate(inputs):
                if name in (H, L):
                    window[k][0] = data[C][i - lookback]
        out[i] = np.asarray(fn(*window), float)[-1]
    return out


CASES = [
    pytest.param(d, s, id=f'{d}-{s}', marks=[pytest.mark.xfail(strict=True, reason='Wilder-sum order')]
                 if (d, s) in OPEN else [])
    for d in MATRIX for s in DATASETS
]


@pytest.mark.parametrize('directive,dataset', CASES)
def test_edge_dataset_matches_talib(directive, dataset):
    data = DATASETS[dataset]
    got = np.asarray(volas.DataFrame(dict(data)).exec(directive), float)
    want = _oracle(directive, data)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=f'{directive} on {dataset}: NaN mask')
    m = ~np.isnan(want)
    if MATRIX[directive][3]:
        np.testing.assert_array_equal(got[m], want[m], err_msg=f'{directive} on {dataset}')
    else:
        np.testing.assert_allclose(got[m], want[m], rtol=1e-9, atol=0, err_msg=f'{directive} on {dataset}')
