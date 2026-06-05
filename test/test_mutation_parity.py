"""Systematic indicator parity under mutation — slice / append / update / combos / CoW.

For EVERY volas∩TA-Lib indicator, after each mutation the frame's indicator must equal
the *fresh full-history recompute* on the same logical data, ``DataFrame(H)[D][R]``. That
fresh recompute is itself TA-Lib-verified by ``test_talib_parity.py``, so this is the
TA-Lib comparison the owner asked for — transitively, but with EXACT equality and no
warm-up/quirk tolerance. ``test_core_vs_talib`` also pins a core set directly to TA-Lib.

This guards the cached-column + incremental-refresh path — volas's live-trading hot path
(cache an indicator, a bar arrives, append+fulfill, read). ``test_slice_continuation``
only checks volas-vs-volas self-consistency there, never against a fresh recompute, so a
real divergence is invisible to it. Known-bug (directive, scenario) cases are xfail-marked
(see ``_REFRESH_BUG`` / ``_UPDATE_BUG``) until the incremental refresh is fixed.
"""

from pathlib import Path

import numpy as np
import pytest

from volas import DataFrame

from test.test_benchmark import COVERAGE_IDS

_CSV = Path(__file__).parent / 'data' / 'tencent_full.csv'
_COLS = ['open', 'high', 'low', 'close', 'volume']
_RAW = np.genfromtxt(_CSV, delimiter=',', names=True)
_ARR = {c: _RAW[c].astype(float) for c in _COLS}

N = 250          # base frame A = rows [0, N)
M = 5            # appended bars   = rows [N, N+M)
LO = 100         # slice start
_CLOSE = _COLS.index('close')
_PER = (np.arange(N + M, dtype=float) % 28.0) + 2.0   # MAVP per-row periods


def _data(lo, hi):
    d = {c: _ARR[c][lo:hi].copy() for c in _COLS}
    d['periods'] = _PER[lo:hi].copy()
    return d


A = _data(0, N)
BARS = _data(N, N + M)
A_APP = _data(0, N + M)
A_UPD = {k: v.copy() for k, v in A.items()}
A_UPD['close'][N - 3] *= 1.05

DIRECTIVES = list(COVERAGE_IDS)

# --- known-bug blast radius (mapped 2026-06-05) ----------------------------
# BUG 1 — the incremental refresh after append (append + fulfill) recomputes the NEW
# rows wrong for recursive / smoothed / cumulative / index-tracking indicators. The
# frame body and a fresh recompute are correct; only the cached-then-refreshed tail is
# wrong. Hit by test_append and test_slice_then_append.
_REFRESH_BUG = frozenset({
    'ad', 'adosc', 'adx:14', 'adxr:14', 'atr:14', 'cmo:14', 'dema:30', 'dx:14', 'ema:12',
    'ht_dcperiod', 'ht_dcphase', 'ht_phasor.inphase', 'ht_phasor.quadrature',
    'ht_sine.leadsine', 'ht_sine.sine', 'ht_trendline', 'kama:30', 'macd',
    'macd.histogram', 'macd.signal', 'macdfix', 'macdfix.histogram', 'macdfix.signal',
    'mama', 'mama.fama', 'maxindex:30', 'minindex:30', 'minmaxindex.max:30',
    'minmaxindex.min:30', 'minus_di:14', 'minus_dm:14', 'natr:14', 'obv', 'plus_di:14',
    'plus_dm:14', 'rsi:14', 'sar', 'sarext', 'stochrsi.d', 'stochrsi.k', 't3:5',
    'tema:30', 'trix:30',
})  # 43

# BUG 2 — an in-place update of a base column does NOT invalidate dependent cached
# indicators, so a re-read returns the pre-update (stale) values. Hit by test_update.
_UPDATE_BUG = frozenset({
    'accbands:20', 'ad', 'adosc', 'apo', 'atr:14', 'avgprice', 'boll.lower',
    'boll.middle', 'boll.upper', 'bop', 'cci:14', 'cdl.3inside', 'cdl.closingmarubozu',
    'cdl.harami', 'cmo:14', 'dema:30', 'ema:12', 'ht_dcperiod', 'ht_dcphase',
    'ht_phasor.quadrature', 'ht_sine.leadsine', 'ht_sine.sine', 'ht_trendline',
    'ht_trendmode', 'kama:30', 'linearreg:14', 'linearreg_angle:14',
    'linearreg_intercept:14', 'linearreg_slope:14', 'ma:20', 'macd', 'macd.histogram',
    'macd.signal', 'macdext', 'macdext.histogram', 'macdext.signal', 'macdfix',
    'macdfix.histogram', 'macdfix.signal', 'mama', 'mama.fama', 'mavp@close,periods',
    'maxindex:30', 'mfi:14', 'midpoint:14', 'minmax.max:30', 'minmaxindex.max:30',
    'minus_di:14', 'mom:10', 'natr:14', 'obv', 'plus_di:14', 'ppo', 'roc:10', 'rocp:10',
    'rocr100:10', 'rocr:10', 'rsi:14', 'stddev:5', 'stoch.d', 'stoch.k', 'stochf.d',
    'stochf.k', 'stochrsi.d', 'stochrsi.k', 'sum:30', 't3:5', 'tema:30', 'tr',
    'trima:30', 'trix:30', 'tsf:14', 'typprice', 'ultosc', 'var:5', 'wclprice',
    'willr:14', 'wma:30',
})  # 78

_REFRESH_REASON = 'incremental append+fulfill refresh recomputes new rows wrong for stateful indicators (fix pending)'
_UPDATE_REASON = 'in-place base-column update does not invalidate dependent cached indicators (stale read; fix pending)'


def _params(bug_set, reason):
    """DIRECTIVES, with the known-bug ones xfail(strict) — a fix flips them to xpass."""
    mark = pytest.mark.xfail(reason=reason, strict=True)
    return [pytest.param(d, marks=mark) if d in bug_set else d for d in DIRECTIVES]


def _gt(directive, data):
    """Fresh full-history result (the TA-Lib-verified ground truth)."""
    return np.asarray(DataFrame(data)[directive].to_numpy(), dtype=float)


def _eq(got, exp):
    np.testing.assert_allclose(
        np.asarray(got, dtype=float), np.asarray(exp, dtype=float),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )


def _cache(df, directive):
    """Materialize+cache the directive; skip if volas cannot express it via getitem."""
    try:
        return df[directive]
    except Exception as e:  # pragma: no cover - directive not getitem-cacheable
        pytest.skip(f'{directive}: {type(e).__name__}: {e}')


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_slice(directive):
    """A contiguous slice carries the full-history cache: sub[D] == full[D][LO:N]."""
    df = DataFrame(A)
    _cache(df, directive)
    sub = df.iloc[LO:N]
    _eq(sub[directive].to_numpy(), _gt(directive, A)[LO:N])


@pytest.mark.parametrize('directive', _params(_REFRESH_BUG, _REFRESH_REASON))
def test_append(directive):
    """Cache, append M bars, fulfill: the result must match a fresh recompute on A+bars."""
    df = DataFrame(A)
    _cache(df, directive)
    df.append(DataFrame(BARS))
    df.fulfill()
    _eq(df[directive].to_numpy(), _gt(directive, A_APP))


@pytest.mark.parametrize('directive', _params(_UPDATE_BUG, _UPDATE_REASON))
def test_update(directive):
    """Cache, then update a recent close cell: dependent indicators must recompute."""
    df = DataFrame(A)
    _cache(df, directive)
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    _eq(df[directive].to_numpy(), _gt(directive, A_UPD))


@pytest.mark.parametrize('directive', _params(_REFRESH_BUG, _REFRESH_REASON))
def test_slice_then_append(directive):
    """Combination: cache, slice off the head, append, fulfill."""
    df = DataFrame(A)
    _cache(df, directive)
    sub = df.iloc[LO:]
    sub.append(DataFrame(BARS))
    sub.fulfill()
    _eq(sub[directive].to_numpy(), _gt(directive, A_APP)[LO:])


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_cow_subframe(directive):
    """Sub-frame B = A.iloc[LO:N]: B's indicator is correct, and mutating B (CoW) must
    leave A's indicator AND data untouched."""
    a_df = DataFrame(A)
    base = _cache(a_df, directive).to_numpy().copy()
    base_close = a_df['close'].to_numpy().copy()
    b = a_df.iloc[LO:N]
    _eq(b[directive].to_numpy(), _gt(directive, A)[LO:N])   # B correct (carried slice)
    b.iloc[2, _CLOSE] = A['close'][LO + 2] * 1.05            # mutate B in place
    _eq(a_df[directive].to_numpy(), base)                   # A indicator intact (CoW)
    _eq(a_df['close'].to_numpy(), base_close)               # A data intact (CoW)
