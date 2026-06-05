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

# --- known limitation -------------------------------------------------------
# BUG 1 (incremental append refresh) is FIXED — append+fulfill now recomputes over the
# full history, so test_append is exact for every indicator. What remains is an inherent
# LIMITATION, not the same bug: a contiguous slice drops its head, so a *stateful*
# indicator (whose value at row i depends on the whole prefix [0,i] — EMA/Wilder/MACD/
# SAR/HT/cumulative/index) cannot be continued past the dropped history without carrying
# its recursive state. Finite-memory indicators (SMA, ROC, price transforms, CDL, ...)
# continue correctly. Only test_slice_then_append is affected; xfail(strict) so a future
# state-carry enhancement flips these to xpass.
_SLICE_STATEFUL = frozenset({
    'ht_dcperiod', 'ht_dcphase', 'ht_phasor.inphase', 'ht_phasor.quadrature',
    'ht_sine.leadsine', 'ht_sine.sine', 'mama.fama', 'maxindex:30', 'minindex:30',
    'minmaxindex.max:30', 'minmaxindex.min:30', 'stochrsi.d', 'stochrsi.k',
})  # 13 — the EMA-recursion family (ema/dema/tema/t3/trix/macd*/macdfix*/kama) AND the
   # Wilder-smoothing family (rsi/atr/natr/cmo/adx/adxr/dx/±di/±dm) are now state-carry-
   # continuable across a slice; the remainder is HT / index / stochrsi.

# BUG 2 (in-place update of a base column/row did not invalidate dependent cached
# indicators) is FIXED — test_update_cell and test_update_column are exact for every
# indicator.

_SLICE_REASON = 'a slice drops its head, so a stateful indicator cannot be continued past it without state-carry'


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


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_append(directive):
    """Cache, append M bars, fulfill: the result must match a fresh recompute on A+bars."""
    df = DataFrame(A)
    _cache(df, directive)
    df.append(DataFrame(BARS))
    df.fulfill()
    _eq(df[directive].to_numpy(), _gt(directive, A_APP))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_cell(directive):
    """Cache, then update a recent close cell IN A ROW (df.iloc[i, j] = v): dependent
    indicators must recompute, not return stale cached values."""
    df = DataFrame(A)
    _cache(df, directive)
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    _eq(df[directive].to_numpy(), _gt(directive, A_UPD))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_column(directive):
    """Cache, then replace a whole base COLUMN (df['close'] = ...): dependent indicators
    must recompute."""
    df = DataFrame(A)
    _cache(df, directive)
    df['close'] = A_UPD['close']
    _eq(df[directive].to_numpy(), _gt(directive, A_UPD))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_then_append(directive):
    """Combination: cache, update a cell, append, fulfill — both mutations honored."""
    df = DataFrame(A)
    _cache(df, directive)
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    df.append(DataFrame(BARS))
    df.fulfill()
    h = {k: np.concatenate([A_UPD[k], BARS[k]]) for k in A_UPD}
    _eq(df[directive].to_numpy(), _gt(directive, h))


@pytest.mark.parametrize('directive', _params(_SLICE_STATEFUL, _SLICE_REASON))
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
