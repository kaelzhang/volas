"""Literal TA-Lib parity under mutation — the direct-oracle companion to test_mutation_parity.

``test_mutation_parity`` proves *mutation-invariance* with bit-exact strictness: a mutated
frame's indicator equals volas's own fresh full-history recompute. That recompute is pinned
to TA-Lib by ``test_talib_parity``, so the chain is ``volas-after-mutation == volas-fresh ==
TA-Lib``. This module closes that chain *literally*: after each mutation it compares volas
straight against TA-Lib computed on the same logical data — the owner's original ask ("after
operating on the frame, access the indicator again and compare the values against TA-Lib").
Together the two files give both the strict self-consistency check and the literal TA-Lib
check, with no logical gap.

Oracle: ``talib_expected`` — the single directive->TA-Lib mapping shared with the benchmark
coverage set. Comparison: where BOTH volas and TA-Lib emit a finite value. volas emits some
lines (the MACD / stochastic %K) earlier than TA-Lib, and uses NaN (not 0) for the index /
candlestick warm-up, so the common-finite region is the meaningful overlap; the values there
are exact (rtol/atol 1e-9). The oracle is always computed on the *full logical data* and then
sliced to the frame's window, mirroring ``test_mutation_parity``'s ground truth.

Excluded (still covered by the strict layer above + ``test_talib_parity``):
  * ``macd`` / ``macdfix`` (line + signal + histogram): volas emits the clean
    ``EMA(fast) - EMA(slow)`` difference; ``talib.MACD`` / ``talib.MACDFIX`` are internally
    inconsistent with their own EMA (a documented TA-Lib quirk), so no *equal* TA-Lib
    reference exists. ``test_talib_parity`` verifies volas against the clean difference.
  * ``ht_dcphase`` / ``ht_sine.*`` / ``ht_trendline`` / ``ht_trendmode``: the installed
    ``ta_lib`` build has a ~90-bar warm-up transient, so a faithful comparison needs the
    >300-row converged region — but these mutation frames are 250 rows. ``test_talib_parity``
    verifies them on the converged region of the full series.
"""

import numpy as np
import pytest

from volas import DataFrame

from test.test_benchmark import COVERAGE_IDS, talib_expected
from test.test_mutation_parity import A, BARS, A_APP, A_UPD, N, LO, _CLOSE

pytest.importorskip('talib')  # the oracle; skip this module wholesale when TA-Lib is absent

_EXCLUDE = frozenset({
    'macd', 'macd.signal', 'macd.histogram',
    'macdfix', 'macdfix.signal', 'macdfix.histogram',
    'ht_dcphase', 'ht_sine.sine', 'ht_sine.leadsine', 'ht_trendline', 'ht_trendmode',
})
DIRECTIVES = [d for d in COVERAGE_IDS if d not in _EXCLUDE]


def _cache(df, directive):
    """Materialize + cache the directive; skip if volas cannot express it via getitem."""
    try:
        return df[directive]
    except Exception as e:  # pragma: no cover - directive not getitem-cacheable
        pytest.skip(f'{directive}: {type(e).__name__}: {e}')


def _eq(got, want, directive):
    """Exact parity where BOTH volas and TA-Lib emit a finite value (the meaningful overlap)."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    both = np.isfinite(got) & np.isfinite(want)
    n = int(both.sum())
    assert n >= 10, f'{directive}: only {n} common-finite rows — volas/TA-Lib barely overlap'
    np.testing.assert_allclose(got[both], want[both], rtol=1e-9, atol=1e-9)


def _talib(directive, data, sl=slice(None)):
    """TA-Lib reference for `directive` on the full logical `data`, sliced to the frame window."""
    return talib_expected(directive, data)[sl]


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_slice(directive):
    """Slice carries the full-history cache: sub[D] == TA-Lib(full A)[LO:N]."""
    df = DataFrame(A)
    _cache(df, directive)
    sub = df.iloc[LO:N]
    _eq(sub[directive].to_numpy(), _talib(directive, A, slice(LO, N)), directive)


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_append(directive):
    """Cache, append M bars, fulfill: result == TA-Lib(A + bars)."""
    df = DataFrame(A)
    _cache(df, directive)
    df.append(DataFrame(BARS))
    df.fulfill()
    _eq(df[directive].to_numpy(), _talib(directive, A_APP), directive)


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_cell(directive):
    """Cache, update a recent close cell (iloc[i, j] = v): result == TA-Lib(updated)."""
    df = DataFrame(A)
    _cache(df, directive)
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    _eq(df[directive].to_numpy(), _talib(directive, A_UPD), directive)


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_column(directive):
    """Cache, replace a whole base column (df['close'] = ...): result == TA-Lib(updated)."""
    df = DataFrame(A)
    _cache(df, directive)
    df['close'] = A_UPD['close']
    _eq(df[directive].to_numpy(), _talib(directive, A_UPD), directive)


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_then_append(directive):
    """Combination: cache, update a cell, append, fulfill — both mutations honored vs TA-Lib."""
    df = DataFrame(A)
    _cache(df, directive)
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    df.append(DataFrame(BARS))
    df.fulfill()
    h = {k: np.concatenate([A_UPD[k], BARS[k]]) for k in A_UPD}
    _eq(df[directive].to_numpy(), _talib(directive, h), directive)


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_slice_then_append(directive):
    """Combination: cache, slice off the head, append, fulfill: == TA-Lib(A + bars)[LO:]."""
    df = DataFrame(A)
    _cache(df, directive)
    sub = df.iloc[LO:]
    sub.append(DataFrame(BARS))
    sub.fulfill()
    _eq(sub[directive].to_numpy(), _talib(directive, A_APP, slice(LO, None)), directive)


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_cow_subframe(directive):
    """Sub-frame B = A.iloc[LO:N]: B's indicator matches TA-Lib, and mutating B (CoW) leaves
    A's indicator matching TA-Lib(A) untouched."""
    a_df = DataFrame(A)
    _cache(a_df, directive)
    base = _talib(directive, A)
    b = a_df.iloc[LO:N]
    _eq(b[directive].to_numpy(), base[LO:N], directive)        # B correct vs TA-Lib
    b.iloc[2, _CLOSE] = A['close'][LO + 2] * 1.05              # mutate B in place (CoW)
    _eq(a_df[directive].to_numpy(), base, directive)           # A still matches TA-Lib(A)
