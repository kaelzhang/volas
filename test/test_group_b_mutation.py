"""Group B mutation parity — self-oracle (the Group B indicators have no TA-Lib counterpart).

The Group B indicators (gap report 2026-06-07 §9) are convention-sensitive market indicators,
deliberately EXCLUDED from the TA-Lib parity / coverage suites. Each indicator's *value* is
verified against its source-pinned reference in ``test_oracle.py``; here we verify that
mutation (the cached-column incremental-refresh hot path) does not corrupt it: after a slice /
append / amend (cell + column) / their combinations / a copy-on-write sub-frame, the cached +
incrementally-refreshed value must equal a fresh full-history recompute.

The finite-memory members (vortex / brar / vr / coppock / rvi / cdp / mike / dkx / wvad /
ttm_squeeze) continue through the windowed-recompute fast-path; the recursive members
(keltner / smi) carry a self-contained state so the append + slice-then-append paths continue
the recursion past a dropped head in O(new rows).
"""

import numpy as np
import pytest

from volas import DataFrame

from test.test_mutation_parity import A, BARS, A_UPD, N, LO, _CLOSE, _gt, _eq

# Grows as each Group B sub-batch lands.
DIRECTIVES = [
    'vortex.plus', 'vortex.minus', 'brar.ar', 'brar.br', 'vr',
    'coppock', 'relative_vigor', 'relative_vigor.signal', 'dkx', 'dkx.ma', 'wvad',
    'cdp', 'cdp.ah', 'cdp.nh', 'cdp.nl', 'cdp.al',
    'mike.weakr', 'mike.midr', 'mike.strongr', 'mike.weaks', 'mike.mids', 'mike.strongs',
    'keltner', 'keltner.upper', 'keltner.lower',
    'stoch_momentum', 'stoch_momentum.signal',
    'ttm_squeeze', 'ttm_squeeze.on',
    'pivot_points', 'pivot_points.r1', 'pivot_points.s1', 'pivot_points.r2',
    'pivot_points.s2', 'pivot_points.r3', 'pivot_points.s3',
    'ichimoku.tenkan', 'ichimoku.kijun', 'ichimoku.senkou_a', 'ichimoku.senkou_b',
    'ichimoku.chikou',
    'wad', 'asi',
]


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_slice(directive):
    """A contiguous slice carries the full-history cache: sub[D] == fresh(A)[LO:N]."""
    df = DataFrame(A)
    df[directive]
    sub = df.iloc[LO:N]
    _eq(sub[directive].to_numpy(), _gt(directive, A)[LO:N])


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_append(directive):
    """Cache, append M bars, fulfill: the result matches a fresh recompute on A+bars."""
    df = DataFrame(A)
    df[directive]
    df.append(DataFrame(BARS))
    df.fulfill()
    h = {k: np.concatenate([A[k], BARS[k]]) for k in A}
    _eq(df[directive].to_numpy(), _gt(directive, h))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_cell(directive):
    """Cache, then amend a recent close cell (iloc[i, j] = v): the indicator must recompute,
    not return stale cached values."""
    df = DataFrame(A)
    df[directive]
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    _eq(df[directive].to_numpy(), _gt(directive, A_UPD))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_column(directive):
    """Cache, then replace a whole base column (df['close'] = ...): the indicator must
    recompute."""
    df = DataFrame(A)
    df[directive]
    df['close'] = A_UPD['close']
    _eq(df[directive].to_numpy(), _gt(directive, A_UPD))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_update_then_append(directive):
    """Combination: cache, amend a cell, append, fulfill — both mutations honored."""
    df = DataFrame(A)
    df[directive]
    df.iloc[N - 3, _CLOSE] = A['close'][N - 3] * 1.05
    df.append(DataFrame(BARS))
    df.fulfill()
    h = {k: np.concatenate([A_UPD[k], BARS[k]]) for k in A_UPD}
    _eq(df[directive].to_numpy(), _gt(directive, h))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_slice_then_append(directive):
    """Combination: cache, slice off the head, append, fulfill — must match a fresh recompute
    on A+bars sliced to [LO:]."""
    df = DataFrame(A)
    df[directive]
    sub = df.iloc[LO:]
    sub.append(DataFrame(BARS))
    sub.fulfill()
    h = {k: np.concatenate([A[k], BARS[k]]) for k in A}
    _eq(sub[directive].to_numpy(), _gt(directive, h)[LO:])


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_cow_subframe(directive):
    """Sub-frame B = A.iloc[LO:N]: B's indicator is correct, and mutating B (CoW) leaves A's
    indicator AND data untouched."""
    a_df = DataFrame(A)
    base = a_df[directive].to_numpy().copy()
    base_close = a_df['close'].to_numpy().copy()
    b = a_df.iloc[LO:N]
    _eq(b[directive].to_numpy(), _gt(directive, A)[LO:N])   # B correct (carried slice)
    b.iloc[2, _CLOSE] = A['close'][LO + 2] * 1.05           # mutate B in place (CoW)
    _eq(a_df[directive].to_numpy(), base)                   # A's indicator intact (CoW)
    _eq(a_df['close'].to_numpy(), base_close)               # A's data intact (CoW)
