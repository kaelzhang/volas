"""Group A mutation parity — self-oracle (the Group A indicators have no TA-Lib counterpart).

The Group A indicators (gap report 2026-06-07) are volas-exclusive, so they are deliberately
EXCLUDED from the TA-Lib parity / coverage suites (``COVERAGE_IDS`` is the volas∩TA-Lib set).
They still need the same systematic mutation check those indicators get — so this file
verifies their correctness UNDER MUTATION against volas's own fresh full-history recompute as
the ground truth (a self-comparison, since there is no external reference). Each indicator's
*value* is verified against its source-pinned reference in ``test_oracle.py``; here we only
verify that mutation (the cached-column incremental-refresh hot path) does not corrupt it.

For every Group A line, after a slice / append / update (cell AND column) / their combinations
/ a copy-on-write sub-frame, the cached + incrementally-refreshed value must equal a fresh
recompute on the same logical data — BIT-EXACTLY. The recursive members (pvt / nvi / pvi /
efi / tsi / mass_index) carry a self-contained state so the append + slice-then-append paths
continue the recursion past a dropped head in O(new rows); the finite-memory members
(psy / dpo / cmf / chop / kst / emv) continue through the windowed-recompute path.
"""

import numpy as np
import pytest

from volas import DataFrame

from test.test_mutation_parity import A, BARS, A_UPD, N, LO, _CLOSE, _gt, _eq

# All 13 Group A directives, in their canonical default-parameter forms.
DIRECTIVES = [
    'psy', 'dpo:20', 'tsi', 'kst', 'crsi', 'chop:14', 'cmf',
    'emv:14', 'efi', 'pvt', 'nvi', 'pvi', 'mass_index',
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
    """Combination: cache, slice off the head, append, fulfill — the carried state continues
    the recursion past the dropped head, matching a fresh recompute on A+bars."""
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
