"""Harness for the source-pinned reference oracle (see ``oracle_reference.py``).

For each non-TA-Lib indicator it (1) checks the reference itself runs and has a finite
converged region (no crash / all-NaN bug), and (2) compares the volas directive against
the reference on the Tencent OHLCV fixture. A directive volas does not yet implement is
skipped, so the suite stays green until each Group A indicator lands and its oracle case
activates automatically. The already-shipped ``bbi`` runs and passes today, validating the
harness end-to-end.
"""

import numpy as np
import pytest

from volas import DataFrame, DirectiveValueError

from . import oracle_reference as oref


# A deterministic synthetic OHLCV. The shipped Tencent test fixture is only 100 rows — too
# short for Connors RSI's 100-period PercentRank — so the oracle uses a seeded 300-bar series
# with all prices > 0 and every bar's high > low (so money-flow / range ratios never divide by
# zero).
@pytest.fixture(scope="module")
def ohlcv():
    rng = np.random.default_rng(20260608)
    n = 300
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    close[150] = close[149]  # one flat bar — exercises the unchanged-close path (Connors RSI streak)
    high = close + rng.uniform(0.3, 2.0, n)
    low = close - rng.uniform(0.3, 2.0, n)
    open_ = low + (high - low) * rng.uniform(0.0, 1.0, n)
    volume = rng.uniform(1e5, 1e6, n)
    return {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}


def _reference(fn, cols):
    out = fn(cols['open'], cols['high'], cols['low'], cols['close'], cols['volume'])
    return np.asarray(out, dtype=float)


@pytest.mark.parametrize('directive,fn,tol', oref.CASES, ids=[c[0] for c in oref.CASES])
def test_reference_executes(ohlcv, directive, fn, tol):
    """The source-pinned reference runs and has a finite converged tail (no crash / all-NaN)."""
    out = _reference(fn, ohlcv)
    assert out.shape == (len(ohlcv['close']),)
    assert np.isfinite(out[-50:]).any(), f'{directive}: reference is all-NaN in the converged tail'


@pytest.mark.parametrize('directive,fn,tol', oref.CASES, ids=[c[0] for c in oref.CASES])
def test_directive_matches_reference(ohlcv, directive, fn, tol):
    """volas's directive matches the source-pinned reference (skips until the indicator lands)."""
    df = DataFrame(dict(ohlcv))
    try:
        got = np.asarray(df[directive].to_numpy(), dtype=float)
    except DirectiveValueError as e:
        if 'unknown command' in str(e):
            pytest.skip(f'{directive}: not yet implemented (oracle pinned; activates on landing)')
        raise
    want = _reference(fn, ohlcv)
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol, equal_nan=True)
