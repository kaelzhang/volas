"""Systematic audit — TradingView / Pine Script `ta.*` indicators added to volas.

These directives fill the Pine-Script gap (2026-06-14 gap analysis): MA family
(vwma / alma / hma / swma) first, oscillators/stats and pivots in later batches.

Oracle policy (SPEC §5):
  * `# pine-formula` — the canonical Pine `ta.*` formula recomputed independently
    in numpy (an *应然* reference, not volas's own output). For vwma / hma / dev
    this also matches pandas-ta (verified during development); for alma / cog /
    swma pandas-ta uses a DIFFERENT convention (floored ALMA offset, opposite
    cog weighting, length-parametric swma), so the Pine formula is the sole
    oracle. Tolerance rtol=1e-9 (the project parity bar).
  * `# equivalence:E2/E3` — df[d] == exec(d); append+fulfill == batch recompute.

Cell IDs:  TV.<directive>.<aspect>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import volas

_CSV = pd.read_csv("test/data/tencent_full.csv")
ARR = {c: _CSV[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume")}
DF = volas.DataFrame(ARR)
C = ARR["close"]
V = ARR["volume"]


def _close(rtol=1e-9):
    def check(directive, ref):
        got = np.asarray(DF[directive].to_numpy(), dtype=float)
        ref = np.asarray(ref, dtype=float)
        m = ~np.isnan(ref)
        np.testing.assert_allclose(got[m], ref[m], rtol=rtol, atol=1e-9)
        # warm-up: every cell the reference leaves NaN must be NaN in volas too
        assert np.isnan(got[~m]).all(), f"{directive}: warm-up NaN mismatch"
    return check


# --- Pine-formula numpy reference implementations ----------------------------

def _wma(x, n):
    x = np.asarray(x, float)
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = np.dot(x[t - n + 1:t + 1], w)
    return out


def _ref_vwma(close, volume, n):
    close, volume = np.asarray(close, float), np.asarray(volume, float)
    out = np.full(len(close), np.nan)
    for t in range(n - 1, len(close)):
        vv = volume[t - n + 1:t + 1].sum()
        out[t] = (close[t - n + 1:t + 1] * volume[t - n + 1:t + 1]).sum() / vv if vv else np.nan
    return out


def _ref_alma(x, n, offset=0.85, sigma=6.0):
    x = np.asarray(x, float)
    m = offset * (n - 1)          # NOT floored — Pine's standard form
    s = n / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(n)])
    w /= w.sum()
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = np.dot(x[t - n + 1:t + 1], w)
    return out


def _ref_hma(x, n):
    half = n // 2
    sq = int(round(np.sqrt(n)))
    raw = 2 * _wma(x, half) - _wma(x, n)
    return _wma(raw, sq)


def _ref_swma(x):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for t in range(3, len(x)):
        out[t] = x[t - 3] / 6 + x[t - 2] * 2 / 6 + x[t - 1] * 2 / 6 + x[t] / 6
    return out


# --- value differential (# pine-formula) -------------------------------------

@pytest.mark.parametrize("n", [10, 20, 30])
def test_vwma_matches_pine(n):
    _close()(f"vwma:{n}", _ref_vwma(C, V, n))


@pytest.mark.parametrize("n,offset,sigma", [(20, 0.85, 6.0), (9, 0.5, 3.0), (30, 0.99, 8.0)])
def test_alma_matches_pine(n, offset, sigma):
    _close()(f"alma:{n},{offset},{sigma}", _ref_alma(C, n, offset, sigma))


@pytest.mark.parametrize("n", [9, 14, 20, 30])
def test_hma_matches_pine(n):
    _close()(f"hma:{n}", _ref_hma(C, n))


def test_swma_matches_pine():
    _close()("swma", _ref_swma(C))


@pytest.mark.parametrize("directive", ["vwma:5000", "alma:5000", "hma:5000"])
def test_period_exceeding_length_is_all_na(directive):
    """A window larger than the data is a valid no-signal column (all NaN),
    like `stddev:99999` — never a panic (P7)."""
    out = DF[directive].to_numpy()
    assert np.isnan(np.asarray(out, dtype=float)).all()


# --- cross-check the documented pandas-ta concordance (skip if not installed) -
def test_vwma_hma_match_pandas_ta():
    pta = pytest.importorskip("pandas_ta")  # noqa: F841
    import warnings
    warnings.simplefilter("ignore")
    np.testing.assert_allclose(
        np.asarray(DF["vwma:20"].to_numpy(), float),
        _CSV.ta.vwma(length=20).to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(
        np.asarray(DF["hma:20"].to_numpy(), float),
        _CSV.ta.hma(length=20).to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)


# --- lookback (warm-up length) -----------------------------------------------
@pytest.mark.parametrize("directive,lb", [
    ("vwma:20", 19), ("alma:20", 19), ("hma:20", 20 + round(20 ** 0.5) - 2), ("swma", 3),
])
def test_lookback(directive, lb):
    assert volas.directive_lookback(directive) == lb


# --- E2 / E3 equivalence ------------------------------------------------------
@pytest.mark.parametrize("directive", ["vwma:20", "alma:20", "hma:20", "swma"])
def test_directive_entries_and_append_refresh(directive):
    # E2: df[d] == exec(d)
    np.testing.assert_array_equal(DF.exec(directive), DF[directive].to_numpy())
    # E3: append+fulfill (probe path) == batch recompute (rtol 1e-9: sliding-sum
    # indicators drift ~1e-13, within the windowed-probe tolerance, like WMA)
    head = volas.DataFrame({k: v[:-1] for k, v in ARR.items()})
    _ = head[directive]
    head.append(volas.DataFrame({k: v[-1:] for k, v in ARR.items()}))
    head.fulfill()
    np.testing.assert_allclose(
        np.asarray(head[directive].to_numpy(), float),
        np.asarray(DF[directive].to_numpy(), float),
        rtol=1e-9, atol=1e-9, equal_nan=True)


# --- guards -------------------------------------------------------------------
def test_guards():
    for bad in ("vwma", "alma", "hma"):           # period is required (no default)
        with pytest.raises(ValueError):
            DF[bad]
    with pytest.raises(ValueError):
        DF["alma:20,1.5"]                          # offset must be in [0, 1]
    with pytest.raises(ValueError):
        DF["alma:20,0.85,0"]                       # sigma must be > 0
