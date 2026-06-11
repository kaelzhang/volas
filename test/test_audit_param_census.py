"""Systematic audit — per-parameter value-category census for parametrized
methods (P8 §6.8), generalised beyond datetime.

For each method, sweep its parameter's value categories (normal / boundary /
degenerate / out-of-range) and diff vs pandas. fillna (T4, 210-cell) and astype
(T8, 7x7) already have full matrices; this covers the rest: clip / round /
quantile / shift / head / tail. Result: all match pandas except clip(lower>upper)
(F19) — i.e. the parametrized surface is sound, pinned against regression.
"""

from __future__ import annotations

import pandas as pd
import pytest

import volas

_V = [1.27, 2.34, 3.55, 4.0, 5.0]
_s = lambda: volas.DataFrame({"x": list(_V)})["x"]
_ps = lambda: pd.Series(list(_V))


def _match(got, want):
    return [round(float(x), 9) for x in got] == [round(float(x), 9) for x in want]


# --- round(decimals): 0 / positive / negative ------------------------------
@pytest.mark.parametrize("dec", [0, 2, -1])
def test_round_decimals_census(dec):
    assert _match(_s().round(dec).to_list(), _ps().round(dec).tolist()), f"round({dec})"


# --- quantile(q): boundary + out-of-range ----------------------------------
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 1.0])
def test_quantile_value_census(q):
    assert _s().quantile(q) == pytest.approx(_ps().quantile(q)), f"quantile({q})"


@pytest.mark.parametrize("q", [1.5, -0.1])
def test_quantile_out_of_range_raises(q):
    with pytest.raises((ValueError, KeyError)):
        _s().quantile(q)


# --- shift(n): 0 / forward / backward / over-length ------------------------
@pytest.mark.parametrize("n", [0, 1, -1, 99, -99])
def test_shift_n_census(n):
    got, want = _s().shift(n), _ps().shift(n)
    assert got.isna().to_list() == [bool(b) for b in want.isna().tolist()], f"shift({n}) NA"


# --- head/tail(n): normal / 0 / negative / over-length ---------------------
@pytest.mark.parametrize("n", [2, 0, -2, 99])
def test_head_tail_n_census(n):
    assert _match(_s().head(n).to_list(), _ps().head(n).tolist()), f"head({n})"
    assert _match(_s().tail(n).to_list(), _ps().tail(n).tolist()), f"tail({n})"


# --- clip(lower, upper): the full bound census -----------------------------
@pytest.mark.parametrize("lo,hi", [(2, 4), (None, 4), (2, None)])
def test_clip_bounds_census(lo, hi):
    assert _match(_s().clip(lo, hi).to_list(), _ps().clip(lo, hi).tolist()), f"clip({lo},{hi})"


# F19 (decision 3, FIXED): clip with lower > upper raises (was a silent collapse
# to `upper`). fail-loud (C5).
def test_clip_inverted_bounds_raises():
    with pytest.raises(ValueError):
        _s().clip(5, 2)
