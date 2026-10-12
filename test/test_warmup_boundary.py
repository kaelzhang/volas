"""The first value of an indicator does not depend on how many rows follow it.

A full computation over the first `n` rows gives, at every row, what a full
computation over more rows gives there; so a frame grown one row at a time
(appended, or folded into a time frame) reads the same as one computed at once.
"""

import numpy as np
import pytest

import volas

OHLCV = ("open", "high", "low", "close", "volume")


def _cols(n):
    x = np.arange(n, dtype=float)
    close = 100.0 + np.sin(x) + 0.05 * x
    return dict(
        open=close - 0.1, high=close + 1.0 + 0.3 * np.cos(x), low=close - 1.0,
        close=close, volume=100.0 + x,
    )


def _bit_exact(got, want, ctx):
    got, want = np.asarray(got, float), np.asarray(want, float)
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    assert same.all(), f"{ctx}: rows {np.flatnonzero(~same)[:5]}: {got[~same][:3]} vs {want[~same][:3]}"


@pytest.mark.parametrize("directive", ["plus_dm:14", "minus_dm:14", "plus_dm:3", "minus_dm:5"])
def test_a_wilder_sum_emits_its_seed_on_exactly_period_rows(directive):
    talib = pytest.importorskip("talib")
    period = int(directive.split(":")[1])
    cols = _cols(period)
    got = np.asarray(volas.DataFrame(cols).exec(directive), float)
    fn = talib.PLUS_DM if directive.startswith("plus") else talib.MINUS_DM
    np.testing.assert_allclose(got, fn(cols["high"], cols["low"], period), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("directive", ["plus_dm:14", "minus_dm:14"])
def test_a_frame_grown_row_by_row_reads_as_one_computed_at_once(directive):
    n = 40
    cols = _cols(n)
    live = volas.DataFrame({k: v[:1] for k, v in cols.items()})
    live[directive]
    for i in range(1, n):
        live.append(volas.DataFrame({k: v[i:i + 1] for k, v in cols.items()}))
        live.fulfill()
        ref = volas.DataFrame({k: v[:i + 1] for k, v in cols.items()})
        _bit_exact(live[directive], ref[directive], f"{directive} rows={i + 1}")


@pytest.mark.parametrize("directive", [
    "ht_trendline", "ht_trendmode", "ht_dcphase", "ht_sine", "ht_sine.leadsine", "ht_dcperiod",
    "ht_phasor", "mama", "cmo:14", "rsi:14", "ema:20", "adosc",
])
def test_an_interior_missing_row_shifts_the_warm_up_on_every_path(directive):
    """A missing row does not exist for a recursive indicator, so it pushes the warm-up
    one row later: a resume past it must keep the rows a full recompute leaves NaN."""
    n = 110
    cols = _cols(n)
    cols["close"][30] = np.nan
    live = volas.DataFrame({k: v[:40] for k, v in cols.items()})
    live[directive]
    for i in range(40, n):
        live.append(volas.DataFrame({k: v[i:i + 1] for k, v in cols.items()}))
        live.fulfill()
        ref = volas.DataFrame({k: v[:i + 1] for k, v in cols.items()})
        _bit_exact(live[directive], ref[directive], f"{directive} rows={i + 1}")
