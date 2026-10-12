"""A live refresh equals a full recompute bit for bit on extreme values too: infinite and
subnormal prices, signed zeros. Each case is one where a capture or a refresh once took a
different path from the full computation (its own seed, its own unguarded sum, its own
tie rule or eviction rule) and so kept a different value."""

import numpy as np
import pytest

import volas


def _bits(a, b):
    """Bit-identical (the sign of a zero included); any two NaNs match."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(((a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b))).all())


def _appended(data, directive, cut):
    """`directive` cached over the first `cut` rows, then one row appended at a time."""
    n = len(next(iter(data.values())))
    df = volas.DataFrame({k: v[:cut] for k, v in data.items()}, index=list(range(cut)))
    df[directive]
    for r in range(cut, n):
        df.append(volas.DataFrame({k: v[r:r + 1] for k, v in data.items()}, index=[r]))
    return np.asarray(df[directive], dtype=float)


def test_mass_index_resumes_its_guarded_sum_over_an_infinite_ratio():
    # subnormal ranges: the double EMA underflows to 0 first, so a ratio is infinite
    data = {"high": np.array([0.0] * 17 + [5e-323] * 3), "low": np.zeros(20)}
    full = volas.DataFrame(data).exec("mass_index:2")
    assert np.isinf(full[-2]) and np.isfinite(full[-1])
    for cut in range(14, 20):
        assert _bits(_appended(data, "mass_index:2", cut), full), cut
    # a same-label replacement of the newest row
    df = volas.DataFrame(data, index=list(range(20)))
    df["mass_index:2"]
    df.append(volas.DataFrame({"high": [5e-323], "low": [0.0]}, index=[19]))
    assert _bits(df["mass_index:2"], full)


@pytest.mark.parametrize("directive", ["atr:2", "natr:2", "atr:5", "natr:5"])
def test_atr_capture_seeds_where_the_full_compute_does(directive):
    # a bar with every price infinite has a NaN true range (inf − inf): the full compute
    # seeds past it, and so must a state captured before the next bar arrives
    cols = ("high", "low", "close")
    rng = np.random.default_rng(3)
    c = rng.random(12) + 10
    data = {"high": c + 1, "low": c - 1, "close": c}
    data = {k: v.copy() for k, v in data.items()}
    for k in cols:
        data[k][1:3] = np.inf
    full = volas.DataFrame(data).exec(directive)
    for cut in range(2, 12):
        assert _bits(_appended(data, directive, cut), full), cut


@pytest.mark.parametrize("directive", ["1 / minmax.min:3", "1 / minmax.max:3", "1 / midpoint:3", "1 / llv:3@close",
                                       "1 / hhv:3@close", "1 / donchian:3@close,close"])
@pytest.mark.parametrize("x", [[1.0, -0.0, 0.0], [-1.0, 0.0, -0.0], [0.0, -0.0, 0.0], [-0.0, 0.0, -0.0],
                               [0.0, 0.0, -0.0], [-0.0, -0.0, 0.0]])
def test_a_later_missing_value_never_flips_an_earlier_zero_extremum(directive, x):
    """Every extremum path resolves a signed-zero tie by total order (`+0 > −0`), so a
    missing value appended later — which once switched the algorithm — changes nothing."""
    clean = volas.DataFrame({"close": x}).exec(directive)
    later = volas.DataFrame({"close": x + [np.nan]}).exec(directive)[:3]
    assert _bits(clean, later)
    df = volas.DataFrame({"close": x})
    df[directive]
    df.append(volas.DataFrame({"close": [np.nan]}))
    assert _bits(np.asarray(df[directive])[:3], clean)


@pytest.mark.parametrize("directive", ["mode:3", "1 / mode:3", "median:3", "1 / median:3"])
def test_a_sorted_window_evicts_the_very_value_that_leaves(directive):
    x = [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    full = volas.DataFrame({"close": x}).exec(directive)
    df = volas.DataFrame({"close": x[:-1]})
    df[directive]
    df.append(volas.DataFrame({"close": x[-1:]}))
    assert _bits(df[directive], full)
    # the last window [+0, +0, 1] is the same whatever came before it
    assert _bits(full[-1:], volas.DataFrame({"close": x[-3:]}).exec(directive)[-1:])
