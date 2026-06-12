"""Systematic audit — T14 (window): rolling / expanding / ewm, pandas-differential.

Owner rulings 2026-06-12: the window API is pandas-aligned COMPATIBILITY surface
(directives are the recommended path; results do not join the directive cache).
`center=True` IS supported (labeling legitimately reads future bars). Time-window
rolling (`rolling('5min')`) is backlogged and must NOT be restarted without an
explicit owner confirmation — multi-timeframe work should run two tf-aware
frames (Cumulator), not window tricks.

Differential method: every method × window/min_periods/center × NA patterns is
compared against pandas 3 on well-conditioned data. Pinned divergences (volas
deliberately better):
  * kurt — volas computes two-pass central moments; pandas's raw power sums lose
    ~8 digits on offset-heavy data (compared at 1e-6 here, exactness pinned
    separately below).
  * skew/kurt with min_periods < 3 — pandas goes permanently NaN after an inner
    NA gap (a pandas state bug); volas keeps emitting correct values.
  * count/nunique return int64 (native-NA), not pandas's float64.

Cell IDs:  T14.<entry>.<method>/<params>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import volas

rng = np.random.default_rng(7)
X = rng.normal(0, 1, 300)
X[[5, 50, 51, 120]] = np.nan
Y = rng.normal(0, 1, 300)
Y[[8, 50, 200]] = np.nan

_VDF = volas.DataFrame({"x": X, "y": Y})
VS, VO = _VDF["x"], _VDF["y"]
PS, PO = pd.Series(X), pd.Series(Y)

METHODS = ["count", "sum", "mean", "median", "min", "max", "var", "std", "sem",
           "skew", "kurt", "nunique", "first", "last"]


def _cmp(a, b, rtol=1e-9):
    a = np.asarray(a.to_numpy(), dtype=float)
    b = np.asarray(b.to_numpy(), dtype=float)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window,mp", [(7, None), (7, 3), (10, 1), (4, 2), (1, None)])
@pytest.mark.parametrize("m", METHODS)
def test_rolling_differential(m, window, mp, center):
    rtol = 1e-6 if m == "kurt" else 1e-9  # pandas's power-sum kurt noise
    _cmp(getattr(VS.rolling(window, min_periods=mp, center=center), m)(),
         getattr(PS.rolling(window, min_periods=mp, center=center), m)(), rtol)


@pytest.mark.parametrize("m", METHODS)
def test_expanding_differential(m):
    rtol = 1e-6 if m == "kurt" else 1e-9
    _cmp(getattr(VS.expanding(3), m)(), getattr(PS.expanding(3), m)(), rtol)


@pytest.mark.parametrize("interp", ["linear", "lower", "higher", "nearest", "midpoint"])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.parametrize("window", [8, 9])  # even + odd (the `nearest` tie axis)
def test_rolling_quantile_differential(q, interp, window):
    _cmp(VS.rolling(window, min_periods=2).quantile(q, interp),
         PS.rolling(window, min_periods=2).quantile(q, interp))


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("pct", [False, True])
@pytest.mark.parametrize("method", ["average", "min", "max"])
def test_rolling_rank_differential(method, pct, ascending):
    _cmp(VS.rolling(9, min_periods=2).rank(method=method, ascending=ascending, pct=pct),
         PS.rolling(9, min_periods=2).rank(method=method, ascending=ascending, pct=pct))


@pytest.mark.parametrize("center", [False, True])
def test_rolling_corr_cov_differential(center):
    _cmp(VS.rolling(15, min_periods=4, center=center).corr(VO),
         PS.rolling(15, min_periods=4, center=center).corr(PO))
    _cmp(VS.rolling(15, min_periods=4, center=center).cov(VO),
         PS.rolling(15, min_periods=4, center=center).cov(PO))


def test_expanding_corr_cov_differential():
    _cmp(VS.expanding(4).corr(VO), PS.expanding(4).corr(PO))
    _cmp(VS.expanding(4).cov(VO), PS.expanding(4).cov(PO))


@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("decay", [{"span": 10}, {"com": 5}, {"halflife": 4}, {"alpha": 0.3}])
def test_ewm_differential(decay, adjust, ignore_na):
    kw = dict(decay, adjust=adjust, ignore_na=ignore_na, min_periods=3)
    for m in ("mean", "var", "std"):
        _cmp(getattr(VS.ewm(**kw), m)(), getattr(PS.ewm(**kw), m)())
    if adjust:
        _cmp(VS.ewm(**kw).sum(), PS.ewm(**kw).sum())
    _cmp(VS.ewm(**kw).corr(VO), PS.ewm(**kw).corr(PO))
    _cmp(VS.ewm(**kw).cov(VO), PS.ewm(**kw).cov(PO))


def test_ewm_sum_adjust_false_raises():
    with pytest.raises((ValueError, NotImplementedError)):
        VS.ewm(span=10, adjust=False).sum()


def test_ewm_decay_param_exclusivity():
    with pytest.raises(ValueError):
        VS.ewm()                       # none given
    with pytest.raises(ValueError):
        VS.ewm(span=10, alpha=0.5)     # two given
    for bad in ({"com": -1.0}, {"span": 0.5}, {"halflife": 0.0}, {"alpha": 1.5}, {"alpha": 0.0}):
        with pytest.raises(ValueError):
            VS.ewm(**bad)


# --- dtype rules (owner ruling: count/nunique int64, first/last dtype-kept) --
def test_count_nunique_are_int64():
    assert VS.rolling(5).count().dtype == "int64"
    assert VS.expanding().nunique().dtype == "int64"
    assert VS.rolling(5).count().isna().to_list()[:4] == [True] * 4  # warm-up NA


def test_first_last_preserve_dtype():
    iv = volas.DataFrame({"i": [1, None, 3, 4]})["i"]
    assert iv.rolling(2, min_periods=1).first().dtype == "int64"
    assert iv.rolling(2, min_periods=1).last().to_list()[2] == 3
    assert getattr(VS.rolling(3).first(), "dtype") == "float64"


def test_aggregations_are_float64():
    for m in ("sum", "mean", "median", "var", "std", "sem", "skew", "kurt"):
        assert getattr(volas.DataFrame({"i": [1, 2, 3, 4]})["i"].rolling(2), m)().dtype == "float64"


# --- guards -------------------------------------------------------------------
def test_window_guards():
    with pytest.raises(ValueError):
        VS.rolling(0)
    with pytest.raises(ValueError):
        VS.rolling(-3)
    with pytest.raises(ValueError):
        VS.rolling(5, min_periods=-1)
    with pytest.raises(ValueError):
        VS.rolling(5, min_periods=6)         # min_periods > window (pandas raises)
    with pytest.raises(ValueError):
        VS.expanding(-1)
    with pytest.raises(ValueError):
        VS.rolling(5).quantile(1.5)
    with pytest.raises(ValueError):
        VS.rolling(5).quantile(0.5, "cubic")
    with pytest.raises(ValueError):
        VS.rolling(5).rank(method="dense")   # pandas rolling rank has no 'dense'
    with pytest.raises(TypeError):
        volas.DataFrame({"s": ["a", "b"]})["s"].rolling(2)
    with pytest.raises(ValueError):
        VS.rolling(5).corr(volas.DataFrame({"z": [1.0, 2.0]})["z"])  # length mismatch


def test_frame_window_str_column_errors():
    df = volas.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
    with pytest.raises(TypeError):
        df.rolling(2).mean()


# --- frame variants delegate per column ---------------------------------------
def test_frame_rolling_matches_series():
    df = volas.DataFrame({"a": X, "b": Y})
    out = df.rolling(7, min_periods=2).mean()
    _cmp(out["a"], PS.rolling(7, min_periods=2).mean())
    _cmp(out["b"], PO.rolling(7, min_periods=2).mean())
    _cmp(df.ewm(span=9).std()["b"], PO.ewm(span=9).std())
    _cmp(df.expanding(2).median()["a"], PS.expanding(2).median())


# --- single source: directive == rolling API, bit-exact ------------------------
@pytest.mark.parametrize("directive,roll", [
    ("median:20", lambda s: s.rolling(20).median()),
    ("quantile:20,0.75", lambda s: s.rolling(20).quantile(0.75)),
    ("rank:20", lambda s: s.rolling(20).rank(pct=True)),
    ("skew:20", lambda s: s.rolling(20).skew()),
    ("kurt:20", lambda s: s.rolling(20).kurt()),
    ("sem:20", lambda s: s.rolling(20).sem()),
])
def test_directive_window_single_source(directive, roll):
    """The six promoted statistics directives and the rolling API share ONE
    kernel — bit-exact on a dense close column (and the directive joins the
    cache + append-refresh, which the rolling API deliberately does not)."""
    arr = {c: 100 + rng.normal(0, 1, 200).cumsum()
           for c in ("open", "high", "low", "close", "volume")}
    df = volas.DataFrame(arr)
    assert np.array_equal(df[directive].to_numpy(), roll(df["close"]).to_numpy(),
                          equal_nan=True)
    assert volas.directive_lookback(directive) == 19


# --- pinned divergences (volas deliberately better than pandas) ----------------
def test_kurt_exact_on_ill_conditioned_window():
    """volas kurt uses two-pass central moments; pandas's raw power sums lose
    ~8 digits when the window mean dwarfs its spread. Pin volas against the
    closed-form value on exactly such a window."""
    b = np.array([1e6 + 1.0, 1e6 + 2.0, 1e6 + 4.0, 1e6 + 10.0])
    got = volas.DataFrame({"x": b})["x"].rolling(4).kurt().to_list()[-1]
    n, m = 4.0, b.mean()
    m2 = ((b - m) ** 2).mean()
    m4 = ((b - m) ** 4).mean()
    want = ((n + 1) * m4 / m2**2 - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3))
    assert got == pytest.approx(want, rel=1e-12)


def test_skew_survives_na_gap_with_low_min_periods():
    """pandas rolling(3, min_periods=0).skew() goes permanently NaN after an
    inner NA gap (a pandas kernel-state bug); volas keeps emitting the correct
    per-window value. Pin the volas behaviour."""
    got = VS.rolling(3, min_periods=0).skew().to_numpy()
    dense_tail = ~np.isnan(X[-3:])
    assert dense_tail.all() and not np.isnan(got[-1])


# --- waived pandas members (recorded; restart needs explicit confirmation) ----
def test_waived_members_absent():
    r = VS.rolling(5)
    for m in ("apply", "agg", "aggregate", "pipe", "win_type", "step", "on",
              "method", "closed"):
        assert not hasattr(r, m), m
    e = VS.ewm(span=5)
    for m in ("online", "times"):
        assert not hasattr(e, m), m


def test_time_window_rolling_backlogged():
    """`rolling('5min')` / timedelta windows are BACKLOG by owner ruling —
    restarting them requires explicit confirmation. Multi-timeframe work should
    maintain two tf-aware frames (Cumulator), not window tricks."""
    with pytest.raises(TypeError):
        VS.rolling("5min")


def test_first_last_all_dtypes():
    """`first` / `last` gather through Column::take_optional — every dtype arm,
    NA-gap and warm-up included."""
    # str / datetime columns are rejected at the rolling() entry, like pandas
    # ("no numeric types to aggregate") — pinned in test_window_guards.
    df = volas.DataFrame({
        "f": [1.5, None, 3.5], "i": [1, None, 3], "b": [True, None, False],
    })
    df["f32"] = df["f"].astype("float32")
    for col, want_first, want_last in [
        ("f", 1.5, 3.5), ("f32", 1.5, 3.5), ("i", 1, 3), ("b", True, False),
    ]:
        s = df[col]
        first = s.rolling(3, min_periods=1).first()
        last = s.rolling(3, min_periods=1).last()
        assert first.dtype == s.dtype and last.dtype == s.dtype, col
        assert first.to_list()[2] == want_first, col
        assert last.to_list()[2] == want_last, col
        # row 0's window holds only row 0 -> both edges are row 0
        assert first.to_list()[0] == last.to_list()[0]
    # an all-NA window is NA
    allna = volas.DataFrame({"f": [None, None, 1.0]})["f"]
    assert allna.rolling(2, min_periods=1).first().isna().to_list() == [True, True, False]


def test_flat_window_guard_branches():
    """A perfectly flat window: skew/kurt have no scale (NaN), corr against a
    constant has no variance (NaN); cov with ddof == count is NaN."""
    flat = volas.DataFrame({"x": [2.0] * 6, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert np.isnan(flat["x"].rolling(4).skew().to_numpy()[-1])
    assert np.isnan(flat["x"].rolling(5).kurt().to_numpy()[-1])
    assert np.isnan(flat["y"].rolling(4).corr(flat["x"]).to_numpy()[-1])
    assert np.isnan(flat["y"].rolling(3, min_periods=2).cov(flat["x"], ddof=2).to_numpy()[1])
