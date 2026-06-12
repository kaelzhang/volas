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
def test_window_surface_set_equality():
    """P8 (1) machine surface differential: pandas's window-class member set
    minus the waived members minus pandas's config-echo attributes must EQUAL
    the volas member set — so a pandas addition, a volas addition, or a waiver
    drift all trip this single set equation (no per-name hasattr hunting)."""
    waived = {"apply", "agg", "aggregate", "pipe",   # arbitrary-Python-per-window
              "win_type", "step", "on", "method", "closed"}  # owner waiver 2026-06-12
    cfg_echo = {"center", "min_periods", "window", "obj", "ndim", "exclusions"}
    pub = lambda o: {m for m in dir(o) if not m.startswith("_")}
    pr = pub(pd.Series(dtype="float64").rolling(2))
    vr = pub(VS.rolling(2))
    assert pr - waived - cfg_echo == vr, (pr - waived - cfg_echo) ^ vr
    pe = pub(pd.Series(dtype="float64").ewm(span=2))
    ve = pub(VS.ewm(span=2))
    ewm_waived = waived | {"online", "times"}
    ewm_cfg = cfg_echo | {"adjust", "alpha", "com", "halflife", "span", "ignore_na"}
    assert pe - ewm_waived - ewm_cfg == ve, (pe - ewm_waived - ewm_cfg) ^ ve


@pytest.mark.parametrize("directive", ["median:20", "quantile:20,0.75", "rank:20",
                                       "skew:20", "kurt:20", "sem:20"])
def test_new_directive_append_refresh_and_entries(directive):
    """The six promoted statistics directives: append+fulfill (probe path)
    equals the batch compute (# equivalence:E3), and the three entries agree
    (df[d] == df.exec(d), lookback consistent — # equivalence:E2)."""
    arr = {c: 100 + rng.normal(0, 1, 120).cumsum()
           for c in ("open", "high", "low", "close", "volume")}
    full = volas.DataFrame(arr)
    want = full[directive].to_numpy()
    np.testing.assert_array_equal(full.exec(directive), want)          # E2
    head = volas.DataFrame({c: v[:-1] for c, v in arr.items()})
    _ = head[directive]
    head.append(volas.DataFrame({c: v[-1:] for c, v in arr.items()}))
    head.fulfill()
    assert np.array_equal(head[directive].to_numpy(), want, equal_nan=True)  # E3


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


# --- self-audit round 2 (2026-06-12): P8 (2) OWAT census on the entry params --
def test_ddof_negative_is_clean_valueerror():
    """R-1: a negative ddof is a clean ValueError, never pyo3's
    unsigned-conversion OverflowError leak (self-audit SA2-1)."""
    for f in (lambda: VS.rolling(3).var(ddof=-1),
              lambda: VS.rolling(3).std(ddof=-1),
              lambda: VS.rolling(3).sem(ddof=-1),
              lambda: VS.rolling(3).cov(VS, ddof=-1),
              lambda: VS.expanding(2).var(ddof=-1),
              lambda: volas.DataFrame({"a": [1.0, 2.0]}).rolling(2).std(ddof=-1)):
        with pytest.raises(ValueError):
            f()


def test_ewm_decay_must_be_finite():
    """SA2-2: a NaN / infinite decay spelling degenerates to alpha=0 / NaN —
    a frozen or all-NA no-signal column. alpha=0 is rejected, so the
    equivalent spellings must be too (same-guard symmetry, # equivalence:E8;
    pandas silently accepts span=nan — a pinned volas fail-loud divergence)."""
    for bad in ({"span": float("nan")}, {"span": float("inf")},
                {"com": float("nan")}, {"com": float("inf")},
                {"halflife": float("inf")}, {"halflife": float("nan")}):
        with pytest.raises(ValueError):
            VS.ewm(**bad)


def test_quantile_nan_q_rejected():
    """q=NaN: pandas silently emits an all-NaN column; volas rejects (C4
    fail-loud, pinned divergence)."""
    with pytest.raises(ValueError):
        VS.rolling(3).quantile(float("nan"))


def test_window_param_irep_and_boundaries():
    """I-rep + V census pins: np.int64 windows work; a window larger than the
    data degrades gracefully to expanding-like (min_periods still gates);
    window == len is exact; bool window is accepted as int 1 (python bool IS
    an int subclass; volas keeps int-parameter behaviour uniform — a pinned
    divergence from pandas's window-only bool rejection)."""
    assert VS.rolling(np.int64(7)).mean().to_list()[6] is not None
    assert VS.rolling(10**9).mean().isna().to_list()[-1]      # min_periods=10^9 never met
    assert not np.isnan(VS.rolling(10**9, min_periods=1).mean().to_numpy()[-1])
    assert VS.rolling(1).mean().to_list()[0] == VS.to_list()[0]
    assert volas.DataFrame({"a": [1.0, 2.0]})["a"].rolling(True).count().to_list() == [1, 1]


def test_window_and_dt_on_empty_and_allna():
    """N2 / N3 states: every kernel family handles the empty and the all-NA
    column without panicking (P7) and with the right shapes."""
    e = volas.DataFrame({"x": np.array([], dtype=float)})["x"]
    assert len(e.rolling(3).median()) == 0
    assert len(e.ewm(span=2).mean()) == 0
    assert len(e.expanding().nunique()) == 0
    allna = volas.DataFrame({"x": [float("nan")] * 4})["x"]
    assert allna.rolling(2).median().isna().to_list() == [True] * 4
    assert allna.rolling(2, min_periods=1).count().to_list()[1:] == [0, 0, 0]
    # dt accessor on empty / all-NaT
    dt_empty = volas.to_datetime(volas.DataFrame({"t": np.array([], dtype="datetime64[ns]")})["t"])
    assert len(dt_empty.dt.year) == 0 and len(dt_empty.dt.floor("D")) == 0
    assert dt_empty.dt.isocalendar().shape == (0, 3)
    nat = volas.to_datetime(volas.DataFrame({"t": [None, None]})["t"])
    assert nat.dt.year.isna().to_list() == [True, True]
    assert nat.dt.day_name().isna().to_list() == [True, True]


def test_frame_window_stale_guard_and_e6():
    """Layer 2 (SA2-3): a frame-level window aggregation is a BULK read — on a
    stale frame (post-append, pre-fulfill) it must carry the same fulfill
    guard as to_numpy / iloc (# equivalence:E8), instead of silently
    aggregating the stale cached-directive column as data. And E6: every
    frame window method equals its Series counterpart per column."""
    arr = {c: 100 + rng.normal(0, 1, 60).cumsum()
           for c in ("open", "high", "low", "close", "volume")}
    d = volas.DataFrame({c: v[:-1] for c, v in arr.items()})
    _ = d["ma:5"]
    d.append(volas.DataFrame({c: v[-1:] for c, v in arr.items()}))
    with pytest.raises(ValueError, match="fulfill"):
        d.rolling(3).mean()                      # stale -> guarded
    d.fulfill()
    out = d.rolling(3).mean()                    # fresh -> ok, incl. ma:5 column
    assert "ma:5" in out.columns
    # E6 across the FULL frame method set (machine loop, not spot checks)
    df = volas.DataFrame({"a": arr["close"], "b": arr["open"]})
    for m in ("count", "nunique", "sum", "mean", "median", "min", "max",
              "var", "std", "sem", "skew", "kurt", "first", "last"):
        fr = getattr(df.rolling(5, min_periods=2), m)()
        se = getattr(df["a"].rolling(5, min_periods=2), m)()
        np.testing.assert_allclose(
            np.asarray(fr["a"].to_numpy(), float), np.asarray(se.to_numpy(), float),
            rtol=1e-12, equal_nan=True, err_msg=m)
    for m in ("mean", "sum", "var", "std"):
        fr = getattr(df.ewm(span=6), m)()
        se = getattr(df["b"].ewm(span=6), m)()
        np.testing.assert_allclose(
            np.asarray(fr["b"].to_numpy(), float), np.asarray(se.to_numpy(), float),
            rtol=1e-12, equal_nan=True, err_msg=m)


def test_window_result_carries_name_and_index():
    """C1: a window result keeps the source's name and row correspondence."""
    df = volas.DataFrame({"t": ["2021-01-01", "2021-01-02"], "v": [1.0, 2.0]})
    df["t"] = volas.to_datetime(df["t"])
    s = df.set_index("t")["v"]
    out = s.rolling(2, min_periods=1).mean()
    assert out.name == "v"
    assert list(out.index) == list(s.index)
