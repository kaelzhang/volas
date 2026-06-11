"""Systematic audit — the owner-confirmed Series/DataFrame align cluster,
behaviour-checked vs pandas (not just existence): reductions, utilities,
windows, interpolation, and the F44/F45 parameters."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import volas

_V = [3.0, 1.0, 2.0, 2.0, None]
_s = lambda: volas.DataFrame({"x": list(_V)})["x"]
_ps = lambda: pd.Series(list(_V))


# --- DataFrame reductions (F8 closed) ---------------------------------------
def test_frame_reductions_match_pandas():
    df = volas.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, None, 6.0]})
    pf = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, None, 6.0]})
    for m in ("sum", "mean", "min", "max", "prod", "var", "std", "median",
              "nunique"):
        got, want = getattr(df, m)(), getattr(pf, m)()
        for c in ("a", "b"):
            assert float(got.loc[c]) == pytest.approx(float(want[c])), f"{m}[{c}]"
    assert df.quantile(0.5).loc["a"] == pf.quantile(0.5)["a"]
    assert df.idxmax().loc["a"] == 2 and df.idxmin().loc["a"] == 0
    assert bool(df.any().loc["a"]) and bool(df.all().loc["a"])


# --- Series utilities --------------------------------------------------------
def test_value_counts():
    s = volas.DataFrame({"x": ["a", "b", "a", "a", None]})["x"]
    vc = s.value_counts()
    assert vc.loc["a"] == 3 and vc.loc["b"] == 1
    assert vc.to_list()[0] == 3                      # most frequent first
    with pytest.raises(TypeError):
        _s().value_counts()                          # float labels -> no float index


def test_mode_isin_between():
    assert _s().mode().to_list() == [2.0]
    assert _s().isin([2.0, 3.0]).to_list() == [True, False, True, True, False]
    assert _s().between(1.5, 2.5).to_list() == [False, False, True, True, False]


def test_replace_keeps_dtype():
    s = volas.DataFrame({"x": [1, 2, 1]})["x"]
    out = s.replace(1, 9)
    assert out.to_list() == [9, 2, 9] and out.dtype == "int64"


def test_nlargest_nsmallest_dupes_monotonic():
    assert _s().nlargest(2).to_list() == [3.0, 2.0]
    assert _s().nsmallest(2).to_list() == [1.0, 2.0]
    assert _s().drop_duplicates().isna().to_list() == [False, False, False, True]
    assert _s().duplicated().to_list() == [False, False, False, True, False]
    assert not _s().is_unique
    inc = volas.DataFrame({"x": [1.0, 2.0, 3.0]})["x"]
    assert inc.is_monotonic_increasing and not inc.is_monotonic_decreasing
    assert not _s().is_monotonic_increasing          # NA -> not monotonic


def test_structure_methods():
    s = volas.DataFrame({"x": [1.0, 2.0]})["x"]
    assert s.rename("y").name == "y" and s.name == "x"
    assert s.copy().to_list() == s.to_list()
    f = s.to_frame()
    assert list(f.columns) == ["x"] and f.shape == (2, 1)
    assert s.to_dict() == {0: 1.0, 1: 2.0}
    assert s.items() == [(0, 1.0), (1, 2.0)]
    r = s.reset_index()                              # default -> 2-col frame
    assert list(r.columns) == ["index", "x"]
    assert s.reset_index(drop=True).to_list() == [1.0, 2.0]
    assert s.iat[1] == 2.0 and s.at[1] == 2.0
    rev = volas.DataFrame({"v": [10.0, 20.0]})
    rev["k"] = volas.DataFrame({"k": [2, 1]})["k"]
    assert rev.set_index("k")["v"].sort_index().to_list() == [20.0, 10.0]


# --- windows vs pandas -------------------------------------------------------
def test_rolling_matches_pandas():
    s, p = _s(), _ps()
    for m in ("mean", "sum", "min", "max", "std", "var"):
        got = getattr(s.rolling(2), m)().to_list()
        want = getattr(p.rolling(2), m)().tolist()
        for g, w in zip(got, want):
            assert (math.isnan(g) and math.isnan(w)) or g == pytest.approx(w), m


def test_expanding_and_ewm_match_pandas():
    s, p = _s(), _ps()
    got = s.expanding().mean().to_list()
    want = p.expanding().mean().tolist()
    for g, w in zip(got, want):
        assert (math.isnan(g) and math.isnan(w)) or g == pytest.approx(w)
    got = s.ewm(span=3).mean().to_list()
    want = p.ewm(span=3).mean().tolist()
    for g, w in zip(got, want):
        assert (math.isnan(g) and math.isnan(w)) or g == pytest.approx(w)


def test_frame_windows():
    df = volas.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    rm = df.rolling(2).mean()
    assert rm["a"].to_list()[1:] == [1.5, 2.5]
    assert df.expanding().sum()["b"].to_list() == [4.0, 9.0, 15.0]
    assert df.ewm(span=2).mean()["a"].to_list()[0] == 1.0


def test_interpolate():
    s = volas.DataFrame({"x": [1.0, None, 3.0, None]})["x"]
    out = s.interpolate()
    assert out.to_list()[:3] == [1.0, 2.0, 3.0]
    assert math.isnan(out.to_list()[3])              # trailing gap stays missing


# --- frame-level utilities ---------------------------------------------------
def test_frame_utilities():
    df = volas.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 1.0, 9.0]})
    assert df.duplicated().to_list() == [False, True, False]
    assert df.drop_duplicates().shape == (2, 2)
    assert df.isin([1.0])["b"].to_list() == [True, True, False]
    assert df.replace(1.0, 7.0)["a"].to_list() == [7.0, 7.0, 2.0]
    assert df.nlargest(1, "b")["b"].to_list() == [9.0]
    assert df.nsmallest(2, "a")["a"].to_list() == [1.0, 1.0]
    assert df.mode()["a"].to_list() == [1.0]
    with pytest.raises(TypeError):
        df.value_counts()                            # multi-col -> no MultiIndex


# --- F45 construction index --------------------------------------------------
def test_dataframe_index_kwarg_kinds():
    df = volas.DataFrame({"a": [1.0, 2.0]}, index=[10, 20])
    assert df.loc[20]["a"] == 2.0
    sdf = volas.DataFrame({"a": [1.0]}, index=["k"])
    assert sdf.loc["k"]["a"] == 1.0
    with pytest.raises(ValueError):
        volas.DataFrame({"a": [1.0, 2.0]}, index=[1, 1])   # duplicate labels rejected
