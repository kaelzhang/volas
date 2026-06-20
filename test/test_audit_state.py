"""Systematic audit — Layer 2: state / sequence / cross-API equivalence laws.

Layer 1 enumerates single calls; Layer 2 covers receiver history, copies,
increments and interop — where ~30% of the historical findings lived. The oracle
here is *metamorphic*: `path_A(cells) ≡ path_B(cells)` (SPEC §2B), so no hand
oracle is needed — a divergence between two paths that must agree IS the finding.

  E1  constructor(Series)  ≡  df[col] = Series
  E2  df[directive] (values) ≡ df.exec(directive)   (three-entry value parity)
  E3  append-incremental   ≡  one-shot cumulate
  E4  batch append         ≡  one-shot cumulate
  E5  copy isolation (mutating one side never touches the other)
  E6  Series.f()           ≡  DataFrame.f() per column
  E7  to_pandas -> from_pandas round-trip  ≡  original
  E8  same-semantics guards behave the same (NaT/monotonic guards are symmetric
      across cumulate / live append / tf-construction)

Cell IDs:  E1/<d>/<n> · E2/<directive> · E5/<d> · E6/<f> · E7/<d>/<n>
"""

from __future__ import annotations

import math

import pytest

import volas
from . import audit_dims as A


def _present(s):
    """(isna-mask, present-values-as-float-or-raw) of a Series."""
    mask = s.isna().to_list()
    vals = []
    for x, m in zip(s.to_list(), mask):
        if m:
            continue
        vals.append(float(x) if isinstance(x, (int, float, bool)) else x)
    return mask, vals


def _equiv(s1, s2, label):
    """Structural Series equivalence: dtype + length + NA-mask + present values
    (avoids depending on `equals`, which is itself under audit)."""
    assert s1.dtype == s2.dtype, f"{label}: dtype {s1.dtype} != {s2.dtype}"
    assert len(s1) == len(s2), f"{label}: len {len(s1)} != {len(s2)}"
    m1, v1 = _present(s1)
    m2, v2 = _present(s2)
    assert m1 == m2, f"{label}: NA mask {m1} != {m2}"
    assert v1 == v2, f"{label}: present values {v1} != {v2}"


# --- E1: constructor(Series) ≡ df[col] = Series ----------------------------
@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_e1_constructor_equals_setitem(d, n):
    s = A.series(d, n)
    constructed = volas.DataFrame({"x": A.series(d, n)})["x"]
    anchor = volas.DataFrame({"k": [0.0] * len(s)})
    anchor["x"] = A.series(d, n)
    _equiv(anchor["x"], constructed, f"E1/{d}/{n}")


# --- E2: df[directive] (values) ≡ df.exec(directive) ; three-entry parity ---
_OHLC = ["open", "high", "low", "close", "volume"]
_DIRECTIVES = ["ma:2", "ma:3", "ema:3", "boll.upper:5,2", "rsi:4"]


@pytest.mark.parametrize("directive", _DIRECTIVES)
def test_e2_directive_entry_parity(directive):
    df = volas.DataFrame({c: [float(i + 1) for i in range(12)] for c in _OHLC})
    via_getitem = df[directive].to_list()
    via_exec = list(df.exec(directive).tolist())   # exec returns an ndarray (raw values)
    assert len(via_getitem) == len(via_exec)
    for a, b in zip(via_getitem, via_exec):
        if a is volas.NA or (isinstance(a, float) and math.isnan(a)):
            assert b is volas.NA or (isinstance(b, float) and math.isnan(b)), f"E2/{directive} NA"
        else:
            assert a == pytest.approx(b), f"E2/{directive} value {a} != {b}"
    # the lookback entry must agree with where the warm-up NaNs actually end.
    lb = volas.directive_lookback(directive)
    leading_na = next((i for i, v in enumerate(via_getitem)
                       if not (v is volas.NA or (isinstance(v, float) and math.isnan(v)))), len(via_getitem))
    assert leading_na == lb, f"E2/{directive}: lookback {lb} != observed warm-up {leading_na}"


# --- E3 / E4: incremental & batch folding ≡ one-shot cumulate ---------------
def _fine():
    t = ["2021-01-01 00:00:00", "2021-01-01 00:01:00", "2021-01-01 00:02:00",
         "2021-01-01 00:05:00", "2021-01-01 00:06:00"]
    n = len(t)
    df = volas.DataFrame({
        "open": [float(i + 1) for i in range(n)],
        "high": [float(i + 1) + 0.5 for i in range(n)],
        "low": [float(i + 1) - 0.5 for i in range(n)],
        "close": [float(i + 1) for i in range(n)],
        "volume": [10.0 * (i + 1) for i in range(n)],
        "t": t,
    })
    df["t"] = volas.to_datetime(df["t"])
    return df.set_index("t")


def test_e3_incremental_matches_one_shot():
    fine = _fine()
    df = fine.iloc[0:1].cumulate("5m")
    for i in range(1, len(fine)):
        df.append(fine.iloc[i:i + 1])
    assert df.equals(fine.cumulate("5m")), "E3: bar-by-bar fold != one-shot cumulate"


def test_e4_batch_matches_one_shot():
    fine = _fine()
    df = fine.iloc[0:1].cumulate("5m")
    df.append(fine.iloc[1:])
    assert df.equals(fine.cumulate("5m")), "E4: batch fold != one-shot cumulate"


# --- E5: copy isolation (mutation never crosses the copy boundary) ---------
@pytest.mark.parametrize("d", A.DTYPES)
def test_e5_copy_isolation(d):
    repl = volas.DataFrame({"r": [9.0, 9.0, 9.0]})["r"]
    # forward: mutate the copy, original is untouched.
    df = volas.DataFrame({"x": A.series(d, "N0")})
    before = df["x"].to_list()
    cp = df.copy()
    cp["x"] = repl
    cp["y"] = repl
    assert df["x"].to_list() == before, f"E5/{d}: copy mutation leaked into original"
    assert list(df.columns) == ["x"], f"E5/{d}: copy column-add leaked into original"
    # reverse: mutate the original, the earlier copy is untouched.
    df2 = volas.DataFrame({"x": A.series(d, "N0")})
    cp2 = df2.copy()
    cp_before = cp2["x"].to_list()
    df2["x"] = repl
    assert cp2["x"].to_list() == cp_before, f"E5/{d}: original mutation leaked into copy"


# --- E6: Series.f() ≡ DataFrame.f() per column -----------------------------
_UNARY = ("abs", "round", "clip", "cumsum", "cumprod", "cummax", "cummin",
          "shift", "diff", "ffill", "bfill")
_SHARED_REDUCE = ("count", "sem", "skew", "kurt")


@pytest.mark.parametrize("f", _UNARY)
def test_e6_unary_series_frame_parity(f):
    col = A.series("f64", "N1")
    df = volas.DataFrame({"x": A.series("f64", "N1")})
    frame_col = getattr(df, f)()["x"]
    series_col = getattr(col, f)()
    _equiv(frame_col, series_col, f"E6/{f}")


@pytest.mark.parametrize("f", _SHARED_REDUCE)
def test_e6_reduction_series_frame_parity(f):
    # 6 distinct values so skew/kurt are defined (kurt needs >=4 points).
    vals = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
    col = volas.DataFrame({"x": list(vals)})["x"]
    df = volas.DataFrame({"x": list(vals)})
    frame_scalar = getattr(df, f)()["x"]
    series_scalar = getattr(col, f)()
    assert frame_scalar == pytest.approx(series_scalar), f"E6/{f}: {frame_scalar} != {series_scalar}"


# --- E7: to_pandas -> from_pandas round-trip ≡ original ---------------------
# The lossless round-trip targets the *nullable* backend; the default numpy
# backend is intentionally legacy-lossy (NA forces int->float etc.) at the
# export boundary, so it is NOT the round-trip oracle (SPEC §10 triage (a)).
def test_e7_default_backend_is_legacy_lossy():
    """An NA-bearing int demotes to float under the default numpy backend —
    documenting the intended boundary behaviour, not a fidelity guarantee."""
    df = A.frame("i64", "N1")
    assert df.to_pandas()["x"].dtype == "float64"           # numpy legacy NA path
    assert df.to_pandas(dtype_backend="numpy_nullable")["x"].dtype == "Int64"  # faithful


# F9/F10 FIXED: from_pandas imports a nullable column through the typed masked
# channel, so the DECLARED dtype survives (Int32 stays int32; an all-NA or empty
# nullable column keeps its dtype instead of decaying to float64). The nullable
# round-trip is now lossless across the full D x N matrix.
@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_e7_pandas_roundtrip(d, n):
    df = A.frame(d, n)
    back = volas.DataFrame.from_pandas(df.to_pandas(dtype_backend="numpy_nullable"))
    _equiv(back["x"], df["x"], f"E7/{d}/{n}")


# --- Layer-2 deep transitions (review CG-8): cached directive lifecycle ------
def test_cached_directive_detracks_on_rename():
    """rename rebuilds the frame and DE-TRACKS cached directive columns: they
    become plain data (still readable), an append NaN-pads them like any plain
    column (visible missing, not silent wrong values), and fulfill leaves them
    alone — the coherent de-track semantics, pinned."""
    cols = ["open", "high", "low", "close", "volume"]
    df = volas.DataFrame({c: [float(i + 1) for i in range(6)] for c in cols})
    cached = df["ma:2"].to_list()                      # materialise + cache
    renamed = df.rename({"close": "px"})
    got = renamed["ma:2"].to_list()                    # plain data survives
    assert all((math.isnan(g) and math.isnan(c)) or g == c for g, c in zip(got, cached))
    renamed.append(volas.DataFrame({(c if c != "close" else "px"): [9.0] for c in cols}))
    renamed.fulfill()
    assert renamed["ma:2"].isna().to_list()[-1] is True   # NaN-padded, visibly


def test_cumulators_typo_fails_loud():
    """A cumulator key that names no column errors (was silently ignored — the
    user believed an aggregation override was applied when it was not)."""
    df = volas.DataFrame({"close": [1.0], "volume": [10.0]})
    df["t"] = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01 00:00:00"]})["t"])
    df = df.set_index("t")
    with pytest.raises(ValueError):
        df.cumulate("5m", cumulators={"volumee": "sum"})    # misspelled
    assert df.cumulate("5m", cumulators={"volume": "sum"}).shape[0] == 1


# --- E8: guard symmetry (same semantics -> same behaviour) -------------------
def test_e8_nat_guard_symmetry():
    """A NaT-bearing timeline is rejected by EVERY entry that builds or extends
    a tf-aware frame — batch cumulate, tf-construction, and the live append —
    never accepted by one and rejected by another (the historical V12 class).
    (The deeper per-entry assertions live in test_cumulate_nat.py; this is the
    audit-level law tying them together.)"""
    import numpy as np
    nat = volas.DataFrame({"close": [1.0, 2.0]})
    nat["t"] = volas.DataFrame(
        {"t": np.array(["2021-01-01", "NaT"], dtype="datetime64[ns]")})["t"]
    nat = nat.set_index("t")
    with pytest.raises(Exception):
        nat.cumulate("5m")                                  # batch entry
    with pytest.raises(Exception):
        volas.DataFrame(nat, time_frame="5m")               # construction entry


def test_e8_monotonic_guard_symmetry():
    """An out-of-order timeline is likewise rejected by both entries (V11/V15)."""
    df = volas.DataFrame({"close": [1.0, 2.0]})
    df["t"] = volas.to_datetime(volas.DataFrame(
        {"t": ["2021-01-01 00:05:00", "2021-01-01 00:00:00"]})["t"])
    df = df.set_index("t")
    with pytest.raises(Exception):
        df.cumulate("5m")
    with pytest.raises(Exception):
        volas.DataFrame(df, time_frame="5m")


# --- E7, tz-aware dimension: the round-trip carries the zone -----------------
@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "+08:00"])
def test_e7_tz_aware_roundtrip(zone):
    """An ANCHORED frame keeps both its instants and its zone across
    to_pandas -> from_pandas — the core face of the two-state tz model."""
    df = volas.DataFrame({"v": [1.0, 2.0]})
    df["t"] = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01", "2021-07-01"]})["t"])
    aware = df.set_index("t").tz_localize(zone)
    back = volas.DataFrame.from_pandas(aware.to_pandas())
    assert back.tz == aware.tz, f"zone lost in round-trip: {back.tz} != {aware.tz}"
    assert back.equals(aware)


def test_e7_naive_roundtrip_stays_naive():
    df = volas.DataFrame({"v": [1.0]})
    df["t"] = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01"]})["t"])
    naive = df.set_index("t")
    assert volas.DataFrame.from_pandas(naive.to_pandas()).tz is None
