"""Contract #26: integer-result paths are exact; float64 math past 2^53 raises.

Comparisons, ``between``, monotonicity, ``nlargest`` / ``nsmallest``, ``min`` /
``max`` / ``sum`` / ``prod`` (Series and DataFrame) and scalar fills compare and
compute in the integers' own dtype — no float64 round trip. An operation whose
result is float64 (mean, var, std, median, quantile, sem, skew, kurt, describe,
corr / cov, the math transforms, true division, the window aggregations) raises
``OverflowError`` for a present int64 with ``|x| > 2^53`` instead of computing on a
rounded input.
"""

import numpy as np
import pytest

import volas

T = 2**53
B = T + 1


def _s(values):
    return volas.DataFrame({"a": values})["a"]


# --- exact comparisons -------------------------------------------------------

def test_int_compare_is_exact():
    s = _s([B])
    assert (s == T).to_list() == [False]
    assert (s > T).to_list() == [True]
    assert (s > _s([T])).to_list() == [True]
    assert (s != float(T)).to_list() == [True]  # int vs float: exact, not via float64
    assert (s > float(T)).to_list() == [True]
    assert (_s([float(T)]) < s).to_list() == [True]
    assert (_s([2]) < 2.5).to_list() == [True]
    assert (_s([1, None]) == _s([1.0, float("nan")])).to_list() == [True, False]
    assert (_s([1, None]) != _s([1, 2])).to_list() == [False, True]
    assert (_s(np.array([3], dtype=np.int32)) == _s([True])).to_list() == [False]


def test_between_is_exact():
    s = _s([T, B, B + 1])
    assert s.between(B, B).to_list() == [False, True, False]
    assert s.between(B, B + 1, inclusive="neither").to_list() == [False, False, False]
    assert s.between(T, B, inclusive="left").to_list() == [True, False, False]
    assert s.between(T, B, inclusive="right").to_list() == [False, True, False]
    with pytest.raises(ValueError, match="inclusive"):
        s.between(1, 2, inclusive="bad")


def test_monotonic_is_exact_and_order_based():
    assert not _s([B, T]).is_monotonic_increasing
    assert _s([B, T]).is_monotonic_decreasing
    assert _s(["a", "b"]).is_monotonic_increasing  # strings by lexical order
    assert not _s(["b", "a"]).is_monotonic_increasing
    assert not _s([1, None, 2]).is_monotonic_increasing
    assert _s([5]).is_monotonic_increasing


def test_frame_nlargest_nsmallest_are_exact():
    df = volas.DataFrame({"a": [T, B, None]})
    assert df.nlargest(1, "a")["a"].to_list() == [B]
    assert df.nsmallest(1, "a")["a"].to_list() == [T]
    assert df.nlargest(5, "a")["a"].to_list() == [B, T]


# --- exact reductions --------------------------------------------------------

def test_frame_reductions_keep_integers_exact():
    df = volas.DataFrame({"a": [B, 1]})
    assert df.sum().dtype == "int64" and df.sum().to_list() == [B + 1]
    assert df.max().to_list() == [B]
    assert df.min().to_list() == [1]
    assert df.prod().to_list() == [B]
    i32 = volas.DataFrame({"a": np.array([1, 2], dtype=np.int32)})
    assert i32.max().dtype == "int32" and i32.sum().dtype == "int64"
    # a mixed frame is float64: exact integers convert, inexact ones raise
    mixed = volas.DataFrame({"a": [2**60, 0], "f": [1.5, 2.0]})
    assert mixed.max().to_list() == [2.0**60, 2.0]
    with pytest.raises(TypeError, match="not exactly representable"):
        volas.DataFrame({"a": [B, 0], "f": [1.5, 2.0]}).max()
    na = volas.DataFrame({"a": ([0, 0], [True, True], "int64"), "b": [1, 2]})
    assert na["a"].dtype == "int64"
    assert na.min().to_list() == [volas.NA, 1]
    with pytest.raises(OverflowError):
        volas.DataFrame({"a": [2**63 - 1, 1]}).sum()


# --- exact scalar fills ------------------------------------------------------

def test_fills_keep_large_integers_exact():
    s = _s([1, None])
    assert s.fillna(B).to_list() == [1, B]
    m = _s([True, False])
    assert s.where(m, B).to_list() == [1, B]
    assert s.mask(m, B).to_list() == [B, volas.NA]
    with pytest.raises(TypeError, match="not exactly representable"):
        _s([1.0, float("nan")]).fillna(B)  # into float64: inexact raises
    with pytest.raises(TypeError, match="not exactly representable"):
        _s([B, None]).fillna(0.5)  # promoting the int column to float64 is inexact
    assert _s([1.0, float("nan")]).fillna(2**60).to_list() == [1.0, 2.0**60]
    df = volas.DataFrame({"a": [1, None]})
    assert df.fillna(B)["a"].to_list() == [1, B]


# --- float64-domain math raises past 2^53 -----------------------------------

@pytest.mark.parametrize(
    "op",
    [
        lambda s: s.mean(), lambda s: s.var(), lambda s: s.std(), lambda s: s.median(),
        lambda s: s.quantile(0.5), lambda s: s.sem(), lambda s: s.skew(), lambda s: s.kurt(),
        lambda s: s.describe(), lambda s: s.sqrt(), lambda s: s.floor(), lambda s: s / 1,
        lambda s: s + 0.5, lambda s: s.corr(s), lambda s: s.cov(s), lambda s: s.interpolate(),
        lambda s: s.clip(0.5, None), lambda s: s // 0,
        lambda s: s.rolling(1).max(), lambda s: s.rolling(1).sum(), lambda s: s.expanding().mean(),
        lambda s: s.rolling(1).nunique(), lambda s: s.rolling(2).corr(s),
        lambda s: s.ewm(alpha=0.5).mean(), lambda s: s.ewm(alpha=0.5).var(),
        lambda s: s.ewm(alpha=0.5).cov(s),
    ],
)
def test_float64_math_past_2_53_raises(op):
    with pytest.raises(OverflowError, match="2\\^53"):
        op(_s([B, 1, 2]))


@pytest.mark.parametrize(
    "op",
    [
        lambda df: df.mean(), lambda df: df.var(), lambda df: df.std(), lambda df: df.median(),
        lambda df: df.quantile(0.5), lambda df: df.sem(), lambda df: df.skew(),
        lambda df: df.kurt(), lambda df: df.rolling(1).mean(), lambda df: df.ewm(alpha=0.5).mean(),
    ],
)
def test_frame_float64_math_past_2_53_raises(op):
    with pytest.raises(OverflowError, match="2\\^53"):
        op(volas.DataFrame({"a": [B, 1, 2]}))


def test_float64_math_inside_2_53_and_exact_paths_still_work():
    s = _s([T, -T, 0])
    assert s.mean() == 0.0
    assert _s([B, 1]).rolling(1).count().to_list() == [1, 1]  # value-independent
    assert _s([B, 1]).rolling(1).first().to_list() == [B, 1]  # dtype-preserving
    assert _s([B]).rank().to_list() == [1.0]  # order-based, not float64 math
