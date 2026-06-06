"""volas trading-safety guards vs pandas footguns (the P1 changes).

A live trading loop must never get a *silently wrong* number or a
silently-taken branch. volas tightens three pandas-shaped behaviours:

  * ``bool(Series / DataFrame)`` raises on an ambiguous truth value
    (pandas-style), with ``Series.any() / .all()`` as the explicit reductions.
  * Series binary ops require a **shared index** — volas aligns by position and
    never silently by label, so two differently-indexed operands raise instead
    of producing a misaligned result.
  * ``df == / !=`` are **element-wise** (a bool DataFrame), not object identity;
    a side effect is that a DataFrame becomes unhashable, matching pandas.
"""

import numpy as np
import pytest

import volas
from volas import DataFrame

nan = float("nan")


def _df(**cols):
    return DataFrame({k: [float(x) for x in v] for k, v in cols.items()})


def _dt_indexed(dates, vals):
    d = DataFrame({"v": [float(x) for x in vals], "t": dates})
    d["t"] = volas.to_datetime(d["t"])
    return d.set_index("t")


# --- bool() ambiguity guard -------------------------------------------------

def test_bool_single_element_series_is_its_truth():
    assert bool(_df(a=[5.0])["a"]) is True
    assert bool(_df(a=[0.0])["a"]) is False


def test_bool_single_element_bool_series():
    assert bool(_df(a=[1.0])["a"] > 0.5) is True
    assert bool(_df(a=[1.0])["a"] > 1.5) is False


def test_bool_multi_element_series_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        bool(_df(a=[1.0, 2.0])["a"])


def test_bool_empty_series_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        bool(_df(a=[])["a"])


def test_bool_dataframe_always_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        bool(_df(a=[1.0]))


# --- any / all (skipna=True) ------------------------------------------------

def test_any_true_when_some_nonzero():
    assert _df(a=[0.0, 0.0, 3.0])["a"].any() is True


def test_any_false_when_all_zero_or_nan():
    assert _df(a=[0.0, 0.0, nan])["a"].any() is False


def test_all_true_skips_nan():
    assert _df(a=[1.0, 2.0, nan])["a"].all() is True


def test_all_false_when_a_value_is_zero():
    assert _df(a=[1.0, 0.0])["a"].all() is False


def test_empty_all_true_any_false():
    s = _df(a=[])["a"]
    assert s.all() is True
    assert s.any() is False


def test_any_all_on_bool_column():
    mask = _df(a=[1.0, 2.0])["a"] > 1.5  # [False, True]
    assert mask.any() is True
    assert mask.all() is False


# --- Series alignment guard -------------------------------------------------

def test_same_frame_columns_align_by_position():
    df = _df(x=[1, 2, 3], y=[4, 5, 6])
    np.testing.assert_array_equal((df["x"] + df["y"]).to_numpy(), [5, 7, 9])


def test_value_equal_index_from_copy_aligns():
    df = _df(x=[1, 2, 3])
    np.testing.assert_array_equal((df["x"] + df.copy()["x"]).to_numpy(), [2, 4, 6])


def test_slices_with_equal_index_align():
    df = _df(x=[1, 2, 3], y=[4, 5, 6])
    np.testing.assert_array_equal(
        (df["x"].iloc[1:] + df["y"].iloc[1:]).to_numpy(), [7, 9]
    )


def test_different_datetime_index_raises_not_misaligns():
    a = _dt_indexed(["2020-01-01", "2020-01-02", "2020-01-03"], [1, 2, 3])
    b = _dt_indexed(["2020-01-02", "2020-01-03", "2020-01-04"], [10, 20, 30])
    with pytest.raises(ValueError, match="different indexes"):
        a["v"] + b["v"]


def test_different_length_series_raises():
    with pytest.raises(ValueError, match="different indexes"):
        _df(a=[1, 2, 3])["a"] + _df(a=[1, 2])["a"]


def test_comparison_guard_also_applies():
    a = _dt_indexed(["2020-01-01", "2020-01-02"], [1, 2])
    b = _dt_indexed(["2020-01-03", "2020-01-04"], [1, 2])
    with pytest.raises(ValueError, match="different indexes"):
        a["v"] > b["v"]


def test_logical_guard_also_applies():
    a = _dt_indexed(["2020-01-01", "2020-01-02"], [1, 0])
    b = _dt_indexed(["2020-01-03", "2020-01-04"], [1, 1])
    with pytest.raises(ValueError, match="different indexes"):
        (a["v"] > 0) & (b["v"] > 0)


def test_scalar_operand_needs_no_alignment():
    np.testing.assert_array_equal((_df(x=[1, 2, 3])["x"] + 10).to_numpy(), [11, 12, 13])


def test_unsupported_operand_raises_typeerror():
    with pytest.raises(TypeError):
        _df(x=[1.0])["x"] + "not a number"


# --- df == / != element-wise ------------------------------------------------

def test_df_eq_df_is_elementwise_bool_frame():
    df = _df(x=[1, 2, 3], y=[4, 5, 9])
    eq = df == _df(x=[1, 0, 3], y=[4, 5, 9])
    assert isinstance(eq, DataFrame)
    np.testing.assert_array_equal(eq["x"].to_numpy(), [True, False, True])
    np.testing.assert_array_equal(eq["y"].to_numpy(), [True, True, True])


def test_df_ne_df():
    np.testing.assert_array_equal(
        (_df(x=[1, 2]) != _df(x=[1, 9]))["x"].to_numpy(), [False, True]
    )


def test_df_eq_scalar_broadcasts():
    eq = _df(x=[1, 2, 2], y=[2, 2, 3]) == 2.0
    np.testing.assert_array_equal(eq["x"].to_numpy(), [False, True, True])
    np.testing.assert_array_equal(eq["y"].to_numpy(), [True, True, False])


def test_df_eq_different_columns_raises():
    with pytest.raises(ValueError, match="different columns"):
        _df(x=[1, 2]) == _df(y=[1, 2])


def test_df_eq_different_index_raises():
    a = _dt_indexed(["2020-01-01", "2020-01-02"], [1, 2])
    b = _dt_indexed(["2020-01-03", "2020-01-04"], [1, 2])
    with pytest.raises(ValueError, match="different indexes"):
        a == b


def test_df_eq_unsupported_operand_raises():
    with pytest.raises(TypeError):
        _df(x=[1.0]) == "nope"


def test_df_is_unhashable_after_elementwise_eq():
    with pytest.raises(TypeError):
        hash(_df(x=[1.0]))
