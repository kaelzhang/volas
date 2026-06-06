"""P2 DataFrame methods: fillna(value=, method=), isna, notna — per-column,
mirroring the Series versions; non-float columns pass through (fillna) or read
as never-missing (isna / notna)."""

import numpy as np
import pytest

from volas import DataFrame

nan = float("nan")


def _mixed():
    # a float column with gaps + a string column (a non-float column)
    return DataFrame({"f": [1.0, nan, 3.0], "s": ["a", "b", "c"]})


# --- fillna -----------------------------------------------------------------

def test_fillna_value_fills_float_columns():
    d = DataFrame({"a": [1.0, nan, 3.0], "b": [nan, 5.0, 6.0]})
    out = d.fillna(0.0)
    np.testing.assert_array_equal(out["a"].to_numpy(), [1, 0, 3])
    np.testing.assert_array_equal(out["b"].to_numpy(), [0, 5, 6])


def test_fillna_ffill_per_column():
    d = DataFrame({"a": [nan, 2.0, nan], "b": [1.0, nan, nan]})
    out = d.fillna(method="ffill")
    np.testing.assert_array_equal(out["a"].to_numpy(), [nan, 2, 2])
    np.testing.assert_array_equal(out["b"].to_numpy(), [1, 1, 1])


def test_fillna_bfill_per_column():
    d = DataFrame({"a": [nan, 2.0, nan]})
    np.testing.assert_array_equal(d.fillna(method="bfill")["a"].to_numpy(), [2, 2, nan])


def test_fillna_leaves_non_float_columns_untouched():
    out = _mixed().fillna(0.0)
    np.testing.assert_array_equal(out["f"].to_numpy(), [1, 0, 3])
    assert list(out["s"].to_numpy()) == ["a", "b", "c"]


def test_fillna_ffill_leaves_non_float_columns_untouched():
    out = _mixed().fillna(method="ffill")
    assert list(out["s"].to_numpy()) == ["a", "b", "c"]


def test_fillna_both_raises():
    with pytest.raises(ValueError, match="not both"):
        DataFrame({"a": [nan]}).fillna(0.0, method="ffill")


def test_fillna_neither_raises():
    with pytest.raises(ValueError):
        DataFrame({"a": [nan]}).fillna()


def test_fillna_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        DataFrame({"a": [nan]}).fillna(method="spline")


# --- isna / notna -----------------------------------------------------------

def test_isna_float_and_nonfloat():
    out = _mixed().isna()
    np.testing.assert_array_equal(out["f"].to_numpy(), [False, True, False])
    np.testing.assert_array_equal(out["s"].to_numpy(), [False, False, False])


def test_notna_float_and_nonfloat():
    out = _mixed().notna()
    np.testing.assert_array_equal(out["f"].to_numpy(), [True, False, True])
    np.testing.assert_array_equal(out["s"].to_numpy(), [True, True, True])
