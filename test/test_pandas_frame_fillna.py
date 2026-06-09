"""DataFrame missing-value methods — ``fillna`` / ``ffill`` / ``bfill`` / ``isna``
/ ``notna`` / ``dropna`` — operate over **every** dtype through the column
validity bitmap (int / bool / str / datetime NA, not only float ``NaN``) and
agree cell-for-cell with the ``Series`` versions.

Regression: these DataFrame-level methods used to be float-only, so they silently
ignored non-float NA — ``isna`` read an int/bool/str hole as present, ``dropna``
never dropped it, and ``fillna`` / ``ffill`` left it unfilled — while the Series
versions handled it. The two surfaces must not disagree on the same data.
"""

import numpy as np
import pytest

import volas
from volas import DataFrame

nan = float("nan")


# --- float fillna / ffill / bfill (the original float path, still valid) ----

def test_fillna_value_fills_float_columns():
    d = DataFrame({"a": [1.0, nan, 3.0], "b": [nan, 5.0, 6.0]})
    out = d.fillna(0.0)
    np.testing.assert_array_equal(out["a"].to_numpy(), [1, 0, 3])
    np.testing.assert_array_equal(out["b"].to_numpy(), [0, 5, 6])


def test_ffill_bfill_float_per_column():
    d = DataFrame({"a": [nan, 2.0, nan], "b": [1.0, nan, nan]})
    np.testing.assert_array_equal(d.ffill()["a"].to_numpy(), [nan, 2, 2])
    np.testing.assert_array_equal(d.ffill()["b"].to_numpy(), [1, 1, 1])
    np.testing.assert_array_equal(d.bfill()["a"].to_numpy(), [2, 2, nan])


def test_fillna_requires_a_value():
    # pandas 3.0 removed fillna(method=); fillna now needs a value
    with pytest.raises(TypeError):
        DataFrame({"a": [nan]}).fillna()


# --- the unified contract: non-float NA is detected, filled, and dropped ----

def _df_na():
    # every dtype carries a hole at row 1 (None -> dtype-preserving NA)
    return DataFrame(
        {"i": [1, None, 3], "b": [True, None, False], "s": ["x", None, "z"], "f": [1.5, None, 3.0]}
    )


def test_isna_notna_detect_every_dtype():
    NA = volas.NA
    isna = _df_na().isna()
    for c in ["i", "b", "s", "f"]:
        assert isna[c].to_list() == [False, True, False], c
    notna = _df_na().notna()
    for c in ["i", "b", "s", "f"]:
        assert notna[c].to_list() == [True, False, True], c
    # a dense (no-hole) non-float column still reads as never missing
    dense = DataFrame({"s": ["a", "b"], "i": [1, 2]})
    assert dense.isna()["s"].to_list() == [False, False]
    assert dense.isna()["i"].to_list() == [False, False]


def test_dataframe_isna_matches_series_isna():
    # the DataFrame surface must agree with the Series surface on identical data
    df = _df_na()
    for c in ["i", "b", "s", "f"]:
        assert df.isna()[c].to_list() == df[c].isna().to_list(), c


def test_fillna_fills_int_and_bool_dtype_preserving():
    # numeric-family columns only (str fillna is covered separately below)
    out = DataFrame({"i": [1, None, 3], "b": [True, None, False], "f": [1.5, None, 3.0]}).fillna(0.0)
    assert out["i"].to_list() == [1, 0, 3] and out["i"].dtype == "int64"  # 0 is integral -> stays int
    assert out["b"].to_list() == [True, False, False] and out["b"].dtype == "bool"  # 0 -> False, keeps bool
    assert out["f"].to_list() == [1.5, 0.0, 3.0]


def test_numeric_fillna_on_str_or_datetime_raises():
    # volas has no object dtype, so a numeric fill cannot apply to a non-numeric
    # column with a missing cell — it raises instead of silently corrupting
    # (str -> all 0.0, datetime -> raw f64 epoch).
    with pytest.raises(TypeError):
        DataFrame({"s": ["x", None, "z"]})["s"].fillna(0)  # Series
    with pytest.raises(TypeError):
        DataFrame({"i": [1, None, 3], "s": ["x", None, "z"]}).fillna(0)  # DataFrame: str col has a hole
    dt = vs_datetime()
    with pytest.raises(TypeError):
        dt.fillna(0)
    # a DENSE (no-hole) str column is untouched by a numeric frame fillna
    out = DataFrame({"i": [1, None, 3], "s": ["a", "b", "c"]}).fillna(0)
    assert out["i"].to_list() == [1, 0, 3] and out["s"].to_list() == ["a", "b", "c"]


def vs_datetime():
    return DataFrame({"t": np.array(["2021-01-01", "NaT"], dtype="datetime64[ns]")})["t"]


def test_ffill_bfill_int_bool_str():
    d = DataFrame({"i": [1, None, None], "s": ["x", None, "z"], "b": [True, None, None]})
    ff = d.ffill()
    assert ff["i"].to_list() == [1, 1, 1]
    assert ff["s"].to_list() == ["x", "x", "z"]
    assert ff["b"].to_list() == [True, True, True]
    # parity with the Series version on the same column
    assert ff["i"].to_list() == DataFrame({"i": [1, None, None]})["i"].ffill().to_list()
    bf = DataFrame({"s": ["x", None, "z"]}).bfill()
    assert bf["s"].to_list() == ["x", "z", "z"]


def test_fillna_ffill_leave_dense_non_float_untouched():
    # a non-float column with no holes is returned unchanged (nothing to fill)
    dense = DataFrame({"f": [1.0, nan, 3.0], "s": ["a", "b", "c"]})
    assert list(dense.fillna(0.0)["s"].to_numpy()) == ["a", "b", "c"]
    assert list(dense.ffill()["s"].to_numpy()) == ["a", "b", "c"]


def test_dropna_counts_every_dtype():
    # how='any': a row with an int NA (no float column to mask it) is now dropped
    d = DataFrame({"i": [1, None, 3], "x": [10, 20, 30]})
    assert len(d.dropna()) == 2
    assert d.dropna()["i"].to_list() == [1, 3]
    # how='all': drop only rows where every column is NA
    d2 = DataFrame({"a": [1, None, None], "b": [10, None, 30]})
    kept = d2.dropna(how="all")
    assert len(kept) == 2 and kept["b"].to_list() == [10, 30]


def test_boolean_mask_from_dataframe_isna_filters_rows():
    # df.isna() is usable as a row mask across dtypes (boolean indexing)
    d = DataFrame({"i": [1, None, 3]})
    only_missing = d[d.isna()["i"]]
    assert len(only_missing) == 1


def test_dropna_invalid_how_raises():
    with pytest.raises(ValueError):
        DataFrame({"a": [1.0, nan]}).dropna(how="bad")
