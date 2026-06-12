"""Appending a frame that is MISSING a (plain) column pads the new rows with
dtype-preserving NA — int+NA, bool+NA, str+NA, datetime+NaT, float+NaN — never
upcasting the dtype (the old int->float64) or raising (the old str/bool/datetime
"no missing-value representation" error).

A *cached directive* column instead keeps its cheap stale placeholder, which
``fulfill`` overwrites — that path is unchanged and exercised by the mutation /
TA-Lib parity suites.
"""

import pytest
import numpy as np

import volas
from volas import DataFrame


def test_append_missing_plain_int_keeps_int_dtype():
    df = DataFrame({"i": [1, 2], "x": [10, 20]})
    out = df.append(DataFrame({"x": [30]}))  # 'i' absent from the appended frame
    assert out["i"].dtype == "int64"  # NOT upcast to float64
    assert out["i"].to_list() == [1, 2, volas.NA]
    assert out["x"].to_list() == [10, 20, 30]


def test_append_missing_plain_str_pads_na():
    df = DataFrame({"i": [1, 2], "s": ["a", "b"]})
    out = df.append(DataFrame({"i": [3]}))  # used to raise "no missing-value representation"
    assert out["s"].dtype == "str"
    assert out["s"].to_list() == ["a", "b", volas.NA]


def test_append_missing_plain_bool_pads_na():
    df = DataFrame({"i": [1, 2], "b": [True, False]})
    out = df.append(DataFrame({"i": [3]}))
    assert out["b"].dtype == "bool"
    assert out["b"].to_list() == [True, False, volas.NA]


def test_append_missing_plain_datetime_pads_nat():
    df = DataFrame({"i": [1, 2], "t": np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[ns]")})
    out = df.append(DataFrame({"i": [3]}))
    assert out["t"].dtype == "datetime64[ns]"
    assert out["t"].isna().to_list() == [False, False, True]


def test_append_missing_plain_float_pads_nan():
    df = DataFrame({"i": [1, 2], "f": [1.0, 2.0]})
    out = df.append(DataFrame({"i": [3]}))
    assert out["f"].isna().to_list() == [False, False, True]


def test_padded_na_participates_in_missing_methods():
    # the padded NA is a first-class missing value: detected, filled, dropped.
    # 'i' is the absent/padded column (-> [1, 2, NA]); 'x' is complete.
    df = DataFrame({"i": [1, 2], "x": [10, 20]})
    out = df.append(DataFrame({"x": [30]}))
    assert out["i"].to_list() == [1, 2, volas.NA]
    assert out.isna()["i"].to_list() == [False, False, True]
    assert out.fillna(0.0)["i"].to_list() == [1, 2, 0]  # padded int NA filled, stays int
    assert out["i"].ffill().to_list() == [1, 2, 2]  # carries the last value forward
    assert len(out.dropna()) == 2  # the padded row is dropped


# --- R4-P2-01: an EXTRA column is rejected (a missing one is padded above) ---


def test_append_extra_column_is_rejected():
    # the inverse of the padding above: an appended frame with a column the target
    # lacks used to be silently dropped (data loss); it now raises.
    df = DataFrame({"x": [1.0]})
    with pytest.raises(ValueError):
        df.append(DataFrame({"x": [2.0], "z": [9.0]}))  # 'z' not in target
    # the original frame is untouched by the rejected append
    assert df.columns == ["x"] and df["x"].to_list() == [1.0]


def test_append_extra_column_rejected_for_row_too():
    df = DataFrame({"x": [1.0], "y": [2.0]})
    extra_row = DataFrame({"x": [3.0], "y": [4.0], "z": [5.0]}).iloc[0]
    with pytest.raises(ValueError):
        df.append(extra_row)


def test_append_same_columns_reordered_still_works():
    # column-name alignment (not position) is unaffected: same names, any order.
    df = DataFrame({"a": [1.0], "b": [2.0]})
    out = df.append(DataFrame({"b": [4.0], "a": [3.0]}))
    assert out["a"].to_list() == [1.0, 3.0] and out["b"].to_list() == [2.0, 4.0]
