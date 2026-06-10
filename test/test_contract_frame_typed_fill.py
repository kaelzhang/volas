"""Contract FU-P1-01 — DataFrame ``fillna`` / ``where`` / ``mask`` share the
Series typed-scalar fill rules, applied per column: a str scalar fills a str
column, a Timestamp / datetime string a datetime column, a number / bool the
numeric family, and ``volas.NA`` is a dtype-preserving no-op. A dense (no-hole /
all-kept) column is untouched, so a numeric fill over a mixed frame skips its
holeless non-numeric columns; a column whose affected cell can't take the fill
raises (C4 — no object dtype to mix types), and the whole frame fails atomically
without a partial write. The DataFrame and Series surfaces must not disagree on
the same column."""

import numpy as np
import pytest
import volas
from volas import DataFrame

NA = volas.NA


# --- fillna: typed scalars per column ---------------------------------------

def test_frame_fillna_str_scalar():
    out = DataFrame({"s": ["a", None, "c"]}).fillna("q")
    assert out["s"].dtype == "str" and out["s"].to_list() == ["a", "q", "c"]


def test_frame_fillna_datetime_string_and_timestamp():
    df = DataFrame({"t": ["2021-01-01", None, "2021-01-03"]}).astype({"t": "datetime64[ns]"})
    out = df.fillna("2021-01-02")
    assert out["t"].dtype == "datetime64[ns]"
    assert out["t"].iloc[1] == np.datetime64("2021-01-02")
    out2 = df.fillna(volas.Timestamp("2021-01-02"))
    assert out2["t"].iloc[1] == np.datetime64("2021-01-02")


def test_frame_fillna_na_is_noop():
    df = DataFrame({"x": [1.0, None, 3.0], "s": ["a", None, "c"]})
    out = df.fillna(NA)
    assert out["x"].isna().to_list() == [False, True, False]
    assert out["s"].isna().to_list() == [False, True, False]


def test_frame_fillna_bool_zero_keeps_bool():
    out = DataFrame({"b": [True, None, False]}).fillna(0.0)
    assert out["b"].dtype == "bool" and out["b"].to_list() == [True, False, False]


# --- fillna: laziness + atomicity over mixed frames -------------------------

def test_frame_fillna_number_skips_holeless_str_fills_numeric():
    # the ubiquitous df.fillna(0) idiom: numeric holes filled, dense str untouched
    out = DataFrame({"x": [1.0, None], "s": ["a", "b"]}).fillna(0)
    assert out["x"].to_list() == [1.0, 0.0] and out["s"].to_list() == ["a", "b"]


def test_frame_fillna_incompatible_hole_raises_atomically():
    df = DataFrame({"x": [1.0, None], "s": ["a", None]})
    with pytest.raises(Exception):
        df.fillna("q")        # "q" can't fill x's numeric hole
    with pytest.raises(Exception):
        df.fillna(0)          # 0 can't fill s's str hole
    # atomic: the original frame is untouched by the failed call
    assert df["x"].isna().to_list() == [False, True]
    assert df["s"].to_list()[0] == "a"


def test_frame_fillna_matches_series_fillna_per_column():
    # the cross-type column is kept dense so the fill only reaches the column it
    # matches (a hole there would correctly make the atomic frame fillna raise).
    df_s = DataFrame({"i": [1, 2, 3], "s": ["x", None, "z"]})
    assert df_s.fillna("q")["s"].to_list() == df_s["s"].fillna("q").to_list()
    df_i = DataFrame({"i": [1, None, 3], "s": ["x", "y", "z"]})
    assert df_i.fillna(9)["i"].to_list() == df_i["i"].fillna(9).to_list()


# --- where / mask: typed scalars + default NA -------------------------------

def _mask(df, bits_per_col):
    return DataFrame({c: bits_per_col for c in df.columns})


def test_frame_where_str_scalar():
    df = DataFrame({"s": ["a", "b", "c"]})
    out = df.where(_mask(df, [True, False, True]), "z")
    assert out["s"].dtype == "str" and out["s"].to_list() == ["a", "z", "c"]


def test_frame_mask_str_scalar():
    df = DataFrame({"s": ["a", "b", "c"]})
    out = df.mask(_mask(df, [True, False, True]), "z")
    assert out["s"].to_list() == ["z", "b", "z"]


def test_frame_where_default_is_dtype_preserving_na():
    df = DataFrame({"s": ["a", "b"]})
    out = df.where(_mask(df, [True, False]))   # no `other` -> NA where False
    assert out["s"].dtype == "str" and out["s"].isna().to_list() == [False, True]


def test_frame_where_all_true_is_unchanged_even_with_wrong_type():
    # lazy: an all-kept column never type-checks the (unused) fill
    df = DataFrame({"x": [1.0, 2.0]})
    out = df.where(_mask(df, [True, True]), "z")
    assert out["x"].to_list() == [1.0, 2.0]


def test_frame_where_incompatible_scalar_on_replaced_cell_raises():
    df = DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(Exception):
        df.where(_mask(df, [True, False]), "z")   # a string into a replaced numeric cell
