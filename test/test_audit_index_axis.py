"""Systematic audit — the I-axis (index) census (P8 §6.8; reviews CG-1).

Label matching via .loc across index kinds (str / int64 / datetime), and the
high-risk index shapes (missing label, duplicate label, datetime partial-string).
Result: full-label lookup is sound across all kinds; the gaps are partial-string
datetime indexing (F28) and duplicate-label .loc (F34).

Cell IDs:  T7.loc/I=<kind> · T7.loc/I=dup · T7.loc/I=partial-str
"""

from __future__ import annotations

import pytest

import volas


def _indexed(keys):
    df = volas.DataFrame({"v": [10.0, 20.0, 30.0]})
    df["k"] = volas.DataFrame({"k": list(keys)})["k"]
    return df.set_index("k")


def _dt_indexed():
    df = volas.DataFrame({"v": [1.0, 2.0, 3.0]})
    df["t"] = volas.to_datetime(volas.DataFrame(
        {"t": ["2021-01-15", "2021-02-15", "2021-03-15"]})["t"])
    return df.set_index("t")


# --- full-label lookup across index kinds (sound) --------------------------
def test_loc_str_index():
    df = _indexed(["a", "b", "c"])
    assert df.loc["b"]["v"] == 20.0
    with pytest.raises(KeyError):
        df.loc["z"]                          # missing label -> clean KeyError


def test_loc_int64_index():
    assert _indexed([10, 20, 30]).loc[20]["v"] == 20.0


def test_loc_datetime_index():
    df = _dt_indexed()
    assert df.loc[volas.Timestamp("2021-02-15")]["v"] == 2.0   # Timestamp label
    assert df.loc["2021-02-15"]["v"] == 2.0                    # full-date string label


# F28 (FIXED): partial-string datetime indexing — a year/month string selects
# the period's rows (pandas parity); a missing period is a clean KeyError.
def test_loc_datetime_partial_string():
    assert _dt_indexed().loc["2021-02"].shape == (1, 1)        # the whole month
    assert _dt_indexed().loc["2021"].shape == (3, 1)           # the whole year
    with pytest.raises(KeyError):
        _dt_indexed().loc["2022"]                              # empty period


# F34 (decision 1B, FIXED with a refinement): set_index REJECTS a duplicate-label
# int64/str column at creation (label access assumes unique labels — the same
# creation-time guard as the NA-label rule V16). A datetime column keys its rows as
# appending them would: a row labelled like the one before re-sends it (the last one
# wins) — the raw fine input of a cumulation keeps its re-send semantics — and any
# other repeated timestamp is rejected like an int64 one.
def test_set_index_rejects_duplicate_labels():
    with pytest.raises((ValueError, KeyError)):
        _indexed([1, 1, 2])              # int64 duplicate label -> rejected


def test_set_index_keys_a_re_sent_datetime_and_rejects_other_repeats():
    df = volas.DataFrame({"v": [1.0, 2.0, 3.0]})
    df["t"] = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01", "2021-01-01", "2021-01-02"]})["t"])
    assert df.set_index("t")["v"].to_list() == [2.0, 3.0]   # the re-sent row wins
    df["t"] = volas.to_datetime(volas.DataFrame({"t": ["2021-01-01", "2021-01-02", "2021-01-01"]})["t"])
    with pytest.raises(ValueError, match="duplicate labels"):
        df.set_index("t")
