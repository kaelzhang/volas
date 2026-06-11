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


# F28 (findings-ledger): partial-string datetime indexing (a month/year selects
# the range) is a core pandas feature; volas leaks the .loc label KeyError.
@pytest.mark.xfail(reason="F28: partial-string datetime indexing missing (label KeyError leak)", strict=True)
def test_loc_datetime_partial_string():
    assert _dt_indexed().loc["2021-02"].shape == (1, 1)        # the whole month


# F34 (decision 1B): volas's index assumes unique labels, so set_index should
# REJECT a duplicate-label column at creation (like it rejects NA-label columns,
# V16) — not allow it and then leak AttributeError on .loc. xfail(strict):
# set_index currently accepts the duplicate; a fix makes it raise -> XPASS flips.
@pytest.mark.xfail(reason="F34: set_index should reject duplicate labels (decision 1B), currently allows", strict=True)
def test_set_index_rejects_duplicate_labels():
    with pytest.raises((ValueError, KeyError)):
        _indexed([1, 1, 2])              # set_index on a column with the duplicate label 1
