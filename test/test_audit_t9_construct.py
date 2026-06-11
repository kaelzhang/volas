"""Systematic audit — T9 (construct): DataFrame / Timestamp construction.

The constructor is the entry to every other subject, so its dtype *inference*
must honour the contract: native-NA integers (a hole does NOT promote int->float
the way legacy pandas does — C2), strings as a real `str` dtype (never object —
C3), and bool/float inferred faithfully.

Cell IDs:  T9.infer/<values> · T9.Timestamp/<value>
"""

from __future__ import annotations

import numpy as np
import pytest

import volas


@pytest.mark.parametrize("values,expected", [
    ([1, 2, 3], "int64"),
    ([1.0, 2, 3], "float64"),
    ([1, None, 3], "int64"),            # native-NA int — no float promotion (# C2)
    ([True, False], "bool"),
    (["a", "b"], "str"),                # a real str dtype, never object (# C3)
    (["a", None, "b"], "str"),
])
def test_dtype_inference(values, expected):
    s = volas.DataFrame({"x": values})["x"]
    assert s.dtype == expected, f"T9.infer/{values}"
    assert "object" not in s.dtype     # C3: object never surfaces


def test_native_na_int_preserves_value_and_mask():
    s = volas.DataFrame({"x": [1, None, 3]})["x"]
    assert s.isna().to_list() == [False, True, False]
    assert [x for x, m in zip(s.to_list(), s.isna().to_list()) if not m] == [1, 3]


# --- Timestamp construction -------------------------------------------------
def test_timestamp_from_string_and_numpy():
    t = volas.Timestamp("2021-06-15 13:45:30")
    assert (t.year, t.month, t.day) == (2021, 6, 15)
    assert t == volas.Timestamp(np.datetime64("2021-06-15T13:45:30"))


# F16 (decision 2): volas has NO NaT scalar, so Timestamp(None) must raise a
# clean ValueError (it currently leaks a `.loc` label-vocabulary KeyError, the
# same class as the fixed F3). Oracle is ValueError-only — NOT NaT (the old
# `is NaT` assertion contradicted decision 2; corrected per review). xfail(strict).
@pytest.mark.xfail(reason="F16: Timestamp(None) leaks label KeyError, should be clean ValueError", strict=True)
def test_timestamp_none_raises_valueerror():
    with pytest.raises(ValueError):
        volas.Timestamp(None)
