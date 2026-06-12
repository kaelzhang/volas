"""Spec-to-spec port of pandas's Series shift / diff / fillna / isna / dropna tests.

Ported from pandas/tests/frame/methods/test_shift.py (Series path via
frame_or_series), pandas/tests/series/methods/test_diff.py, test_fillna.py,
test_isna.py and test_dropna.py — restricted to the float64 / bool, default
integer index Series volas models. Every ``expected`` was produced by pandas.

volas detail differences (allowed by the porting brief, root cause noted):
  * ``shift`` takes an integer period only (no ``freq=`` / ``fill_value=``);
    only the integer-period, NaN-fill cases are portable.
  * ``fillna`` takes a scalar float only (no ``method=`` ffill/bfill, no dict);
    only the scalar-fill cases are portable.
"""


import numpy as np
import pytest

import volas

nan = float("nan")


def _series(values):
    return volas.DataFrame({"x": values})["x"]


def _vals(series):
    return np.asarray(series.to_numpy(), dtype=float)


def _idx(series):
    return np.asarray(series.index).tolist()


def _assert_vals(series, expected):
    got = _vals(series)
    exp = np.asarray([float(x) for x in expected], dtype=float)
    assert got.shape == exp.shape, f"{got.shape} != {exp.shape}"
    assert np.array_equal(got, exp, equal_nan=True), f"{got.tolist()} != {expected}"


# (id, values, method, args, expected) — same-length result.
ELEMENTWISE_CASES = [
    # -- shift: positive / negative / zero / >= len / preserves NaN --
    ("shift_pos_1", [0.0, 1.0, 2.0, 3.0, 4.0], "shift", (1,), [nan, 0.0, 1.0, 2.0, 3.0]),
    ("shift_pos_2", [0.0, 1.0, 2.0, 3.0, 4.0], "shift", (2,), [nan, nan, 0.0, 1.0, 2.0]),
    ("shift_neg_1", [0.0, 1.0, 2.0, 3.0, 4.0], "shift", (-1,), [1.0, 2.0, 3.0, 4.0, nan]),
    ("shift_neg_2", [0.0, 1.0, 2.0, 3.0, 4.0], "shift", (-2,), [2.0, 3.0, 4.0, nan, nan]),
    ("shift_zero", [0.0, 1.0, 2.0, 3.0, 4.0], "shift", (0,), [0.0, 1.0, 2.0, 3.0, 4.0]),
    ("shift_pos_eq_len", [1.0, 2.0, 3.0], "shift", (3,), [nan, nan, nan]),
    ("shift_pos_gt_len", [1.0, 2.0, 3.0], "shift", (5,), [nan, nan, nan]),
    ("shift_neg_eq_len", [1.0, 2.0, 3.0], "shift", (-3,), [nan, nan, nan]),
    ("shift_neg_gt_len", [1.0, 2.0, 3.0], "shift", (-5,), [nan, nan, nan]),
    (
        "shift_5",
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "shift",
        (5,),
        [nan, nan, nan, nan, nan, 0.0, 1.0, 2.0],
    ),
    ("shift_preserves_nan", [nan, 1.0, nan, 3.0], "shift", (1,), [nan, nan, 1.0, nan]),
    ("shift_empty", [], "shift", (1,), []),
    # -- diff: diff(n) == s - s.shift(n) --
    ("diff_default_1", [0.0, 1.0, 2.0, 3.0, 4.0], "diff", (1,), [nan, 1.0, 1.0, 1.0, 1.0]),
    ("diff_n1", [1.0, 3.0, 6.0, 10.0], "diff", (1,), [nan, 2.0, 3.0, 4.0]),
    ("diff_n2", [0.0, 1.0, 2.0, 3.0, 4.0], "diff", (2,), [nan, nan, 2.0, 2.0, 2.0]),
    ("diff_neg1", [0.0, 1.0, 2.0, 3.0, 4.0], "diff", (-1,), [-1.0, -1.0, -1.0, -1.0, nan]),
    ("diff_zero", [0.0, 1.0, 2.0, 3.0, 4.0], "diff", (0,), [0.0, 0.0, 0.0, 0.0, 0.0]),
    ("diff_n_eq_len", [1.0, 2.0, 3.0], "diff", (3,), [nan, nan, nan]),
    ("diff_n_gt_len", [1.0, 2.0, 3.0], "diff", (5,), [nan, nan, nan]),
    ("diff_with_nan", [1.0, nan, 3.0, 6.0], "diff", (1,), [nan, nan, nan, 3.0]),
    # -- fillna: scalar fill --
    ("fillna_scattered", [0.0, 1.0, nan, 3.0, 4.0], "fillna", (5.0,), [0.0, 1.0, 5.0, 3.0, 4.0]),
    ("fillna_zero", [0.0, 1.0, nan, nan, 4.0], "fillna", (0.0,), [0.0, 1.0, 0.0, 0.0, 4.0]),
    ("fillna_no_nan", [0.0, 1.0, 2.0, 3.0, 4.0], "fillna", (9.0,), [0.0, 1.0, 2.0, 3.0, 4.0]),
    ("fillna_all_nan", [nan, nan, nan], "fillna", (999.0,), [999.0, 999.0, 999.0]),
    (
        "fillna_lead_trail_interior",
        [nan, 1.0, nan, 3.0, nan],
        "fillna",
        (0.0,),
        [0.0, 1.0, 0.0, 3.0, 0.0],
    ),
    ("fillna_single_nan", [nan], "fillna", (1.0,), [1.0]),
    (
        "fillna_negative",
        [0.0, 1.0, nan, nan, 4.0],
        "fillna",
        (-0.3,),
        [0.0, 1.0, -0.3, -0.3, 4.0],
    ),
    ("fillna_empty", [], "fillna", (0.0,), []),
    # -- isna / notna --
    (
        "isna_mixed",
        [0.0, 5.4, 3.0, nan, -0.001],
        "isna",
        (),
        [False, False, False, True, False],
    ),
    (
        "notna_mixed",
        [0.0, 5.4, 3.0, nan, -0.001],
        "notna",
        (),
        [True, True, True, False, True],
    ),
    ("isna_no_nan", [1.0, 2.0, 3.0], "isna", (), [False, False, False]),
    ("isna_all_nan", [nan, nan, nan], "isna", (), [True, True, True]),
    ("notna_all_nan", [nan, nan, nan], "notna", (), [False, False, False]),
    ("isna_lead_trail", [nan, 1.0, 2.0, nan], "isna", (), [True, False, False, True]),
    ("isna_empty", [], "isna", (), []),
]


@pytest.mark.parametrize(
    "values,method,args,expected",
    [c[1:] for c in ELEMENTWISE_CASES],
    ids=[c[0] for c in ELEMENTWISE_CASES],
)
def test_series_method_matches_pandas(values, method, args, expected):
    _assert_vals(getattr(_series(values), method)(*args), expected)


def test_diff_on_bool_series_raises():
    # diff is subtraction, and bool - bool is unsupported (use ^), so a bool
    # Series's diff raises rather than funnelling through float64 (contract C4,
    # consistent with bool-bool subtraction).
    with pytest.raises(Exception):
        _series([False, True, True, False, False]).diff(1)


# (id, values, expected_values, surviving_index) — dropna returns a shorter
# Series with the survivors' original index labels preserved.
DROPNA_CASES = [
    ("dropna_interior", [nan, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1, 2, 3]),
    ("dropna_leading", [nan, nan, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2, 3, 4]),
    ("dropna_trailing", [1.0, 2.0, 3.0, nan, nan], [1.0, 2.0, 3.0], [0, 1, 2]),
    ("dropna_interleaved", [1.0, nan, 2.0, nan, 3.0], [1.0, 2.0, 3.0], [0, 2, 4]),
    ("dropna_no_nan", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0, 1, 2]),
    ("dropna_all_nan", [nan, nan, nan], [], []),
    ("dropna_empty", [], [], []),
    ("dropna_single_kept", [nan, 7.0, nan], [7.0], [1]),
]


@pytest.mark.parametrize(
    "values,expected_values,surviving_index",
    [c[1:] for c in DROPNA_CASES],
    ids=[c[0] for c in DROPNA_CASES],
)
def test_dropna_matches_pandas(values, expected_values, surviving_index):
    out = _series(values).dropna()
    _assert_vals(out, expected_values)
    assert _idx(out) == surviving_index
