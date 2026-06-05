"""Spec-to-spec port of pandas's Series reduction tests.

Ported from pandas/tests/reductions/test_reductions.py,
pandas/tests/reductions/test_stat_reductions.py and
pandas/tests/series/test_reductions.py — restricted to the float64 / bool
Series with a default integer index that volas models. Every ``expected``
value below was produced by pandas itself (the oracle), so each case asserts
exact volas↔pandas parity for sum / mean / min / max / std / var / median.

volas detail differences (allowed by the porting brief, root cause noted):
  * ``std`` / ``var`` are sample statistics with ddof=1 and take no ``ddof``
    argument (pandas exposes ``ddof=``); the ddof=0 population variant is not
    portable and is omitted.
  * the sum of an empty / all-NaN Series is ``-0.0`` in volas, which is IEEE
    equal to pandas's ``0.0`` (``-0.0 == 0.0``); compared with ``==`` here.
"""

import math

import pytest

import volas

nan = float("nan")
inf = float("inf")


def _series(values):
    return volas.DataFrame({"x": values})["x"]


def _close(got, exp):
    if isinstance(exp, float) and math.isnan(exp):
        return math.isnan(got)
    if isinstance(exp, float) and math.isinf(exp):
        return got == exp
    return abs(got - exp) <= 1e-9 + 1e-9 * abs(exp)


# (id, values, op, expected) — expected computed by pandas.
REDUCTION_CASES = [
    # -- empty Series (GH#9422): sum is 0, the rest are NaN --
    ("empty_sum", [], "sum", 0.0),
    ("empty_mean", [], "mean", nan),
    ("empty_std", [], "std", nan),
    ("empty_var", [], "var", nan),
    ("empty_median", [], "median", nan),
    ("empty_min", [], "min", nan),
    ("empty_max", [], "max", nan),
    # -- all-NaN Series: sum skips to 0, the rest are NaN --
    ("allnan1_sum", [nan], "sum", 0.0),
    ("allnan1_mean", [nan], "mean", nan),
    ("allnan1_std", [nan], "std", nan),
    ("allnan1_median", [nan], "median", nan),
    ("allnan3_sum", [nan, nan, nan], "sum", 0.0),
    ("allnan3_mean", [nan, nan, nan], "mean", nan),
    ("allnan3_var", [nan, nan, nan], "var", nan),
    ("allnan3_median", [nan, nan, nan], "median", nan),
    # -- single element: std/var are NaN because ddof=1 (n-ddof=0) --
    ("single_sum", [5.0], "sum", 5.0),
    ("single_mean", [5.0], "mean", 5.0),
    ("single_median", [5.0], "median", 5.0),
    ("single_min", [5.0], "min", 5.0),
    ("single_max", [5.0], "max", 5.0),
    ("single_std_ddof1", [5.0], "std", nan),
    ("single_var_ddof1", [5.0], "var", nan),
    # -- mixed NaN + finite: NaN is skipped --
    ("nan_one_sum", [nan, 1.0], "sum", 1.0),
    ("nan_one_mean", [nan, 1.0], "mean", 1.0),
    ("nan_one_min", [nan, 1.0], "min", 1.0),
    ("nan_one_max", [nan, 1.0], "max", 1.0),
    ("nan_one_std", [nan, 1.0], "std", nan),
    ("one_nan_sum", [1.0, nan], "sum", 1.0),
    # -- canonical arange(20) base, no NaN --
    ("range20_sum", [float(i) for i in range(20)], "sum", 190.0),
    ("range20_mean", [float(i) for i in range(20)], "mean", 9.5),
    ("range20_median", [float(i) for i in range(20)], "median", 9.5),
    ("range20_std", [float(i) for i in range(20)], "std", 5.916079783099616),
    ("range20_var", [float(i) for i in range(20)], "var", 35.0),
    ("range20_min", [float(i) for i in range(20)], "min", 0.0),
    ("range20_max", [float(i) for i in range(20)], "max", 19.0),
    # -- arange(20) with the canonical [5:15] block NaN'd; survivors
    #    [0,1,2,3,4,15,16,17,18,19] (the _check_stat_op series) --
    (
        "range20nan_sum",
        [0.0, 1.0, 2.0, 3.0, 4.0] + [nan] * 10 + [15.0, 16.0, 17.0, 18.0, 19.0],
        "sum",
        95.0,
    ),
    (
        "range20nan_mean",
        [0.0, 1.0, 2.0, 3.0, 4.0] + [nan] * 10 + [15.0, 16.0, 17.0, 18.0, 19.0],
        "mean",
        9.5,
    ),
    (
        "range20nan_std",
        [0.0, 1.0, 2.0, 3.0, 4.0] + [nan] * 10 + [15.0, 16.0, 17.0, 18.0, 19.0],
        "std",
        8.045012257431447,
    ),
    (
        "range20nan_var",
        [0.0, 1.0, 2.0, 3.0, 4.0] + [nan] * 10 + [15.0, 16.0, 17.0, 18.0, 19.0],
        "var",
        64.72222222222223,
    ),
    (
        "range20nan_median",
        [0.0, 1.0, 2.0, 3.0, 4.0] + [nan] * 10 + [15.0, 16.0, 17.0, 18.0, 19.0],
        "median",
        9.5,
    ),
    # -- std/var with the textbook [1..5] sample (ddof=1) --
    ("v12345_mean", [1.0, 2.0, 3.0, 4.0, 5.0], "mean", 3.0),
    ("v12345_var_ddof1", [1.0, 2.0, 3.0, 4.0, 5.0], "var", 2.5),
    ("v12345_std_ddof1", [1.0, 2.0, 3.0, 4.0, 5.0], "std", 1.5811388300841898),
    # -- zero-variance (identical values) is 0.0, not NaN, for n>1 --
    ("equal_two_var", [2.0, 2.0], "var", 0.0),
    ("equal_two_std", [2.0, 2.0], "std", 0.0),
    # -- signed / negative values --
    ("neg4_sum", [-1.0, -2.0, -3.0, -4.0], "sum", -10.0),
    ("neg4_mean", [-1.0, -2.0, -3.0, -4.0], "mean", -2.5),
    ("neg4_median", [-1.0, -2.0, -3.0, -4.0], "median", -2.5),
    ("neg4_min", [-1.0, -2.0, -3.0, -4.0], "min", -4.0),
    ("neg4_max", [-1.0, -2.0, -3.0, -4.0], "max", -1.0),
    ("neg4_var", [-1.0, -2.0, -3.0, -4.0], "var", 1.6666666666666667),
    ("signed5_sum", [-2.0, -1.0, 0.0, 1.0, 2.0], "sum", 0.0),
    ("signed5_mean", [-2.0, -1.0, 0.0, 1.0, 2.0], "mean", 0.0),
    ("signed5_median", [-2.0, -1.0, 0.0, 1.0, 2.0], "median", 0.0),
    ("signed5_std", [-2.0, -1.0, 0.0, 1.0, 2.0], "std", 1.5811388300841898),
    # -- median: even vs odd count, and NaN-skipping changing the parity --
    ("median_odd", [1.0, 2.0, 3.0], "median", 2.0),
    ("median_even", [1.0, 2.0, 3.0, 4.0], "median", 2.5),
    ("median_nan_to_odd", [1.0, 2.0, nan, 4.0], "median", 2.0),
    ("median_nan_to_odd2", [1.0, 2.0, 3.0, nan, 4.0, 5.0], "median", 3.0),
    ("median_nan_to_even", [1.0, 2.0, 3.0, 4.0, nan], "median", 2.5),
    # -- infinity flows through sum --
    ("sum_posinf", [1.0, 2.0, inf], "sum", inf),
    ("sum_neginf", [1.0, 2.0, -inf], "sum", -inf),
    # -- bool Series reductions (volas supports a bool dtype) --
    ("bool_sum", [True, False, True, True], "sum", 3.0),
    ("bool_mean", [True, False, True, True], "mean", 0.75),
    ("bool_min", [True, False, True, True], "min", 0.0),
    ("bool_max", [True, False, True, True], "max", 1.0),
    ("bool_median", [True, False, True, True], "median", 1.0),
    ("bool_std", [True, False, True, True], "std", 0.5),
    ("bool_var", [True, False, True, True], "var", 0.25),
    ("bool_all_true_std", [True, True, True], "std", 0.0),
    ("bool_all_true_mean", [True, True, True], "mean", 1.0),
]


@pytest.mark.parametrize(
    "values,op,expected",
    [c[1:] for c in REDUCTION_CASES],
    ids=[c[0] for c in REDUCTION_CASES],
)
def test_reduction_matches_pandas(values, op, expected):
    got = getattr(_series(values), op)()
    assert _close(got, expected), f"{op}({values}) = {got!r}, expected {expected!r}"
