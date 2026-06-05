"""Spec-to-spec port of pandas's Series binary-op tests.

Ported from pandas/tests/series/test_arithmetic.py and
pandas/tests/series/test_logical_ops.py — restricted to the float64 / bool,
same-length, default-index Series that volas models. Every ``expected`` list
was produced by pandas itself, so each case asserts exact volas↔pandas parity.

volas detail differences (allowed by the porting brief, root cause noted):
  * volas binary ops are positional/elementwise on equal-length operands; it
    does not align on index labels, so the alignment-across-labels cases in
    pandas are not ported (they describe a feature volas deliberately omits).
  * volas implements ``+ - * /`` only (no ``// % ** divmod``) and ``& | ^ ~``
    are bool-only (no integer bitwise), matching its float64/bool model.
"""

import math
import operator

import numpy as np
import pytest

import volas

nan = float("nan")
inf = float("inf")

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}


def _series(values):
    return volas.DataFrame({"x": values})["x"]


def _apply(left, right, op, mode):
    ls = _series(left)
    if mode == "unary":
        return ~ls
    r = _series(right) if isinstance(right, list) else right
    if mode == "reflected":  # e.g. 2.0 - s  ->  s.__rsub__(2.0)
        return OPS[op](r, ls)
    return OPS[op](ls, r)


def _check(left, right, op, mode, expected):
    got = np.asarray(_apply(left, right, op, mode).to_numpy(), dtype=float)
    exp = np.asarray([float(x) for x in expected], dtype=float)
    assert got.shape == exp.shape, f"{got.shape} != {exp.shape}"
    assert np.array_equal(got, exp, equal_nan=True), f"{op} -> {got.tolist()} != {exp}"


# (id, left, right, op, mode, expected)
ARITH_CASES = [
    # -- division by zero: 1/0=inf, -1/0=-inf, 0/0=nan (IEEE float division) --
    ("div_pos_by_zero", [1.0], 0.0, "/", "normal", [inf]),
    ("div_neg_by_zero", [-1.0], 0.0, "/", "normal", [-inf]),
    ("div_zero_by_zero", [0.0], 0.0, "/", "normal", [nan]),
    (
        "div_by_zero_vec",
        [1.0, -1.0, 0.0, 2.0, -3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        "/",
        "normal",
        [inf, -inf, nan, inf, -inf],
    ),
    ("div_scalar_zero_mixed", [1.0, -1.0, 0.0], 0.0, "/", "normal", [inf, -inf, nan]),
    ("rdiv_by_zero_left", [0.0, 0.0, 0.0], 1.0, "/", "reflected", [inf, inf, inf]),
    # -- NaN propagation through + - * / (vector op vector) --
    ("add_nan", [1.0, nan, 3.0], [nan, 2.0, 3.0], "+", "normal", [nan, nan, 6.0]),
    ("sub_nan", [1.0, nan, 3.0], [nan, 2.0, 3.0], "-", "normal", [nan, nan, 0.0]),
    ("mul_nan", [1.0, nan, 3.0], [nan, 2.0, 3.0], "*", "normal", [nan, nan, 9.0]),
    ("div_nan", [1.0, nan, 3.0], [nan, 2.0, 3.0], "/", "normal", [nan, nan, 1.0]),
    # -- scalar broadcast --
    ("add_scalar", [1.0, 2.0, 3.0], 2.0, "+", "normal", [3.0, 4.0, 5.0]),
    ("sub_scalar", [1.0, 2.0, 3.0], 2.0, "-", "normal", [-1.0, 0.0, 1.0]),
    ("mul_scalar", [1.0, 2.0, 3.0], 2.0, "*", "normal", [2.0, 4.0, 6.0]),
    # -- reflected scalar (scalar on the left) --
    ("radd_scalar", [1.0, 2.0, 3.0], 2.0, "+", "reflected", [3.0, 4.0, 5.0]),
    ("rsub_scalar", [1.0, 2.0, 3.0], 2.0, "-", "reflected", [1.0, 0.0, -1.0]),
    ("rsub_scalar_10", [1.0, 2.0, 3.0], 10.0, "-", "reflected", [9.0, 8.0, 7.0]),
    ("rmul_scalar", [1.0, 2.0, 3.0], 2.0, "*", "reflected", [2.0, 4.0, 6.0]),
    (
        "rtruediv_scalar",
        [1.0, 2.0, 3.0],
        2.0,
        "/",
        "reflected",
        [2.0, 1.0, 0.6666666666666666],
    ),
]

CMP_CASES = [
    # -- comparison involving NaN is always False except != --
    ("cmp_lt_nan", [1.0, nan, 3.0], [1.0, 2.0, 2.0], "<", "normal", [False, False, False]),
    ("cmp_le_nan", [1.0, nan, 3.0], [1.0, 2.0, 2.0], "<=", "normal", [True, False, False]),
    ("cmp_eq_nan", [1.0, nan, 3.0], [1.0, 2.0, 2.0], "==", "normal", [True, False, False]),
    ("cmp_ne_nan", [1.0, nan, 3.0], [1.0, 2.0, 2.0], "!=", "normal", [False, True, True]),
    ("cmp_ge_nan", [1.0, nan, 3.0], [1.0, 2.0, 2.0], ">=", "normal", [True, False, True]),
    ("cmp_gt_nan", [1.0, nan, 3.0], [1.0, 2.0, 2.0], ">", "normal", [False, False, True]),
    ("cmp_eq_nan_scalar", [1.0, nan, 3.0], 2.0, "==", "normal", [False, False, False]),
    ("cmp_ne_nan_scalar", [1.0, nan, 3.0], 2.0, "!=", "normal", [True, True, True]),
    ("cmp_eq_nan_vs_nan", [nan], [nan], "==", "normal", [False]),
    ("cmp_ne_nan_vs_nan", [nan], [nan], "!=", "normal", [True]),
    # -- elementwise == / != and the ordered ops --
    (
        "eq_basic",
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 9.0, 3.0, 9.0],
        "==",
        "normal",
        [True, False, True, False],
    ),
    (
        "ne_basic",
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 9.0, 3.0, 9.0],
        "!=",
        "normal",
        [False, True, False, True],
    ),
    ("le_vec", [1.0, 3.0, 2.0], [2.0, 2.0, 2.0], "<=", "normal", [True, False, True]),
    ("lt_vec", [1.0, 3.0, 2.0], [2.0, 2.0, 2.0], "<", "normal", [True, False, False]),
    ("ge_vec", [1.0, 3.0, 2.0], [2.0, 2.0, 2.0], ">=", "normal", [False, True, True]),
    ("gt_vec", [1.0, 3.0, 2.0], [2.0, 2.0, 2.0], ">", "normal", [False, True, False]),
]

LOGICAL_CASES = [
    # -- bool & / | / ^ (vector op vector), truth tables --
    (
        "and_2x2",
        [True, True, False, False],
        [True, False, True, False],
        "&",
        "normal",
        [True, False, False, False],
    ),
    (
        "or_2x2",
        [True, True, False, False],
        [True, False, True, False],
        "|",
        "normal",
        [True, True, True, False],
    ),
    (
        "xor_2x2",
        [True, True, False, False],
        [True, False, True, False],
        "^",
        "normal",
        [False, True, True, False],
    ),
    # -- unary invert --
    ("not_t", [True, True, False, False], None, "&", "unary", [False, False, True, True]),
    ("not_u", [True, False, True, False], None, "&", "unary", [False, True, False, True]),
    ("not_allT", [True, True, True], None, "&", "unary", [False, False, False]),
    ("not_allF", [False, False, False], None, "&", "unary", [True, True, True]),
    # -- all-True / all-False operands against a mix --
    (
        "and_allT_mix",
        [True, True, True],
        [True, False, True],
        "&",
        "normal",
        [True, False, True],
    ),
    (
        "or_allF_mix",
        [False, False, False],
        [True, False, True],
        "|",
        "normal",
        [True, False, True],
    ),
    (
        "xor_allT_mix",
        [True, True, True],
        [True, False, True],
        "^",
        "normal",
        [False, True, False],
    ),
    # -- bool op scalar bool --
    ("and_scalar_true", [True, False, True], True, "&", "normal", [True, False, True]),
    ("and_scalar_false", [True, False, True], False, "&", "normal", [False, False, False]),
    ("or_scalar_false", [True, False, True], False, "|", "normal", [True, False, True]),
    ("xor_scalar_true", [True, False, True], True, "^", "normal", [False, True, False]),
    # -- reflected scalar bool (scalar on the left -> __rand__ / __ror__ / __rxor__) --
    ("rand_scalar_true", [True, False, True], True, "&", "reflected", [True, False, True]),
    ("ror_scalar_false", [True, False, True], False, "|", "reflected", [True, False, True]),
    ("rxor_scalar_true", [True, False, True], True, "^", "reflected", [False, True, False]),
    # -- 5-element truth table --
    (
        "and_5elem",
        [True, True, True, False, True],
        [True, False, False, True, False],
        "&",
        "normal",
        [True, False, False, False, False],
    ),
    (
        "or_5elem",
        [True, True, True, False, True],
        [True, False, False, True, False],
        "|",
        "normal",
        [True, True, True, True, True],
    ),
    (
        "xor_5elem",
        [True, True, True, False, True],
        [True, False, False, True, False],
        "^",
        "normal",
        [False, True, True, True, True],
    ),
]

ALL_CASES = ARITH_CASES + CMP_CASES + LOGICAL_CASES


@pytest.mark.parametrize(
    "left,right,op,mode,expected",
    [c[1:] for c in ALL_CASES],
    ids=[c[0] for c in ALL_CASES],
)
def test_binary_op_matches_pandas(left, right, op, mode, expected):
    _check(left, right, op, mode, expected)
