"""Contract #25: integer overflow raises, identically in debug and release builds.

pandas / NumPy wrap int64 arithmetic silently (``max + 1 == min``); volas raises an
``OverflowError`` instead, from every integer kernel: element-wise ``+ - *``, the
``sum`` / ``prod`` reductions, ``cumsum`` / ``cumprod``, ``diff``, ``abs``, ``round``
and floor division. A missing cell's placeholder never trips the check.
"""

import numpy as np
import pytest

import volas

I64_MAX = 2**63 - 1
I64_MIN = -(2**63)


def _s(values, dtype=np.int64):
    return volas.DataFrame({"a": np.array(values, dtype=dtype)})["a"]


@pytest.mark.parametrize(
    "op",
    [
        lambda: _s([I64_MAX]) + 1,
        lambda: _s([I64_MIN]) - 1,
        lambda: 1 - _s([I64_MIN]),
        lambda: _s([I64_MAX]) * 2,
        lambda: _s([I64_MAX]) + _s([1]),
        lambda: _s([I64_MAX, 1]).sum(),
        lambda: _s([2**62, 4]).prod(),
        lambda: _s([I64_MAX, 1]).cumsum(),
        lambda: _s([2**62, 4]).cumprod(),
        lambda: _s([I64_MIN, 1]).diff(),
        lambda: _s([I64_MIN]).abs(),
        lambda: _s([I64_MAX]).round(-1),
        # 10^19 itself is past int64: |x| > 5e18 rounds to +-1e19
        lambda: _s([6 * 10**18]).round(-19),
        lambda: _s([-6 * 10**18]).round(-19),
        lambda: _s([5 * 10**18 + 1]).round(-19),
        lambda: volas.DataFrame({"a": np.array([6 * 10**18])}).round(-19),
        lambda: _s([I64_MIN]) // -1,
        lambda: _s([2**31 - 1], np.int32) + _s([1], np.int32),
        lambda: _s([2**31 - 1, 1], np.int32).cumsum(),
        lambda: _s([-(2**31)], np.int32).abs(),
        lambda: volas.DataFrame({"a": np.array([I64_MAX, 1])}).cumsum(),
        lambda: volas.DataFrame({"a": np.array([I64_MIN])}).abs(),
    ],
)
def test_integer_overflow_raises(op):
    with pytest.raises(OverflowError, match="overflow"):
        op()


def test_in_range_results_are_exact():
    assert (_s([I64_MAX - 1]) + 1).to_list() == [I64_MAX]
    assert _s([I64_MAX - 1, 1]).sum() == I64_MAX
    assert _s([2**31 - 1, 1], np.int32).sum() == 2**31  # int32 sums accumulate in int64
    assert _s([-5, 3]).abs().to_list() == [5, 3]


def test_round_at_the_int64_decimal_edge():
    # exact half-to-even at the 10^19 grid: the tie and everything inside it go to 0
    edge = [5 * 10**18, -5 * 10**18, 5 * 10**18 - 1, I64_MIN + 5 * 10**18, 0]
    assert _s(edge).round(-19).to_list() == [0, 0, 0, 0, 0]
    assert _s([I64_MAX, I64_MIN]).round(-20).to_list() == [0, 0]
    assert _s([I64_MAX, I64_MIN]).round(-(2**31 - 1)).to_list() == [0, 0]
    assert _s([6 * 10**18, -6 * 10**18]).round(-18).to_list() == [6 * 10**18, -6 * 10**18]
    assert _s([2**31 - 1], np.int32).round(-10).to_list() == [0]


def test_missing_cells_do_not_overflow():
    # the masked construction channel keeps I64_MAX as the missing cell's payload
    s = volas.DataFrame({"a": ([I64_MAX, 1], [True, False], "int64")})["a"]
    assert s.sum() == 1
    assert s.cumsum().to_list() == [volas.NA, 1]
    assert (s + 1).to_list() == [volas.NA, 2]
    assert (s * 2).to_list() == [volas.NA, 2]
