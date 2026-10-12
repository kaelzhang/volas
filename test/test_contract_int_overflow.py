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


def test_missing_cells_do_not_overflow():
    # the masked construction channel keeps I64_MAX as the missing cell's payload
    s = volas.DataFrame({"a": ([I64_MAX, 1], [True, False], "int64")})["a"]
    assert s.sum() == 1
    assert s.cumsum().to_list() == [volas.NA, 1]
    assert (s + 1).to_list() == [volas.NA, 2]
    assert (s * 2).to_list() == [volas.NA, 2]
