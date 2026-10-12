"""Contract #27 (Series half): boolean conversion carries validity.

A missing value (float NaN, int / bool NA) never becomes a present ``True`` /
``False``: logical ops follow Kleene logic, ``~NA`` is NA, ``bool()`` of a
missing single value raises, and Series / DataFrame ``any`` / ``all`` read the
same validity-aware truth values.
"""

import pytest

import volas

NA = volas.NA
nan = float("nan")


def _s(values):
    return volas.DataFrame({"a": values})["a"]


@pytest.mark.parametrize("missing", [nan, None, NA])
def test_logical_with_a_missing_scalar_is_kleene(missing):
    s = _s([True, False])
    assert (s & missing).to_list() == [NA, False]
    assert (s | missing).to_list() == [True, NA]
    assert (s ^ missing).to_list() == [NA, NA]


def test_logical_with_missing_numeric_cells_is_kleene():
    b = _s([True, True, False])
    assert (b & _s([nan, 1.0, nan])).to_list() == [NA, True, False]
    assert (b | _s([None, 0, None])).to_list() == [True, True, NA]
    assert (~_s([None, 1])).to_list() == [NA, False]
    assert (~_s([nan, 0.0])).to_list() == [NA, True]
    # a payload under an Arrow-style NA slot is never read
    masked = volas.DataFrame({"a": ([7, 0], [True, False], "int64")})["a"]
    assert (~masked).to_list() == [NA, True]


def test_bool_of_a_missing_single_value_raises():
    with pytest.raises(TypeError, match="NA is ambiguous"):
        bool(_s([nan]))
    with pytest.raises(TypeError, match="NA is ambiguous"):
        bool(_s([None, True]).iloc[0:1])
    assert bool(_s([1.0])) is True
    assert bool(_s([0])) is False


def test_any_all_read_validity_consistently():
    assert _s([nan, 0.0]).any() == False  # noqa: E712 — a missing cell is skipped
    assert _s([nan, 1.0]).all() == True  # noqa: E712
    df = volas.DataFrame({"f": [nan, 0.0], "s": ["", "x"], "i": [None, 2]})
    assert df.any().index.tolist() == ["f", "i"]  # str is not truth-valued, like Series.any
    assert df.any().to_list() == [False, True]
    assert df.all().to_list() == [False, True]
    with pytest.raises(TypeError):
        df["s"].any()

