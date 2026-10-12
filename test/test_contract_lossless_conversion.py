"""Contract #20: one lossless conversion — exact converts, inexact raises.

Target-typed scalar writes (``iat`` / ``iloc`` / ``at`` / dict ``append``), column
ingress (list / tuple / NumPy uint64 / object / complex / Decimal / mixed int+float),
``astype`` and ``read_csv`` inference never silently round or saturate a value.
"""

import decimal

import numpy as np
import pytest

import volas

B = 2**53 + 1  # the first integer float64 cannot hold


def _col(values):
    return volas.DataFrame({"x": values})["x"]


# --- target-typed scalar writes ---------------------------------------------

@pytest.mark.parametrize("write", ["iat", "iloc", "at"])
def test_inexact_int_into_float64_cell_raises(write):
    df = volas.DataFrame({"f": [0.0, 1.0]})
    acc = getattr(df, write)
    key = (0, 0) if write != "at" else (0, "f")
    with pytest.raises(TypeError, match="not exactly representable as float64"):
        acc[key] = B
    assert df["f"].to_list() == [0.0, 1.0]
    acc[key] = 2**60  # exactly representable: stored exactly
    assert df["f"].to_list()[0] == 2.0**60


def test_float32_cell_writes():
    df = volas.DataFrame({"f": np.array([0.0], dtype=np.float32)})
    with pytest.raises(TypeError, match="float32"):
        df.iat[0, 0] = 2**24 + 1  # an integer float32 cannot hold
    with pytest.raises(OverflowError, match="float32"):
        df.iat[0, 0] = 1e300  # would become inf
    df.iat[0, 0] = 0.1  # a float rounds to the nearest float32 (float precision)
    assert df["f"].to_list()[0] == pytest.approx(0.1)


def test_int_cell_writes_are_lossless():
    df = volas.DataFrame({"i": [1, 2]})
    df.iat[0, 0] = 5.0  # integral float: exact
    assert df["i"].to_list() == [5, 2]
    with pytest.raises(TypeError):
        df.iat[0, 0] = 2.5
    with pytest.raises(OverflowError):
        df.iat[0, 0] = 1e20
    with pytest.raises(OverflowError):
        df.iat[0, 0] = np.uint64(2**63 + 1)
    df.iat[1, 0] = np.int64(7)
    assert df["i"].to_list() == [5, 7]


def test_int32_cell_write_out_of_range_raises():
    df = volas.DataFrame({"i": np.array([1], dtype=np.int32)})
    with pytest.raises(OverflowError, match="int32"):
        df.iat[0, 0] = 2**40


def test_huge_python_int_into_float_cell():
    df = volas.DataFrame({"f": [0.0]})
    df.iat[0, 0] = 2**64  # exact in float64
    assert df["f"].to_list() == [2.0**64]
    with pytest.raises(TypeError, match="not exactly representable"):
        df.iat[0, 0] = 2**64 + 1
    with pytest.raises(OverflowError):
        df.iat[0, 0] = 10**400  # past the float64 range


def test_dict_append_converts_losslessly():
    df = volas.DataFrame({"f": [0.0], "i": [1]})
    with pytest.raises(TypeError, match="not exactly representable"):
        df.append({"f": B, "i": 2})
    assert len(df) == 1
    df.append({"f": 2**60, "i": 3.0})  # both exact
    assert df["f"].to_list()[1] == 2.0**60
    assert df["i"].to_list() == [1, 3]


# --- column ingress ---------------------------------------------------------

@pytest.mark.parametrize(
    "values",
    [
        [B, 1.5],
        [B, float("nan")],
        (B, 1.5),
        np.array([B, 1.5], dtype=object),
    ],
)
def test_mixed_int_float_ingress_raises_when_inexact(values):
    with pytest.raises(TypeError, match="not exactly representable as float64"):
        _col(values)


def test_mixed_int_float_ingress_converts_when_exact():
    assert _col([2**60, 1.5]).to_list() == [2.0**60, 1.5]
    assert _col((1, 2.5)).to_list() == [1.0, 2.5]


@pytest.mark.parametrize(
    "values",
    [
        (2**63 + 1,),
        np.array([2**63, 2**63 + 1], dtype=np.uint64),
        np.array([2**70 + 1], dtype=object),
        [np.uint64(2**63 + 1)],
    ],
)
def test_integers_past_int64_raise(values):
    with pytest.raises(OverflowError, match="int64"):
        _col(values)


def test_unsigned_arrays_widen_exactly():
    assert _col(np.array([1, 2**63 - 1], dtype=np.uint64)).to_list() == [1, 2**63 - 1]
    s = _col(np.array([1, 2**32 - 1], dtype=np.uint32))
    assert s.dtype == "int64" and s.to_list() == [1, 2**32 - 1]


@pytest.mark.parametrize(
    "values",
    [(decimal.Decimal("9007199254740993"),), np.array([decimal.Decimal("1.5")], dtype=object)],
)
def test_decimal_ingress_raises(values):
    with pytest.raises(TypeError, match="Decimal"):
        _col(values)


@pytest.mark.parametrize("values", [np.array([1 + 2j]), [1 + 2j]])
def test_complex_ingress_raises(values):
    with pytest.raises(TypeError, match="complex"):
        _col(values)


def test_object_and_tuple_ingress_keep_list_inference():
    assert _col(np.array([1, None, 3], dtype=object)).to_list() == [1, volas.NA, 3]
    assert _col(("a", "b")).to_list() == ["a", "b"]


# --- astype -----------------------------------------------------------------

def test_astype_float_to_int_out_of_range_raises():
    with pytest.raises(OverflowError, match="int64"):
        _col([1e20]).astype("int64")
    with pytest.raises(OverflowError):
        _col(np.array([-1e30], dtype=np.float32)).astype("int64")
    assert _col([2.7, -2.7]).astype("int64").to_list() == [2, -2]  # truncation kept


def test_astype_int_to_float_is_exact_or_raises():
    with pytest.raises(TypeError, match="not exactly representable as float64"):
        _col([B]).astype("float64")
    with pytest.raises(TypeError, match="float32"):
        _col([2**24 + 1]).astype("float32")
    assert _col([2**60]).astype("float64").to_list() == [2.0**60]
    with pytest.raises(OverflowError, match="float32"):
        _col([1e300]).astype("float32")


# --- read_csv inference -------------------------------------------------------

def _csv(tmp_path, text, **kw):
    p = tmp_path / "a.csv"
    p.write_text(text)
    return volas.read_csv(str(p), **kw)["a"]


@pytest.mark.parametrize(
    "text, values",
    [
        ("a\n9223372036854775809\n", ["9223372036854775809"]),  # past int64
        ("a\n9007199254740993\n1.5\n", ["9007199254740993", "1.5"]),  # past 2^53 + float
    ],
)
def test_csv_integer_float64_cannot_hold_stays_text(tmp_path, text, values):
    s = _csv(tmp_path, text)
    assert s.dtype == "str" and s.to_list() == values


def test_csv_exact_integers_mixed_with_floats_are_float64(tmp_path):
    s = _csv(tmp_path, "a\n9007199254740992\n-007\n1.5\n18446744073709551616\n")
    assert s.dtype == "float64"
    assert s.to_list() == [2.0**53, -7.0, 1.5, 2.0**64]


def test_csv_nan_text_without_default_na_stays_text(tmp_path):
    s = _csv(tmp_path, "a\n1\nnan\n", keep_default_na=False)
    assert s.dtype == "str" and s.to_list() == ["1", "nan"]
    assert _csv(tmp_path, "a\n1\nnan\n").to_list() == [1, volas.NA]  # default NA token


def test_csv_bool_inference_honors_na(tmp_path):
    s = _csv(tmp_path, "a\nTrue\nFalse\n", na_values=["True"])
    assert s.dtype == "bool" and s.to_list() == [volas.NA, False]
    s = _csv(tmp_path, "a\ntrue\nNA\nFALSE\n")
    assert s.dtype == "bool" and s.to_list() == [True, volas.NA, False]


def test_float_epoch_past_int64_raises():
    with pytest.raises(ValueError, match="epoch"):
        _col([1e30]).astype("datetime64[ns]")
