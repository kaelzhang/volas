"""volas.NA — the native missing-value model (PDEP-16 aligned, no object dtype).

int/bool columns keep their dtype and carry a `volas.NA` in vacated cells
(shift / diff gap), where pandas 3.0 upcasts to float/object. `volas.NA` is a
pure user-facing symbol; storage stays dtype-optimal underneath.
"""

import numpy as np
import pytest

import volas


def _int_na():
    # int64 column with a leading NA from shift: [NA, 1, 2]
    return volas.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})["a"].shift(1)


def test_na_singleton_repr_and_bool():
    assert repr(volas.NA) == "<NA>"
    assert volas.NA is volas.NA
    with pytest.raises(TypeError):
        bool(volas.NA)


def test_int_shift_keeps_dtype_with_na():
    s = _int_na()
    assert s.dtype == "int64"
    assert s[0] is volas.NA
    assert s[1] == 1 and isinstance(s[1], np.int64)
    assert s.isna().to_numpy().tolist() == [True, False, False]
    assert s.notna().to_numpy().tolist() == [False, True, True]


def test_na_element_access_per_dtype():
    si = volas.DataFrame({"a": np.array([1, 2, 3], dtype=np.int32)})["a"].shift(1)
    assert si.dtype == "int32" and si[0] is volas.NA and si[1] == 1
    sb = (volas.DataFrame({"a": np.array([1, 0, 1], dtype=np.int64)})["a"] > 0).shift(1)
    assert sb.dtype == "bool" and sb[0] is volas.NA
    assert bool(sb[1]) is True  # the present cell is a real np.bool_


def test_na_to_numpy_and_to_list():
    s = _int_na()  # NA, 1, 2
    arr = s.to_numpy()
    assert arr.dtype == np.float64  # numpy can't hold NA -> float64 + nan
    assert np.isnan(arr[0]) and arr[1] == 1.0
    lst = s.to_list()
    assert lst[0] is volas.NA and lst[1] == 1


def test_na_reductions_skip_missing():
    s = _int_na()  # NA, 1, 2
    assert s.sum() == 3 and isinstance(s.sum(), np.int64)
    assert s.min() == 1 and s.max() == 2
    assert s.dropna().to_list() == [1, 2]


def test_na_fillna_promotes_only_when_needed():
    s = _int_na()  # NA, 1, 2
    assert s.fillna(0).to_list() == [0, 1, 2]
    assert s.fillna(0).dtype == "int64"
    assert s.fillna(2.5).dtype == "float64"  # a non-integral fill promotes to float


def test_na_directional_fill():
    s = _int_na()  # NA, 1, 2
    assert s.bfill().to_list() == [1, 1, 2]  # backfill the leading gap
    assert s.ffill()[0] is volas.NA  # nothing to carry forward into the leading gap
    assert s.ffill().dtype == "int64"


def test_na_diff_keeps_int():
    s = volas.DataFrame({"a": np.array([1, 3, 6], dtype=np.int64)})["a"].diff()
    assert s.dtype == "int64"
    assert s[0] is volas.NA and s[1] == 2 and s[2] == 3


def test_na_construction_from_none():
    # None in a list carries a missing value; int/bool keep their dtype + NA where
    # pandas 3.0 upcasts to float/object.
    s = volas.DataFrame({"a": [1, None, 3]})["a"]
    assert s.dtype == "int64"
    lst = s.to_list()
    assert lst[0] == 1 and lst[1] is volas.NA and lst[2] == 3
    assert s.sum() == 4
    b = volas.DataFrame({"a": [True, None, False]})["a"]
    assert b.dtype == "bool" and b[1] is volas.NA
    f = volas.DataFrame({"a": [1.5, None, 3.0]})["a"]
    assert f.dtype == "float64" and np.isnan(f.to_numpy()[1])
    # an all-None column, and a NaN-containing list, are float (NaN is a float value)
    assert volas.DataFrame({"a": [None, None]})["a"].dtype == "float64"
    assert volas.DataFrame({"a": [1, np.nan, 3]})["a"].dtype == "float64"
    # volas.NA itself in a list (round-tripping to_list output) is recognised too
    s2 = volas.DataFrame({"a": [1, volas.NA, 3]})["a"]
    assert s2.dtype == "int64" and s2.to_list()[1] is volas.NA
    rt = volas.DataFrame({"a": [1, None, 3]})["a"].to_list()  # contains volas.NA
    assert volas.DataFrame({"a": rt})["a"].to_list()[1] is volas.NA


def test_na_to_pandas_dtype_backend():
    df = volas.DataFrame({"i": [1, None, 3], "b": [True, None, False], "f": [1.5, None, 2.5]})
    # default 'numpy': numpy can't hold NA, so int/bool become float64 + NaN
    p = df.to_pandas()
    assert str(p["i"].dtype) == "float64" and str(p["f"].dtype) == "float64"
    # 'numpy_nullable': faithful masked Int64 / boolean (float stays numpy)
    pn = df.to_pandas(dtype_backend="numpy_nullable")
    assert str(pn["i"].dtype) == "Int64"
    assert str(pn["b"].dtype) == "boolean"
    assert str(pn["f"].dtype) == "float64"
    assert pn["i"].isna().tolist() == [False, True, False]
    # an int32 column -> nullable Int32
    fi32 = volas.DataFrame({"a": np.array([1, 2, 3], dtype=np.int32)}).shift()
    assert str(fi32.to_pandas(dtype_backend="numpy_nullable")["a"].dtype) == "Int32"
    with pytest.raises(ValueError):
        df.to_pandas(dtype_backend="arrow")


def test_na_from_pandas_nullable_round_trip():
    import pandas as pd

    pdf = pd.DataFrame(
        {
            "i": pd.array([1, None, 3], dtype="Int64"),
            "b": pd.array([True, None, False], dtype="boolean"),
        }
    )
    v = volas.from_pandas(pdf)
    assert v["i"].dtype == "int64" and v["i"][1] is volas.NA
    assert v["b"].dtype == "bool" and v["b"][1] is volas.NA
    # lossless round-trip: volas int+NA -> pandas nullable -> volas int+NA
    orig = volas.DataFrame({"i": [1, None, 3]})
    rt = volas.from_pandas(orig.to_pandas(dtype_backend="numpy_nullable"))
    assert rt["i"].dtype == "int64"
    lst = rt["i"].to_list()
    assert lst[0] == 1 and lst[1] is volas.NA and lst[2] == 3


def test_na_astype_carries_validity():
    NA = volas.NA
    s = volas.DataFrame({"a": [1, None, 3]})["a"]  # int64 + NA
    # int -> int32 / bool keeps the missing cell (was: error / silently False)
    assert s.astype("int32").dtype == "int32"
    assert s.astype("int32").to_list() == [1, NA, 3]
    assert s.astype("bool").to_list() == [True, NA, True]
    # int -> float keeps NA as NaN
    assert np.isnan(s.astype("float64").to_numpy()[1])
    # a present out-of-range value still errors
    big = volas.DataFrame({"a": np.array([3_000_000_000], dtype=np.int64)})["a"]
    with pytest.raises(Exception):
        big.astype("int32")


def test_na_bool_logic_kleene():
    NA = volas.NA
    b = volas.DataFrame({"a": [True, None, False]})["a"]  # T, NA, F
    # ~ propagates NA (was silently !placeholder -> True)
    assert (~b).to_list() == [False, NA, True]
    # & / | use Kleene three-valued logic
    assert (b & True).to_list() == [True, NA, False]  # NA & True  = NA
    assert (b & False).to_list() == [False, False, False]  # NA & False = False
    assert (b | True).to_list() == [True, True, True]  # NA | True  = True
    assert (b | False).to_list() == [True, NA, False]  # NA | False = NA
    assert (b ^ True).to_list() == [False, NA, True]  # XOR is NA if either is NA
    # a dense mask is unchanged
    import numpy as np

    c = volas.DataFrame({"a": np.arange(5.0)})["a"]
    assert ((c > 1) & (c < 3)).to_list() == [False, False, True, False, False]


def test_na_str_column():
    NA = volas.NA
    s = volas.DataFrame({"a": ["x", None, "z"]})["a"]
    assert s.dtype == "str"
    assert s.to_list() == ["x", NA, "z"]
    assert s[1] is NA
    assert s.isna().to_numpy().tolist() == [False, True, False]
    assert s.dropna().to_list() == ["x", "z"]
    assert s.shift(1).to_list() == [NA, "x", NA]
    arr = s.to_numpy()  # object array with None at the missing cell
    assert arr[0] == "x" and arr[1] is None
    assert "<NA>" in repr(s)
    # round-trip through a pandas nullable string column
    import pandas as pd

    pdf = pd.DataFrame({"a": pd.array(["x", None, "z"], dtype="string")})
    v = volas.from_pandas(pdf)
    assert v["a"].dtype == "str" and v["a"][1] is NA


def test_na_str_directional_fill():
    # Regression: a str column with a hole used to panic in Column::fill_dir
    # (`unreachable!("str has no missing value")`). str now fills like every dtype.
    NA = volas.NA

    def col(vals):
        return volas.DataFrame({"a": vals})["a"]

    # ffill carries the last valid value forward; a leading gap stays NA
    assert col(["x", None, None, "z"]).ffill().to_list() == ["x", "x", "x", "z"]
    assert col([None, "a", None]).ffill().to_list() == [NA, "a", "a"]
    # bfill carries the next valid value back; a trailing gap stays NA
    assert col(["x", None, None, "z"]).bfill().to_list() == ["x", "z", "z", "z"]
    assert col([None, "a", None]).bfill().to_list() == ["a", "a", NA]
    # a fully-missing str column fills to itself; a dense column is unchanged
    all_na = volas.DataFrame({"a": ["x", "y"]})["a"].shift(2)  # [NA, NA], still str
    assert all_na.dtype == "str" and all_na.ffill().to_list() == [NA, NA]
    assert col(["a", "b"]).bfill().to_list() == ["a", "b"]
    # dtype is preserved throughout
    assert col(["x", None, "z"]).ffill().dtype == "str"


def test_na_display_symbol():
    # Every dtype prints a missing cell as the single <NA> symbol — a float NaN,
    # an int/bool/str NA, and a datetime NaT all render identically (storage and
    # element access stay dtype-specific; only the console display is unified).
    assert "<NA>" in repr(volas.DataFrame({"a": [1, None, 3]})["a"])
    assert "<NA>" in repr(volas.DataFrame({"a": [True, None, False]})["a"])
    f = repr(volas.DataFrame({"a": [1.5, None, 3.0]})["a"])
    assert "<NA>" in f and "NaN" not in f
    sd = repr(
        volas.DataFrame(
            {"a": np.array(["2021-01-01", "NaT", "2021-01-03"], dtype="datetime64[ns]")}
        )["a"]
    )
    assert "<NA>" in sd and "NaT" not in sd  # NaT no longer renders as a garbage date
    # a user-supplied na_rep overrides the default for every dtype, uniformly
    assert "NULL" in volas.DataFrame({"a": [1.5, None, 3.0]}).to_string(na_rep="NULL")
