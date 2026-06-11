"""Systematic audit — T4 (fill / conditional select): Series.fillna.

The full inherent×parameter matrix: receiver dtype (D=7) × NA-state (N=5) ×
fill-value type (P=6) = 210 cells. The oracle is the *intended* behaviour
(contract C2/C4 + the typed-scalar-fill design), independent of what volas
currently does — a failing cell is a real deviation, not a moved goalpost.

Cell IDs (cited per parametrization id):  FILL:<dtype>/<N>/<param>
"""

from __future__ import annotations

import pytest

import volas
from . import audit_dims as A

# P — the fill-value parameter types (one representative value each). `ts` is an
# actual volas.Timestamp (a distinct datetime scalar type) so it is unambiguous —
# a datetime *string* would just be a valid str fill for a str column.
FILL_PARAMS = {
    "int": 0,                          # integral number
    "float": 2.5,                      # fractional number
    "bool": True,                      # bool
    "str": "z",                        # non-datetime string
    "ts": volas.Timestamp("2021-06-15"),  # the datetime scalar
    "NA": volas.NA,                    # the missing singleton
}

_F = {"f64": "float64", "f32": "float32", "i64": "int64", "i32": "int32"}


def _fill_oracle(d: str, n: str, p: str):
    """Intended outcome of `series(d, n).fillna(FILL_PARAMS[p])`.

    Returns ("ok", out_dtype) or ("raise", ExceptionClass).
    """
    want = A._DTYPE_STR[d]
    # No holes -> lazy no-op: the fill is never applied, so its type is irrelevant
    # and the dtype is unchanged (N0 dense, N3 empty, N4 single non-NA).
    if n in ("N0", "N3", "N4"):
        return ("ok", want)
    # N1 / N2 have holes the fill must fit.
    if p == "NA":
        return ("ok", want)  # filling a hole with NA is identity
    if d in ("f64", "f32"):
        # a float column absorbs any number/bool fill, keeping its float dtype.
        return ("ok", want) if p in ("int", "float", "bool") else ("raise", TypeError)
    if d in ("i64", "i32"):
        if p == "int":
            return ("ok", want)          # integral fill stays int
        if p == "float":
            return ("ok", "float64")     # fractional fill promotes to float64
        if p == "bool":
            return ("ok", want)          # 0/1 bool stays int
        return ("raise", TypeError)      # str / ts into a numeric column
    if d == "bool":
        if p == "bool" or p == "int":    # bool, or 0/1 (FILL_PARAMS int=0), keeps bool
            return ("ok", "bool")
        # F40 (decision 4): a non-0/1 number (float 2.5) into a bool column is an
        # error, not a silent promotion to float — C3/C4 dtype honesty.
        return ("raise", TypeError)      # float / str / ts into a bool column
    if d == "str":
        return ("ok", "str") if p == "str" else ("raise", TypeError)
    if d == "datetime":
        if p == "ts":
            return ("ok", "datetime64[ns]")
        # a string is the right *type* for a datetime fill but an invalid *value*
        # (ValueError); a number is the wrong type entirely (TypeError).
        return ("raise", ValueError) if p == "str" else ("raise", TypeError)
    raise AssertionError(d)  # pragma: no cover


@pytest.mark.parametrize("p", list(FILL_PARAMS))
@pytest.mark.parametrize("n", A.NA_STATES)
@pytest.mark.parametrize("d", A.DTYPES)
def test_fillna_matrix(d, n, p):
    s = A.series(d, n)
    value = FILL_PARAMS[p]
    outcome = _fill_oracle(d, n, p)

    if outcome[0] == "raise":
        with pytest.raises(outcome[1]):
            s.fillna(value)
        return

    out = s.fillna(value)
    assert out.dtype == outcome[1], f"FILL:{d}/{n}/{p} dtype {out.dtype} != {outcome[1]}"
    assert len(out) == len(s)
    # a no-hole receiver is returned unchanged; a holed one has no NA left after a
    # non-NA fill, and is unchanged after an NA fill.
    if n in ("N0", "N3", "N4") or p == "NA":
        assert out.isna().to_list() == s.isna().to_list()
    else:
        assert not any(out.isna().to_list())
