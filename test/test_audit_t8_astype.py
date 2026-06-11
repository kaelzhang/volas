"""Systematic audit — T8 (boundary): Series.astype, the dtype-conversion grid.

Matrix: source dtype (D=7) × target dtype (7) × NA-state (N=5) = 245 cells.

Oracle policy (SPEC §5 provenance, §10 triage):
  * numeric×numeric — the N-basis values are integer-valued (1/2/3), so every
    numeric cast is *exact*; outcome is ok, dtype is the target, and NA is
    preserved (`# C2` native-NA, value `# pandas`-trivial).
  * str(non-numeric "a"/"b"/"c") -> numeric/datetime — pandas raises on the
    unparseable token (`# pandas`); volas must too (a guard, not a crash → P7).
  * <anything> -> str — stringify is defined in outcome (ok, dtype=str) but the
    exact string *format* is a volas design choice; value deferred.
  * everything else (num↔datetime epoch convention, num→bool truthiness,
    str→bool) — the contract is silent → `rejected_no_oracle`, surfaced by the
    backlog meta-test, NOT hand-guessed.

Cell IDs:  ASTYPE:<src>-><dst>/N=<n>
"""

from __future__ import annotations

import pytest

from . import audit_dims as A

_NUMERIC = set(A.NUMERIC)  # f64 f32 i64 i32 bool


def _astype_oracle(src: str, dst: str):
    """Intended outcome of `series(src, n).astype(dst)` — independent of N.

    Returns one of:
      ("ok", dst)        numeric cast, value-preserving, NA-preserving
      ("identity", dst)  src==dst non-op (exact, incl. str/datetime)
      ("stringify",)     ok + dtype=str, value format deferred
      ("raise", Exc)     a guard rejects the conversion
      ("defer", reason)  rejected_no_oracle — owner decision pending
    """
    if src == dst:
        return ("identity", dst)                         # trivial no-op
    if src in _NUMERIC and dst in _NUMERIC:
        if dst == "bool":
            return ("defer", "numeric->bool truthiness undecided")  # 2,3 lose info
        return ("ok", dst)                               # # pandas exact + # C2
    if src == "str" and dst in ("f64", "f32", "i64", "i32"):
        return ("raise", ValueError)                     # "a" unparseable  # pandas
    if src == "str" and dst == "datetime":
        return ("raise", ValueError)                     # "a" not a date   # pandas
    if src == "str" and dst == "bool":
        return ("defer", "str->bool parse policy undecided")
    if dst == "str":                                     # numeric/bool/datetime -> str
        return ("stringify",)                            # ok+str; format deferred
    if {src, dst} & {"datetime"}:
        return ("defer", f"{src}->{dst} epoch convention undecided")
    return ("defer", f"{src}->{dst} cross-family undecided")  # pragma: no cover


# The frozen rejected_no_oracle backlog: cells whose intended behaviour the
# contract has not yet ruled (SPEC §5). A new deferral can't sneak in silently,
# and resolving one *forces* removing it here. Owner decision pending.
_PENDING = frozenset(
    (s, d)
    for s in A.DTYPES
    for d in A.DTYPES
    if _astype_oracle(s, d)[0] == "defer"
)


# F4 FIXED: stringify carries EVERY source's missing cells into the str target's
# validity (incl the in-band float-NaN / datetime-NaT sentinels) — no more
# literal 'NaN'/'NaT' placeholder collapse. (float->INT with NA correctly raises
# ValueError, contract line 40 — handled below.)
_FLOAT = ("f64", "f32")
_INT = ("i64", "i32")


def _ids():
    for s in A.DTYPES:
        for d in A.DTYPES:
            for n in A.NA_STATES:
                yield pytest.param(s, d, n, id=f"{s}->{d}/N={n}")


@pytest.mark.parametrize("src,dst,n", list(_ids()))
def test_astype_matrix(src, dst, n):
    kind = _astype_oracle(src, dst)
    if kind[0] == "defer":
        pytest.skip(f"rejected_no_oracle: {kind[1]} (tracked by backlog meta-test)")

    s = A.series(src, n)

    # float NaN cannot be represented as an integer: NA-bearing float->int must
    # raise (# C4 / contract line 40), while a dense float->int converts exactly.
    if src in _FLOAT and dst in _INT and n in ("N1", "N2"):
        with pytest.raises(ValueError):
            s.astype(A._DTYPE_STR[dst])
        return

    if kind[0] == "raise":
        # an all-NA / empty column has nothing to parse, so the guard may not
        # fire — only assert the guard on columns that actually carry values.
        if n in ("N2", "N3"):
            pytest.skip("no present value to trigger the parse guard")
        with pytest.raises(kind[1]):
            s.astype(A._DTYPE_STR[dst])
        return

    out = s.astype(A._DTYPE_STR[dst])
    assert len(out) == len(s)
    assert out.isna().to_list() == s.isna().to_list(), "NA must be preserved (# C2)"

    if kind[0] == "stringify":
        assert out.dtype == "str"
        return

    # ok / identity: dtype is the target, values are preserved.
    assert out.dtype == A._DTYPE_STR[dst]
    if kind[0] == "identity" and src in ("str", "datetime"):
        assert out.to_list() == s.to_list()             # exact, no numeric coercion
        return
    src_vals, out_vals = s.to_list(), out.to_list()
    mask = out.isna().to_list()
    for sv, ov, m in zip(src_vals, out_vals, mask):
        if not m:
            assert float(ov) == float(sv), f"value drift {sv!r} -> {ov!r}"


def test_astype_oracle_backlog():
    """The rejected_no_oracle set is exactly the known-pending set (SPEC §5).

    This is the loud surface for un-ruled astype conversions — see
    `tasks/04/audits/findings-ledger.md`. Resolving any cell means giving it an
    oracle above AND deleting it from this frozen set.
    """
    expected = {
        # numeric -> bool: truthiness (2,3 -> True loses information)
        ("f64", "bool"), ("f32", "bool"), ("i64", "bool"), ("i32", "bool"),
        ("str", "bool"),
        # datetime <-> numeric: epoch unit/sign convention unruled
        ("f64", "datetime"), ("f32", "datetime"), ("i64", "datetime"),
        ("i32", "datetime"), ("bool", "datetime"),
        ("datetime", "f64"), ("datetime", "f32"), ("datetime", "i64"),
        ("datetime", "i32"), ("datetime", "bool"),
    }
    assert _PENDING == expected, (
        f"astype backlog drift — new: {_PENDING - expected}, "
        f"resolved (delete from set): {expected - _PENDING}"
    )
