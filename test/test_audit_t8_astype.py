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
  * numeric -> bool (owner ruling 2026-06-12) — pandas-nullable truthiness:
    nonzero -> True (incl ±inf), zero -> False, NA stays NA (a float NaN IS NA);
    never numpy's NaN-is-truthy footgun.
  * str -> bool (owner ruling) — no parse policy: raises.
  * datetime <-> numeric (owner-approved disposition) — the exact-ns channels
    are kept (i64 -> datetime epoch-ns; f64/i64 with an explicit unit suffix is
    the documented ingestion divergence; datetime -> i64 exact ns, NaT -> NA per
    C2 where pandas raises); every lossy/ambiguous pair raises (i32/f32/bool <->
    datetime, datetime -> f64/f32/i32/bool).

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
      ("boolify",)       numeric -> bool truthiness, NA-preserving
      ("epoch", dir)     i64/f64 <-> datetime via exact epoch nanoseconds
      ("raise", Exc)     a guard rejects the conversion
    """
    if src == dst:
        return ("identity", dst)                         # trivial no-op
    if src in _NUMERIC and dst in _NUMERIC:
        if dst == "bool":
            return ("boolify",)                          # nonzero -> True, NA -> NA
        return ("ok", dst)                               # # pandas exact + # C2
    if src == "str" and dst in ("f64", "f32", "i64", "i32"):
        return ("raise", ValueError)                     # "a" unparseable  # pandas
    if src == "str" and dst == "datetime":
        return ("raise", ValueError)                     # "a" not a date   # pandas
    if src == "str" and dst == "bool":
        return ("raise", TypeError)                      # no parse policy (owner)
    if dst == "str":                                     # numeric/bool/datetime -> str
        return ("stringify",)                            # ok+str; format deferred
    if dst == "datetime":
        # i64 -> datetime is the exact epoch-ns read (# pandas); f64 keeps the
        # documented float-epoch ingestion divergence (truncated to ns). The
        # narrow/unordered sources (i32 / f32 / bool) raise.
        return ("epoch", "to") if src in ("i64", "f64") else ("raise", TypeError)
    if src == "datetime":
        # datetime -> i64 is the exact ns; NaT -> NA per C2 (pandas raises).
        # Every lossy target (f64/f32 quantize past 2^53, i32 overflows, bool
        # is meaningless) raises.
        return ("epoch", "from") if dst == "i64" else ("raise", TypeError)
    return ("raise", TypeError)  # pragma: no cover — no remaining pair


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
    s = A.series(src, n)

    # float NaN cannot be represented as an integer: NA-bearing float->int must
    # raise (# C4 / contract line 40), while a dense float->int converts exactly.
    if src in _FLOAT and dst in _INT and n in ("N1", "N2"):
        with pytest.raises(ValueError):
            s.astype(A._DTYPE_STR[dst])
        return

    if kind[0] == "raise":
        # str -> numeric/datetime is a PER-VALUE parse guard: an all-NA / empty
        # column has nothing to parse and converts to all-NA instead of raising
        # (documented: blank cells are NA). Every other guard is dtype-level
        # and fires even on an empty column.
        if src == "str" and dst != "bool" and n in ("N2", "N3"):
            pytest.skip("per-value parse guard: nothing to parse in N2/N3")
        with pytest.raises(kind[1]):
            s.astype(A._DTYPE_STR[dst])
        return

    out = s.astype(A._DTYPE_STR[dst])
    assert len(out) == len(s)
    assert out.isna().to_list() == s.isna().to_list(), "NA must be preserved (# C2)"

    if kind[0] == "stringify":
        assert out.dtype == "str"
        return

    if kind[0] == "boolify":
        # pandas-nullable truthiness (# pandas-nullable / owner ruling): the
        # basis values 1/2/3 (or True/False/True) -> nonzero, NA stays NA.
        assert out.dtype == "bool"
        for sv, ov, m in zip(s.to_list(), out.to_list(), out.isna().to_list()):
            if not m:
                assert ov is (float(sv) != 0.0)
        return

    if kind[0] == "epoch":
        # the exact epoch-ns channel: int64 ns <-> datetime round-trips losslessly.
        if kind[1] == "to":
            assert out.dtype == "datetime64[ns]"
            back = out.astype("int64")
            for sv, bv, m in zip(s.to_list(), back.to_list(), back.isna().to_list()):
                if not m:
                    assert int(bv) == int(float(sv))     # f64 basis is integer-valued
        else:
            assert out.dtype == "int64"
            assert [str(x) for x in out.astype("datetime64[ns]").to_list()] == \
                   [str(x) for x in s.to_list()]         # exact ns round-trip
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


def test_astype_oracle_resolved():
    """Every astype cell now has an oracle (owner rulings 2026-06-12) — the
    rejected_no_oracle backlog is EMPTY. A new deferral cannot reappear without
    failing here."""
    kinds = {_astype_oracle(a, b)[0] for a in A.DTYPES for b in A.DTYPES}
    assert kinds == {"identity", "ok", "boolify", "epoch", "raise", "stringify"}
