"""Systematic audit — the SIGNATURE MANIFEST machine layer (SPEC §6.8②).

For every method volas shares with pandas, the parameter sets are diffed by
reflection; EVERY pandas parameter volas lacks must be dispositioned here —
a global category waiver, a per-method waiver, or an `align` entry (strict
backlog). A newly implemented method, or a new pandas parameter, lands in the
differential automatically and FAILS this test until dispositioned: this is
the mechanism that ends the recurring kwargs debt (F44, then R-1..R-6).

Disposition vocabulary:
  GLOBAL_OUT     parameter names that are out-of-scope across the whole API by
                 volas design (each with its reason below).
  PER_METHOD     extra waived params for one method (naming divergence, fixed
                 policy, or a feature volas deliberately lacks).
  ALIGN          (Class.method, param) pairs volas SHOULD gain — strict-xfail.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

import volas

_S = volas.DataFrame({"x": [1.0]})["x"]
_DF = volas.DataFrame({"x": [1.0]})
_PS = pd.Series(dtype="float64")
_PDF = pd.DataFrame()

# --- global category waivers (volas-wide design decisions) -------------------
GLOBAL_OUT = {
    "axis",            # volas is column-wise only; no axis switching
    "inplace",         # immutable-by-default: every op returns a new object
    "level", "sort_remaining", "col_level", "col_fill",   # no MultiIndex
    "copy", "deep",    # Arc copy-on-write makes the copy knob meaningless
    "skipna",          # NA-skip is the single reduction semantic (C2)
    "numeric_only", "bool_only",   # typed columns; non-numeric are skipped/serror by design
    "ignore_index",    # the index always rides along; reset explicitly
    "kind",            # sort algorithm choice — implementation detail
    "key",             # arbitrary python sort keys break the typed kernel (like apply)
    "errors",          # errors='ignore' silences failures — anti-C4 (drop's own
                       # errors= is volas-implemented and not in the gap set)
    "regex",           # string-pattern ops are out-of-scope (.str family)
    "limit_area", "limit_direction",   # fill micro-policies, out-of-scope v1
    "freq", "suffix",  # period/frequency shifting via the index — use cumulate
    "fill_value",      # shift fill: vacated cells are NA (C2); fill after, explicitly
    "min_count",       # sum/prod NA quorum — volas reduces what is present
    "interpolation",   # quantile interpolation fixed to linear (pandas default)
    "ddof",            # ddof fixed to 1 (sample statistics, pandas default)
    "min_periods",     # (corr/ewm) fixed policy; rolling's own min_periods IS implemented
    "method",          # corr=pearson only / interpolate=linear only / window engine knob
    "dropna",          # mode/nunique always drop NA; value_counts(dropna=False) errors loudly
    "into",            # to_dict container class — plain dict only
    "na_value",        # to_numpy NA representation is fixed by the NA model (F17)
    "ambiguous", "nonexistent",   # DST gap/fold are REJECTED, never resolved (F33)
    "times", "adjust", "ignore_na", "com", "halflife", "alpha",  # ewm: span-only (R-6)
    "center", "closed", "win_type", "on", "step",                # rolling micro-knobs
    "keep",            # nlargest/nsmallest tie policy: first-k only (drop_duplicates'
                       # own keep= IS implemented and not in the gap set)
    "allow_duplicates",  # unique-column contract is unconditional
    "bins",            # value_counts binning -> use cut/round first
    "percentiles", "include", "exclude",   # describe: the fixed 8-stat summary
    "thresh", "how", "subset",   # dropna/duplicated row policies: whole-row semantics v1
    "verify_integrity",  # set_index ALWAYS verifies (unique-label guard) — not optional
}

# --- per-method waivers (naming divergence or a fixed local policy) ----------
PER_METHOD = {
    "Series.shift": {"periods"},      # volas names it n (positional parity)
    "Series.diff": {"periods"},       # volas names it n
    "DataFrame.shift": {"periods"},
    "DataFrame.diff": {"periods"},
    "DataFrame.astype": {"dtype"},    # volas: astype(mapping) per-column form
    "DataFrame.drop": {"columns", "index"},  # volas: labels + axis form only
    "DataFrame.nlargest": {"columns"},        # volas: column (one ranking column)
    "DataFrame.nsmallest": {"columns"},
    "Series.rename": {"index"},       # scalar-name rename only
    "DataFrame.rename": {"mapper", "index"},  # columns-dict rename only
    "Series.reset_index": {"name"},
    "DataFrame.reset_index": {"names"},
    "DataFrame.set_index": {"drop", "append"},  # always-consume, single-key form
    "Series.to_string": {"buf", "dtype", "header", "index", "length", "min_rows", "name"},
    "DataFrame.to_string": {"buf", "col_space", "decimal", "encoding", "formatters",
                            "index_names", "justify", "line_width", "max_cols",
                            "max_colwidth", "sparsify"},
    "DataFrame.to_csv": {"chunksize", "compression", "date_format", "decimal",
                         "doublequote", "encoding", "escapechar", "index_label",
                         "lineterminator", "mode", "path_or_buf", "quotechar",
                         "quoting", "storage_options"},
    "DataFrame.value_counts": {"subset", "normalize", "sort", "ascending"},
    # ^ frame-level value_counts delegates to the single column; kwargs live there
    "Series.ffill": {"limit"}, "Series.bfill": {"limit"},
    "DataFrame.ffill": {"limit"}, "DataFrame.bfill": {"limit"},
    "Series.interpolate": {"limit"}, "DataFrame.interpolate": {"limit"},
    # ^ run-length caps on directional fills: micro-policy, out-of-scope v1
    #   (fillna's own limit= IS implemented — kept out of GLOBAL_OUT for that)
    "Series.sort_index": {"na_position"},     # index labels are NA-free by guard
    "DataFrame.sort_index": {"na_position"},  # (NaT-bearing datetime sorts last, fixed)
}

# --- align backlog: params volas SHOULD gain (strict; currently none) --------
ALIGN: set[tuple[str, str]] = set()


def _params(obj, m):
    try:
        return {p for p in inspect.signature(getattr(obj, m)).parameters
                if p not in ("self", "args", "kwargs")}
    except (ValueError, TypeError):
        return None


def _gaps():
    out = {}
    for label, vo, po in (("Series", _S, _PS), ("DataFrame", _DF, _PDF)):
        common = ({m for m in dir(vo) if not m.startswith("_")}
                  & {m for m in dir(po) if not m.startswith("_")})
        for m in sorted(common):
            vp, pp = _params(vo, m), _params(po, m)
            if vp is None or pp is None:
                continue
            missing = pp - vp
            if missing:
                out[f"{label}.{m}"] = missing
    return out


def test_every_parameter_gap_is_dispositioned():
    """The machine layer: no pandas parameter volas lacks may be undispositioned."""
    undisposed = {}
    for key, missing in _gaps().items():
        rest = missing - GLOBAL_OUT - PER_METHOD.get(key, set())
        rest -= {p for (k, p) in ALIGN if k == key}
        if rest:
            undisposed[key] = sorted(rest)
    assert not undisposed, (
        "undispositioned parameter gaps (add to GLOBAL_OUT / PER_METHOD as a "
        f"reasoned waiver, or to ALIGN as backlog): {undisposed}"
    )


def test_per_method_waivers_are_not_stale():
    """A waived param that volas later implements must leave the waiver table."""
    gaps = _gaps()
    stale = {}
    for key, waived in PER_METHOD.items():
        actual = gaps.get(key, set())
        gone = waived - actual - GLOBAL_OUT
        if gone:
            stale[key] = sorted(gone)
    assert not stale, f"stale per-method waivers (param now implemented?): {stale}"


@pytest.mark.parametrize("key,param", sorted(ALIGN))
@pytest.mark.xfail(reason="signature align-backlog: parameter not yet implemented", strict=True)
def test_signature_align_backlog(key, param):
    label, m = key.split(".")
    obj = _S if label == "Series" else _DF
    assert param in _params(obj, m)
