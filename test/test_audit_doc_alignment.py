"""Systematic audit — §6.7 doc alignment: the docs are an oracle, machine-checked.

The README / PANDAS-DIFFERENCES / INDICATORS docs promise behaviour, so they are
themselves an oracle and must agree with the runtime (the `unify` README
mis-description was exactly this kind of drift, missed by eyeballing). Checks:

  1. coverage   — every public class + key method is documented somewhere.
  2. promises   — the docs' HEADLINE behavioural claims are pinned directly to
                  the runtime (the NA model, guard claims). NOTE the scope
                  honestly: `>>>` doctest transcripts are NOT golden-executed
                  (they are human-authored narratives with shared state and
                  numpy-2 reprs); full doctest-golden remains explicitly out of
                  scope of this module — SPEC §6.7.2 is implemented as
                  headline-promise pins + runnable fenced blocks.
  3. runnable   — every *self-sufficient* plain ```py example still executes
                  (catches API drift like a removed/renamed method); excluded
                  blocks are counted and reported so coverage can't silently shrink.
  4. remove     — the §6.6 removals must leave no dangling doc references.

This module is self-justifying: reading the contract+README through this lens is
what corrected the F5/F6/F11 NA-surface oracle (see findings-ledger).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import volas  # noqa: F401 - in scope for exec'd doc examples

_REPO = Path(__file__).resolve().parent.parent
_DOCS = ["README.md", "PANDAS-DIFFERENCES.md", "INDICATORS.md", "DEVELOPMENT.md"]
_BLOCK = re.compile(r"```py\n(.*?)```", re.S)

# a plain block is runnable on its own only if it imports volas, builds its own
# data, and pulls in nothing external (a file, a pandas frame, a prior block's
# variables, or a `>>>` REPL transcript).
_EXTERNAL = ("read_csv(", ".csv", ".tsv", "pandas_df", "from_pandas(pandas")


def _blocks():
    for d in _DOCS:
        p = _REPO / d
        if p.exists():
            for i, blk in enumerate(_BLOCK.findall(p.read_text())):
                yield f"{d}#{i}", blk




def _runnable_blocks():
    out = []
    for lbl, b in _blocks():
        if ">>>" in b or ("import volas" not in b and "from volas" not in b):
            continue
        if "DataFrame({" not in b and "Timestamp(" not in b:
            continue                                  # builds no own data
        if any(tok in b for tok in _EXTERNAL):
            continue
        out.append((lbl, b))
    return out


def _doc_text():
    return "\n".join((_REPO / d).read_text() for d in _DOCS if (_REPO / d).exists())


# --- 1. coverage: doc ⊇ public API -----------------------------------------
def test_runnable_selection_is_reported():
    """The runnable subset is a SELECTION: pin how many fenced blocks exist vs
    run, so a doc rewrite can't silently shrink executable coverage to zero."""
    total = len(list(_blocks()))
    runnable = len(_runnable_blocks())
    assert total >= 90 and runnable >= 3, f"doc blocks drifted: {runnable}/{total} runnable"


def test_core_classes_documented():
    text = _doc_text()
    for cls in ("DataFrame", "Series", "Timestamp", "TimeFrame", "volas.NA"):
        assert cls in text, f"public surface {cls} is undocumented (doc ⊇ API)"


def test_key_methods_documented():
    text = _doc_text()
    for m in ("fillna", "astype", "to_numpy", "to_pandas", "cumulate", "isna", "exec"):
        assert m in text, f"method {m} is undocumented"


# --- 2. description-O consistency: the docs' headline behavioural promises ---
# The PANDAS-DIFFERENCES / README narratives are human-authored (numpy-2.x repr,
# error-demos as prose, shared REPL state), so they don't run cleanly under
# doctest; instead the *promises they make* are pinned directly to the runtime.
# These are exactly the claims that corrected the F5/F6/F11 NA-surface oracle.
def test_documented_na_model_claims():
    # "int stays int; the hole is volas.NA" (PANDAS-DIFFERENCES).
    s = volas.DataFrame({"qty": [1, None, 3]})["qty"]
    assert s.dtype == "int64"
    assert s.to_list() == [1, volas.NA, 3]
    # "a float hole stays np.nan" (README) — the dtype-specific NA surface.
    import math
    f = volas.DataFrame({"x": [1.0, None, 3.0]})["x"]
    assert math.isnan(f.to_list()[1])
    # "the +1 survives" where pandas (2**53+1 -> float) loses it.
    big = volas.DataFrame({"x": [2 ** 53 + 1, None]})["x"]
    assert big.to_list()[0] == 2 ** 53 + 1


def test_documented_guard_claims():
    # "an extra column is rejected so data is never silently dropped".
    with pytest.raises(ValueError):
        volas.DataFrame({"x": [1.0]}).append(volas.DataFrame({"x": [2.0], "z": [9.0]}))
    # reductions skip NA: documented `sum() -> 4` for [1, NA, 3].
    assert volas.DataFrame({"x": [1, None, 3]})["x"].sum() == 4


# --- 3. runnable plain examples (catch API drift) --------------------------
@pytest.mark.parametrize("label,block", _runnable_blocks(),
                         ids=[lbl for lbl, _ in _runnable_blocks()])
def test_doc_example_runs(label, block):
    demonstrates_error = bool(re.search(r"#.*\b(raises?|TypeError|ValueError|Error)\b", block))
    try:
        exec(compile(block, label, "exec"), {})
    except Exception as e:
        if demonstrates_error:
            return
        pytest.fail(f"doc example {label} no longer runs: {type(e).__name__}: {e}")


# --- 4. remove-candidates: doc must not outlive the API --------------------
def test_remove_candidates_doc_state():
    """§6.6 removals LANDED (unify / tolist deleted): the docs must not retain
    dangling references to the removed APIs."""
    text = _doc_text()
    assert "tf.unify" not in text and ".unify(" not in text, "dangling unify doc reference"
    assert ".tolist(" not in text, "dangling tolist doc reference"


def test_talib_parity_table_rows_have_a_talib_original():
    """The 'TA-Lib-compatible directives' table is strictly upstream
    correspondences: every row's `TA-Lib original` column must name a real
    TA-Lib function — a `—` placeholder means a volas-native command was
    appended to the wrong section (it belongs under 'Built-in Commands for
    Statistics'; caught by owner review 2026-06-12)."""
    import pathlib
    text = pathlib.Path("INDICATORS.md").read_text()
    table = text[text.index("## TA-Lib-compatible directives"):]
    offenders = [
        line.split("|")[1].strip()
        for line in table.splitlines()
        if line.startswith("| `") and line.split("|")[2].strip() in ("—", "-", "")
    ]
    assert not offenders, f"native commands in the TA-Lib parity table: {offenders}"
