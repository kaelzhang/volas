"""Systematic audit — P8 layer-1 signature sublayer (review NF-C).

The method-name surface differential is blind to *parameter*-level gaps: a method
volas has, but missing a pandas kwarg. volas exposes reflectable signatures
(pyo3 text sigs), so the gap is machine-detectable. The verified-important
missing kwargs are align-backlog; the bulk of pandas's kwargs are out-of-scope.

Cell IDs:  T-sig.<receiver>.<method>/kw=<param>
"""

from __future__ import annotations

import inspect

import pytest

import volas

_S = volas.DataFrame({"x": [3.0, 1.0, 2.0]})["x"]
_DF = volas.DataFrame({"a": [3.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})


def _params(fn):
    try:
        return set(inspect.signature(fn).parameters)
    except (ValueError, TypeError):  # pragma: no cover
        return None


def test_signature_reflection_is_feasible():
    """The sublayer is possible: volas methods expose reflectable signatures,
    so a pandas-vs-volas parameter differential is machine-generable (P8)."""
    assert _params(_S.sort_values) == {"ascending"}
    assert "na_position" not in _params(_S.sort_values)   # the gap, detected


# F44 (align-backlog): pandas kwargs volas lacks, verified-important.
_SIG_ALIGN = [
    ("Series.sort_values/na_position", lambda: _S.sort_values(na_position="first")),
    ("Series.rank/na_option", lambda: _S.rank(na_option="top")),
    ("Series.fillna/limit", lambda: _S.fillna(0, limit=1)),
    ("DataFrame.drop/errors", lambda: _DF.drop(["a"], axis=1, errors="raise")),
]


@pytest.mark.parametrize("label,call", _SIG_ALIGN, ids=[l for l, _ in _SIG_ALIGN])
@pytest.mark.xfail(reason="F44: pandas kwarg not accepted by volas (signature-level gap)", strict=True)
def test_signature_align_backlog_kwarg(label, call):
    call()                                                # currently TypeError: unexpected keyword
