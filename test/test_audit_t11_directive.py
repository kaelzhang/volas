"""Systematic audit — T11 (directive): parser / schema / guard robustness.

Per SPEC §8 the directive subject is *dimension-reduced*: indicator value
correctness is owned by the talib / stock-pandas parity suites (not duplicated
here), and the three-entry value parity + warm-up lookback is the E2 law
(test_audit_state.py). What remains for the matrix is the guard surface —
malformed directives must raise a typed DirectiveError, never panic (P7).

Cell IDs:  T11.guard/<directive> · T11.exc-hierarchy
"""

from __future__ import annotations

import pytest

import volas

_OHLC = ["open", "high", "low", "close", "volume"]


def _frame():
    return volas.DataFrame({c: [float(i + 1) for i in range(8)] for c in _OHLC})


@pytest.mark.parametrize("directive,exc,fragment", [
    ("", volas.DirectiveSyntaxError, "empty"),
    ("a >", volas.DirectiveSyntaxError, "unexpected end"),
    ("ma:0", volas.DirectiveValueError, ">= 1"),
    ("ma:-1", volas.DirectiveValueError, ">= 1"),
    ("notacommand", volas.DirectiveValueError, "unknown command"),
])
def test_malformed_directive_raises_typed(directive, exc, fragment):
    with pytest.raises(exc) as ei:
        _frame().exec(directive)
    assert fragment in str(ei.value), f"T11.guard/{directive!r}: message lost its diagnostic"


def test_exception_hierarchy():
    # both concrete errors descend from DirectiveError, which is a ValueError —
    # so `except ValueError` (the pandas-shaped catch) still works.
    assert issubclass(volas.DirectiveSyntaxError, volas.DirectiveError)
    assert issubclass(volas.DirectiveValueError, volas.DirectiveError)
    assert issubclass(volas.DirectiveError, ValueError)


def test_no_panic_on_adversarial_directives():
    """A spread of hostile inputs must surface as DirectiveError, never a pyo3
    PanicException (P7) and never a silent wrong answer."""
    for d in ("ma:", ":5", "ma:1.5", "ma:99999999999999999999", "()", "@@@"):
        with pytest.raises(volas.DirectiveError):
            _frame().exec(d)


def test_trailing_comma_is_lenient_not_a_crash():
    """`ma:2,` is parsed leniently as `ma:2` (a trailing empty arg is tolerated)
    — documented as intended leniency: a valid value, never a panic (P7). Not a
    finding; pinned so a future strictness change is a conscious diff."""
    out = _frame().exec("ma:2,")
    assert list(out.tolist())[1:4] == [1.5, 2.5, 3.5]
