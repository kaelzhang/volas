"""Negative static type checks: every line below MUST be flagged by the type checker.

Run under ``mypy --strict`` (which enables ``warn_unused_ignores``): if a wrong usage is
NOT caught, its ``# type: ignore`` becomes unused and mypy fails the build — proving the
types actually catch misuse rather than being permissive ``Any``. Intentionally invalid,
so it is never executed (no ``test_`` name; pytest skips it).
"""

from __future__ import annotations

import numpy as np

from volas import DataFrame
from volas_rs import _directive_facts

df = DataFrame({"close": np.arange(5.0)})

df.nonexistent_method()  # type: ignore[attr-defined]
df["close"].nonexistent_attr  # type: ignore[attr-defined]
df.head("not-an-int")  # type: ignore[arg-type]
DataFrame(123)  # type: ignore[arg-type]
_wrong: int = df["close"]  # type: ignore[assignment]

_directive_facts(123)  # type: ignore[call-overload]
_directive_facts("ema:2", columns=[123])  # type: ignore[list-item]
_directive_facts("ema:2", columns=["close"], shadow=123)  # type: ignore[call-overload]
_directive_facts("ema:2", columns=iter(["close"]))  # type: ignore[call-overload]
_directive_facts("ema:2")["lookback"]  # type: ignore[typeddict-item]
_directive_facts("ema:2", ["close"])["resume"]  # type: ignore[typeddict-item]
_wrong_fact: str = _directive_facts("ema:2", ["close"])["shadows"]  # type: ignore[assignment]
