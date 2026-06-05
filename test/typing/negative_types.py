"""Negative static type checks: every line below MUST be flagged by the type checker.

Run under ``mypy --strict`` (which enables ``warn_unused_ignores``): if a wrong usage is
NOT caught, its ``# type: ignore`` becomes unused and mypy fails the build — proving the
types actually catch misuse rather than being permissive ``Any``. Intentionally invalid,
so it is never executed (no ``test_`` name; pytest skips it).
"""

from __future__ import annotations

import numpy as np

from volas import DataFrame

df = DataFrame({"close": np.arange(5.0)})

df.nonexistent_method()  # type: ignore[attr-defined]
df["close"].nonexistent_attr  # type: ignore[attr-defined]
df.head("not-an-int")  # type: ignore[arg-type]
DataFrame(123)  # type: ignore[arg-type]
_wrong: int = df["close"]  # type: ignore[assignment]
