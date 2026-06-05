"""Positive static type checks for volas — the type checker is the oracle.

``assert_type(expr, T)`` makes mypy/pyright fail unless ``expr`` is inferred as exactly
``T``. These pin the dynamic, @overload-driven surfaces (getitem / exec / indexers).
Run under ``mypy --strict`` and ``pyright``; ``assert_type`` is a runtime no-op, so the
file also executes cleanly as a smoke test. Not collected by pytest (no ``test_`` name).
"""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

import volas
from volas import DataFrame, Row, Series, Timestamp, TimeFrame

df = DataFrame({"open": np.arange(10.0), "close": np.arange(10.0)})

# DataFrame.__getitem__ overloads: column/directive -> Series; mask/slice/list -> frame.
assert_type(df["close"], Series)
assert_type(df["ma:5"], Series)
assert_type(df[df["close"] > 1.0], DataFrame)
assert_type(df[5:9], DataFrame)
assert_type(df[["open", "close"]], DataFrame)

# exec returns a NumPy array, NOT a Series (the correction stubtest surfaced).
assert_type(df.exec("ma:5"), npt.NDArray[Any])

# Series surfaces
s = df["close"]
assert_type(s[0], float | int | bool | str | Timestamp)
assert_type(s[1:5], Series)
assert_type(s + 1, Series)
assert_type(s * 2.0, Series)
assert_type(s > 0.0, Series)
assert_type(~(s > 0.0), Series)
assert_type(s.mean(), float)
assert_type(s.shift(1), Series)
assert_type(s.sqrt(), Series)
assert_type(s.iloc[0], float | int | bool | str | Timestamp)
assert_type(s.to_numpy(), npt.NDArray[Any])

# indexers
assert_type(df.iloc[0], Row)
assert_type(df.iloc[0:5], DataFrame)
assert_type(df.iat[0, 0], float | int | bool | str | Timestamp)

# misc precise returns
assert_type(df.copy(), DataFrame)
assert_type(df.shape, tuple[int, int])
assert_type(df.columns, list[str])
assert_type(df.head(), DataFrame)
assert_type(volas.read_csv("x.csv"), DataFrame)
assert_type(TimeFrame.m5, TimeFrame)  # preset is a TimeFrame instance, not a method
