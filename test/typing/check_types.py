"""Positive static type checks for volas — the type checker is the oracle.

``assert_type(expr, T)`` makes mypy/pyright fail unless ``expr`` is inferred as exactly
``T``. These pin the dynamic, @overload-driven surfaces (getitem / exec / indexers).
Run under ``mypy --strict`` and ``pyright``; ``assert_type`` is a runtime no-op, so the
file also executes cleanly as a smoke test. Not collected by pytest (no ``test_`` name).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, assert_type

import numpy as np
import numpy.typing as npt

import volas
import volas_rs
from volas import DataFrame, Row, Series, Timestamp, TimeFrame
from volas_rs import NAType

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
assert_type(s[0], float | int | bool | str | Timestamp | NAType)
assert_type(s[1:5], Series)
assert_type(s + 1, Series)
assert_type(s * 2.0, Series)
assert_type(s > 0.0, Series)
assert_type(~(s > 0.0), Series)
assert_type(s.mean(), np.float64)
assert_type(s.shift(1), Series)
assert_type(s.sqrt(), Series)
assert_type(s.iloc[0], float | int | bool | str | Timestamp | NAType)
assert_type(s.to_numpy(), npt.NDArray[Any])

# indexers
assert_type(df.iloc[0], Row)
assert_type(df.iloc[0:5], DataFrame)
assert_type(df.iat[0, 0], float | int | bool | str | Timestamp | NAType)

# misc precise returns
assert_type(df.copy(), DataFrame)
assert_type(df.shape, tuple[int, int])
assert_type(df.columns, list[str])
assert_type(df.head(), DataFrame)
assert_type(volas.read_csv("x.csv"), DataFrame)
assert_type(TimeFrame.m5, TimeFrame)  # preset is a TimeFrame instance, not a method

# Typed scalar fills accepted by where / mask / fillna (FU-P1-01 / FU-P2-04): the
# widened stubs must accept a str / Timestamp / volas.NA / bool fill, not just a
# number — these calls work at runtime but previously failed mypy --strict.
mask = s > 0.0
assert_type(s.where(mask, "z"), Series)
assert_type(s.where(mask, Timestamp("2021-01-01")), Series)
assert_type(s.where(mask, volas.NA), Series)
assert_type(s.mask(mask, True), Series)
assert_type(s.fillna("z"), Series)
assert_type(s.fillna(volas.NA), Series)

fmask = df.isna()
assert_type(df.where(fmask, "z"), DataFrame)
assert_type(df.where(fmask, volas.NA), DataFrame)
assert_type(df.mask(fmask, 0), DataFrame)
assert_type(df.fillna("z"), DataFrame)
assert_type(df.fillna(volas.NA), DataFrame)

# Directive facts have two distinct schemas, selected by columns=None vs a sequence.
command_facts = volas_rs._directive_facts("ema:2")
assert_type(command_facts["history"], Literal["window", "accumulating", "recursive"])
assert_type(command_facts["resume"], Literal["recompute", "rows", "step"])
assert_type(command_facts["accumulates"], bool)
assert_type(command_facts["position"], bool)
assert_type(command_facts["bare"], bool)
assert_type(volas_rs._directive_facts("ema:2", None, "ema:2")["bare"], bool)

bound_facts = volas_rs._directive_facts("ema:2+close", columns=["close"], shadow="EMA:02")
assert_type(bound_facts["lookback"], int)
assert_type(bound_facts["history_class"], Literal[1, 2, 3])
assert_type(bound_facts["shadows"], bool)
assert_type(volas_rs._directive_facts("ema:2+close", ("close",), None)["lookback"], int)
assert_type(volas_rs._directive_facts("1", columns=[])["history_class"], Literal[1, 2, 3])
assert_type(volas_rs._directive_facts("ema:2", shadow=None)["position"], bool)


def check_optional_fact_columns(columns: Sequence[str] | None) -> None:
    # A caller's optional schema must also resolve across both overloads.
    volas_rs._directive_facts("ema:2", columns, shadow=None)
