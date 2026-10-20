"""Frozen, executable evaluation workloads. Never imported by PGO training.

Use build(kind, length, shape, directive, window) to obtain (run, setup).
All setup state is fresh. Timers may exclude setup, never a measured operation.
"""

import numpy as np
import pyarrow as pa

from volas import DataFrame
from scripts.pgo_train import HOLDOUT
from test.bench_workloads import COLUMNS, generated_ohlcv

SHAPES = ("reversal", "flat", "sparse-missing")
RESEND_EVERY = 23  # differs from the training stream's 17-bar schedule


def data(length, shape):
    values = generated_ohlcv(length, HOLDOUT["seed"])
    t = np.arange(length, dtype=float)
    if shape == "reversal":
        close = 100.0 + 0.01 * np.minimum(t, length - 1 - t) + 0.2 * np.sin(t * 0.07)
    elif shape == "flat":
        close = np.full(length, 100.0)
    elif shape == "sparse-missing":
        close = values["close"].copy()
    else:
        raise ValueError(f"unknown hold-out shape: {shape}")
    values.update(close=close, open=close + 0.1, high=close + 1.0, low=close - 1.0)
    if shape == "flat":
        for name in COLUMNS[:-1]:
            values[name] = np.full(length, 100.)
    if shape == "sparse-missing":
        for name in COLUMNS:
            values[name][31::113] = np.nan
    return values


def build(kind, length=257, shape="reversal", directive="ma:7", window=47):
    if (
        length not in HOLDOUT["lengths"]
        or directive not in HOLDOUT["directives"]
        or window not in HOLDOUT["windows"]
    ):
        raise ValueError("hold-out configuration not in the frozen inventory")
    values = data(length, shape)
    if kind == "batch":

        def setup():
            return (DataFrame(values),), {}

        def run(df):
            return df.exec(directive, dtype="float64")

        return run, setup
    if kind == "fold":
        index = np.datetime64("2021-03-01", "ns") + np.arange(length) * np.timedelta64(
            5, "m"
        )
        bars = [
            DataFrame(
                {c: a[i : i + 1] for c, a in values.items()}, index=index[i : i + 1]
            )
            for i in range(1, length)
        ]

        def setup():
            df = DataFrame(
                {c: a[:1] for c, a in values.items()},
                index=index[:1],
                time_frame="1h",
                window=window,
                max_lookback=[directive],
            )
            df[directive]
            return (df,), {}

        def run(df):
            for i, bar in enumerate(bars):
                df.append(bar)
                df.fulfill()
                if i % RESEND_EVERY == 0:
                    df.append(bar)
                    df.fulfill()
            return df[directive].to_numpy(dtype="float64")

        return run, setup
    if kind == "interop":
        array = pa.array(values["close"], mask=~np.isfinite(values["close"]))
        half = length // 2
        chunked = pa.chunked_array([array.slice(0, half), array.slice(half)]).slice(
            1, length - 2
        )
        table = pa.table({"close": chunked})

        def setup():
            return (), {}

        def run():
            frame = DataFrame.from_arrow(table)
            dense = DataFrame({"close": values["close"][::2]})
            return (
                frame["close"].to_numpy(),
                dense["close"].to_numpy(),
                np.from_dlpack(dense["close"]),
            )

        return run, setup
    raise ValueError(f"unknown hold-out operation: {kind}")
