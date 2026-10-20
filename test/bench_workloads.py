"""Pure volas factories shared by benchmarks and deterministic PGO training."""

import numpy as np
from volas import DataFrame

COLUMNS = ("open", "high", "low", "close", "volume")


def generated_ohlcv(n, seed=20260606):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = close + rng.normal(0.0, 0.2, n)
    high = np.maximum(open_, close) + rng.random(n) * 1.5
    low = np.minimum(open_, close) - rng.random(n) * 1.5
    volume = np.abs(rng.normal(1.0e6, 2.0e5, n)) + 1.0
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "periods": (np.arange(n, dtype=float) % 29.0) + 2.0,
    }


def append(data, indicator):
    history = {c: a[:-2] for c, a in data.items()}
    warm = DataFrame({c: a[-2:-1] for c, a in data.items()})
    bar = DataFrame({c: a[-1:] for c, a in data.items()})

    def setup():
        df = DataFrame(history)
        df[indicator]
        df.append(warm)
        df.fulfill()
        return (df,), {}

    def run(df):
        df.append(bar)
        df.fulfill()

    return run, setup


def interop(length):
    import pyarrow as pa
    from volas import Series

    values = np.arange(length, dtype="float64")
    columns = {name: values.copy() for name in COLUMNS}
    table = pa.table(columns)
    array = table["close"].chunk(0)
    frame = DataFrame(columns)
    series = frame["close"]
    return {
        "frame_from_arrow": lambda: DataFrame.from_arrow(table),
        "frame_to_arrow": frame.to_arrow,
        "series_from_arrow": lambda: Series.from_arrow(array),
        "series_to_arrow": series.to_arrow,
        "dlpack_import": lambda: DataFrame({"close": np.from_dlpack(values)}),
        "dlpack_export": lambda: np.from_dlpack(series),
    }


def stream(data, directives, window=None, time_frame=None):
    n = len(data["close"])
    index = np.datetime64("2020-01-01", "ns") + np.arange(n) * np.timedelta64(1, "m")
    bars = [
        DataFrame({c: a[i : i + 1] for c, a in data.items()}, index=index[i : i + 1])
        for i in range(1, n)
    ]

    def setup():
        kwargs = {"index": index[:1]}
        if window is not None:
            kwargs.update(window=window, max_lookback=directives)
        if time_frame is not None:
            kwargs["time_frame"] = time_frame
        df = DataFrame({c: a[:1] for c, a in data.items()}, **kwargs)
        for d in directives:
            df[d]
        return (df,), {}

    def run(df):
        for i, bar in enumerate(bars):
            df.append(bar)
            df.fulfill()
            if time_frame and i % 17 == 0:
                df.append(bar)
                df.fulfill()
        return df

    return run, setup
