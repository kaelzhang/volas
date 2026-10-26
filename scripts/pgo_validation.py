"""Independent operation expectations for deterministic training factories."""

import numpy as np

COLUMNS = ("open", "high", "low", "close", "volume")


class Results:
    def __init__(self, digest):
        self.digest = digest

    def equal(self, actual, expected):
        actual, expected = (
            np.asarray(actual, dtype="float64"),
            np.asarray(expected, dtype="float64"),
        )
        if actual.shape != expected.shape:
            raise AssertionError((actual.shape, expected.shape))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(actual, expected)
        self.digest.update(str(actual.shape).encode())
        self.digest.update(np.isnan(actual).tobytes())
        self.digest.update(np.nan_to_num(actual).tobytes())

    def frame(self, frame, expected, directives=(), tail=None):
        from volas import DataFrame

        size = len(next(iter(expected.values())))
        tail = size if tail is None else min(tail, size)
        if len(frame) != tail:
            raise AssertionError()
        for c, values in expected.items():
            self.equal(frame[c].to_numpy(dtype="float64"), values[-tail:])
        reference = DataFrame(expected)
        for d in directives:
            self.equal(
                frame[d].to_numpy(dtype="float64"),
                reference.exec(d, dtype="float64")[-tail:],
            )


def fold(data, period=15):
    n = len(data["close"])
    starts = np.arange(0, n, period)
    ends = np.minimum(starts + period, n) - 1
    return {
        "open": data["open"][starts],
        "high": np.maximum.reduceat(data["high"], starts),
        "low": np.minimum.reduceat(data["low"], starts),
        "close": data["close"][ends],
        "volume": np.add.reduceat(data["volume"], starts),
    }


def live_expected(data, phase):
    from test.bench_tf_live import _history

    kind = phase.removesuffix("_20k")
    total = len(data["close"]) * (10 if phase.endswith("_20k") else 1)
    history = _history(data, COLUMNS, total)
    n = total - 2
    if kind in ("plain_append", "rollover"):
        return history
    if kind == "plain_replace":
        return {c: np.r_[a[:n], a[n] * 1.0005] for c, a in history.items()}
    out = {c: a[:n].copy() for c, a in history.items()}
    members = [n - 1, n, n + 1] if kind == "forming" else [n - 1, n]
    scales = np.ones(len(members))
    if kind == "resend":
        scales[-1] = 1.0005
    for c, a in history.items():
        v = a[members] * scales
        out[c][-1] = (
            v[0]
            if c == "open"
            else np.max(v)
            if c == "high"
            else np.min(v)
            if c == "low"
            else v[-1]
            if c == "close"
            else sum(v)
        )
    return out
