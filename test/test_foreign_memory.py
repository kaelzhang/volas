"""Explicit Arrow snapshots freeze bytes; exports lend read-only views.

Explicit immutable borrowing never permits writes through producer aliases.
"""

import numpy as np
import pytest

from volas import DataFrame, Series

from test.exact import assert_exact

pa = pytest.importorskip("pyarrow")

N = 160
_rng = np.random.default_rng(3)
CLOSE = 100.0 + np.cumsum(_rng.standard_normal(N))


def _bit_exact(got, want):
    assert_exact(got, want)


def _borrowed_arrow(values):
    """An Arrow float64 array over `values`' own bytes (no copy)."""
    arr = pa.Array.from_buffers(pa.float64(), len(values), [None, pa.py_buffer(values)])
    assert arr.buffers()[1].address == values.ctypes.data
    return arr


@pytest.mark.parametrize("directive", ["ema:5", "rsi:5", "macd", "ht_trendline"])
def test_an_arrow_source_write_cannot_change_the_import_snapshot(directive):
    source = CLOSE.copy()
    df = DataFrame.from_arrow(pa.table({"close": _borrowed_arrow(source)}), copy=True)
    clean = np.asarray(df.exec(directive, dtype='float64'), float)  # the frame has now looked at its bytes
    _bit_exact(clean, DataFrame({"close": CLOSE.copy()}).exec(directive))
    source[100] = np.nan  # written behind the frame's back
    got = df.exec(directive)
    _bit_exact(got, clean)
    assert not np.isnan(np.asarray(got, float)[100])


def test_an_imported_series_keeps_its_snapshot():
    source = CLOSE.copy()
    s = Series.from_arrow(_borrowed_arrow(source), name="close", copy=True)
    df = DataFrame({"close": s})
    df.exec("ema:5")
    source[50] = np.nan
    # The Series owns the import snapshot; the frame shares that stable column.
    now = np.asarray(df["close"], float).copy()
    assert not np.isnan(now[50])
    _bit_exact(df.exec("ema:5"), DataFrame({"close": now}).exec("ema:5"))


def test_dlpack_and_arrow_exports_are_read_only():
    df = DataFrame({"close": CLOSE.copy()})
    df.exec("ema:5")
    views = [
        np.from_dlpack(df["close"]),
        pa.array(df["close"]).to_numpy(zero_copy_only=True),
        pa.table(df)["close"].chunk(0).to_numpy(zero_copy_only=True),
    ]
    for v in views:
        assert not v.flags.writeable
        with pytest.raises(ValueError):
            v[0] = np.nan
        with pytest.raises(ValueError):
            v.flags.writeable = True
    _bit_exact(df.exec("ema:5"), DataFrame({"close": CLOSE.copy()}).exec("ema:5"))


def test_numpy_import_and_export_copy():
    source = CLOSE.copy()
    df = DataFrame({"close": source})
    before = np.asarray(df.exec("ema:5", dtype='float64'), float)
    source[10] = np.nan  # the frame copied on construction
    out = df["close"].to_numpy()
    out[20] = np.nan  # and `to_numpy` hands out a copy
    _bit_exact(df.exec("ema:5"), before)
