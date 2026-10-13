"""Bytes volas does not own can never carry a stale "no missing value" fact.

A float column remembers whether it holds a NaN, so a recursive indicator over a frame
known to be complete skips the missing-row check. That is only sound while the bytes
cannot change behind the frame's back:

- a zero-copy **import** (Arrow — `from_arrow`, the C-Data / C-Stream protocols)
  borrows the producer's buffer, which the producer may still write: the fact is never
  remembered for it, so a row made missing through the source is skipped exactly like
  one the frame was built with;
- a zero-copy **export** (DLPack, Arrow) lends a read-only view: nothing can write
  through it;
- NumPy import and `to_numpy` copy, so neither side can reach the other's bytes.
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
def test_a_row_made_missing_through_an_arrow_source_is_skipped(directive):
    source = CLOSE.copy()
    df = DataFrame.from_arrow(pa.table({"close": _borrowed_arrow(source)}))
    clean = np.asarray(df.exec(directive, dtype='float64'), float)  # the frame has now looked at its bytes
    _bit_exact(clean, DataFrame({"close": CLOSE.copy()}).exec(directive))
    source[100] = np.nan  # written behind the frame's back
    got = df.exec(directive)
    _bit_exact(got, DataFrame({"close": source.copy()}).exec(directive))
    assert np.isnan(np.asarray(got, float)[100])


def test_a_borrowed_series_is_rescanned_too():
    source = CLOSE.copy()
    s = Series.from_arrow(_borrowed_arrow(source), name="close")
    df = DataFrame({"close": s})
    df.exec("ema:5")
    source[50] = np.nan
    # the frame shares the borrowed buffer, so the write is visible — and skipped
    now = np.asarray(df["close"], float).copy()
    assert np.isnan(now[50])
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
