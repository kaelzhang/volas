"""A resume applies its full kernel's guards: degenerate inputs resume bit-exactly.

A recursive kernel's full computation guards its divisions and degenerate windows (a
zero prior price, a flat range, a zero true range, a zero volume, …). A resume that
re-implemented the arithmetic without the guard would diverge exactly there — the
TRIX of a zero series resumed to NaN where the full computation gives 0. Every guard
has one owner shared by both paths; this test feeds degenerate series through the
live path one row at a time and holds every refresh to a full recompute, bit for bit.
"""

import numpy as np
import pytest

from volas import DataFrame

from .test_window_locality import RECURSIVE, SUBCOMMANDS

from test.exact import assert_exact

_KEYS = sorted(set(RECURSIVE.values()) | {SUBCOMMANDS[k] for k in SUBCOMMANDS if k.split(".", 1)[0] in RECURSIVE})


def _series(kind, n=90):
    i = np.arange(n, dtype=float)
    if kind == "zero":
        c = np.zeros(n)
    elif kind == "flat":
        c = np.full(n, 7.0)
    elif kind == "zero-then-walk":
        c = np.where(i < 40, 0.0, 100.0 + np.sin(i))
    else:  # "steps": flat runs with jumps, crossing zero
        c = np.repeat([0.0, 5.0, 5.0, -3.0, 0.0, 2.0], n // 6 + 1)[:n]
    flat_bars = kind in ("zero", "flat")
    return {
        "open": c.copy(),
        "high": c if flat_bars else c + (i % 3 == 0),
        "low": c.copy(),
        "close": c.copy(),
        "volume": np.where(i % 4 == 0, 0.0, 100.0),
    }


def _bits(got, want, ctx):
    assert_exact(got, want, ctx)


@pytest.mark.parametrize("kind", ["zero", "flat", "zero-then-walk", "steps"])
@pytest.mark.parametrize("directive", _KEYS)
def test_a_degenerate_series_resumes_bit_exactly(directive, kind):
    data = _series(kind)
    n = len(data["close"])
    live = DataFrame({k: v[:30] for k, v in data.items()})
    live[directive]
    for i in range(30, n):
        live.append(DataFrame({k: v[i:i + 1] for k, v in data.items()}))
        live.fulfill()
    full = DataFrame(data).exec(directive)
    _bits(live[directive], full, f"{directive} {kind}")


@pytest.mark.parametrize("directive", ["trix:2", "trix:2@(close - open)"])
def test_trix_of_a_zero_base_is_zero_on_every_path(directive):
    live = DataFrame({"close": [0.0] * 6, "open": [0.0] * 6})
    live[directive]
    live.append(DataFrame({"close": [0.0], "open": [0.0]}))
    full = DataFrame({"close": [0.0] * 7, "open": [0.0] * 7}).exec(directive)
    _bits(live[directive], full, directive)
    assert np.asarray(full)[-1] == 0.0
