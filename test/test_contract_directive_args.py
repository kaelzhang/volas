"""Contract V17 — every directive argument is validated at the boundary against
its parameter's real domain: a value whose output is degenerate by construction
(a zero window's all-NaN column, a single-sample variance's constant zero, an
out-of-domain seed / limit, a divide-by-zero scaling factor) raises
DirectiveValueError instead of silently producing a valid-shaped column with no
signal. A period merely longer than the data stays legal (pure warm-up NaN)."""

import numpy as np
import pytest
from volas import DataFrame


def _df():
    n = 30
    c = np.arange(1.0, n + 1)
    return DataFrame(
        {"open": c - 0.5, "high": c + 1, "low": c - 1, "close": c, "volume": c * 100}
    )


@pytest.mark.parametrize("directive", [
    "ma:0",                  # zero window -> all-NaN by construction
    "rsi:0",
    "atr:0",
    "kdj.k:0",
    "repeat:0@(close>10)",   # used to underflow (panic) in the kernel
    "change:0",              # used to underflow (panic) in the kernel
    "linearreg:1",           # regression slope undefined on 1 sample
    "var:1",                 # single-sample variance identically 0
    "correl:1@close,high",   # single-sample correlation is 0/0
    "cci:1",                 # numerator identically 0 at window 1
    "aroon.up:1",            # identically 100 at window 1
    "hv:1",                  # stddev of one log return -> all-NaN
    "change:1",              # a bar vs itself -> identically 0
    "ma:5,9",                # matype selector past 8
    "kdj.k:9,3,150",         # %K seed outside [0, 100]
    "t3:5,1.5",              # T3 vfactor outside [0, 1]
    "mama:1.5,0.05",         # MAMA limit outside [0.01, 0.99] -> all-NaN
    "sar:-0.1,0.2",          # negative acceleration walks the stop away
    "asi:0",                 # the formula divides by the limit-move argument
    "increase:3,0",          # direction must be +1 or -1
    "hv:10,1d,0",            # annualization trading-days must be >= 1
    "boll.upper:20,nan",     # a NaN multiplier would poison the whole band
    "mavp:30,2@close,close", # inverted period limits (used to panic in clamp)
])
def test_invalid_directive_argument_raises(directive):
    with pytest.raises(Exception):
        _df()[directive]


@pytest.mark.parametrize("directive", [
    "ma:1",                  # a 1-period MA is the series itself — legal
    "atr:1",                 # ATR(1) is the true range — legal
    "ma:999999999999",       # longer than the data: pure warm-up NaN — legal
    "t3:5,0", "t3:5,1",      # vfactor domain edges
    "kdj.k:9,3,0", "kdj.k:9,3,100",   # seed domain edges
    "mama:0.01,0.01", "mama:0.99,0.99",  # limit domain edges
    "sar:0,0",               # zero acceleration: a static stop — legal
    "boll.upper:20,-2",      # a negative multiplier flips the band (TA-Lib nbdev)
    "increase:3,-1",         # falling-run direction
    "var:2", "linearreg:2", "change:2", "aroon.up:2",  # second-moment minima
])
def test_domain_edge_values_still_execute(directive):
    assert len(_df()[directive]) == 30
