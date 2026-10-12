"""Which finite-window commands a last-window recompute reproduces bit for bit.

A finite-window command whose full computation does not carry anything from row to
row (each output a function of its own window alone) gives the same value over any
window that holds its lookback — so a refresh may recompute just the last window. One
whose full computation carries a running accumulator (a rolling sum, a rolling sum of
squares, …) resumes from a carried state instead, which is cheaper than re-summing a
window. Its sums are exact (each the correctly rounded sum of its window), so it too
equals its last window bit for bit. This test pins that every command the engine
recomputes over a window is exact there, and that the accumulating ones are the ones
the engine resumes.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame

from .test_window_locality import ABSOLUTE_POSITION, FINITE, SUBCOMMANDS, _command, _directive

ROOT = Path(__file__).resolve().parent.parent
N = 400
_rng = np.random.default_rng(11)
# Full-precision values (no short binary fractions), so any accumulated rounding shows.
_c = 1000.0 + np.pi * np.cumsum(_rng.standard_normal(N))
DATA = dict(
    open=_c + _rng.uniform(-0.5, 0.5, N) * np.e, high=_c + 1.0 + _rng.uniform(0, 1, N) * np.e,
    low=_c - 1.0 - _rng.uniform(0, 1, N) * np.e, close=_c.copy(),
    volume=_rng.uniform(100, 900, N) * np.sqrt(2.0), periods=np.full(N, 5.0),
)

# Runs of huge and tiny values: a compensated pair cannot hold every bit of a tiny window
# right after a huge run, so the running sums must notice and re-sum exactly.
_RUNS = np.where((np.arange(N) // 13) % 2 == 0, 1e12, 1e-8)
WIDE = {k: (v if k == "periods" else v * _RUNS) for k, v in DATA.items()}

# A copy with a few missing and infinite rows: the careful (window-local) path.
GAPPY = {k: v.copy() for k, v in DATA.items()}
for _col, _rows in (("close", [120, 121, 250]), ("high", [180]), ("volume", [300]), ("low", [181])):
    GAPPY[_col][_rows] = np.nan
GAPPY["close"][200] = np.inf

# The index family emits positions and never takes the window path.
_KEYS = sorted(k for k in FINITE if k not in ABSOLUTE_POSITION) + sorted(
    k for k in SUBCOMMANDS if _command(k) in FINITE and _command(k) not in ABSOLUTE_POSITION
)

# The finite-window commands whose full computation carries a running accumulator: these
# resume from a state.
ACCUMULATING = {
    "accbands", "apo", "bbi", "bbw", "beta", "bias", "boll", "cci", "cmf", "cog", "coppock",
    "correl", "dkx", "dma", "dpo", "emv", "hma", "hv", "imi", "kst", "linearreg",
    "linearreg_angle", "linearreg_intercept", "linearreg_slope", "ma", "macdext", "mavp", "mfi",
    "ppo", "relative_vigor", "stddev", "stoch", "stochf", "sum", "trima", "tsf", "ttm_squeeze",
    "ultosc", "var", "vr", "vwma", "wma", "wvad",
    "accbands.lower", "accbands.upper", "boll.lower", "boll.u", "boll.upper", "dkx.ma",
    "dma.ama", "dma.ddd", "macdext.histogram", "macdext.signal", "relative_vigor.signal",
    "stoch.d", "stoch.k", "stochf.d", "ttm_squeeze.on",
}


def _window_matches(directive, data):
    """Whether computing over a window of rows that ends at the last row reproduces the
    full computation, the window a slice keeping its rows' absolute positions (as the
    engine's window recompute does — second moments shift by the value at a row of
    absolute position, see docs/indicator-inputs.md)."""
    frame = DataFrame(data)
    full = np.asarray(frame.exec(directive), dtype=float)
    lb = volas.directive_lookback(directive)
    win = 2 * (lb + 1)
    for vr in range(win + 1, N):
        start = vr - win
        part = frame.iloc[start:]
        tail = np.asarray(part.exec(directive), dtype=float)[vr - start:]
        want = full[vr:]
        same = (tail == want) | (np.isnan(tail) & np.isnan(want))
        if not same.all():
            return False
    return True


def test_the_accumulating_commands_are_the_ones_the_engine_resumes():
    """`spec::accumulates` (which decides that a refresh resumes rather than recomputes
    a window) names exactly the commands this test finds accumulating."""
    src = (ROOT / "crates/volas-directive/src/spec.rs").read_text()
    body = re.search(r"pub fn accumulates\(.*?\{(.*?)\n\}", src, re.S).group(1)
    named = set(re.findall(r'"([a-z0-9_]+)" =>', body))
    named |= set(re.findall(r'"([a-z0-9_]+)"', re.search(r"matches!\(\s*name,(.*?)\)", body, re.S).group(1)))
    assert named == {k.split(".")[0] for k in ACCUMULATING}


@pytest.mark.parametrize("key", [k for k in _KEYS if k not in ACCUMULATING])
def test_a_last_window_recompute_is_exact_unless_the_kernel_accumulates(key):
    directive = _directive(key)
    for data in (DATA, GAPPY, WIDE):
        assert _window_matches(directive, data), directive


@pytest.mark.parametrize("key", ["ma", "sum", "var", "linearreg", "wma", "cci", "mfi", "ultosc"])
def test_an_accumulating_command_is_its_last_window_because_its_sums_are_exact(key):
    """Even on runs of huge and tiny values, a running sum's value is the exact sum of
    its window (re-summed whenever it could not hold it), and a second moment's shift is
    the value at a row of absolute position, so an accumulating command computed over
    its last window alone (a slice, keeping positions) gives the full computation's
    bits."""
    for data in (DATA, GAPPY, WIDE):
        assert _window_matches(_directive(key), data), key


# --- the accumulating commands resume bit-exactly on every live path ----------------

def _bit_exact(got, want, ctx):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    if not same.all():
        bad = int(np.flatnonzero(~same)[0])
        raise AssertionError(f"{ctx}: row {bad}: {got[bad]!r} vs {want[bad]!r}")


def _rows(data, lo, hi, labels):
    return DataFrame({k: v[lo:hi] for k, v in data.items()}, index=labels[lo:hi])


_LABELS = (np.datetime64("2026-01-01T00:00", "ns") + np.arange(N) * np.timedelta64(1, "m"))


@pytest.mark.parametrize("data_name", ["clean", "gappy", "wide"])
@pytest.mark.parametrize("key", sorted(ACCUMULATING))
def test_an_accumulating_command_resumes_bit_exactly(key, data_name):
    """Appends one row at a time, replaces the newest row (with a value, with a NaN),
    then a windowed twin: every refresh equals a full recompute bit for bit."""
    data = {"clean": DATA, "gappy": GAPPY, "wide": WIDE}[data_name]
    directive = _directive(key)
    seed = 100
    live = _rows(data, 0, seed, _LABELS)
    win = DataFrame(_rows(data, 0, 3, _LABELS), window=4, max_lookback=[directive])
    live[directive]
    win[directive]
    for i in range(3, seed):
        win.append(_rows(data, i, i + 1, _LABELS))
        win.fulfill()
    for i in range(seed, N):
        live.append(_rows(data, i, i + 1, _LABELS))
        win.append(_rows(data, i, i + 1, _LABELS))
        if i % 3 == 0:  # re-sent: a different value, then a missing one, then the real row
            for alt in ({**{k: v[i:i + 1] * 1.001 for k, v in data.items()}},
                        {**{k: v[i:i + 1] for k, v in data.items()}, "close": np.array([np.nan])}):
                for frame in (live, win):
                    frame.append(DataFrame(alt, index=_LABELS[i:i + 1]))
                live.fulfill()
                ref = DataFrame({k: np.asarray(live[k]) for k in data}, index=np.asarray(live.index))
                _bit_exact(live[directive], ref.exec(directive), f"{directive} replaced @{i}")
            for frame in (live, win):
                frame.append(_rows(data, i, i + 1, _LABELS))
        live.fulfill()
        win.fulfill()
        if i % 7 == 0 or i == N - 1:
            ref = DataFrame({k: v[:i + 1] for k, v in data.items()}, index=_LABELS[:i + 1])
            _bit_exact(live[directive], ref.exec(directive), f"{directive} @{i}")
            w = np.asarray(win[directive], dtype=float)
            _bit_exact(w, np.asarray(live[directive], dtype=float)[-len(w):], f"{directive} windowed @{i}")
