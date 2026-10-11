"""Generators for bars: schemas (all 7 dtypes, nullable), value regimes, timestamps.

Values are drawn as a *regime* (walk, constant, NaN in one input, ±inf, extreme
magnitudes, negative prices, ...) plus a numpy seed, so a shrunk counterexample keeps
its data class and stays small. Every row is a ``{column: value}`` dict whose missing
cells are ``None`` (any dtype) — the ledger's representation-independent form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st

OHLC = ('open', 'high', 'low', 'close')
NS = 1_000_000_000
# A Sunday-morning US DST start (2026-03-08 07:00 UTC = 02:00 EST -> 03:00 EDT) and a
# plain Monday; aware frames are generated around both.
EPOCHS = (1_772_953_200 * NS, 1_767_571_200 * NS)
TIMEZONES = (None, 'UTC', 'Asia/Shanghai', 'America/New_York')


@dataclass(frozen=True)
class Schema:
    price: str  # dtype of open/high/low/close: float64 | float32 | int64
    volume: str  # float64 | int64 | int32
    extras: tuple[tuple[str, str], ...]  # (name, dtype) of non-OHLCV data columns
    index: str  # 'datetime' | 'range'
    tz: str | None

    @property
    def columns(self) -> tuple[str, ...]:
        return OHLC + ('volume',) + tuple(n for n, _ in self.extras)

    def dtype(self, name: str) -> str:
        if name in OHLC:
            return self.price
        if name == 'volume':
            return self.volume
        return dict(self.extras)[name]

    @property
    def float_columns(self) -> tuple[str, ...]:
        """Numeric columns a directive may read as a series."""
        return tuple(c for c in self.columns if self.dtype(c) in ('float64', 'float32', 'int64', 'int32'))


EXTRA_TYPES = (
    ('x', 'float64'), ('n', 'int64'), ('flag', 'bool'), ('tag', 'str'), ('stamp', 'datetime64[ns]'),
    ('m', 'int32'),
)


@st.composite
def schemas(draw, index: str | None = None, extras: bool = True, price_floats_only: bool = False) -> Schema:
    price = 'float64' if price_floats_only else draw(st.sampled_from(['float64', 'float64', 'float32', 'int64']))
    volume = draw(st.sampled_from(['float64', 'float64', 'int64', 'int32']))
    chosen = tuple(e for e in EXTRA_TYPES if extras and draw(st.integers(0, 3)) == 0)
    kind = index or draw(st.sampled_from(['datetime', 'datetime', 'range']))
    tz = draw(st.sampled_from(TIMEZONES)) if kind == 'datetime' else None
    return Schema(price, volume, chosen, kind, tz)


REGIMES = ('walk', 'walk', 'walk', 'constant', 'zero', 'nan_one', 'nan_all', 'inf', 'extreme', 'negative',
           'tiny')


@st.composite
def regimes(draw) -> tuple[str, int]:
    return draw(st.sampled_from(REGIMES)), draw(st.integers(0, 2**16))


EXTREME_SCALES = (1e300, 1e-300, 5e-324, 1e15)


def make_rows(schema: Schema, regime: tuple[str, int], n: int, prev_close: float | None,
              scale: float | None = None) -> list[dict]:
    """``n`` bar rows of ``regime`` continuing from ``prev_close``; ``scale`` pins the
    magnitude of the 'extreme' regime (otherwise drawn per batch)."""
    name, seed = regime
    rng = np.random.default_rng(seed)
    finite = prev_close is not None and np.isfinite(prev_close) and 1e-3 < abs(prev_close) < 1e6
    base = prev_close if finite else 100.0
    close = base + np.cumsum(rng.standard_normal(n))
    o, c = close + 0.3 * rng.standard_normal(n), close
    h = np.maximum(o, c) + np.abs(rng.standard_normal(n)) * 0.5 + 0.05
    l = np.minimum(o, c) - np.abs(rng.standard_normal(n)) * 0.5 - 0.05
    v = rng.uniform(1.0, 1000.0, n)
    cols = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
    if name == 'constant':
        for k in OHLC:
            cols[k] = np.full(n, round(base))
        cols['volume'] = np.where(rng.random(n) < 0.5, 0.0, v)
    elif name == 'zero':
        for k in OHLC:
            cols[k] = np.zeros(n)
    elif name == 'extreme':
        scale = scale or rng.choice(EXTREME_SCALES)
        for k in OHLC:
            cols[k] = cols[k] / base * scale
        cols['volume'] = v * scale
    elif name == 'negative':
        for k in OHLC:
            cols[k] = cols[k] - 2 * base
    elif name == 'tiny':
        for k in OHLC:
            cols[k] = cols[k] / base * 1e-6
    rows = []
    for i in range(n):
        row = {k: float(cols[k][i]) for k in cols}
        if name == 'nan_one':
            row[('open', 'high', 'low', 'close', 'volume')[int(rng.integers(0, 5))]] = None
        elif name == 'nan_all':
            for k in OHLC + ('volume',):
                row[k] = None
        elif name == 'inf':
            row[('high', 'low', 'close', 'volume')[int(rng.integers(0, 4))]] = float(rng.choice([np.inf, -np.inf]))
        rows.append(_typed(schema, row, rng))
    return rows


def _typed(schema: Schema, row: dict, rng: np.random.Generator) -> dict:
    out = {}
    for col in schema.columns:
        dtype = schema.dtype(col)
        value = row.get(col)
        if col not in row:
            value = _extra_value(dtype, rng)
        if dtype in ('int64', 'int32'):
            if value is None or not np.isfinite(value):
                value = None
            else:
                bound = 2**31 - 1 if dtype == 'int32' else 2**53
                value = int(np.clip(round(value), -bound, bound))
        elif dtype == 'float32' and value is not None:
            value = float(np.float32(value))
        out[col] = value
    return out


def _extra_value(dtype: str, rng: np.random.Generator):
    if rng.random() < 0.2:
        return None
    if dtype == 'float64':
        return float(rng.standard_normal() * 10)
    if dtype in ('int64', 'int32'):
        return int(rng.integers(-1000, 1000))
    if dtype == 'bool':
        return bool(rng.random() < 0.5)
    if dtype == 'str':
        return str(rng.choice(['a', 'b', 'é', '', 'long-' * 3]))
    return int(EPOCHS[1] + rng.integers(0, 10**6) * NS)  # datetime64[ns] as epoch-ns


STEP_KINDS = ('1m', '1m', '1s', 'sub', 'irregular', 'resend')


@st.composite
def steps(draw) -> str:
    return draw(st.sampled_from(STEP_KINDS))


def next_timestamps(last: int | None, n: int, kind: str, seed: int, start: int) -> list[int]:
    """``n`` timestamps (epoch ns) after ``last`` per ``kind``; ``resend`` repeats the
    last timestamp once, then continues by minutes."""
    rng = np.random.default_rng(seed)
    t = start if last is None else last
    out = []
    for i in range(n):
        if kind == 'resend' and i == 0 and last is not None:
            out.append(t)
            continue
        if kind == '1s':
            t += NS
        elif kind == 'sub':
            t += int(rng.integers(1, 4)) * NS // 4 + 7  # sub-second, odd ns
        elif kind == 'irregular':
            t += int(rng.integers(1, 400)) * 60 * NS
        else:
            t += 60 * NS
        out.append(t)
    return out
