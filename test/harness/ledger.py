"""The external event ledger and the oracles derived from it.

The ledger records every *accepted* event (bars, cell writes, added plain columns)
outside volas, in plain Python. Reference values never read the frame under test:

* base data: the ledger rows themselves;
* plain-frame indicators: a fresh ``volas.DataFrame`` built from the ledger rows
  (only the columns the directive was bound against), then ``frame[d]`` — a
  from-scratch full compute, the path every incremental one must equal;
* tf frames: :func:`aggregate` re-derives the coarse bars from the ledger of fine
  bars (period-start labels, first / max / min / last / sum, same-timestamp
  replacement) without calling ``cumulate`` / ``aggregate_period``;
* windowed frames: the full-history reference projected onto the last ``window``
  rows (the unbounded twin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import volas

from . import build, contracts
from .data import OHLC, Schema


@dataclass
class Ledger:
    """Accepted rows of a plain frame (or the coarse rows of a tf frame)."""

    schema: Schema
    rows: list[dict] = field(default_factory=list)
    index: list[int] = field(default_factory=list)  # epoch ns, or absolute range labels
    user: dict[str, str] = field(default_factory=dict)  # plain columns added by setitem -> dtype

    # Whether the frame's index is still a RangeIndex (a bare RangeIndex bar then
    # takes the next labels); a Row / sliced bar / slice turns it into explicit labels.
    range_kind: bool = True

    def copy(self) -> 'Ledger':
        return Ledger(self.schema, [dict(r) for r in self.rows], list(self.index), dict(self.user),
                      self.range_kind)

    def coarse(self) -> 'Ledger':
        """The rows the frame shows (a plain frame shows its rows as they are)."""
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        return self.schema.columns + tuple(self.user)

    def dtype(self, col: str) -> str:
        return self.user[col] if col in self.user else self.schema.dtype(col)

    def accepts(self, labels: list[int] | None, retained: int | None = None) -> bool:
        """Whether the frame takes rows labelled ``labels`` (``contracts.
        APPEND_KEYED_BY_LABEL``): row by row, a label equal to the last replaces it; one
        before the last raises on a DatetimeIndex; on an integer index a label an
        earlier row (or an earlier row of the batch) has raises. A bounded frame knows only
        the labels of the ``retained`` rows it still holds."""
        if labels is None or not contracts.APPEND_KEYED_BY_LABEL:
            return True
        prev = self.index[-1] if self.index else None
        seen = set(self.index if retained is None else self.index[len(self.index) - retained:])
        for lab in labels:
            if prev is not None and lab != prev:
                if self.schema.index == 'datetime' and lab < prev:
                    return False
                if self.schema.index != 'datetime' and lab in seen:
                    return False
            seen.add(lab)
            prev = lab
        return True

    def append(self, rows: list[dict], index: list[int] | None) -> None:
        """Record accepted rows; ``index=None`` is a bare RangeIndex bar, whose labels
        continue the absolute sequence (``last + 1``). A row labelled like the last
        replaces it (see :meth:`accepts`)."""
        if index is None:
            start = self.index[-1] + 1 if self.index else 0
            index = list(range(start, start + len(rows)))
            keyed = False
        else:
            keyed = contracts.APPEND_KEYED_BY_LABEL
        for r, lab in zip(rows, index):
            full = dict(r)
            for c in self.user:
                full.setdefault(c, None)
            for c in self.schema.columns:
                full.setdefault(c, None)
            if keyed and self.index and lab == self.index[-1]:
                self.rows[-1] = full
                continue
            if keyed and self.schema.index == 'range':
                self.range_kind = False
            self.rows.append(full)
            self.index.append(lab)

    def write(self, pos: int, col: str, value) -> None:
        self.rows[pos][col] = value

    def add_column(self, name: str, values: list, dtype: str = 'float64') -> None:
        self.user[name] = dtype
        for r, v in zip(self.rows, values):
            r[name] = v

    def slice(self, start: int) -> 'Ledger':
        out = self.copy()
        out.rows, out.index = out.rows[start:], out.index[start:]
        return out

    def values(self, col: str, start: int = 0) -> np.ndarray:
        """The column as the frame should hold it (float NaN / None for NA)."""
        dtype = self.dtype(col)
        vals = [r[col] for r in self.rows[start:]]
        if dtype in ('float64', 'float32'):
            return np.array([np.nan if v is None else v for v in vals], dtype=dtype)
        return np.array(vals, dtype=object)

    def frame(self, columns=None, start: int = 0) -> volas.DataFrame:
        """A fresh frame of the ledger rows ``[start:]`` (the reference frame): the
        whole history sliced at ``start``, so its rows keep their absolute positions
        (where sliding sums re-anchor) exactly as a slice of the frame under test does."""
        columns = tuple(columns or self.columns)
        schema = Schema(self.schema.price, self.schema.volume,
                        tuple((c, self.dtype(c)) for c in columns if c not in self.schema.columns[:5]),
                        self.schema.index, self.schema.tz)
        index = None if self.schema.index == 'range' else self.index
        full = build.frame(schema, self.rows, index, columns=columns)
        return full.iloc[start:] if start else full


def reference(ledger: Ledger, directive: str, env: tuple[str, ...], start: int = 0,
              evaluate: bool = False):
    """``(values, error)`` of ``directive`` computed from scratch over the ledger rows
    ``[start:]`` restricted to the columns ``env`` it was bound against. ``evaluate``
    always evaluates the directive (``exec``), never reading a same-named column."""
    ref = ledger.frame(env, start)
    try:
        return np.asarray(ref.exec(directive) if evaluate else ref[directive]), None
    except Exception as e:  # noqa: BLE001 - the error class is compared, not judged
        return None, type(e)


# --- tf: independent re-aggregation of fine bars ---------------------------------------

TF_NS = {'1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400}
_TZ_OFFSET = {None: 0, 'UTC': 0, 'Asia/Shanghai': 8 * 3600}
AGG = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}


def period_start(ts: int, tf: str, tz: str | None) -> int:
    """The period-start label (epoch ns) of ``ts`` on ``tf``'s midnight-anchored grid
    in the frame's (fixed-offset) zone."""
    step = TF_NS[tf] * 10**9
    off = _TZ_OFFSET[tz] * 10**9
    local = ts + off
    return local - local % step - off


def _agg_float(op: str, values: list):
    nan = [v is None or (isinstance(v, float) and math.isnan(v)) for v in values]
    vals = [math.nan if n else v for v, n in zip(values, nan)]
    if op == 'first':
        return vals[0] if contracts.TF_FLOAT_FIRST_LAST_KEEP_NAN else next((v for v, n in zip(vals, nan) if not n), math.nan)
    if op == 'last':
        return vals[-1] if contracts.TF_FLOAT_FIRST_LAST_KEEP_NAN else next((v for v, n in zip(reversed(vals), reversed(nan)) if not n), math.nan)
    if op in ('max', 'min'):
        present = [v for v, n in zip(vals, nan) if not n]
        if not contracts.TF_FLOAT_MAX_MIN_SKIP_NAN and any(nan):
            return math.nan
        if not present and contracts.avoids('all-nan-fold-extreme'):
            return -math.inf if op == 'max' else math.inf  # the known current result
        return (max if op == 'max' else min)(present) if present else math.nan
    if contracts.TF_FLOAT_SUM_PROPAGATES_NAN and any(nan):
        return math.nan
    total = 0.0
    for v, n in zip(vals, nan):
        if not n:
            total += v  # plain left-to-right addition (Python's sum() compensates)
    return total


def _agg_int(op: str, values: list):
    present = [v for v in values if v is not None] if contracts.TF_INT_AGG_SKIPS_NA else values
    if not present or any(v is None for v in present):
        return None
    if op == 'first':
        return present[0]
    if op == 'last':
        return present[-1]
    if op == 'max':
        return max(present)
    if op == 'min':
        return min(present)
    return sum(present)


def aggregate(schema: Schema, fine: list[tuple[int, dict]], tf: str) -> tuple[list[int], list[dict]]:
    """Coarse ``(labels, rows)`` of the fine bars ``[(ts, row)]`` (already deduplicated)."""
    labels: list[int] = []
    groups: list[list[dict]] = []
    for ts, row in fine:
        key = period_start(ts, tf, schema.tz)
        if not labels or labels[-1] != key:
            labels.append(key)
            groups.append([])
        groups[-1].append(row)
    rows = []
    for g in groups:
        out = {}
        for col in schema.columns:
            op = AGG.get(col, 'last')
            vals = [r[col] for r in g]
            if schema.dtype(col) in ('float64', 'float32'):
                v = _agg_float(op, vals)
                out[col] = None if math.isnan(v) else v
            else:
                out[col] = _agg_int(op, vals)
        rows.append(out)
    return labels, rows


@dataclass
class TfLedger:
    """Closed seed bars plus the accepted fine bars of a tf frame."""

    schema: Schema
    tf: str
    seed_index: list[int] = field(default_factory=list)
    seed_rows: list[dict] = field(default_factory=list)
    fine: list[tuple[int, dict]] = field(default_factory=list)

    def copy(self) -> 'TfLedger':
        return TfLedger(self.schema, self.tf, list(self.seed_index), [dict(r) for r in self.seed_rows],
                        [(t, dict(r)) for t, r in self.fine])

    def accepts(self, ts: int) -> bool:
        """Whether the fold takes a bar at ``ts`` (documented: no out-of-order bar)."""
        last = self.fine[-1][0] if self.fine else (self.seed_index[-1] if self.seed_index else None)
        return last is None or ts >= last

    def add(self, ts: int, row: dict) -> None:
        full = {c: row.get(c) for c in self.schema.columns}
        if self.fine and self.fine[-1][0] == ts:
            self.fine[-1] = (ts, full)  # a re-sent bar replaces the one it repeats
        else:
            self.fine.append((ts, full))

    def coarse(self) -> Ledger:
        labels, rows = aggregate(self.schema, self.fine, self.tf)
        out = Ledger(self.schema)
        out.rows = [dict(r) for r in self.seed_rows] + rows
        out.index = list(self.seed_index) + labels
        return out


def last_close(rows: list[dict]) -> float | None:
    for r in reversed(rows):
        v = r.get('close')
        if v is not None and math.isfinite(v):
            return float(v)
    return None


__all__ = ['Ledger', 'TfLedger', 'aggregate', 'period_start', 'reference', 'last_close', 'OHLC']
