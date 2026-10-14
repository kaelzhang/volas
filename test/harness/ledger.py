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

from . import build, contracts, observe
from .data import OHLC, Schema


def _columns(schema: Schema, user: dict[str, str]) -> tuple[str, ...]:
    return schema.columns + tuple(c for c in user if c not in schema.columns)


def _record_dtype(schema: Schema, user: dict[str, str], name: str, dtype: str) -> None:
    # Whole-column replacement adopts the incoming type, including for base data.
    if name not in schema.columns or dtype != schema.dtype(name):
        user[name] = dtype
    else:
        user.pop(name, None)


@dataclass
class Ledger:
    """Accepted rows of a plain frame (or the coarse rows of a tf frame)."""

    schema: Schema
    rows: list[dict] = field(default_factory=list)
    index: list[int] = field(default_factory=list)  # epoch ns, or absolute range labels
    user: dict[str, str] = field(default_factory=dict)  # added columns / replacement dtype overrides

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
        return _columns(self.schema, self.user)

    def dtype(self, col: str) -> str:
        return self.user[col] if col in self.user else self.schema.dtype(col)

    def accepts(self, labels: list[int] | None, retained: int | None = None, count: int = 0) -> bool:
        """Whether the frame takes rows labelled ``labels`` (``contracts.
        APPEND_KEYED_BY_LABEL``): row by row, a label equal to the last replaces it; one
        before the last raises on a DatetimeIndex; on an integer index a label an
        earlier row (or an earlier row of the batch) has raises. ``count`` unlabelled
        rows (``labels=None``) take ``last + 1 ...``, none of which an integer index may
        already hold. A bounded frame knows only the labels of the ``retained`` rows it
        still holds."""
        if not contracts.APPEND_KEYED_BY_LABEL:
            return True
        seen = set(self.index if retained is None else self.index[len(self.index) - retained:])
        if labels is None:
            start = self.index[-1] + 1 if self.index else 0
            return self.schema.index == 'datetime' or not seen & set(range(start, start + count))
        prev = self.index[-1] if self.index else None
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
        self.set_column(name, values, dtype=dtype)

    def set_column(self, name: str, values: list, start: int = 0,
                   dtype: str = 'float64') -> None:
        """Record a whole-column write; retain history outside the visible rows.

        Unlike a cell write, replacement takes the incoming column's dtype.
        The product has already checked that retained values can survive it.
        """
        _record_dtype(self.schema, self.user, name, dtype)
        for r in self.rows:
            r.setdefault(name, None)
        for r, v in zip(self.rows[start:], values):
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
        full = build.frame(schema, self.rows, index, columns=columns,
                           dtypes={c: self.dtype(c) for c in columns})
        return full.iloc[start:] if start else full


def reference(ledger: Ledger, directive: str, env: tuple[str, ...], start: int = 0,
              evaluate: bool = False):
    """``(values, error)`` of ``directive`` computed from scratch over the ledger rows
    ``[start:]`` restricted to the columns ``env`` it was bound against. ``evaluate``
    always evaluates the directive (``exec``), never reading a same-named column."""
    ref = ledger.frame(env, start)
    try:
        return (observe.exec_values(ref, directive) if evaluate
                else observe.column_values_series(ref[directive])), None
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


def _aggregate(op: str, values: list):
    """A period's aggregate: ``None`` (missing) as soon as any value is missing."""
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
        return None
    if op == 'first':
        return values[0]
    if op == 'last':
        return values[-1]
    if op in ('max', 'min'):
        return (max if op == 'max' else min)(values)
    total = values[0]
    for v in values[1:]:
        total += v  # plain left-to-right addition (Python's sum() compensates)
    return total


def aggregate(schema: Schema, fine: list[tuple[int, dict]], tf: str,
              user: dict[str, str] | None = None) -> tuple[list[int], list[dict]]:
    """Coarse ``(labels, rows)`` of the fine bars ``[(ts, row)]`` (already deduplicated);
    ``user`` names plain float columns added to the frame (aggregated by ``last``)."""
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
        for col in _columns(schema, user or {}):
            op = AGG.get(col, 'last')
            vals = [r.get(col) for r in g]
            dtype = user[col] if user and col in user else schema.dtype(col)
            if dtype in ('float64', 'float32'):
                vals = [None if v is None else float(v) for v in vals]  # float addition
            out[col] = _aggregate(op, vals)
        rows.append(out)
    return labels, rows


@dataclass
class TfLedger:
    """Closed seed bars plus the accepted fine bars of a tf frame.

    Writes (docs/cumulation.md, "Writing to a time-frame frame") freeze the closed rows
    they reach into seed rows, as written, while the open period keeps its fine bars:
    a column that becomes data enters them as missing, its written forming cell standing
    (``override``) until the next bar of the period folds it to missing; a new forming
    value of a column the bars carry rewrites the period's one bar (with several the
    frame refuses it, see :meth:`refuses`)."""

    schema: Schema
    tf: str
    seed_index: list[int] = field(default_factory=list)
    seed_rows: list[dict] = field(default_factory=list)
    fine: list[tuple[int, dict]] = field(default_factory=list)
    user: dict[str, str] = field(default_factory=dict)  # added columns / replacement dtype overrides
    override: dict = field(default_factory=dict)  # forming cells written over missing bar values

    def copy(self) -> 'TfLedger':
        return TfLedger(self.schema, self.tf, list(self.seed_index), [dict(r) for r in self.seed_rows],
                        [(t, dict(r)) for t, r in self.fine], dict(self.user), dict(self.override))

    def _key(self, ts: int) -> int:
        return period_start(ts, self.tf, self.schema.tz)

    def _open_bars(self) -> list[tuple[int, dict]]:
        """The open period's fine bars: the fine bars of the last period, or the last seed
        row at its label (a constructed forming bar)."""
        if self.fine:
            key = self._key(self.fine[-1][0])
            return [(t, r) for t, r in self.fine if self._key(t) == key]
        return [(self.seed_index[-1], self.seed_rows[-1])]

    def carried(self, col: str) -> bool:
        """Whether the open period's bars carry ``col`` (a data column already there)."""
        return col in self.schema.columns or col in self.user

    def refuses(self, col: str, value) -> bool:
        """Whether the frame refuses a forming write of ``value`` into ``col``: a new
        forming value of a column the bars carry, over several fine bars."""
        coarse = self.coarse()
        if not coarse.rows or not self.carried(col):
            return False
        return not _same(coarse.rows[-1].get(col), value) and len(self._open_bars()) > 1

    def write(self, writes: dict[int, dict], columns: dict[str, str] | None = None) -> None:
        """Record accepted writes ``{coarse position: {column: value}}``; ``columns`` the
        plain columns they turn into data."""
        coarse = self.coarse()
        if not coarse.rows:
            return
        last = len(coarse.rows) - 1
        open_bars = [(t, dict(r)) for t, r in self._open_bars()]
        rows = [dict(r) for r in coarse.rows]
        for pos, cells in writes.items():
            rows[pos].update(cells)
        for row in rows:  # a new column is missing on the rows not written (a window's margin)
            for col in columns or {}:
                row.setdefault(col, None)
        for col, value in writes.get(last, {}).items():
            if columns and col in columns:
                self.override[col] = value  # the bars hold it as missing
            elif self.carried(col) and not _same(coarse.rows[last].get(col), value):
                self.override.pop(col, None)
                open_bars[0][1][col] = value  # the period's one bar is rewritten
        for col in columns or {}:
            self.override.setdefault(col, rows[last].get(col))
        self.user.update(columns or {})
        self.seed_rows, self.seed_index = rows[:last], list(coarse.index[:last])
        self.fine = open_bars

    def set_column(self, name: str, values: list, start: int = 0,
                   dtype: str = 'float64') -> None:
        """Whole-column writes share dtype ownership with the plain ledger."""
        existing = self.carried(name)
        self.write({start + i: {name: v} for i, v in enumerate(values)},
                   columns=None if existing else {name: dtype})
        _record_dtype(self.schema, self.user, name, dtype)

    def accepts(self, ts: int) -> bool:
        """Whether the fold takes a bar at ``ts`` (documented: no out-of-order bar)."""
        last = self.fine[-1][0] if self.fine else (self.seed_index[-1] if self.seed_index else None)
        return last is None or ts >= last

    def add(self, ts: int, row: dict) -> None:
        if self.override:
            if self._key(ts) != self._key(self._open_bars()[-1][0]):
                # the forming row closes as written: it is final
                coarse = self.coarse()
                self.seed_rows, self.seed_index = [dict(r) for r in coarse.rows], list(coarse.index)
                self.fine = []
            self.override = {}
        full = {c: row.get(c) for c in _columns(self.schema, self.user)}
        if self.fine and self.fine[-1][0] == ts:
            self.fine[-1] = (ts, full)  # a re-sent bar replaces the one it repeats
        else:
            self.fine.append((ts, full))

    def coarse(self) -> Ledger:
        labels, rows = aggregate(self.schema, self.fine, self.tf, self.user)
        out = Ledger(self.schema)
        out.rows = [dict(r) for r in self.seed_rows] + rows
        out.index = list(self.seed_index) + labels
        out.user = dict(self.user)
        if self.override and out.rows:
            out.rows[-1].update(self.override)
        return out


def _same(a, b) -> bool:
    """Cell equality as the frame judges a forming write (missing equals missing)."""
    missing = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
    return (missing(a) and missing(b)) or (not missing(a) and not missing(b) and a == b)


def last_close(rows: list[dict]) -> float | None:
    for r in reversed(rows):
        v = r.get('close')
        if v is not None and math.isfinite(v):
            return float(v)
    return None


__all__ = ['Ledger', 'TfLedger', 'aggregate', 'period_start', 'reference', 'last_close', 'OHLC']
