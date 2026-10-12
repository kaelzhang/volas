"""The plain-frame machine: appends in every form, rejected appends, cell / column
writes, name collisions, slices, and appends / comparisons across snapshots."""

from __future__ import annotations

import re

import numpy as np
from hypothesis import strategies as st
from hypothesis.stateful import initialize, precondition, rule

import volas

from . import build, contracts, inventory, observe
from .directives import stored_bare_names
from .data import EPOCHS, NS, make_rows, next_timestamps, regimes, schemas, steps
from .ledger import Ledger, last_close
from .machine_base import Entry, FrameMachine

APPEND_FORMS = ('arrow', 'arrow', 'numpy', 'sliced', 'row', 'dict')
BAD_APPENDS = ('dtype', 'extra_column', 'index_kind', 'naive_onto_aware', 'aware_onto_naive', 'nat',
               'str_index', 'stale_write')


class PlainMachine(FrameMachine):

    def build_frame(self, schema, rows, index):
        return build.frame(schema, rows, index)

    @initialize(schema=schemas(), n=st.integers(0, 40), regime=regimes(), step=steps(),
                epoch=st.sampled_from(EPOCHS))
    def init(self, schema, n, regime, step, epoch):
        self.setup(schema, n, regime, step, epoch)

    def setup(self, schema, n, regime, step, epoch):
        self.store = Ledger(schema)
        self.pin_scale(regime)
        steered, scale = self.steer(regime)
        rows = make_rows(schema, steered, n, None, scale)
        index = None if schema.index == 'range' else next_timestamps(None, n, step, regime[1], epoch)
        self.df = self.build_frame(schema, rows, index)
        self.store.append(rows, index)
        self.log(f'init {schema} n={n} {regime} {step}')

    # --- appends -----------------------------------------------------------------------

    def _new_bars(self, k, regime, step):
        schema = self.store.schema
        steered, scale = self.steer(regime)
        rows = make_rows(schema, steered, k, last_close(self.store.rows), scale)
        if schema.index == 'range':
            return rows, None
        last = self.store.index[-1] if self.store.index else None
        return rows, next_timestamps(last, k, step, regime[1], EPOCHS[0])

    def retained(self) -> int | None:
        """How many rows the frame physically holds: the ledger rows from the slice's
        start (a slice is a frame of its own rows only)."""
        return len(self.store.index) - self.offset

    def _pend_stale(self):
        if self.coarse().rows:
            self.stale |= set(self.cached)

    @rule(k=st.sampled_from([1, 1, 1, 2, 3, 7]), form=st.sampled_from(APPEND_FORMS), regime=regimes(),
          step=steps())
    def append(self, k, form, regime, step):
        schema = self.store.schema
        if form in ('row', 'dict'):
            k = 1
        if form == 'sliced' and schema.index == 'range':
            form = 'arrow'
        if form == 'dict' and schema.index == 'range':
            form = 'numpy'  # an unnamed RangeIndex frame takes no bar dict
        rows, index = self._new_bars(k, regime, step)
        bar = build.bars(schema, rows, index, form, pad=2, extra=tuple(self.store.user))
        labels = index
        if schema.index == 'range' and form == 'row':
            labels = [0]  # a Row carries its own label (0 in its one-row source frame)
        elif schema.index == 'range' and not self.store.range_kind:
            labels = None  # a bare RangeIndex bar continues the explicit labels

        def commit(result):
            assert result is self.df, 'append returns the frame itself'
            self.store.append(rows, labels)
            if k:
                self._pend_stale()

        expect = None if self.store.accepts(labels, self.retained(), len(rows)) else ValueError
        self.attempt(lambda: self.df.append(bar), expect_error=expect, commit=commit,
                     label=f'append {form} k={k}')

    @rule(kind=st.sampled_from(BAD_APPENDS), regime=regimes(), multi=st.booleans())
    def append_rejected(self, kind, regime, multi):
        """An append the contract rejects, then the model is unchanged (I8 / I12)."""
        schema = self.store.schema
        rows, index = self._new_bars(2 if multi else 1, regime, '1m')
        if kind == 'dtype':
            # One column carries values of another dtype that its own cannot hold
            # exactly (an int past float64's exact range, a fraction for an int), so
            # the lossless append conversion raises (contract #20): any column, the rows
            # are checked whole before the first one grows.
            if not self.store.accepts(index, self.retained(), len(rows)):
                return  # the labels are refused first (the index is checked before values)
            numeric = [c for c in schema.columns if schema.dtype(c) in ('float64', 'float32', 'int64', 'int32')]
            col = numeric[regime[1] % len(numeric)]
            # built with the lossy dtype directly, so no assignment can round it first
            lossy, lossy_dtype = ((2**53 + 1, 'int64') if schema.dtype(col) in ('float64', 'float32')
                                  else (0.5, 'float64'))
            rows = [{**r, col: lossy} for r in rows]
            bar = build.frame(schema, rows, index, dtypes={col: lossy_dtype})
            self.attempt(lambda: self.df.append(bar), expect_error=TypeError, label=f'append dtype {col}')
            return
        if kind == 'extra_column':
            bar = build.frame(schema, rows, index)
            bar['zz_extra'] = np.zeros(len(bar))
            self.attempt(lambda: self.df.append(bar), expect_error=ValueError, label='append extra column')
            return
        if kind == 'stale_write':
            if not self.stale or not len(self.df):
                return
            col = schema.columns[0]
            self.attempt(lambda: self.df.iloc.__setitem__((0, 0), 1.0), expect_error=ValueError,
                         label=f'iloc write {col} while stale')
            return
        if not contracts.APPEND_FOREIGN_INDEX_RAISES:
            return
        if kind == 'str_index':
            plain = build.frame(schema, rows, None)
            bar = volas.DataFrame({'k': [f'k{i}' for i in range(len(plain))],
                                   **{c: plain.get_column(c) for c in plain.columns}}).set_index('k')
            self.attempt(lambda: self.df.append(bar), expect_error=ValueError, label='append str index')
            return
        if schema.index == 'range':
            if kind != 'index_kind':
                return
            bar = build.frame(schema, rows, next_timestamps(None, len(rows), '1m', 0, EPOCHS[1]))
        elif kind == 'index_kind':
            bar = build.frame(schema, rows, None)
        elif kind == 'naive_onto_aware':
            if schema.tz is None:
                return
            bar = build.frame(type(schema)(schema.price, schema.volume, schema.extras, 'datetime', None),
                              rows, index)
        elif kind == 'aware_onto_naive':
            if schema.tz is not None:
                return
            bar = build.frame(type(schema)(schema.price, schema.volume, schema.extras, 'datetime', 'UTC'),
                              rows, index)
        else:  # nat
            bar = build.frame(schema, rows, index)
            bar = _nat_index(bar, schema)
        self.attempt(lambda: self.df.append(bar), expect_error=ValueError, label=f'append {kind}')

    # --- writes ------------------------------------------------------------------------------

    @precondition(lambda self: len(self.df) > 0)
    @rule(data=st.data(), via=st.sampled_from(['iloc', 'iat', 'at', 'loc']))
    def write_cell(self, data, via):
        """A positional / label write into a data column (invalidates every cache)."""
        if self.stale or self._positions_carried():
            return  # a stale write is covered by append_rejected('stale_write')
        coarse = self.coarse()
        col = data.draw(st.sampled_from([c for c in coarse.columns
                                         if coarse.dtype(c) in ('float64', 'float32', 'int64', 'int32')]))
        pos = data.draw(st.integers(0, len(self.df) - 1))
        dtype = coarse.dtype(col)
        if dtype in ('float64', 'float32'):
            huge = [] if contracts.avoids('sliding-sum-magnitude') else [1e300]
            value = data.draw(st.sampled_from([np.nan, 0.0, -1.5, 101.25] + huge))
            stored = None if np.isnan(value) else float(np.float32(value)) if dtype == 'float32' else value
            if dtype == 'float32' and value == 1e300:
                return  # out of float32 range: a lossy-write contract of its own
        else:
            value = data.draw(st.sampled_from([0, -7, 12345]))
            stored = value
        label = self.df.index[pos] if via in ('at', 'loc') else None
        if via in ('at', 'loc') and np.count_nonzero(np.asarray(self.df.index) == label) != 1:
            return  # a duplicate label addresses several rows
        if via in ('at', 'loc') and isinstance(label, np.datetime64) and self.df.tz is not None:
            label = volas.Timestamp(int(label.astype('int64')), tz='UTC').tz_convert(self.df.tz)
        j = self.df.columns.index(col)
        target = {'iloc': lambda: self.df.iloc.__setitem__((pos, j), value),
                  'iat': lambda: self.df.iat.__setitem__((pos, j), value),
                  'at': lambda: self.df.at.__setitem__((label, col), value),
                  'loc': lambda: self.df.loc.__setitem__((label, col), value)}[via]

        def commit(_):
            self.store.write(self.visible_start() + pos, col, stored)
            self._invalidate_all()

        self.attempt(target, commit=commit, label=f'{via} write {col}[{pos}]={value!r}')

    def _positions_carried(self) -> bool:
        """Whether a cached position-valued directive (the index family) sits on a
        slice: an invalidating write would recompute it with the slice's origin
        (absolute positions) while a fresh cache on the same slice is relative — an
        undecided history contract, so such writes are not generated."""
        from .inventory import position_valued
        heads = position_valued()
        return self.offset > 0 and any(
            re.search(rf'\b{re.escape(h)}\b', n) for n in self.cached for h in heads)

    def _invalidate_all(self):
        # A write drops every carried state: each cached column recomputes over the
        # rows the frame retains (a slice's history restarts at its first row).
        start = self.visible_start()
        self.cached = {n: Entry(e.name, e.directive, e.env, start) if self.window is None else e
                       for n, e in self.cached.items()}
        self.stale = set(self.cached) if len(self.df) else set()

    @precondition(lambda self: len(self.df) > 0)
    @rule(data=st.data())
    def write_computed(self, data):
        """A cell write into a cached float column: it becomes plain data."""
        if self.stale or not self.cached or self._positions_carried():
            return
        name = data.draw(st.sampled_from(sorted(self.cached)))
        want, error = self.oracle(self.cached[name])
        if error is not None or want.dtype != np.float64:
            return
        pos = data.draw(st.integers(0, len(self.df) - 1))
        j = self.df.columns.index(name)
        # The column keeps the cells it holds (already checked against the reference,
        # within the documented tolerance) as plain data.
        held = self.df.get_column(name).to_numpy()

        def commit(_):
            values = [None] * self.visible_start() + [None if np.isnan(v) else float(v) for v in held]
            values[self.visible_start() + pos] = -3.25
            del self.cached[name]
            self.store.add_column(name, values)
            if self.window is not None:
                self.opaque.add(name)  # its hidden cells stay as the cache held them
            self._invalidate_all()

        self.attempt(lambda: self.df.iloc.__setitem__((pos, j), -3.25), commit=commit,
                     label=f'iloc write cached {name}[{pos}]')

    @rule(data=st.data(), kind=st.sampled_from(['new', 'command', 'canonical', 'existing']))
    def set_column(self, data, kind):
        """``df[name] = values``: a new plain column, one named like a bare command or
        a canonical directive, or a replacement of an existing column."""
        if self.stale or self._positions_carried():
            return
        from .inventory import executable_lines
        if kind == 'new':
            name = data.draw(st.sampled_from(['u0', 'u1']))
        elif kind == 'command':
            name = data.draw(st.sampled_from([ln.head for ln in executable_lines()
                                              if ln.canonical == ln.head and '.' not in ln.head]))
        elif kind == 'canonical':
            name = data.draw(st.sampled_from([ln.canonical for ln in executable_lines()][:40]))
        else:
            name = data.draw(st.sampled_from(list(self.df.columns)))
        bare = {ln.head for ln in executable_lines() if ln.canonical == ln.head and '.' not in ln.head}
        if name in bare and any(name in stored_bare_names(key) for key in self.cached if key != name):
            if contracts.COLUMN_UNDER_A_CACHED_COMMAND_RAISES:
                self.attempt(lambda: self.df.__setitem__(name, np.zeros(len(self.df))),
                             expect_error=ValueError, label=f'df[{name!r}] = ... under a cached command')
            return
        if name in self.coarse().schema.columns and self.coarse().dtype(name) not in ('float64',):
            return
        was_cached = name in self.cached
        values = np.round(np.linspace(-1, 1, len(self.df)) * data.draw(st.integers(1, 9)), 3)

        existing = name in self.df.columns

        def commit(_):
            stored = [None] * self.visible_start() + [float(v) for v in values]
            if self.window is not None and was_cached:
                self.opaque.add(name)  # its hidden cells stay as the cache held them
            if name in self.cached:
                del self.cached[name]
            if name in self.coarse().schema.columns:
                for i, v in enumerate(stored[self.visible_start():]):
                    self.store.write(self.visible_start() + i, name, v)
            else:
                self.store.add_column(name, stored)
            if existing:
                self._invalidate_all()  # a replaced column may feed any cache
            # A brand-new column leaves every cache as it is (each keeps its binding).

        self.attempt(lambda: self.df.__setitem__(name, values), commit=commit, label=f'df[{name!r}] = ...')

    # --- derived frames ---------------------------------------------------------------------

    @precondition(lambda self: len(self.df) > 1)
    @rule(data=st.data())
    def slice_continue(self, data):
        """``df = df.iloc[a:]`` and keep going on the slice (I11)."""
        if self.window is not None:
            return
        if contracts.avoids('slice-recursive-rewarm') and any(
                _names(e.name) & inventory.unbounded_memory() or inventory.recursive(e.name)
                for e in self.cached.values()):
            return  # a recursive cached column is not continued across the slice
        a = data.draw(st.integers(1, len(self.df) - 1))
        if self.stale:
            self.attempt(lambda: self.df.iloc[a:], expect_error=ValueError, label='slice while stale')
            return
        # A column the slice is too short to keep computed stays as the plain data it
        # holds (fresh here: a slice of a stale frame raises).
        held = {n: (observe.column_values_series(s), s.dtype)
                for n, s in ((n, self.df.get_column(n)) for n in self.cached)}
        start = self.visible_start()

        def commit(result):
            self.df = result
            self.offset += a
            self.store.range_kind = False
            new_len = len(self.df)
            for name in list(self.cached):
                if contracts.SLICE_CACHED_COLUMNS_CONTINUE and volas.directive_lookback(name) <= new_len:
                    continue
                values, dtype = held[name]
                del self.cached[name]
                cells = [None] * start + [None if (isinstance(v, float) and np.isnan(v)) else v
                                          for v in values.tolist()]
                self.store.add_column(name, cells, 'bool' if dtype == 'bool' else 'float64')

        self.attempt(lambda: self.df.iloc[a:], commit=commit, label=f'iloc[{a}:]')

    @precondition(lambda self: self.snapshots)
    @rule(data=st.data())
    def append_snapshot(self, data):
        """Append another frame (a snapshot): rejected while that source is stale."""
        src_df, src_store, src_cached, src_stale, src_offset = self.snapshots[
            data.draw(st.integers(0, len(self.snapshots) - 1))]
        if self.window is not None:
            return  # a windowed source's hidden margin: an open contract question
        if src_store.schema != self.store.schema or src_store.user != self.store.user:
            return
        if set(src_df.columns) - set(self.df.columns):
            self.attempt(lambda: self.df.append(src_df), expect_error=ValueError, label='append snapshot (extra)')
            return
        if contracts.APPEND_CACHED_COLUMNS_RAISES and any(c in self.cached for c in src_df.columns):
            self.attempt(lambda: self.df.append(src_df), expect_error=ValueError,
                         label='append snapshot (cached columns)')
            return
        if any(c in src_cached for c in src_df.columns):
            return
        if src_stale:
            if contracts.APPEND_STALE_SOURCE_RAISES:
                self.attempt(lambda: self.df.append(src_df), expect_error=ValueError,
                             label='append stale snapshot')
            return
        rows = [dict(r) for r in src_store.rows[src_offset:]]
        # A source still on its RangeIndex is a bare RangeIndex bar (labels continue);
        # any other source carries its own labels.
        bare = self.store.schema.index == 'range' and src_store.range_kind
        labels = None if bare else list(src_store.index[src_offset:])

        def commit(_):
            self.store.append(rows, labels)
            if rows:
                self._pend_stale()

        expect = None if self.store.accepts(labels, self.retained(), len(rows)) else ValueError
        self.attempt(lambda: self.df.append(src_df), expect_error=expect, commit=commit,
                     label='append snapshot')

    @precondition(lambda self: self.snapshots)
    @rule(data=st.data())
    def compare_snapshot(self, data):
        """``df == other`` reads both frames: it raises while either is stale."""
        other, other_store, _, other_stale, _ = self.snapshots[data.draw(st.integers(0, len(self.snapshots) - 1))]
        if other.columns != self.df.columns or len(other) != len(self.df) or \
                other_store.range_kind != self.store.range_kind or \
                not np.array_equal(np.asarray(other.index), np.asarray(self.df.index)):
            return  # differently-labelled (or differently-kinded) operands are rejected by design
        stale = bool(self.stale or other_stale) and contracts.COMPARE_STALE_RHS_RAISES
        before = observe.raw_state(other)

        def commit(result):
            assert result.shape == self.df.shape
            assert observe.diff_state(before, observe.raw_state(other)) == [], 'comparison mutated its operand'

        self.attempt(lambda: self.df == other, expect_error=ValueError if stale else None, commit=commit,
                     label='df == snapshot')


def _names(text: str) -> set[str]:
    """The command names a directive mentions (lower-cased, sub-commands dropped)."""
    return {m.split('.')[0] for m in re.findall(r'[a-z_][a-z_0-9.]*', text.lower())}


def _nat_index(df: volas.DataFrame, schema) -> volas.DataFrame:
    labels = np.asarray(df.index).astype('datetime64[ns]').copy()
    labels[-1] = np.datetime64('NaT')
    out = volas.DataFrame({c: df.get_column(c) for c in df.columns}, index=labels)
    return out._set_index_tz(schema.tz) if schema.tz else out


__all__ = ['PlainMachine', 'NS']
