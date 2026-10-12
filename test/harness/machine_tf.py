"""The tf-aware (live fold) machine: fine bars folded into a forming coarse bar,
checked against an independent re-aggregation of the ledger of fine bars — fine bars
with re-sent timestamps, bars leaving out a data column, and writes to the coarse rows
(the forming one included) among the ops."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.stateful import initialize, precondition, rule

import volas

from . import build, contracts
from .data import EPOCHS, NS, Schema, make_rows, regimes
from .ledger import TF_NS, TfLedger, last_close, period_start
from .machine_base import FrameMachine

TFS = ('5m', '15m', '1h')
FOLD_FORMS = ('arrow', 'arrow', 'numpy', 'row', 'dict', 'sliced')
# How the next fine bar's time relates to the last one.
MOVES = ('same', 'same', 'same', 'resend', 'next', 'gap', 'sub')


@st.composite
def tf_schemas(draw) -> Schema:
    volume = draw(st.sampled_from(['float64', 'float64', 'int64']))
    extras = (('x', 'float64'),) if draw(st.booleans()) else ()
    tz = draw(st.sampled_from([None, 'UTC', 'Asia/Shanghai']))
    return Schema('float64', volume, extras, 'datetime', tz)


class TfMachine(FrameMachine):

    def wrap(self, df: volas.DataFrame) -> volas.DataFrame:
        """Hook for the windowed subclass: ``df`` is already tf-aware (its forming bar
        included); the subclass re-wraps it with a window."""
        return df

    @initialize(schema=tf_schemas(), tf=st.sampled_from(TFS), mode=st.sampled_from(['cumulate', 'construct']),
                n=st.integers(1, 30), regime=regimes(), epoch=st.sampled_from(EPOCHS),
                resent=st.lists(st.booleans(), min_size=30, max_size=30))
    def init(self, schema, tf, mode, n, regime, epoch, resent):
        self.setup(schema, tf, mode, n, regime, epoch, resent)

    def setup(self, schema, tf, mode, n, regime, epoch, resent=()):
        self.store = TfLedger(schema, tf)
        self.pin_scale(regime)
        steered, scale = self.steer(regime)
        rows = make_rows(schema, steered, n, None, scale)
        step = TF_NS[tf] * NS
        start = period_start(epoch, tf, schema.tz)
        if mode == 'cumulate':
            # the fed bars may re-send a timestamp (consecutive equal stamps: last wins)
            ts = [start + 7]
            for i in range(1, n):
                ts.append(ts[-1] if i < len(resent) and resent[i] else ts[-1] + 60 * NS)
            for t, r in zip(ts, rows):
                self.store.add(t, r)
            self.df = self.wrap(build.frame(schema, rows, ts).cumulate(tf))
        else:
            labels = [start + i * step for i in range(n)]
            self.store.seed_index, self.store.seed_rows = labels, rows
            self.df = self.wrap(volas.DataFrame(build.frame(schema, rows, labels), time_frame=tf))
        self.log(f'init {mode} {tf} {schema} n={n} {regime}')

    def _last_ts(self) -> int:
        st_ = self.store
        return st_.fine[-1][0] if st_.fine else st_.seed_index[-1]

    def _next(self, last: int, move: str, after_seed: bool = False) -> int:
        step = TF_NS[self.store.tf] * NS
        if after_seed and not contracts.TF_SEED_LAST_PERIOD_GENERATED:
            # after closed seed bars, start in a later period
            return period_start(last, self.store.tf, self.store.schema.tz) + step + 60 * NS * (move == 'same')
        if move == 'resend':
            return last
        if move == 'same':
            return last + 60 * NS
        if move == 'sub':
            return last + NS // 3
        if move == 'next':
            return period_start(last, self.store.tf, self.store.schema.tz) + step
        return period_start(last, self.store.tf, self.store.schema.tz) + 3 * step + 120 * NS

    @rule(k=st.sampled_from([1, 1, 1, 2, 4]), moves=st.lists(st.sampled_from(MOVES), min_size=4, max_size=4),
          form=st.sampled_from(FOLD_FORMS), regime=regimes(),
          omit=st.sampled_from([(), (), (), ('volume',), ('high',), ('close',), ('x',)]))
    def fold(self, k, moves, form, regime, omit):
        """Fold fine bars: same period, re-sent, next period, after a gap, sub-second —
        possibly leaving a data column out (it is a missing value of the bar)."""
        schema = self.store.schema
        omit = tuple(c for c in omit if c in schema.columns)
        if form in ('row', 'dict'):
            k = 1
        steered, scale = self.steer(regime)
        prev = [r for _, r in self.store.fine] or self.store.seed_rows
        rows = make_rows(schema, steered, k, last_close(prev), scale)
        ts, last = [], self._last_ts()
        for i in range(k):
            last = self._next(last, moves[i], after_seed=i == 0 and not self.store.fine)
            ts.append(last)
        bar = build.bars(schema, rows, ts, form, pad=2, omit=omit)

        def commit(_):
            for t, r in zip(ts, rows):
                self.store.add(t, {c: v for c, v in r.items() if c not in omit})
            if self.cached:
                self.stale |= set(self.cached)

        self.attempt(lambda: self.df.append(bar), commit=commit, label=f'fold {form} {moves[:k]}')

    @rule(kind=st.sampled_from(['out_of_order', 'nat', 'range_index', 'extra_column', 'dtype']),
          regime=regimes())
    def fold_rejected(self, kind, regime):
        """A fold the contract rejects: the frame, its forming bar and its caches are
        untouched, and the next valid fold behaves as if it never happened (I8)."""
        schema = self.store.schema
        rows = make_rows(schema, regime, 2, None)
        last = self._last_ts()
        after_seed = not self.store.fine
        ts = [self._next(last, 'same', after_seed), self._next(last, 'next', after_seed) + 60 * NS]
        if kind == 'out_of_order':
            if not self.store.fine:
                return
            ts = [last - 60 * NS, last + 60 * NS]
            ts.sort(reverse=True)  # the second bar precedes the first: validated before any fold
            bar = build.frame(schema, rows, ts)
        elif kind == 'nat':
            bar = build.frame(schema, rows[:1], ts[:1])
            from .machine_plain import _nat_index
            bar = _nat_index(bar, schema)
        elif kind == 'range_index':
            bar = build.frame(schema, rows, None)
        elif kind == 'extra_column':
            bar = build.frame(schema, rows, ts)
            bar['zz_extra'] = [0.0] * len(bar)
        else:
            # a value the close column cannot hold exactly (the lossless conversion raises)
            bar = build.frame(schema, rows, ts)
            bar['close'] = (np.full(len(bar), 2**53 + 1, dtype='int64') if schema.dtype('close') in ('float64', 'float32')
                            else np.full(len(bar), 0.5))
        expect = TypeError if kind == 'dtype' else ValueError
        self.attempt(lambda: self.df.append(bar), expect_error=expect, label=f'fold rejected {kind}')


    # --- writes (docs/cumulation.md, "Writing to a time-frame frame") ----------------------

    def _writable(self) -> bool:
        # a stale frame refuses writes (the plain machine covers it); a windowed frame's
        # writes restart carried state on the retained rows only while that is open
        return not self.stale and not (self.window is not None and contracts.avoids('windowed-drift'))

    def _wrote(self) -> None:
        if self.cached:
            self.stale |= set(self.cached)

    @precondition(lambda self: len(self.df) > 0)
    @rule(data=st.data(), via=st.sampled_from(['iloc', 'iat', 'at', 'loc', 'iloc_twice', 'loc_twice']),
          forming=st.booleans())
    def write_cell(self, data, via, forming):
        """A cell write into a data column of a closed row or of the forming row — also
        twice to the same cell in one write, where the last value is the one left."""
        if not self._writable():
            return
        coarse = self.coarse()
        col = data.draw(st.sampled_from([c for c in coarse.columns if coarse.dtype(c) in ('float64', 'int64')]))
        last = len(self.df) - 1
        pos = last if forming else data.draw(st.integers(0, last))
        value = data.draw(st.sampled_from([0.0, -1.5, 101.25, np.nan] if coarse.dtype(col) == 'float64'
                                          else [0, -7, 12345]))
        stored = None if isinstance(value, float) and np.isnan(value) else value
        j = self.df.columns.index(col)
        label = self.df.index[pos]
        if self.df.tz is not None:
            label = volas.Timestamp(int(np.datetime64(label, 'ns').astype('int64')), tz='UTC').tz_convert(self.df.tz)
        first = data.draw(st.sampled_from([7.0, 2.5] if coarse.dtype(col) == 'float64' else [3, 9]))
        target = {'iloc_twice': lambda: self.df.iloc.__setitem__(([pos, pos], j), [first, value]),
                  'loc_twice': lambda: self.df.loc.__setitem__(([label, label], col), [first, value]),
                  'iloc': lambda: self.df.iloc.__setitem__((pos, j), value),
                  'iat': lambda: self.df.iat.__setitem__((pos, j), value),
                  'at': lambda: self.df.at.__setitem__((label, col), value),
                  'loc': lambda: self.df.loc.__setitem__((label, col), value)}[via]

        def commit(_):
            self.store.write({pos: {col: stored}})
            self._wrote()

        refused = pos == last and self.store.refuses(col, stored)
        self.attempt(target, expect_error=ValueError if refused else None, commit=commit,
                     label=f'{via} write {col}[{pos}]={value!r}')

    @precondition(lambda self: len(self.df) > 0)
    @rule(data=st.data(), name=st.sampled_from(['u0', 'u1', 'close', 'x']))
    def set_column(self, data, name):
        """``df[name] = values``: a new plain float column, or a float data column replaced."""
        if not self._writable() or self.window is not None:
            return
        coarse = self.coarse()
        if name in coarse.columns and coarse.dtype(name) != 'float64':
            return
        if name in ('close', 'x') and name not in coarse.columns:
            return
        values = np.round(np.linspace(-1, 1, len(self.df)) * data.draw(st.integers(1, 9)), 3)
        existing = name in self.df.columns

        def commit(_):
            self.store.write({i: {name: float(v)} for i, v in enumerate(values)},
                             columns=None if existing else {name: 'float64'})
            if existing:
                self._wrote()  # a replaced column may feed any cache

        refused = existing and self.store.refuses(name, float(values[-1]))
        self.attempt(lambda: self.df.__setitem__(name, values), expect_error=ValueError if refused else None,
                     commit=commit, label=f'df[{name!r}] = ...')

    @precondition(lambda self: len(self.df) > 0)
    @rule(data=st.data())
    def write_computed(self, data):
        """A cell write into a cached float column: it becomes plain data the fine bars
        never carried, so they hold it as missing."""
        if not self._writable() or not self.cached:
            return
        name = data.draw(st.sampled_from(sorted(self.cached)))
        want, error = self.oracle(self.cached[name])
        if error is not None or want.dtype != np.float64:
            return
        pos = data.draw(st.integers(0, len(self.df) - 1))
        j = self.df.columns.index(name)
        held = self.df.get_column(name).to_numpy()

        def commit(_):
            cells = {i: {name: None if np.isnan(v) else float(v)} for i, v in enumerate(held)}
            cells[pos] = {name: -3.25}
            del self.cached[name]
            self.store.write(cells, columns={name: 'float64'})
            self._wrote()

        self.attempt(lambda: self.df.iloc.__setitem__((pos, j), -3.25), commit=commit,
                     label=f'iloc write cached {name}[{pos}]')


__all__ = ['TfMachine']
