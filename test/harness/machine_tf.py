"""The tf-aware (live fold) machine: fine bars folded into a forming coarse bar,
checked against an independent re-aggregation of the ledger of fine bars."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.stateful import initialize, rule

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
                n=st.integers(1, 30), regime=regimes(), epoch=st.sampled_from(EPOCHS))
    def init(self, schema, tf, mode, n, regime, epoch):
        self.setup(schema, tf, mode, n, regime, epoch)

    def setup(self, schema, tf, mode, n, regime, epoch):
        self.store = TfLedger(schema, tf)
        self.pin_scale(regime)
        steered, scale = self.steer(regime)
        rows = make_rows(schema, steered, n, None, scale)
        step = TF_NS[tf] * NS
        start = period_start(epoch, tf, schema.tz)
        if mode == 'cumulate':
            ts = [start + i * 60 * NS + 7 for i in range(n)]
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
          form=st.sampled_from(FOLD_FORMS), regime=regimes())
    def fold(self, k, moves, form, regime):
        """Fold fine bars: same period, re-sent, next period, after a gap, sub-second."""
        schema = self.store.schema
        if form in ('row', 'dict'):
            k = 1
        steered, scale = self.steer(regime)
        prev = [r for _, r in self.store.fine] or self.store.seed_rows
        rows = make_rows(schema, steered, k, last_close(prev), scale)
        ts, last = [], self._last_ts()
        for i in range(k):
            last = self._next(last, moves[i], after_seed=i == 0 and not self.store.fine)
            ts.append(last)
        bar = build.bars(schema, rows, ts, form, pad=2)

        def commit(_):
            for t, r in zip(ts, rows):
                self.store.add(t, r)
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
            if contracts.avoids('append-atomicity'):
                return  # a fold that fails part-way leaves the forming state changed
            # a value the close column cannot hold exactly (the lossless conversion raises)
            bar = build.frame(schema, rows, ts)
            bar['close'] = (np.full(len(bar), 2**53 + 1, dtype='int64') if schema.dtype('close') in ('float64', 'float32')
                            else np.full(len(bar), 0.5))
        expect = TypeError if kind == 'dtype' else ValueError
        self.attempt(lambda: self.df.append(bar), expect_error=expect, label=f'fold rejected {kind}')


__all__ = ['TfMachine']
