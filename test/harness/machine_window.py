"""Bounded rolling-window machines (plain and tf): the visible rows must equal the
last ``window`` rows of an unbounded history fed the same events (I11), and the
``ready`` / margin bookkeeping must follow the documented rule."""

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.stateful import initialize, invariant

import volas

from . import build
from .data import EPOCHS, regimes, schemas, steps
from .machine_plain import PlainMachine
from .machine_tf import TFS, TfMachine, tf_schemas

WINDOWS = st.integers(1, 12)
MARGINS = st.integers(0, 30)


class _Windowed:
    """The window bookkeeping shared by both windowed machines."""

    window: int
    max_lookback: int

    @invariant()
    def window_bookkeeping(self):
        if self.df is None:
            return
        total = len(self.coarse().rows)
        assert len(self.df) == min(total, self.window), f'{len(self.df)} visible rows of {total}'
        physical = self.df._physical_height
        assert min(total, self.window + self.max_lookback) <= physical <= total, \
            f'physical {physical} outside [{min(total, self.window + self.max_lookback)}, {total}]'
        assert self.df.ready == (total >= self.window + self.max_lookback), \
            f'ready={self.df.ready} with {total} rows, window {self.window} + margin {self.max_lookback}'


class WindowedMachine(_Windowed, PlainMachine):

    def retained(self):
        return self.df._physical_height

    def build_frame(self, schema, rows, index):
        return volas.DataFrame(build.frame(schema, rows, index), window=self.window,
                               max_lookback=self.max_lookback)

    @initialize(schema=schemas(), n=st.integers(0, 40), regime=regimes(), step=steps(),
                epoch=st.sampled_from(EPOCHS), window=WINDOWS, margin=MARGINS)
    def init(self, schema, n, regime, step, epoch, window, margin):
        self.window, self.max_lookback = window, margin
        self.setup(schema, n, regime, step, epoch)
        self.log(f'window={window} margin={margin}')


class TfWindowedMachine(_Windowed, TfMachine):

    def wrap(self, df):
        # Without `time_frame=` the tf frame keeps its forming bar (re-stating the
        # frame would close it; see contracts.TF_SEED_LAST_PERIOD_GENERATED).
        return volas.DataFrame(df, window=self.window, max_lookback=self.max_lookback)

    @initialize(schema=tf_schemas(), tf=st.sampled_from(TFS), mode=st.sampled_from(['cumulate', 'construct']),
                n=st.integers(1, 30), regime=regimes(), epoch=st.sampled_from(EPOCHS), window=WINDOWS,
                margin=MARGINS)
    def init(self, schema, tf, mode, n, regime, epoch, window, margin):
        self.window, self.max_lookback = window, margin
        self.setup(schema, tf, mode, n, regime, epoch)
        self.log(f'window={window} margin={margin}')
