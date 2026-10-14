"""Column-write events in the ledger preserve data outside the visible window."""

import pytest
import numpy as np

from .harness.data import Schema, EPOCHS
from .harness.ledger import Ledger, TfLedger
from .harness.machine_window import WindowedMachine

from test.exact import assert_exact


class _Draws:
    def __init__(self, *answers):
        self.answers = iter(answers)

    def draw(self, _strategy):
        return next(self.answers)


@pytest.mark.parametrize('kind', ['new', 'existing'])
@pytest.mark.parametrize('directive', ['cdp@open+donchian:2@u0', 'pivot_points.r1@u0'])
def test_replacing_a_user_column_preserves_the_ledgers_hidden_history(directive, kind):
    machine = WindowedMachine()
    schema = Schema('float64', 'float64', (), 'range', None)
    machine.init(schema, 1, ('walk', 0), '1m', EPOCHS[0], 1, 1)
    machine.set_column(_Draws('u1', 1), 'new')
    machine.set_column(_Draws('u0', 1), 'new')
    machine.append(1, 'arrow', ('walk', 0), '1m')
    assert_exact(machine.df[directive].to_numpy(),
                                  machine.store.frame()[directive].to_numpy()[-1:])
    machine.set_column(_Draws('u0', 1), kind)
    assert machine.store.rows[0]['u0'] == -1.0
    want = machine.store.frame()[directive].to_numpy()[-1:]
    assert_exact(machine.df[directive].to_numpy(), want)
    machine.teardown()


@pytest.mark.parametrize('tf', [False, True], ids=['plain', 'tf'])
@pytest.mark.parametrize('name', ['flag', 'u0', 'close'])
def test_ledger_column_replacement_adopts_dtype_and_preserves_unwritten_rows(tf, name):
    schema = Schema('float64', 'float64', (('flag', 'bool'),), 'range', None)
    rows = [dict(open=1.0, high=2.0, low=0.0, close=1.0, volume=3.0,
                 flag=True, u0=True) for _ in range(3)]
    if tf:
        ledger = TfLedger(schema, '5m', seed_index=[0, 1, 2], seed_rows=rows,
                          user={'u0': 'bool'})
    else:
        ledger = Ledger(schema, rows, [0, 1, 2], {'u0': 'bool'})
    ledger.set_column(name, [-1.5, 1.25], start=1, dtype='float64')
    coarse = ledger.coarse()
    assert coarse.dtype(name) == 'float64'
    assert coarse.columns.count(name) == 1
    assert coarse.rows[0][name] == 1.0
    assert_exact(coarse.frame().get_column(name).to_numpy(), [1.0, -1.5, 1.25])
    snapshot = ledger.copy().coarse()
    assert snapshot.dtype(name) == 'float64'
    coarse.write(1, name, 0.5)
    assert coarse.dtype(name) == 'float64'  # cell writes keep storage dtype
    coarse.set_column(name, [2, 3, 4], dtype='int32')
    assert coarse.dtype(name) == 'int32'
    assert_exact(coarse.frame().get_column(name).to_numpy(), np.array([2, 3, 4], dtype='int32'))


def test_cached_bool_slice_projected_to_data_then_replaced_with_floats():
    from .harness.directives import Directive
    from .harness.machine_plain import PlainMachine

    machine = PlainMachine()
    schema = Schema('float64', 'float64', (), 'range', None)
    machine.init(schema, 3, ('walk', 0), '1m', EPOCHS[0])
    machine.cache(_Draws(Directive('close>0', 'bool')))
    name = next(iter(machine.cached))
    held = machine.store.frame().exec('close>0').tolist()
    machine.slice_continue(_Draws(1))
    # Contiguous slices continue caches on main; a projection explicitly turns
    # the cached values into plain data under the same name.
    machine.df = machine.df[list(machine.df.columns)]
    machine.store.add_column(name, held, 'bool')
    machine.cached.clear()
    assert not machine.df.is_computed(name)
    assert machine.store.dtype(name) == 'bool'
    machine.set_column(_Draws(name, 1), 'existing')
    assert machine.df.dtypes[name] == 'float64'
    assert machine.store.dtype(name) == 'float64'
    assert_exact(machine.store.frame().get_column(name).to_numpy()[1:], [-1.0, 1.0])
    machine.consistent()
    machine.teardown()
