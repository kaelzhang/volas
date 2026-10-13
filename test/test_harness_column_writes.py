"""Column-write events in the ledger preserve data outside the visible window."""

import pytest

from .harness.data import Schema, EPOCHS
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
