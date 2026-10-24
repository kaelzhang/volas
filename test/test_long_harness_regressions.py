"""Deterministic operation traces reduced by the release differential harness."""

import numpy as np
import pytest
import volas

from .exact import assert_exact
from .harness.data import Schema, EPOCHS
from .harness.directives import Directive
from .harness.machine_plain import PlainMachine
from .harness.machine_tf import TfMachine
from .harness.machine_window import WindowedMachine, TfWindowedMachine
from .test_harness_column_writes import _Draws


def schema(index='datetime'):
    return Schema('float64', 'float64', (), index, None)


def cache(machine, text, kind='num'):
    machine.cache(_Draws(Directive(text, kind, commands=1)))
    machine.consistent()


@pytest.mark.parametrize('text', [
    'style.onneck@(-(accbands@open+bop))',
    '(open)<=((open+accbands@open)*(TYPPRICE))',
])
def test_bound_expression_data_preserves_grouping(text):
    df = volas.DataFrame({'open': [2.0] * 40, 'high': [4.0] * 40,
                          'low': [1.0] * 40, 'close': [3.0] * 40, 'volume': [5.0] * 40})
    name = 'accbands@open+bop' if 'onneck' in text else 'open+accbands@open'
    df[name] = np.arange(40, dtype=float)
    result = df[text]
    assert result.name == volas.directive_stringify(text)
    assert_exact(result.to_numpy(dtype='object'), df.exec(result.name, dtype='object'))


@pytest.mark.parametrize('text,kind', [('(open)+(ACCBANDS)', 'num'), ('(open)<=(ACCBANDS)', 'bool')])
def test_harness_reparse_keeps_physical_command_owner(text, kind):
    machine = TfMachine()
    machine.init(schema(), '5m', 'cumulate', 1, ('walk', 0), EPOCHS[0], [False] * 30)
    cache(machine, 'ACCBANDS')
    machine.write_computed(_Draws('accbands', 0))
    cache(machine, text, kind)
    machine.teardown()


def test_harness_late_request_binds_physical_command_owner():
    machine = WindowedMachine()
    machine.init(schema('range'), 0, ('walk', 0), '1m', EPOCHS[0], 1, 0)
    machine.set_column(_Draws('accbands', 1), 'command')
    cache(machine, '(open)+(ACCBANDS)')
    machine.teardown()


def test_ttm_squeeze_slice_write_keeps_original_warmup():
    machine = PlainMachine()
    machine.init(schema(), 0, ('walk', 0), '1m', EPOCHS[0])
    for _ in range(3):
        machine.append(1, 'arrow', ('walk', 0), '1m')
    cache(machine, 'ACCBANDS@open')
    machine.slice_continue(_Draws(1))
    cache(machine, 'TTM_SQUEEZE:2')
    machine.slice_continue(_Draws(1))
    machine.write_cell(_Draws('open', 0, np.nan), 'iloc')
    machine.consistent()
    machine.teardown()


def test_candle_window_missing_operand_matches_full_history():
    machine = WindowedMachine()
    machine.init(schema('range'), 8, ('walk', 0), '1m', EPOCHS[0], 1, 7)
    machine.set_column(_Draws('iii', 1), 'command')
    cache(machine, '(0)+(style.gapsidesidewhite@iii)')
    machine.teardown()


def test_candle_slice_write_matches_full_history():
    machine = PlainMachine()
    machine.init(schema(), 13, ('walk', 0), '1m', EPOCHS[0])
    cache(machine, 'ACCBANDS@open')
    machine.slice_continue(_Draws(1))
    cache(machine, 'AD@open')
    cache(machine, 'style.2crows@open')
    machine.slice_continue(_Draws(1))
    machine.write_computed(_Draws('ad@open', 0))
    machine.consistent()
    machine.teardown()


def test_harness_cached_candle_dependency_refuses_data_shadow():
    machine = PlainMachine()
    machine.init(schema(), 0, ('walk', 0), '1m', EPOCHS[0])
    cache(machine, 'AD@open')
    cache(machine, 'style.2crows@open')
    cache(machine, '(open)+(style.2crows@open)')
    machine.set_column(_Draws('style.2crows', 1), 'existing')
    machine.consistent()
    machine.teardown()


def test_harness_forming_write_preserves_explicit_positive_zero():
    machine = TfWindowedMachine()
    machine.init(schema(), '5m', 'construct', 12, ('walk', 0), EPOCHS[0], 2, 11)
    cache(machine, '(open)*(-(style.onneck))')
    machine.write_computed(_Draws('open*-style.onneck', 0))
    machine.consistent()
    machine.write_cell(_Draws('open*-style.onneck', 0.0, 7.0), 'iloc', True)
    machine.consistent()
    machine.teardown()


@pytest.mark.parametrize('text,lookback,history', [
    ('(open)+(ACCBANDS)', 0, 1),
    ('ema:2@(ACCBANDS)', 1, 3),
    ('ma:3@(open+accbands)', 2, 2),
    ('open//ACCBANDS', 1, 1),
])
def test_harness_facts_follow_physical_binding(text, lookback, history):
    import volas_rs
    facts = volas_rs._directive_facts(text, columns=['open', 'ACCBANDS'])
    assert facts['lookback'] == lookback
    assert facts['history_class'] == history
    assert not facts['shadows']


@pytest.mark.parametrize('value', [0.0, -0.0])
def test_unary_negative_zero_is_bit_exact(value):
    df = volas.DataFrame({'open': [1.0, 2.0], 'style.onneck': [value, value]})
    assert_exact(df['open*-style.onneck'].to_numpy(), np.array([-value, -value * 2]))


def test_candle_missing_operand_is_unknown_until_window_recovers():
    n = 20
    df = volas.DataFrame({'open': [2.] * n, 'high': [4.] * n,
                          'low': [1.] * n, 'close': [3.] * n, 'volume': [5.] * n})
    df['iii'] = [np.nan] * 7 + [2.] * (n - 7)
    values = df['0+style.gapsidesidewhite@iii'].to_numpy()
    assert np.isnan(values[:14]).all()
    assert np.isfinite(values[14:]).all()


def test_cog_short_slice_last_row_resend_keeps_original_reach():
    # Pinned minimal trace from the original slice/re-send counterexample.
    # The first Row(label=0) appends; the second replaces that latest label.
    x = np.arange(10, dtype=float) + 100
    def frame(a, b):
        return volas.DataFrame({'open': x[a:b]}, index=list(range(a, b)))
    df = frame(0, 4).iloc[1:]
    df.append(frame(4, 8))
    df.append(volas.DataFrame({'open': [108.0]}).iloc[0])
    df['COG:8@open']
    df = df.iloc[1:]
    df.append(volas.DataFrame({'open': [109.0]}).iloc[0])
    got = df['COG:8@open'].to_numpy()
    want = volas.DataFrame({'open': np.r_[x[1:8], 109.0]})['COG:8@open'].to_numpy()[1:]
    assert np.isfinite(want[-1])
    assert_exact(got, want)
