"""Regression contracts for window declarations."""

import numpy as np
import pytest

import volas


@pytest.mark.parametrize('column', ['close', 'custom'])
def test_raw_column_declaration_has_zero_lookback_and_survives_compaction(column):
    frame = volas.DataFrame({column: [1.0, 2.0, 3.0]}, window=1, max_lookback=[column])
    assert not frame.is_computed(column)
    assert volas.directive_lookback(column) == 0
    for value in range(4, 12):
        frame.append({column: float(value)})
        assert frame.exec(column).tolist() == [float(value)]
        assert frame[column].to_list() == [float(value)]
        assert frame.ready
    assert frame._physical_height < 4



def test_raw_column_and_indicator_declarations_keep_each_own_name():
    frame = volas.DataFrame({'close': [1., 2., 3.]}, window=1,
                           max_lookback=['close', 'ema:2'])
    frame.append({'close': 4.})
    assert not frame.is_computed('close')
    assert frame.is_computed('ema:2')
    assert frame['ema:2'].to_numpy().view(np.uint64).tolist() == (
        volas.DataFrame({'close': [1., 2., 3., 4.]})['ema:2'].to_numpy()[-1:].view(np.uint64).tolist())



@pytest.mark.parametrize('column', ['ma', 'kdj'])
def test_raw_column_named_like_a_command_requiring_arguments_can_be_declared(column):
    # Without a usable bare command, this name is unambiguously a data column.
    data = {column: [1., 2., 3.]}
    assert volas.DataFrame(data).exec(column).tolist() == [1., 2., 3.]
    frame = volas.DataFrame(data, window=1, max_lookback=[column])
    frame.append({column: 4.})
    assert frame.exec(column).tolist() == [4.]
    assert not frame.is_computed(column)


@pytest.mark.parametrize('name', ['change', 'ema:2'])
def test_window_admission_uses_physical_directive_data_bindings(name):
    frame = volas.DataFrame({name: [1., 2., 3.]}, window=1, max_lookback=0)
    for spelling in (name, name.upper()):
        assert frame.exec(spelling).tolist() == [3.]
        assert frame[f'{spelling}+0'].to_list() == [3.]
        assert frame.exec(f'hhv:1@({spelling})').tolist() == [3.]
