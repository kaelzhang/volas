"""Regression contracts for period one binding."""

import pytest

import volas


@pytest.mark.parametrize('directive', ['chop:1', 'ttm_squeeze:1,-5.0,-5.0', 'ttm_squeeze.on:1'])
@pytest.mark.parametrize('boundary', ['cache', 'exec', 'lookback', 'stringify', 'declare'])
def test_degenerate_period_one_is_rejected_at_binding(directive, boundary):
    data = {'high': [3., 4., 5.], 'low': [1., 2., 3.], 'close': [2., 3., 4.]}
    frame = volas.DataFrame(data)
    with pytest.raises(volas.DirectiveValueError, match=r'argument #0 must be an integer >= 2'):
        if boundary == 'cache':
            frame[directive]
        elif boundary == 'exec':
            frame.exec(directive)
        elif boundary == 'lookback':
            volas.directive_lookback(directive)
        elif boundary == 'stringify':
            volas.directive_stringify(directive)
        else:
            volas.DataFrame(data, window=1, max_lookback=[directive])

