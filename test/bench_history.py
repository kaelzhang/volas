"""Million-row history ownership boundaries and their immediate / warm continuation."""
import numpy as np
import pytest
from volas import DataFrame


DIRECTIVES = ['ema:12', 'ema:5@(ma:200)']
OPERATIONS = ['tail', 'iloc', 'window']


def source(directive):
    parent = DataFrame({'close': 100 + np.sin(np.arange(1_048_576) * .01)})
    parent[directive]
    return parent


def derive(parent, directive, operation):
    if operation == 'tail':
        return parent.tail(5)
    if operation == 'iloc':
        return parent.iloc[524_283:524_288]
    return DataFrame(parent, window=5, max_lookback=[directive])


@pytest.mark.parametrize('directive', DIRECTIVES)
@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('candidate', ['volas'])
def test_api_history_derive(benchmark, directive, operation, candidate):
    parent = source(directive)
    benchmark(derive, parent, directive, operation)


@pytest.mark.parametrize('directive', DIRECTIVES)
@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('candidate', ['volas'])
def test_api_history_continue(benchmark, directive, operation, warm, candidate):
    parent = source(directive)
    template = derive(parent, directive, operation)
    if warm:
        template.append({'close': 103.})
        template.fulfill()

    def setup():
        # The first continuation must include lazy history-owner initialization;
        # copying a template would materialize that memo outside the timer.
        frame = template.copy() if warm else derive(parent, directive, operation)
        return (frame,), {}

    def run(frame):
        frame.append({'close': 104.})
        frame.fulfill()

    benchmark.pedantic(run, setup=setup, rounds=100, iterations=1)
