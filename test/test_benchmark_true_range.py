"""Million-row range kernels and their steady-state single-bar append cost."""

import numpy as np
import pytest

import volas

DIRECTIVES = ['atr:14', 'natr:14', 'adx:14', 'sar', 'sarext']


@pytest.fixture(scope='module')
def range_prices():
    t = np.arange(1_000_000, dtype=float)
    c = 100. + np.sin(t * .03) + np.cos(t * .017)
    return {'high': c + 1.5, 'low': c - .5, 'close': c}


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_range_batch_million(benchmark, range_prices, directive):
    df = volas.DataFrame(range_prices)
    benchmark(lambda: df.exec(directive))


@pytest.mark.parametrize('directive', DIRECTIVES)
def test_range_single_row(benchmark, range_prices, directive):
    # Match the main benchmark's append setup: warm the capacity growth once,
    # then measure append + refresh with the cached state already established.
    history = {k: v[:1998] for k, v in range_prices.items()}
    warm = volas.DataFrame({k: v[1998:1999] for k, v in range_prices.items()})
    bar = volas.DataFrame({k: v[1999:2000] for k, v in range_prices.items()})

    def setup():
        df = volas.DataFrame(history)
        df[directive]
        df.append(warm)
        df.fulfill()
        return (df,), {}

    def run(df):
        df.append(bar)
        df.fulfill()

    benchmark.pedantic(run, setup=setup, rounds=300, iterations=1, warmup_rounds=10)
