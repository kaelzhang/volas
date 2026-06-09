"""Group E wrapper equivalence — bias / dma are formula-identical to apo / ppo.

bias and dma are dominant China-market names whose formulas reduce exactly to existing
directives: ``bias:N`` ≡ ``ppo:1,N,0`` (close's percentage deviation from its N-period SMA),
and dma's DDD line ≡ ``apo:fast,slow,0`` (fast SMA − slow SMA), with AMA = the M-period SMA of
that line. These tests pin each wrapper to its underlying directive BIT-FOR-BIT, so the
wrappers inherit apo/ppo's TA-Lib-verified correctness (test_talib_parity / test_core_vs_talib)
and can never silently drift from them.
"""

import numpy as np
import pytest

from volas import DataFrame


@pytest.fixture(scope='module')
def df():
    rng = np.random.default_rng(20260609)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))
    return DataFrame({'close': close})


def _bits_equal(a, b):
    # assert_array_equal compares element-wise with NaN == NaN — exact, no tolerance.
    np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


@pytest.mark.parametrize('n', [6, 12, 24])
def test_bias_equals_ppo(df, n):
    """bias:N is bit-identical to ppo:1,N,0."""
    _bits_equal(df[f'bias:{n}'].to_numpy(), df[f'ppo:1,{n},0'].to_numpy())


def test_bias_default_period(df):
    """bias with no argument uses the China BIAS1 default N=6."""
    _bits_equal(df['bias'].to_numpy(), df['ppo:1,6,0'].to_numpy())


@pytest.mark.parametrize('fast,slow', [(10, 50), (5, 20), (3, 10)])
def test_dma_line_equals_apo(df, fast, slow):
    """dma's DDD line is bit-identical to apo:fast,slow,0; dma.ddd is an alias of it."""
    apo = df[f'apo:{fast},{slow},0'].to_numpy()
    _bits_equal(df[f'dma:{fast},{slow}'].to_numpy(), apo)
    _bits_equal(df[f'dma.ddd:{fast},{slow}'].to_numpy(), apo)


def test_dma_default_periods(df):
    """dma with no arguments uses the China defaults fast=10, slow=50."""
    _bits_equal(df['dma'].to_numpy(), df['apo:10,50,0'].to_numpy())


@pytest.mark.parametrize('fast,slow,m', [(10, 50, 10), (5, 20, 6)])
def test_dma_ama_is_sma_of_line(df, fast, slow, m):
    """dma.ama is bit-identical to the M-period SMA of the DDD line."""
    line = df[f'dma:{fast},{slow}'].to_numpy()
    sma_of_line = DataFrame({'close': line})[f'ma:{m}'].to_numpy()
    _bits_equal(df[f'dma.ama:{fast},{slow},{m}'].to_numpy(), sma_of_line)


def test_unknown_subcommands_rejected(df):
    """Only the DDD / AMA outputs exist; an unknown sub-command is a directive error."""
    for bad in ['dma.bogus', 'bias.foo']:
        with pytest.raises(Exception):
            df[bad]
