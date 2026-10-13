"""A declared continuation keeps the entire directive across bounded history events."""
import numpy as np
import pytest
from volas import DataFrame


LINES = [f'{outer}@({operand})' for outer in ['ema:2', 'rsi:2']
         for operand in ['maxindex:2', 'minindex:3']]
LINES += ['ema:5@(ema:3)', 'close>ema:5', 'close-ma:5+obv', 'ema:3-ema:5']


def equal(got, want):
    a, b = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert ((a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b))).all(), (a, b)


@pytest.mark.parametrize('d', ['ema:2@(maxindex:2)', 'ema:2@(minindex:2)', 'rsi:2@(maxindex:2)'])
def test_position_operand_visible_write(d):
    x = np.arange(1., 31.)
    w = DataFrame({'close': x}, window=3, max_lookback=[d])
    w.iat[1, 0] = 0.
    x[-2] = 0.
    equal(w[d], DataFrame({'close': x}).exec(d)[-3:])


@pytest.mark.parametrize('d', LINES)
@pytest.mark.parametrize('tf', [None, '1m'])
@pytest.mark.parametrize('gaps', ['complete', 'periodic', 'long'])
def test_declared_chain_events(d, tf, gaps):
    x = 100 + np.sin(np.arange(130.) * .7) * 10
    if gaps == 'periodic':
        x[::7] = np.nan
    elif gaps == 'long':
        x[45:85] = np.nan
    volume = np.arange(130, dtype=np.int64) + 10
    def frame(a, b):
        return DataFrame({'close': x[a:b], 'volume': volume[a:b]},
                         index=np.datetime64('2024-01-01', 'ns') + np.arange(a, b) * np.timedelta64(1, 'm'))
    kw = {} if tf is None else {'time_frame': tf}
    w = DataFrame(frame(0, 30), window=3, max_lookback=[d], **kw)
    u = DataFrame(frame(0, 30), **kw)
    for i in range(30, 130):
        w.append(frame(i, i+1)); u.append(frame(i, i+1))
        equal(w[d], u.exec(d, dtype='float64')[-3:])
        if i % 7 == 0:
            x[i] += 3.
            w.append(frame(i, i+1)); u.append(frame(i, i+1))
            equal(w[d], u.exec(d, dtype='float64')[-3:])
        if i % 11 == 0:
            w.iat[1, 0] = 0.; u.iat[len(u)-2, 0] = 0.
            equal(w[d], u.exec(d, dtype='float64')[-3:])
    assert w._physical_height < 2 * (3 + __import__('volas').directive_lookback(d))
