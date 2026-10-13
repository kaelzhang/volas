"""A contiguous tail slice continues its cached columns as the parent would (R15):
appending to it, or replacing its last row, gives what the full history gives — even
when the slice keeps exactly a directive's lookback rows, where a finite-window
indicator no longer holds the row its window drops."""

import numpy as np
import pytest

import volas

from test.exact import same_exact

rng = np.random.default_rng(7)
N = 40
CLOSE = 100 + np.cumsum(rng.standard_normal(N + 2))
SIZES = [N - 9, N - 8, N - 7]  # new rows at, and off, the sliding re-anchor positions
HIGH = CLOSE + rng.random(N + 2)
LOW = CLOSE - rng.random(N + 2)
VOLUME = rng.uniform(1, 100, N + 2)


def frame(a, b):
    return volas.DataFrame({'high': HIGH[a:b], 'low': LOW[a:b], 'close': CLOSE[a:b],
                            'volume': VOLUME[a:b]}, index=list(range(a, b)))


DIRECTIVES = ['ma:5', 'boll.lower:5', 'vwma:5', 'wma:5', 'linearreg:5', 'correl:5@high,close',
              'ema:5', 'rsi:5', 'stddev:5']


@pytest.mark.parametrize('directive', DIRECTIVES)
@pytest.mark.parametrize('extra', [0, 1, 2])
@pytest.mark.parametrize('n', SIZES)
def test_a_tail_slice_continues_like_the_full_history(directive, extra, n):
    keep = volas.directive_lookback(directive) + extra
    df = frame(0, n)
    df[directive]
    tail = df.iloc[n - keep:]
    tail.append(frame(n, n + 1))
    want = frame(0, n + 1).exec(directive)
    assert same_exact(tail[directive].to_numpy(), want[-len(tail):])


@pytest.mark.parametrize('directive', ['ema:3', 'rsi:3', 'macd'])
def test_replacing_the_last_row_of_a_lookback_long_slice_keeps_the_cached_rows(directive):
    keep = volas.directive_lookback(directive)
    df = frame(0, N)
    df[directive]
    tail = df.iloc[N - keep:].copy()
    before = tail[directive].to_numpy()
    tail.append(frame(N - 1, N))  # the last row, re-sent unchanged
    tail.fulfill()
    assert same_exact(tail[directive].to_numpy(), before)
    tail.append(frame(N, N + 1))
    want = frame(0, N + 1).exec(directive)
    assert same_exact(tail[directive].to_numpy(), want[-len(tail):])
