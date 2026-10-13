"""Short publication histories fall back to the continuation warm-up owner."""

import numpy as np
import pytest
import volas

from test.exact import assert_exact


@pytest.mark.parametrize('directive', ['stochrsi.fastk:2,', 'stochrsi.d:2,5,8',
                                     'mavp:2,8,1@close,periods', 'macdext.signal:2,1,3,1,8,0',
                                     'stoch.d:2,3,0,8,0'])
@pytest.mark.parametrize('tf', [False, True])
def test_short_frames_publish_only_complete_continuation_states(directive, tf):
    nmax = volas.directive_lookback(directive) + 3
    close = np.array([100.0 + (i * 7) % 13 for i in range(nmax + 2)])
    columns = {k: close + j for j, k in enumerate(['open', 'high', 'low', 'close', 'volume'])}
    columns['periods'] = np.full(len(close), 3.0)
    index = np.datetime64('2026-01-01', 'ns') + np.arange(len(close)) * np.timedelta64(60, 's')
    def frame(n):
        return volas.DataFrame({k: v[:n] for k, v in columns.items()}, index=index[:n],
                               **({'time_frame': '1m'} if tf else {}))
    for n in range(nmax + 1):
        df = frame(n)
        df[directive]
        bar = volas.DataFrame({k: v[n:n + 1] for k, v in columns.items()}, index=index[n:n + 1])
        df.append(bar)
        assert_exact(df[directive].to_numpy(), frame(n + 1)[directive].to_numpy())

@pytest.mark.parametrize('complete', range(3, 7))
def test_short_complete_history_uses_warmup_state_inside_a_large_frame(complete):
    # A large physical frame enters the batch publisher while its gathered history
    # is still shorter than StochRSI's tail reach.
    close = np.r_[np.full(30, np.nan), np.arange(complete, dtype=float)]
    df = volas.DataFrame({'close': close})
    df['stochrsi.fastk:2,']
    df.append(volas.DataFrame({'close': [9.0]}))
    expected = volas.DataFrame({'close': np.r_[close, 9.0]})['stochrsi.fastk:2,']
    assert_exact(df['stochrsi.fastk:2,'].to_numpy(), expected.to_numpy())
