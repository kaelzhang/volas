"""Owned derived histories continue every registered cached shape bit for bit."""
import numpy as np
import pytest

from volas import DataFrame
from test.exact import assert_exact
from test.test_infinity_missing import DIRECTIVES


def data(n, missing):
    x = np.arange(n, dtype=float)
    close = 100 + np.sin(x * .07) + x * .001
    out = {'open': close - .3, 'high': close + 1, 'low': close - 1,
           'close': close, 'volume': 1000 + x % 71, 'periods': 2 + x % 12}
    if missing:
        for column in ('open', 'high', 'low', 'close', 'volume'):
            out[column][[1, 511, 512, 1023, 1149, 2050]] = np.nan
    return out


@pytest.mark.parametrize('directive', DIRECTIVES)
@pytest.mark.parametrize('missing', [False, True])
def test_bounded_owners_continue_every_cached_shape(directive, missing):
    n = 2053
    values = data(n + 24, missing)

    def frame(a, b):
        return DataFrame({k: v[a:b] for k, v in values.items()}, index=list(range(a, b)))

    parent = frame(0, n)
    parent[directive]
    # Large visible slices must keep their internal anchors for nested derivation.
    selections = [(parent.tail(5), n, 2), (parent.iloc[1152:1157], 1157, 2),
                  (parent.iloc[200:1800].iloc[950:955], 1155, 2),
                  (DataFrame(parent, window=5, max_lookback=[directive]), n, 20)]
    del parent
    for short, end, count in selections:
        for step in range(count):
            short.append(frame(end + step, end + step + 1))
            short.fulfill()
            if step in (0, count - 1):
                want = frame(0, end + step + 1).exec(directive, dtype='float64')[-len(short):]
                assert_exact(short[directive].to_numpy(dtype='float64'), want,
                             f'{directive}, missing={missing}, end={end}, step={step}')
        # Same-last-row replacement also consumes the before-last continuation.
        short.append(frame(end + count - 1, end + count))
        short.fulfill()
        assert_exact(short[directive].to_numpy(dtype='float64'), want, directive + ' replacement')


@pytest.mark.parametrize('operation', ['copy', 'drop'])
def test_derived_branches_publish_future_anchors_independently(operation):
    n = 2053
    seed = 100 + np.sin(np.arange(n) * .01)
    parent = DataFrame({'close': seed, 'unused': seed})
    directive = 'ema:1000'
    parent[directive]
    left = parent.tail(5)
    right = left.copy() if operation == 'copy' else left.drop(['unused'], axis=1)
    left_rows = 100 + np.sin(np.arange(1800) * .05)
    right_rows = 150 + np.cos(np.arange(1800) * .04)
    for branch, rows in [(left, left_rows), (right, right_rows)]:
        branch.append(DataFrame({'close': rows}))
        branch.fulfill()
    # Replaying one branch publishes anchors after the shared seed. They must
    # never seed a different branch's lazy continuation.
    first = left.iloc[1500:1505]
    first.append({'close': 90.})
    first.fulfill()
    second = right.iloc[1500:1505]
    second.append({'close': 91.})
    second.fulfill()
    expected = DataFrame({'close': np.r_[seed, right_rows[:1500], 91.]}).exec(directive)[-6:]
    assert_exact(second[directive].to_numpy(), expected, operation + ' branch anchors')


@pytest.mark.parametrize('directive', ['ema:10000', 'ema:5@(ma:4000)'])
def test_wide_warm_history_chunks_continue_bit_exactly(directive):
    values = 100 + np.sin(np.arange(25_000) * .03)
    parent = DataFrame({'close': values})
    parent[directive]  # the batch pass publishes both warm and seeded snapshots
    short = parent.iloc[5000:5005]
    del parent
    short.append(DataFrame({'close': values[5005:10_200]}))
    short.fulfill()
    expected = DataFrame({'close': values[:10_200]}).exec(directive)[-len(short):]
    assert_exact(short[directive].to_numpy(), expected, directive + ' growing chunks')
