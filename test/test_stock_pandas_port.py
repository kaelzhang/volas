"""Ports of stock-pandas test cases not yet covered explicitly by volas.

volas is the Rust successor of stock-pandas and already mirrors most of its suite
(see test_commands / test_cumulation / test_indexing / test_directive_errors /
test_manipulate). These fill the remaining behavioural gaps.

stock-pandas-only surface is intentionally NOT part of volas and is not ported:
backend switching (volas is Rust-only), ``define_command`` / custom command
registration, the ``_stock_columns_info_map`` internals, ``cum_append`` from a
dict / Series / list (volas's ``Cumulator.append`` takes a DataFrame),
the ``source=`` constructor, callable indexing (``df[lambda d: ...]``), and the
``DirectiveNonSenseWarning``.
"""

from pathlib import Path

import numpy as np

import volas

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


def _df():
    return volas.read_csv(TENCENT)


def test_compound_two_command_directive():
    # stock-pandas test_parse::test_column_with_two_command_real_case — a boolean
    # directive whose *both* operands are indicator commands carrying their own args.
    df = _df()
    got = np.asarray(df.exec('ma:10 > boll.upper:20'), dtype=float)
    ma10 = np.asarray(df.exec('ma:10'), dtype=float)
    upper = np.asarray(df.exec('boll.upper:20'), dtype=float)
    # NaN > x is False (0.0), matching numpy — so a plain elementwise compare suffices.
    np.testing.assert_array_equal(got, (ma10 > upper).astype(float))


def test_iloc_slice_with_step_and_negative_bounds():
    # stock-pandas test_truncated::test_slice_with_step / _negative_start / _negative_end:
    # iloc slicing supports a step and negative bounds, selecting the same rows numpy does.
    df = _df()
    close = df['close'].to_numpy()
    np.testing.assert_array_equal(df.iloc[0::2]['close'].to_numpy(), close[0::2])
    np.testing.assert_array_equal(df.iloc[-3:]['close'].to_numpy(), close[-3:])
    np.testing.assert_array_equal(df.iloc[:-3]['close'].to_numpy(), close[:-3])
    np.testing.assert_array_equal(df.iloc[2:9:2]['close'].to_numpy(), close[2:9:2])


def test_indicator_recomputes_over_a_sliced_frame():
    # stock-pandas truncated / column_info intent: after a shape-changing slice, an
    # indicator recomputes over the *sliced* frame (volas drops cached computed columns
    # on a slice, so the result is fresh and correct rather than a stale tail).
    df = _df()
    sliced = df.iloc[10:40]
    c = sliced['close'].to_numpy()
    exp = np.full(len(c), np.nan)
    for i in range(4, len(c)):
        exp[i] = c[i - 4: i + 1].mean()  # SMA(5) over the sliced close
    np.testing.assert_allclose(
        np.asarray(sliced.exec('ma:5'), dtype=float), exp, rtol=1e-12, atol=1e-12, equal_nan=True
    )
