"""A row appended under the last row's label replaces it; under an earlier label it
raises; without a label it takes the next one.

After any replacement every cached column — recursive, finite-window, stateless,
explicit `@` operand, nested, boolean, sub-line — equals a from-scratch full
recompute bit-exactly, on plain, time-frame and windowed frames, with datetime and
integer labels, for single and repeated replacements, replacements followed by
appends, and replacement rows holding a missing value.
"""

import numpy as np
import pytest

from volas import DataFrame, Timestamp, to_datetime

from .test_window_locality import FINITE, RECURSIVE, SUBCOMMANDS

from test.exact import assert_exact

OHLCV = ("open", "high", "low", "close", "volume")
T0 = np.datetime64("2026-01-01T00:00", "ns")
MINUTE = np.timedelta64(1, "m")

# Every recursive command is held bit-exact here.
_RECURSIVE = sorted(
    set(RECURSIVE.values()) | {SUBCOMMANDS[k] for k in SUBCOMMANDS if k.split(".")[0] in RECURSIVE}
)
_FINITE = ["ma:5", "wma:5", "boll.upper", "stddev:5", "llv:5", "rsv:5", "cci:5", "maxindex:5",
           "linearreg:5", "mfi:5", "willr:5", "kst", "ichimoku.kijun", "pivothigh:5,3",
           "ttm_squeeze", "ttm_squeeze.on"]
_STATELESS = ["avgprice", "medprice", "tr", "mom:5", "roc:5", "bop"]
_OPERANDS = ["rsi:5@open", "ema:5@(close + open)", "ema:5@(rsi:5)", "atr:5@high,low,open",
             "ma:5@(close - open)"]
_BOOLEAN = ["close > ma:5", "close > open", "(ema:5 > ma:10) & (close > open)"]
SHAPES = sorted(set(_RECURSIVE + _FINITE + _STATELESS + _OPERANDS + _BOOLEAN))


def _bar(i, salt=0.0):
    x = float(i) + salt
    c = 100.0 + 0.2 * x + 3.0 * np.sin(1.1 * x)
    return {"open": c - 0.3, "high": c + 1.0 + 0.5 * np.cos(0.7 * x) ** 2,
            "low": c - 1.0 - 0.5 * np.sin(0.3 * x) ** 2, "close": c,
            "volume": 1000.0 + 37.0 * abs(np.sin(x))}


def _frame_of(rows, labels):
    return DataFrame({k: np.array([r[k] for r in rows]) for k in OHLCV}, index=labels)


def _np(x):
    """Values to compare: a bool Series as float64 (a missing truth -> NaN, so its
    position is compared too), anything else as it exports."""
    if getattr(x, "dtype", None) == "bool" and not isinstance(x, np.ndarray):
        return x.to_numpy(dtype="float64")
    return np.asarray(x)


def _same(got, want, ctx):
    assert_exact(_np(got), _np(want), ctx)


def _full(live):
    """The frame's own data, recomputed from scratch."""
    return DataFrame({c: np.asarray(live[c]) for c in OHLCV}, index=np.asarray(live.index))


# --- the rule --------------------------------------------------------------------

def test_a_resent_bar_replaces_the_last_row():
    times = ['2026-01-01 17:00', '2026-01-01 17:01', '2026-01-01 17:02']
    df = DataFrame({'time': to_datetime(times), 'close': [10.0, 11.0, 12.0]}).set_index('time')
    df['ma:2']
    df.append({'time': Timestamp('2026-01-01 17:02'), 'close': 12.5})
    assert len(df) == 3
    assert df['close'].iloc[-1] == 12.5
    assert df['ma:2'].iloc[-1] == 11.75


@pytest.mark.parametrize("minute", [2, 3.5])
def test_a_datetime_label_before_the_last_raises_and_leaves_the_frame_unchanged(minute):
    df = _frame_of([_bar(i) for i in range(5)], T0 + np.arange(5) * MINUTE)
    df['ema:3']
    before = df['close'].to_numpy().copy()
    at = T0 + np.timedelta64(int(minute * 60), "s")
    with pytest.raises(ValueError, match="only increases"):
        df.append(_frame_of([_bar(9)], np.array([at])))
    # in a batch, a later row going back raises too, atomically
    batch = _frame_of([_bar(5), _bar(6), _bar(7)], T0 + np.array([5, 6, 4]) * MINUTE)
    with pytest.raises(ValueError, match="only increases"):
        df.append(batch)
    assert len(df) == 5
    _same(df['close'], before, "unchanged")
    _same(df['ema:3'], _full(df)['ema:3'], "ema unchanged")


def test_an_integer_index_stays_unique():
    df = _frame_of([_bar(i) for i in range(5)], np.arange(0, 50, 10))
    df['ema:3']
    with pytest.raises(ValueError, match="earlier row"):
        df.append(_frame_of([_bar(9)], np.array([20])))
    assert len(df) == 5
    # a new label before the last is appended
    df.append(_frame_of([_bar(9)], np.array([25])))
    assert len(df) == 6
    _same(df['ema:3'], _full(df)['ema:3'], "ema")


def test_a_batch_replaces_and_appends_in_order():
    df = _frame_of([_bar(i) for i in range(5)], T0 + np.arange(5) * MINUTE)
    df['ema:3']
    # a batch never repeats a label: the re-sent 5 is the first row of the next batch
    df.append(_frame_of([_bar(4, 0.5), _bar(5)], T0 + np.array([4, 5]) * MINUTE))
    df.append(_frame_of([_bar(5, 0.3), _bar(6)], T0 + np.array([5, 6]) * MINUTE))
    assert len(df) == 7
    assert df['close'].iloc[-2] == _bar(5, 0.3)["close"]
    _same(df['ema:3'], _full(df)['ema:3'], "batch")


def test_unlabelled_rows_take_the_next_labels():
    df = DataFrame({k: [v] for k, v in _bar(0).items()})
    df.append(DataFrame({k: [v] for k, v in _bar(1).items()}))
    df.append(DataFrame({k: [v] for k, v in _bar(2).items()}))
    assert list(np.asarray(df.index)) == [0, 1, 2]


def test_a_replacing_row_without_a_column_makes_it_missing():
    df = DataFrame({'close': [1.0, 2.0], 'flag': [True, False], 'name': ['a', 'b'],
                    'n': [1, 2]}, index=[10, 11])
    df.append(DataFrame({'close': [5.0], 'name': ['zz']}, index=[11]))
    assert len(df) == 2
    assert df['close'].iloc[-1] == 5.0 and df['name'].iloc[-1] == 'zz'
    assert df['flag'].isna().iloc[-1] and df['n'].isna().iloc[-1]
    df.append(DataFrame({'close': [6.0], 'flag': [True], 'name': ['q'], 'n': [7]}, index=[11]))
    assert df['flag'].iloc[-1] and df['n'].iloc[-1] == 7 and df['name'].iloc[-1] == 'q'


# --- every cached shape stays exact ---------------------------------------------

# op: ('append', salt) | ('replace', salt) | ('replace_nan', salt)
_OPS = ([("append", 0.0)] * 2 + [("replace", 0.37), ("replace", -0.61), ("append", 0.0),
        ("replace_nan", 0.2), ("replace", 0.11), ("append", 0.0), ("append", 0.0),
        ("replace", 0.9)] * 3)


def _run_plain(directive, labels_of, window=None):
    # A windowed frame starts from a seed inside its retention and runs long enough to
    # compact many times; it is compared with its unbounded twin.
    n0, ops = (60, _OPS) if window is None else (3, _OPS * 16)
    rows = [_bar(i) for i in range(n0)]
    live = _frame_of(rows, labels_of(np.arange(n0)))
    twin = _frame_of(rows, labels_of(np.arange(n0)))
    if window is not None:
        live = DataFrame(live, window=window, max_lookback=[directive])
    live[directive]
    twin[directive]
    i = n0
    for step, (op, salt) in enumerate(ops):
        if op == "append":
            pos, bar = i, _bar(i)
            i += 1
        else:
            pos, bar = i - 1, _bar(i - 1, salt)
            if op == "replace_nan":
                bar = dict(bar, close=np.nan)
        add = _frame_of([bar], labels_of(np.array([pos])))
        live.append(add)
        twin.append(add)
        live.fulfill()
        twin.fulfill()
        ctx = f"{directive} step={step} {op}"
        if window is None:
            _same(live[directive], _full(live)[directive], ctx)
        elif step % 9 == 8:
            w = _np(live[directive])
            _same(w, _np(twin[directive])[-len(w):], ctx + " windowed")
    _same(twin[directive], _full(twin)[directive], f"{directive} unbounded end")


@pytest.mark.parametrize("directive", SHAPES)
def test_plain_datetime_frame_replacements_are_exact(directive):
    _run_plain(directive, lambda k: T0 + k * MINUTE)


@pytest.mark.parametrize("directive", SHAPES)
def test_plain_integer_frame_replacements_are_exact(directive):
    _run_plain(directive, lambda k: (1000 + 10 * k).astype(np.int64))


@pytest.mark.parametrize("directive", SHAPES)
def test_windowed_frame_replacements_match_the_unbounded_frame(directive):
    _run_plain(directive, lambda k: T0 + k * MINUTE, window=4)


def _run_tf(directive, window=None):
    # a 15m frame fed 1m bars, each re-sent twice (once with a missing close) while it
    # forms, then 15m snapshots stamped at the period start re-sent while they form
    n_seed = 40
    seed = [_bar(i) for i in range(n_seed)]
    labels = T0 + np.arange(n_seed) * 15 * MINUTE
    kw = {} if window is None else {"window": window, "max_lookback": [directive]}
    if window is not None:
        seed, labels = seed[:3], labels[:3]
        n_seed = 3
    live = DataFrame(_frame_of(seed, labels), time_frame="15m", **kw)
    twin = DataFrame(_frame_of(seed, labels), time_frame="15m")
    live[directive]
    twin[directive]
    minute = 15 * (n_seed - 1) + 1
    feed = []
    for k in range(50 if window is None else 900):
        m = minute + k
        feed += [(m, _bar(m)), (m, _bar(m, 0.4)), (m, dict(_bar(m, 0.2), close=np.nan)),
                 (m, _bar(m, 0.1))]
    p0 = (feed[-1][0]) // 15 + 1
    for p in range(p0, p0 + 4):
        for salt in (0.0, 0.3, -0.2):
            feed.append((15 * p, _bar(15 * p, salt)))
    for step, (m, bar) in enumerate(feed):
        add = _frame_of([bar], T0 + np.array([m]) * MINUTE)
        live.append(add)
        twin.append(add)
        live.fulfill()
        twin.fulfill()
        if step % (7 if window is None else 61) == 6 or step == len(feed) - 1:
            ctx = f"{directive} step={step}"
            if window is None:
                _same(live[directive], _full(live)[directive], ctx)
            else:
                w = _np(live[directive])
                _same(w, _np(twin[directive])[-len(w):], ctx + " windowed")
    _same(twin[directive], _full(twin)[directive], f"{directive} unbounded end")


@pytest.mark.parametrize("directive", SHAPES)
def test_tf_frame_resends_are_exact(directive):
    _run_tf(directive)


@pytest.mark.parametrize("directive", SHAPES)
def test_windowed_tf_frame_resends_match_the_unbounded_frame(directive):
    _run_tf(directive, window=4)


@pytest.mark.parametrize("directive", ["ema:10", "rsi:14", "ma:5", "macd", "mfi:4 > 0"])
def test_a_cached_boolean_column_keeps_the_fold_exact(directive):
    """A cached boolean directive (a Bool column) must not turn every fold into a
    whole-frame invalidation: the other columns stay exact, windowed or not."""
    for window in (None, 8):
        n_seed = 40
        seed = [_bar(i) for i in range(n_seed)]
        labels = T0 + np.arange(n_seed) * 15 * MINUTE
        kw = {} if window is None else {"window": window, "max_lookback": [directive, "ma:40"]}
        live = DataFrame(_frame_of(seed, labels), time_frame="15m", **kw)
        twin = DataFrame(_frame_of(seed, labels), time_frame="15m")
        for df in (live, twin):
            df["close > open"]
            df[directive]
        for k in range(1, 120):
            m = 15 * (n_seed - 1) + k
            for salt in (0.0, 0.25):
                add = _frame_of([_bar(m, salt)], T0 + np.array([m]) * MINUTE)
                live.append(add)
                twin.append(add)
                live.fulfill()
                twin.fulfill()
        for d in (directive, "close > open"):
            w = _np(live[d])
            _same(w, _np(twin[d])[-len(w):], f"{d} window={window}")
            _same(twin[d], _full(twin)[d], f"{d} full")


def test_finite_shapes_are_finite_commands():
    assert all(k.split(":")[0].split(".")[0] in FINITE for k in _FINITE + _STATELESS)
