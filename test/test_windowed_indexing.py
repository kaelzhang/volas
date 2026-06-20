"""Windowed-frame indexing parity: every row-index-dependent API on a windowed
frame must equal the SAME API on the equivalent unbounded frame's last-M rows.

The windowed change reroutes ~40 surfaces through a logical-M view (and offsets the
indexer set paths). The risk is not only ``iloc``/``loc`` themselves but every API
whose result depends on the row index (labels, positions, reductions' index, the
``index`` column from ``reset_index``, …). This module pins each one against an
oracle: ``oracle = unbounded.iloc[-M:]`` — a plain frame holding exactly the rows the
windowed frame should expose, with the same labels. We feed enough bars to force
several compactions, and run the whole matrix on BOTH a DatetimeIndex and a
RangeIndex (the two slice → label-kind paths).
"""

import numpy as np
import pytest

from volas import DataFrame

WINDOW = 30
LOOKBACK = 14          # atr:14 -> capacity 44, 2*capacity 88
N = 500                # >> 2*capacity: forces ~5 compactions
SEED = 20


def _ohlcv(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    openp = close + rng.standard_normal(n) * 0.1
    vol = rng.integers(1, 1000, n).astype(float)
    return dict(open=openp, high=high, low=low, close=close, volume=vol)


def _dt(n, start=0):
    base = np.datetime64('2020-01-01T00:00:00')
    return (base + (start + np.arange(n)) * np.timedelta64(1, 's')).astype('datetime64[ns]')


def _make(index_kind):
    """Return (windowed_frame, oracle_tail_frame) fed the same N bars."""
    data = _ohlcv(N, seed=11)
    if index_kind == 'datetime':
        idx_full = _dt(N)
        seed_idx, bar_idx = idx_full[:SEED], idx_full
    else:  # range
        idx_full = None
        seed_idx, bar_idx = None, None

    if index_kind == 'datetime':
        wf = DataFrame({k: v[:SEED] for k, v in data.items()}, index=seed_idx,
                       window=WINDOW, max_lookback=['atr:14'])
    else:
        wf = DataFrame({k: v[:SEED] for k, v in data.items()},
                       window=WINDOW, max_lookback=['atr:14'])
    for k in range(SEED, N):
        if index_kind == 'datetime':
            one = DataFrame({c: [data[c][k]] for c in data}, index=np.array([bar_idx[k]]))
        else:
            one = DataFrame({c: [data[c][k]] for c in data})
        wf.append(one)
        _ = wf['atr:14']                       # keep the cache fresh each bar

    if index_kind == 'datetime':
        uf = DataFrame(data, index=idx_full)
    else:
        uf = DataFrame(data)
    _ = uf['atr:14']
    oracle = uf.iloc[-WINDOW:]                  # plain frame: the last M rows + labels
    return wf, oracle


@pytest.fixture(params=['datetime', 'range'])
def pair(request):
    return _make(request.param)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _eq_values(av, bv):
    assert av.shape == bv.shape, f"shape {av.shape} != {bv.shape}"
    if np.issubdtype(av.dtype, np.floating) and np.issubdtype(bv.dtype, np.floating):
        assert np.allclose(av, bv, equal_nan=True), f"values differ:\n{av}\n{bv}"
    else:
        assert np.array_equal(av, bv), f"values differ:\n{av}\n{bv}"


def _eq_series(a, b):
    assert np.array_equal(a.index, b.index), f"series index mismatch:\n{a.index}\n{b.index}"
    _eq_values(a.to_numpy(), b.to_numpy())


def _eq_frame(a, b):
    assert list(a.columns) == list(b.columns), f"columns {list(a.columns)} != {list(b.columns)}"
    assert np.array_equal(a.index, b.index), f"index mismatch:\n{a.index}\n{b.index}"
    for col in a.columns:
        _eq_values(a[col].to_numpy(), b[col].to_numpy())


# --------------------------------------------------------------------------- #
# scalar shape / index                                                        #
# --------------------------------------------------------------------------- #

def test_len_shape_index(pair):
    wf, oracle = pair
    assert len(wf) == len(oracle) == WINDOW
    assert wf.shape == oracle.shape
    assert np.array_equal(wf.index, oracle.index)
    assert list(wf.columns) == list(oracle.columns)


# --------------------------------------------------------------------------- #
# __getitem__ : column / list / mask / slice                                  #
# --------------------------------------------------------------------------- #

def test_getitem_column_and_indicator(pair):
    wf, oracle = pair
    for col in ['close', 'high', 'atr:14']:
        _eq_series(wf[col], oracle[col])


def test_getitem_list(pair):
    wf, oracle = pair
    _eq_frame(wf[['close', 'high', 'atr:14']], oracle[['close', 'high', 'atr:14']])


def test_getitem_bool_mask_series(pair):
    wf, oracle = pair
    thr = float(np.nanmedian(wf['close'].to_numpy()))
    _eq_frame(wf[wf['close'] > thr], oracle[oracle['close'] > thr])


def test_getitem_bool_mask_numpy(pair):
    wf, oracle = pair
    mask = wf['close'].to_numpy() > float(np.nanmedian(wf['close'].to_numpy()))
    _eq_frame(wf[mask], oracle[mask])


def test_getitem_positional_slice(pair):
    wf, oracle = pair
    _eq_frame(wf[5:20], oracle[5:20])
    _eq_frame(wf[:10], oracle[:10])
    _eq_frame(wf[-5:], oracle[-5:])


# --------------------------------------------------------------------------- #
# iloc                                                                         #
# --------------------------------------------------------------------------- #

def test_iloc_scalar_row(pair):
    wf, oracle = pair
    for i in (0, 5, WINDOW - 1, -1, -WINDOW):
        assert wf.iloc[i]['close'] == oracle.iloc[i]['close']
        assert np.array_equal(wf.iloc[i].to_numpy(), oracle.iloc[i].to_numpy())


def test_iloc_slice(pair):
    wf, oracle = pair
    _eq_frame(wf.iloc[3:18], oracle.iloc[3:18])
    _eq_frame(wf.iloc[:7], oracle.iloc[:7])
    _eq_frame(wf.iloc[-4:], oracle.iloc[-4:])


def test_iloc_strided_slice(pair):
    wf, oracle = pair
    _eq_frame(wf.iloc[0:WINDOW:3], oracle.iloc[0:WINDOW:3])


def test_iloc_int_list(pair):
    wf, oracle = pair
    sel = [0, 2, 5, WINDOW - 1]
    _eq_frame(wf.iloc[sel], oracle.iloc[sel])


def test_iloc_bool_mask(pair):
    wf, oracle = pair
    mask = (wf['close'].to_numpy() > float(np.nanmedian(wf['close'].to_numpy()))).tolist()
    _eq_frame(wf.iloc[mask], oracle.iloc[mask])


def test_iloc_2d(pair):
    wf, oracle = pair
    assert wf.iloc[2, 3] == oracle.iloc[2, 3]
    _eq_frame(wf.iloc[1:5, 0:3], oracle.iloc[1:5, 0:3])


def test_iat(pair):
    wf, oracle = pair
    for (i, j) in [(0, 0), (5, 3), (-1, 4), (WINDOW - 1, 1)]:
        assert wf.iat[i, j] == oracle.iat[i, j]


# --------------------------------------------------------------------------- #
# loc / at (label based)                                                       #
# --------------------------------------------------------------------------- #

def test_loc_label_scalar(pair):
    wf, oracle = pair
    labels = list(wf.index)
    for lab in (labels[0], labels[7], labels[-1]):
        assert wf.loc[lab]['close'] == oracle.loc[lab]['close']


def test_loc_label_slice(pair):
    wf, oracle = pair
    labels = list(wf.index)
    _eq_frame(wf.loc[labels[3]:labels[15]], oracle.loc[labels[3]:labels[15]])


def test_loc_label_list(pair):
    wf, oracle = pair
    labels = list(wf.index)
    sel = [labels[1], labels[4], labels[20]]
    _eq_frame(wf.loc[sel], oracle.loc[sel])


def test_loc_bool_mask(pair):
    wf, oracle = pair
    thr = float(np.nanmedian(wf['close'].to_numpy()))
    mask = wf['close'] > thr
    omask = oracle['close'] > thr
    _eq_frame(wf.loc[mask], oracle.loc[omask])


def test_at_label(pair):
    wf, oracle = pair
    labels = list(wf.index)
    assert wf.at[labels[3], 'close'] == oracle.at[labels[3], 'close']
    assert wf.at[labels[-1], 'high'] == oracle.at[labels[-1], 'high']


def test_loc_margin_label_not_addressable():
    # a label that fell out of the window must NOT resolve.
    wf, _ = _make('datetime')
    margin_ts = _dt(1, start=0)[0]   # the very first bar's timestamp, long compacted away
    with pytest.raises(KeyError):
        _ = wf.loc[margin_ts]


# --------------------------------------------------------------------------- #
# index-derived results (the "APIs related to indexing" the change can break)  #
# --------------------------------------------------------------------------- #

def test_idxmax_idxmin_labels(pair):
    wf, oracle = pair
    _eq_series(wf.idxmax(), oracle.idxmax())
    _eq_series(wf.idxmin(), oracle.idxmin())


def test_reset_index(pair):
    wf, oracle = pair
    _eq_frame(wf.reset_index(), oracle.reset_index())


def test_sort_index(pair):
    wf, oracle = pair
    _eq_frame(wf.sort_index(), oracle.sort_index())
    _eq_frame(wf.sort_index(ascending=False), oracle.sort_index(ascending=False))


def test_drop_row_label(pair):
    wf, oracle = pair
    labels = list(wf.index)
    _eq_frame(wf.drop([labels[2], labels[6]]), oracle.drop([labels[2], labels[6]]))


def test_drop_column(pair):
    wf, oracle = pair
    _eq_frame(wf.drop(['volume'], axis=1), oracle.drop(['volume'], axis=1))


def test_head_tail(pair):
    wf, oracle = pair
    _eq_frame(wf.head(4), oracle.head(4))
    _eq_frame(wf.tail(4), oracle.tail(4))
    _eq_frame(wf.head(-3), oracle.head(-3))
    _eq_frame(wf.tail(-3), oracle.tail(-3))


def test_to_csv_index_labels(pair):
    wf, oracle = pair
    assert wf.to_csv() == oracle.to_csv()


def test_to_numpy(pair):
    wf, oracle = pair
    assert np.allclose(wf.to_numpy(), oracle.to_numpy(), equal_nan=True)


def test_to_pandas_index(pair):
    pytest.importorskip("pandas")
    wf, oracle = pair
    a, b = wf.to_pandas(), oracle.to_pandas()
    assert np.array_equal(a.index.to_numpy(), b.index.to_numpy())
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# set paths: the windowed offset must hit the same logical cell as the oracle  #
# --------------------------------------------------------------------------- #

# A set into a column an indicator depends on stales the cached directive (the
# existing ensure_fresh-after-write contract, windowed or not), so the set tests
# fulfill() before the positional read — exactly what a caller does.

def test_iloc_setitem_scalar_and_array(pair):
    wf, oracle = pair
    wf.iloc[2, 3] = -1.0
    oracle.iloc[2, 3] = -1.0
    wf.fulfill(); oracle.fulfill()
    wf.iloc[5:8, 0] = [7.0, 8.0, 9.0]
    oracle.iloc[5:8, 0] = [7.0, 8.0, 9.0]
    wf.fulfill(); oracle.fulfill()
    _eq_frame(wf[['open', 'close']], oracle[['open', 'close']])


def test_iat_setitem(pair):
    wf, oracle = pair
    wf.iat[-1, 3] = 42.0
    oracle.iat[-1, 3] = 42.0
    wf.fulfill(); oracle.fulfill()
    assert wf.iat[-1, 3] == oracle.iat[-1, 3] == 42.0


def test_loc_setitem(pair):
    wf, oracle = pair
    labels = list(wf.index)
    wf.loc[labels[4], 'close'] = -5.0
    oracle.loc[labels[4], 'close'] = -5.0
    wf.fulfill(); oracle.fulfill()
    _eq_series(wf['close'], oracle['close'])


def test_at_setitem(pair):
    wf, oracle = pair
    labels = list(wf.index)
    wf.at[labels[6], 'high'] = -7.0
    oracle.at[labels[6], 'high'] = -7.0
    wf.fulfill(); oracle.fulfill()
    assert wf.at[labels[6], 'high'] == oracle.at[labels[6], 'high'] == -7.0


# --------------------------------------------------------------------------- #
# derive / reduce family — every result that carries the row index            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("op", [
    lambda d: d.cumsum(), lambda d: d.diff(), lambda d: d.shift(2),
    lambda d: d.rank(), lambda d: d.abs(), lambda d: d.clip(lower=99.0),
    lambda d: d.round(1), lambda d: d.isna(), lambda d: d.notna(),
    lambda d: d.ffill(), lambda d: d.bfill(),
])
def test_columnwise_transforms_keep_logical_index(pair, op):
    wf, oracle = pair
    _eq_frame(op(wf), op(oracle))


def test_nlargest_nsmallest(pair):
    wf, oracle = pair
    _eq_frame(wf.nlargest(5, 'close'), oracle.nlargest(5, 'close'))
    _eq_frame(wf.nsmallest(5, 'high'), oracle.nsmallest(5, 'high'))


def test_duplicated_and_drop_duplicates(pair):
    wf, oracle = pair
    _eq_series(wf.duplicated(), oracle.duplicated())
    _eq_frame(wf.drop_duplicates(), oracle.drop_duplicates())


def test_where_mask(pair):
    wf, oracle = pair
    _eq_frame(wf.where(wf.notna()), oracle.where(oracle.notna()))


def test_set_index_routes_through_logical(pair):
    # OHLCV columns are all float (no valid index column), so set_index must reject
    # them — and a windowed frame must reject identically to the oracle (proving the
    # call routes through the logical view, not the physical buffer).
    wf, oracle = pair
    with pytest.raises(TypeError):
        wf.set_index('close')
    with pytest.raises(TypeError):
        oracle.set_index('close')


def test_astype(pair):
    wf, oracle = pair
    _eq_frame(wf.astype({'volume': 'float32'}), oracle.astype({'volume': 'float32'}))


def test_setitem_does_not_corrupt_margin(pair):
    # an in-window set must not change len / shape / the hidden margin's bookkeeping.
    wf, _ = pair
    phys = wf._physical_height
    wf.iloc[0, 0] = 123.0
    wf.fulfill()
    assert wf._physical_height == phys
    assert len(wf) == WINDOW
    assert wf.iloc[0, 0] == 123.0
