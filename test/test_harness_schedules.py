"""Deterministic differential schedules over the whole directive inventory, plus the
read-surface and representation checks the property machines also assert.

Each test compares against the independent ledger oracle (``harness/ledger.py``):
a frame rebuilt from the accepted events, never the frame under test.
"""

import numpy as np
import pytest

import volas

from .harness import budget, build, observe
from .harness.data import EPOCHS, NS, Schema, make_rows
from .harness.ledger import Ledger, reference
from .harness.schedules import OHLCV, locality_failures, plain_schedule, sweep_directives, tf_schedule


def test_the_sweep_covers_the_inventory():
    assert budget.ONLY or len(sweep_directives()) > 300


@pytest.mark.parametrize('data_class', ['walk', 'nan', 'inf'])
@pytest.mark.parametrize('mode', ['one', 'read', 'batch', 'quiet'])
def test_appends_match_a_full_recompute(data_class, mode):
    """Every cached line and composite, refreshed after single-bar / multi-bar appends
    on clean data and with a NaN / inf cell, equals the from-scratch reference."""
    failures = plain_schedule(data_class, mode)
    assert not failures, f'{len(failures)} mismatch(es):\n' + '\n'.join(failures[:20])


@pytest.mark.parametrize('data_class', ['nan', 'inf'])
def test_a_non_finite_input_does_not_outlive_its_window(data_class):
    """Finite-memory lines: once past the lookback, a NaN / inf input earlier in the
    history no longer changes the output (full compute == compute without it)."""
    failures = locality_failures(data_class)
    assert not failures, f'{len(failures)} line(s) keep it:\n' + '\n'.join(failures[:20])


@pytest.mark.parametrize('tf', ['5m', '15m'])
@pytest.mark.parametrize('mode', ['cumulate', 'construct'])
@pytest.mark.parametrize('cache_when', ['start', 'forming'])
@pytest.mark.parametrize('bools', [False, True], ids=['numeric', 'with-bools'])
def test_tf_folds_match_the_reaggregated_ledger(tf, mode, cache_when, bools):
    """Live folds (forming updates, rollovers, a frame built from closed bars, a
    column cached while a bar is forming) equal the independent re-aggregation."""
    failures = tf_schedule(tf, mode, cache_when, bools)
    assert not failures, f'{len(failures)} mismatch(es):\n' + '\n'.join(failures[:20])


def test_canonical_names_reparse_to_the_same_values():
    """A cached column's name is a directive of the same meaning (a refresh re-parses
    it), and canonicalization is idempotent."""
    rows = make_rows(OHLCV, ('walk', 3), 50, None)
    index = [EPOCHS[1] + i * 60 * NS for i in range(50)]
    ledger = Ledger(OHLCV)
    ledger.append(rows, index)
    ref = ledger.frame()
    failures = []
    for d in sweep_directives():
        try:
            want = np.asarray(ref.exec(d))
        except (volas.DirectiveError, ValueError, TypeError):
            continue
        name = volas.directive_stringify(d)
        if volas.directive_stringify(name) != name:
            failures.append(f'{d}: stringify not idempotent ({name!r})')
            continue
        try:
            again = np.asarray(ref.exec(name))
        except Exception as e:  # noqa: BLE001 - reported
            failures.append(f'{d}: its canonical name {name!r} does not re-parse ({e})')
            continue
        try:
            observe.same_indicator(again, want, f'{d} vs {name}')
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, f'{len(failures)} failure(s):\n' + '\n'.join(failures[:20])


# Every non-refreshing reader of a whole frame (README "df.fulfill()": every read
# other than a directive-column read raises while a cached column is stale).
BULK_READERS = {
    'to_numpy': lambda df: df.to_numpy(),
    'iloc_row': lambda df: df.iloc[-1],
    'iloc_slice': lambda df: df.iloc[1:],
    'loc_row': lambda df: df.loc[df.index[-1]],
    'head': lambda df: df.head(2),
    'tail': lambda df: df.tail(2),
    'equals': lambda df: df.equals(df.copy()),
    'to_pandas': lambda df: df.to_pandas(),
    'to_arrow': lambda df: df.to_arrow(),
    'to_csv': lambda df: df.to_csv(),
    'repr': lambda df: repr(df),
    'sum': lambda df: df.sum(),
    'describe': lambda df: df.describe(),
    'rolling': lambda df: df.rolling(2).mean(),
    'sort_index': lambda df: df.sort_index(),
    'reset_index': lambda df: df.reset_index(),
    'drop_duplicates': lambda df: df.drop_duplicates(),
    'duplicated': lambda df: df.duplicated(),
    'nsmallest': lambda df: df.nsmallest(2, 'close'),
    'nlargest': lambda df: df.nlargest(2, 'close'),
    'astype': lambda df: df.astype({'close': 'float32'}),
    'fill_into': lambda df: df.fill_into(np.empty(df.shape)),
}


@pytest.mark.parametrize('reader', sorted(BULK_READERS))
def test_a_stale_frame_rejects_every_bulk_reader(reader):
    rows = make_rows(OHLCV, ('walk', 4), 12, None)
    index = [EPOCHS[1] + i * 60 * NS for i in range(12)]
    df = build.frame(OHLCV, rows[:11], index[:11])
    df['ema:3']
    df.append(build.frame(OHLCV, rows[11:], index[11:]))
    with pytest.raises(ValueError, match='stale'):
        BULK_READERS[reader](df)
    df.fulfill()
    invalid_order = False
    if reader in ('nlargest', 'nsmallest'):
        values = np.array([row['close'] for row in rows])
        order = np.argsort(-values if reader == 'nlargest' else values, kind='stable')[:2]
        labels = np.asarray(index)[order]
        invalid_order = np.any(labels[1:] <= labels[:-1])
    if invalid_order:
        # Freshness is checked first; after refresh, index admission owns
        # the independent error for a value ordering that reverses time.
        with pytest.raises(ValueError, match='monotonic'):
            BULK_READERS[reader](df)
    else:
        BULK_READERS[reader](df)


def test_windowed_range_index_labels_continue_across_compaction():
    """A windowed RangeIndex frame's labels stay absolute: the last ``window`` labels
    of the unbounded sequence, through every front-drop."""
    schema = Schema('float64', 'float64', (), 'range', None)
    rows = make_rows(schema, ('walk', 2), 40, None)
    df = volas.DataFrame(build.frame(schema, rows[:5], None), window=4, max_lookback=2)
    for i in range(5, 40):
        df.append(build.frame(schema, rows[i:i + 1], None))
        assert list(np.asarray(df.index)) == list(range(max(0, i + 1 - 4), i + 1)), f'after bar {i}'


def test_a_refold_that_fills_the_last_missing_int_keeps_an_int_export():
    """Validity stays canonical: a later bar never fills a missing aggregate (any
    missing input makes it missing), but once the bar that held the only missing int
    cell is re-sent with a value, the period re-aggregates dense — it exports as
    int64 and equals the dense reference."""
    schema = Schema('float64', 'float64', (('n', 'int64'),), 'datetime', None)
    t0 = EPOCHS[1]
    first = {'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'volume': 1.0, 'n': None}
    second = {'open': 1.2, 'high': 2.5, 'low': 1.0, 'close': 2.0, 'volume': 1.0, 'n': 7}
    df = build.frame(schema, [first], [t0]).cumulate('5m')
    df.append(build.frame(schema, [second], [t0 + 60 * NS]))
    assert df.get_column('n').to_list() == [volas.NA]  # stays missing
    df = build.frame(schema, [first], [t0]).cumulate('5m')
    df.append(build.frame(schema, [dict(first, n=3)], [t0]))  # the bar re-sent with a value
    df.append(build.frame(schema, [second], [t0 + 60 * NS]))
    n = df.get_column('n')
    assert n.equals(volas.DataFrame({'n': np.array([7])}, index=df.index)['n'])
    observe.structure(df)


def test_a_bool_column_with_na_exports_only_with_a_stated_fill():
    """The NA model at the NumPy boundary: a bool column with a missing cell has no
    bare NumPy form (a placeholder would read True or False); its object export
    carries volas.NA, its float export NaN."""
    df = volas.DataFrame({'close': [1.0, 2.0, 3.0], 'flag': [True, None, False]})
    assert df.dtypes['flag'] == 'bool'
    with pytest.raises(ValueError, match='bool column with missing values'):
        df.get_column('flag').to_numpy()
    out = df.get_column('flag').to_numpy(dtype='object')
    assert out[0] == True and out[2] == False and out[1] is volas.NA  # noqa: E712
    assert np.isnan(df.get_column('flag').to_numpy(dtype='float64')[1])
    observe.structure(df)


def test_a_lossy_coarse_volume_beside_a_missing_price_is_an_error():
    """A long-budget counterexample: on an Asia/Shanghai ledger with an int64 volume past
    2^53 (exact in int64, no exact float64) and a missing price, the from-scratch
    reference of a composite over `ad@open` crashed — the missing-row rerun read an
    empty volume. It is the error the same frame raises without the missing price."""
    schema = Schema('float64', 'int64', (('x', 'float64'),), 'datetime', 'Asia/Shanghai')
    index = [1772953200000000000, 1772954100000000000, 1772954400000000000, 1772954700000000000]
    rows = [dict(open=10.0 + i, high=11.0 + i, low=9.0 + i, close=10.5 + i, volume=100 + i, x=0.5)
            for i in range(4)]
    rows[2]['volume'] = 2**53 + 1
    directive = '-((ad@open)+(ACCBANDS@high))'
    clean = Ledger(schema)
    clean.append([dict(r) for r in rows], index)
    rows[1]['open'] = None
    ledger = Ledger(schema)
    ledger.append(rows, index)
    assert reference(clean, directive, schema.columns) == (None, OverflowError)
    assert reference(ledger, directive, schema.columns) == (None, OverflowError)
