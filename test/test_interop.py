"""pandas interop + to_csv (audit SP-6 / PD-19)."""

import numpy as np
import pandas as pd
import volas
from volas import DataFrame, to_datetime


def test_to_pandas():
    df = DataFrame({'a': [1., 2., 3.], 'b': np.array([10, 20, 30], dtype=np.int64)})
    pdf = df.to_pandas()
    assert isinstance(pdf, pd.DataFrame)
    assert pdf['a'].tolist() == [1., 2., 3.]
    assert pdf['b'].tolist() == [10, 20, 30]


def test_from_pandas_roundtrip():
    pdf = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    df = volas.DataFrame.from_pandas(pdf)
    assert df.columns == ['a', 'b']
    assert df['a'].to_list() == [1., 2.]


def test_from_pandas_datetime_index():
    pdf = pd.DataFrame({'close': [1., 2.]},
                       index=pd.to_datetime(['2020-01-01', '2020-01-02']))
    df = volas.DataFrame.from_pandas(pdf)
    assert str(df.index.dtype) == 'datetime64[ns]'
    assert df['close'].to_list() == [1., 2.]


def test_from_pandas_datetime_column_native():
    # a datetime *column* (not the index) is carried natively as datetime64 (no string round-trip)
    pdf = pd.DataFrame({'when': pd.to_datetime(['2020-01-01 00:00']), 'c': [1.0]})
    df = volas.DataFrame.from_pandas(pdf)
    assert str(df['when'].dtype) == 'datetime64[ns]'
    assert df['when'].to_numpy()[0] == np.datetime64('2020-01-01 00:00:00')


def test_datetime64_array_column_ingested():
    # pandas-aligned: a dict value that is a datetime64 array becomes a datetime column.
    # A non-ns unit ([s]) exercises the normalise-to-ns path.
    arr = np.array(['2021-01-04 09:30', '2021-01-04 09:31'], dtype='datetime64[s]')
    df = DataFrame({'t': arr, 'c': [1.0, 2.0]})
    assert str(df['t'].dtype) == 'datetime64[ns]'
    assert (df.set_index('t').index.astype('datetime64[ns]')[0]
            == np.datetime64('2021-01-04 09:30:00'))


def test_to_pandas_tz_aware_index():
    # a tz-aware volas frame exports a tz-aware pandas index (faithful, not UTC-naive)
    df = DataFrame({'t': ['2021-01-04 09:30:00'], 'c': [1.0]})
    df['t'] = to_datetime(df['t'])
    df = df.set_index('t').tz_localize('America/New_York')
    pdf = df.to_pandas()
    assert str(pdf.index.tz) == 'America/New_York'
    assert pdf.index[0] == pd.Timestamp('2021-01-04 09:30:00', tz='America/New_York')


def test_from_pandas_tz_aware_named():
    idx = pd.to_datetime(['2021-01-04 09:30']).tz_localize('America/New_York')
    df = volas.DataFrame.from_pandas(pd.DataFrame({'c': [1.0]}, index=idx))
    assert df.tz == 'America/New_York'
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2021-01-04 14:30:00')


def test_from_pandas_tz_aware_fixed_offset():
    # pandas renders a fixed offset as 'UTC+08:00'; from_pandas normalises it back to '+08:00'
    idx = pd.to_datetime(['2021-01-04 09:30']).tz_localize('+08:00')
    df = volas.DataFrame.from_pandas(pd.DataFrame({'c': [1.0]}, index=idx))
    assert df.tz == '+08:00'
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2021-01-04 01:30:00')


def test_tz_roundtrip_volas_pandas_volas():
    df = DataFrame({'t': ['2021-01-04 09:30:00', '2021-01-04 09:31:00'], 'c': [1.0, 2.0]})
    df['t'] = to_datetime(df['t'])
    df = df.set_index('t').tz_localize('+08:00')
    back = volas.DataFrame.from_pandas(df.to_pandas())
    assert back.tz == '+08:00'
    assert (back.index.astype('datetime64[ns]') == df.index.astype('datetime64[ns]')).all()
    assert back['c'].to_list() == [1.0, 2.0]


def test_to_csv_string_and_file(tmp_path):
    df = DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    s = df.to_csv()
    # an unnamed index writes an empty header field (pandas 3.0), not "index"
    assert s.splitlines()[0] == ',a,b'
    p = tmp_path / "o.csv"
    assert df.to_csv(str(p), index=False) is None
    lines = p.read_text().splitlines()
    assert lines[0] == 'a,b'
    df2 = volas.read_csv(str(p))            # round-trips
    assert df2.columns == ['a', 'b']


def test_to_csv_accepts_pathlike(tmp_path):
    """to_csv's path takes a str, a pathlib.Path, or any os.PathLike (pandas parity)."""
    df = DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    p = tmp_path / "p.csv"
    assert df.to_csv(p, index=False) is None      # pathlib.Path
    assert p.read_text().splitlines()[0] == 'a,b'

    class _PathLike:
        def __fspath__(self):
            return str(tmp_path / "q.csv")

    assert df.to_csv(_PathLike(), index=False) is None
    assert (tmp_path / "q.csv").read_text() == p.read_text()


def test_equals_ignores_index_name():
    # pandas .equals ignores the index name; volas matches (same values, different name -> equal)
    a = DataFrame({'A': [1.0, 2.0], 'k': [7, 8]}).set_index('k')
    b = DataFrame({'A': [1.0, 2.0], 'j': [7, 8]}).set_index('j')
    assert a.equals(b)


def test_to_pandas_carries_index_name():
    v = DataFrame({'A': [1.0, 2.0], 'k': [7, 8]}).set_index('k')
    assert v.to_pandas().index.name == 'k'


def test_to_csv_named_index_round_trips(tmp_path):
    v = DataFrame({'A': [1.0, 2.0], 'k': [7, 8]}).set_index('k')
    csv = v.to_csv()
    assert csv.splitlines()[0] == 'k,A'          # the index name (pandas parity)
    p = tmp_path / "named.csv"
    p.write_text(csv)
    back = volas.read_csv(str(p), index_col='k')  # restores the named index
    assert back.columns == ['A']
    assert back.reset_index().columns[0] == 'k'


def test_read_csv_empty_header_is_unnamed(tmp_path):
    # an empty header field becomes "Unnamed: {i}" (pandas parity)
    p = tmp_path / "empty.csv"
    p.write_text(",A\n0,1.0\n1,2.0\n")
    assert volas.read_csv(str(p)).columns == ['Unnamed: 0', 'A']
