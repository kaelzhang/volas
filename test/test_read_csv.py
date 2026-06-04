"""Tests for ``volas.read_csv``.

``read_csv`` is a pandas capability that stock-pandas does not have, so these
cases are ported from pandas's parser test suite (``pandas/tests/io/parser``)
and adapted to volas's native, path-based API. Where volas supports the same
behaviour, results are checked against ``pandas.read_csv`` directly; volas only
implements the pandas-subset relevant to OHLCV time-series (dtype inference,
default NA tokens, quoting, and an optional ``date_col`` -> ``DatetimeIndex``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import volas

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


def write_csv(tmp_path, text, name='t.csv'):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# --- parity with pandas on the real fixture ---------------------------------

def test_basic_shape_and_columns():
    df = volas.read_csv(TENCENT)
    pf = pd.read_csv(TENCENT)
    assert df.shape == pf.shape
    assert df.columns == list(pf.columns)


def test_numeric_columns_match_pandas():
    df = volas.read_csv(TENCENT)
    pf = pd.read_csv(TENCENT)
    for col in ['open', 'close', 'high', 'low', 'volume', 'turnover',
                'pe_ratio', 'turnover_rate', 'last_close']:
        np.testing.assert_allclose(
            np.asarray(df[col], dtype=float),
            pf[col].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_object_column_matches_pandas():
    df = volas.read_csv(TENCENT)
    pf = pd.read_csv(TENCENT)
    assert df['time_key'].dtype == 'object'
    assert list(np.asarray(df['time_key'])) == list(pf['time_key'].astype(str))


# --- dtype inference --------------------------------------------------------

def test_int_vs_float_inference(tmp_path):
    path = write_csv(tmp_path, "A,B\n1.0,1\n2.0,2\n3.0,3\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'float64'
    assert df['B'].dtype == 'int64'
    np.testing.assert_array_equal(np.asarray(df['A']), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(df['B']), [1, 2, 3])


def test_mixed_int_and_float_upcasts(tmp_path):
    path = write_csv(tmp_path, "A\n1\n2.5\n3\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'float64'
    np.testing.assert_array_equal(np.asarray(df['A']), [1.0, 2.5, 3.0])


def test_blank_upcasts_int_to_float_nan(tmp_path):
    path = write_csv(tmp_path, "A,B\n1,1\n,2\n3,3\n")
    df = volas.read_csv(path)
    pf = pd.read_csv(path)
    assert df['A'].dtype == 'float64'
    assert df['B'].dtype == 'int64'
    np.testing.assert_allclose(
        np.asarray(df['A']), pf['A'].to_numpy(float), equal_nan=True
    )


@pytest.mark.parametrize('token', ['NA', 'null', 'N/A', 'NaN', 'None', 'nan'])
def test_default_na_tokens_become_nan(tmp_path, token):
    path = write_csv(tmp_path, f"A\n1\n{token}\n2\n")
    df = volas.read_csv(path)
    pf = pd.read_csv(path)
    assert df['A'].dtype == 'float64'
    np.testing.assert_allclose(
        np.asarray(df['A']), pf['A'].to_numpy(float), equal_nan=True
    )


def test_all_blank_column_is_float_nan(tmp_path):
    path = write_csv(tmp_path, "A,B\n,1\n,2\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'float64'
    assert np.isnan(np.asarray(df['A'])).all()
    np.testing.assert_array_equal(np.asarray(df['B']), [1, 2])


def test_bool_inference(tmp_path):
    path = write_csv(tmp_path, "A,B\nTrue,1\nFalse,2\nTrue,3\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'bool'
    np.testing.assert_array_equal(np.asarray(df['A']), [True, False, True])


def test_bool_lowercase(tmp_path):
    path = write_csv(tmp_path, "A\ntrue\nfalse\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'bool'
    np.testing.assert_array_equal(np.asarray(df['A']), [True, False])


def test_string_object_column(tmp_path):
    path = write_csv(tmp_path, "A,B\nx,1\ny,2\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'object'
    assert list(np.asarray(df['A'])) == ['x', 'y']


def test_scientific_and_negative_floats(tmp_path):
    path = write_csv(tmp_path, "A\n-1.5\n1e3\n")
    df = volas.read_csv(path)
    assert df['A'].dtype == 'float64'
    np.testing.assert_array_equal(np.asarray(df['A']), [-1.5, 1000.0])


# --- structure / quoting / edges -------------------------------------------

def test_quoted_field_with_comma(tmp_path):
    path = write_csv(tmp_path, 'A,B\n"x,y",1\n"z",2\n')
    df = volas.read_csv(path)
    pf = pd.read_csv(path)
    assert list(np.asarray(df['A'])) == ['x,y', 'z']
    assert list(np.asarray(df['A'])) == list(pf['A'].astype(str))
    np.testing.assert_array_equal(np.asarray(df['B']), [1, 2])


def test_header_only_is_zero_rows(tmp_path):
    path = write_csv(tmp_path, "A,B\n")
    df = volas.read_csv(path)
    assert df.shape == (0, 2)
    assert df.columns == ['A', 'B']


def test_single_column(tmp_path):
    path = write_csv(tmp_path, "A\n1\n2\n3\n")
    df = volas.read_csv(path)
    assert df.shape == (3, 1)
    np.testing.assert_array_equal(np.asarray(df['A']), [1, 2, 3])


def test_whitespace_around_numbers_trimmed(tmp_path):
    path = write_csv(tmp_path, "A,B\n 1 , 2 \n3,4\n")
    df = volas.read_csv(path)
    np.testing.assert_array_equal(np.asarray(df['A']), [1, 3])
    np.testing.assert_array_equal(np.asarray(df['B']), [2, 4])


# --- sep / delimiter / header / na_values ----------------------------------

def test_custom_separator(tmp_path):
    path = write_csv(tmp_path, "A\tB\n1\t2\n3\t4\n")
    df = volas.read_csv(path, sep='\t')
    np.testing.assert_array_equal(np.asarray(df['A']), [1, 3])
    np.testing.assert_array_equal(np.asarray(df['B']), [2, 4])


def test_semicolon_delimiter(tmp_path):
    path = write_csv(tmp_path, "A;B\n1;2\n")
    df = volas.read_csv(path, delimiter=';')
    assert df.columns == ['A', 'B']
    np.testing.assert_array_equal(np.asarray(df['A']), [1])


def test_multichar_sep_raises(tmp_path):
    path = write_csv(tmp_path, "A,B\n1,2\n")
    with pytest.raises(Exception):
        volas.read_csv(path, sep=', ')


def test_header_none_generates_positional_names(tmp_path):
    path = write_csv(tmp_path, "1,2,3\n4,5,6\n")
    df = volas.read_csv(path, header=None)
    assert df.columns == ['0', '1', '2']
    assert df.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(df['0']), [1, 4])


def test_custom_na_values(tmp_path):
    path = write_csv(tmp_path, "A\n1\nMISSING\n3\n")
    df = volas.read_csv(path, na_values='MISSING')
    pf = pd.read_csv(path, na_values='MISSING')
    assert df['A'].dtype == 'float64'
    np.testing.assert_allclose(
        np.asarray(df['A']), pf['A'].to_numpy(float), equal_nan=True
    )


def test_keep_default_na_false_keeps_token_as_string(tmp_path):
    path = write_csv(tmp_path, "A\nx\nNA\ny\n")
    df = volas.read_csv(path, keep_default_na=False)
    assert df['A'].dtype == 'object'
    assert list(np.asarray(df['A'])) == ['x', 'NA', 'y']


# --- parse_dates + index_col ------------------------------------------------

def test_parse_dates_with_index_col_makes_datetime_index():
    df = volas.read_csv(TENCENT, parse_dates=['time_key'], index_col='time_key')
    pf = pd.read_csv(TENCENT)
    assert 'time_key' not in df.columns
    assert str(df.index.dtype) == 'datetime64[ns]'
    expected = pd.to_datetime(pf['time_key']).to_numpy()
    assert (df.index == expected).all()


def test_parse_dates_without_index_keeps_datetime_column():
    df = volas.read_csv(TENCENT, parse_dates=['time_key'])
    pf = pd.read_csv(TENCENT)
    assert 'time_key' in df.columns
    assert df['time_key'].dtype == 'datetime64[ns]'
    expected = pd.to_datetime(pf['time_key']).to_numpy()
    assert (np.asarray(df['time_key']) == expected).all()


def test_index_col_by_integer_position():
    df = volas.read_csv(TENCENT, parse_dates=['time_key'], index_col=0)
    assert 'time_key' not in df.columns
    assert str(df.index.dtype) == 'datetime64[ns]'


def test_index_col_on_int_column():
    df = volas.read_csv(TENCENT, index_col='volume')
    pf = pd.read_csv(TENCENT)
    assert 'volume' not in df.columns
    np.testing.assert_array_equal(np.asarray(df.index), pf['volume'].to_numpy())


# --- error paths ------------------------------------------------------------

def test_missing_file_raises():
    with pytest.raises(Exception):
        volas.read_csv('/no/such/file/volas_missing.csv')


def test_unparsable_parse_dates_raises(tmp_path):
    path = write_csv(tmp_path, "time_key,a\nnot-a-date,1\n")
    with pytest.raises(Exception):
        volas.read_csv(path, parse_dates=['time_key'])


def test_parse_dates_on_non_string_column_raises(tmp_path):
    path = write_csv(tmp_path, "t,a\n1,2\n3,4\n")
    with pytest.raises(Exception):
        volas.read_csv(path, parse_dates=['t'])


def test_index_col_on_string_column_builds_string_index(tmp_path):
    # a string column moved into the index yields a (pandas object) string index
    path = write_csv(tmp_path, "k,a\nx,1\ny,2\n")
    df = volas.read_csv(path, index_col='k')
    assert list(df.index) == ['x', 'y']
    assert df.loc['y']['a'] == 2
