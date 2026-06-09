"""pandas-parity tests for ``repr`` / ``str`` / ``to_string`` / ``_repr_html_``.

The expected strings match pandas 3.0's console output, captured and inlined so
the suite stays pandas-free — with one deliberate divergence: volas prints every
missing value as the single ``<NA>`` symbol (``volas.NA``), where pandas prints
``NaN`` for a numpy float column and ``NaT`` for datetime. ``repr`` and ``str``
are identical for every type (as in pandas).
"""

import numpy as np
import pytest

import volas

nan = float("nan")


def vdf(d):
    return volas.DataFrame({k: list(v) for k, v in d.items()})


def vdt(dates, **cols):
    d = {**{k: list(v) for k, v in cols.items()}, "t": np.array(dates, dtype="datetime64[ns]")}
    return volas.DataFrame(d).set_index("t")


# --- DataFrame repr (== str) -------------------------------------------------

DF_CASES = {
    "basic": (
        lambda: vdf({"open": [1.0, 2.0, 3.0], "close": [4.0, 5.0, 6.0], "volume": [100, 200, 300]}),
        "   open  close  volume\n0   1.0    4.0     100\n1   2.0    5.0     200\n2   3.0    6.0     300",
    ),
    "float_decimals": (  # per-column decimals; 0.1+0.2 rounds and trims to 0.3
        lambda: vdf({"a": [1.5, 2.25, 3.125], "b": [0.1 + 0.2, 1.0, 100.0]}),
        "       a      b\n0  1.500    0.3\n1  2.250    1.0\n2  3.125  100.0",
    ),
    "nan": (lambda: vdf({"a": [1.0, nan, 3.0]}), "     a\n0  1.0\n1 <NA>\n2  3.0"),
    "negative": (lambda: vdf({"x": [1.0, -1.0]}), "     x\n0  1.0\n1 -1.0"),
    "negative_int": (lambda: vdf({"x": [1, -20, 3]}), "    x\n0   1\n1 -20\n2   3"),
    "bool_col": (
        lambda: vdf({"b": [True, False], "a": [1.0, 2.0]}),
        "       b    a\n0   True  1.0\n1  False  2.0",
    ),
    "str_col": (lambda: vdf({"s": ["x", "yy"], "a": [1, 2]}), "    s  a\n0   x  1\n1  yy  2"),
    "named_int_index": (
        lambda: vdf({"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]}).set_index("k"),
        "     A\nk     \n7  1.0\n8  2.0\n9  3.0",
    ),
    "named_str_index": (
        lambda: vdf({"a": [1, 2], "sym": ["x", "yy"]}).set_index("sym"),
        "     a\nsym   \nx    1\nyy   2",
    ),
    "datetime_index_dates": (  # all-midnight -> date-only labels
        lambda: vdt(["2020-01-01", "2020-01-02"], close=[1.0, 2.0]),
        "            close\nt                \n2020-01-01    1.0\n2020-01-02    2.0",
    ),
    "datetime_index_times": (
        lambda: vdt(["2020-01-01 09:30", "2020-01-01 09:31"], p=[1.0, 2.0]),
        "                       p\nt                       \n2020-01-01 09:30:00  1.0\n2020-01-01 09:31:00  2.0",
    ),
    "empty": (lambda: vdf({"a": []}), "Empty DataFrame\nColumns: [a]\nIndex: []"),
    "truncated": (
        lambda: vdf({"a": list(range(100))}),
        "     a\n0    0\n1    1\n2    2\n3    3\n4    4\n..  ..\n95  95\n96  96\n97  97\n98  98\n99  99\n\n[100 rows x 1 columns]",
    ),
}


@pytest.mark.parametrize("name", list(DF_CASES))
def test_dataframe_repr_and_str(name):
    build, expected = DF_CASES[name]
    df = build()
    assert repr(df) == expected
    assert str(df) == expected  # repr == str, as in pandas


# --- DataFrame.to_string (params) --------------------------------------------

def _df():
    return vdf({"open": [1.0, 2.0, 3.0], "close": [4.5, 5.0, nan]})


TO_STRING_CASES = {
    "default": ({}, "   open  close\n0   1.0    4.5\n1   2.0    5.0\n2   3.0   <NA>"),
    "na_rep": ({"na_rep": "-"}, "   open  close\n0   1.0    4.5\n1   2.0    5.0\n2   3.0      -"),
    "float_format": (
        {"float_format": "%.2f"},
        "   open  close\n0  1.00   4.50\n1  2.00   5.00\n2  3.00   <NA>",
    ),
    "no_header": ({"header": False}, "0  1.0  4.5\n1  2.0  5.0\n2  3.0 <NA>"),
    "no_index": ({"index": False}, " open  close\n  1.0    4.5\n  2.0    5.0\n  3.0   <NA>"),
    "columns": ({"columns": ["close"]}, "   close\n0    4.5\n1    5.0\n2   <NA>"),
    "show_dimensions": (
        {"show_dimensions": True},
        "   open  close\n0   1.0    4.5\n1   2.0    5.0\n2   3.0   <NA>\n\n[3 rows x 2 columns]",
    ),
}


@pytest.mark.parametrize("name", list(TO_STRING_CASES))
def test_dataframe_to_string(name):
    kwargs, expected = TO_STRING_CASES[name]
    assert _df().to_string(**kwargs) == expected


def test_to_string_default_is_not_truncated():
    # 70 rows > display.max_rows(60), but to_string default shows them all
    assert vdf({"a": list(range(70))}).to_string().count("\n") == 70  # header + 70 rows - 1


def test_to_string_max_rows_truncates():
    # to_string truncation shows no dimensions footer (only show_dimensions=True does)
    assert vdf({"a": list(range(10))}).to_string(max_rows=4) == "    a\n0   0\n1   1\n.. ..\n8   8\n9   9"


def test_to_string_empty_columns_lists_index():
    # selecting no columns yields the empty-frame placeholder, listing the index
    assert vdf({"a": [1, 2, 3]}).to_string(columns=[]) == "Empty DataFrame\nColumns: []\nIndex: [0, 1, 2]"


def test_datetime_data_column():
    # a datetime *column* (not the index): no leading space, full timestamps
    df = volas.DataFrame(
        {"t": np.array(["2020-01-01 09:30", "2020-01-02 16:00"], dtype="datetime64[ns]"), "v": [1.0, 2.0]}
    )
    assert repr(df) == "                    t    v\n0 2020-01-01 09:30:00  1.0\n1 2020-01-02 16:00:00  2.0"


def test_to_string_unknown_column_raises():
    with pytest.raises(KeyError):
        _df().to_string(columns=["missing"])


def test_to_string_bad_float_format_raises():
    with pytest.raises(ValueError):
        _df().to_string(float_format="%q")


# --- Series repr (== str) ----------------------------------------------------

SERIES_CASES = {
    "float": (lambda: vdf({"open": [1.0, 2.0, 3.0]})["open"], "0    1.0\n1    2.0\n2    3.0\nName: open, dtype: float64"),
    "int": (lambda: vdf({"v": [10, 20, 30]})["v"], "0    10\n1    20\n2    30\nName: v, dtype: int64"),
    "bool": (lambda: vdf({"x": [True, False]})["x"], "0     True\n1    False\nName: x, dtype: bool"),
    "str": (lambda: vdf({"sym": ["a", "bb"]})["sym"], "0     a\n1    bb\nName: sym, dtype: str"),
    "negative": (lambda: vdf({"d": [1.0, -2.5]})["d"], "0    1.0\n1   -2.5\nName: d, dtype: float64"),
    "single": (lambda: vdf({"x": [1.0]})["x"], "0    1.0\nName: x, dtype: float64"),
    "named_index": (
        lambda: vdf({"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]}).set_index("k")["A"],
        "k\n7    1.0\n8    2.0\n9    3.0\nName: A, dtype: float64",
    ),
    "truncated": (
        lambda: vdf({"v": list(range(100))})["v"],
        "0      0\n1      1\n2      2\n3      3\n4      4\n      ..\n95    95\n96    96\n97    97\n98    98\n99    99\nName: v, Length: 100, dtype: int64",
    ),
}


@pytest.mark.parametrize("name", list(SERIES_CASES))
def test_series_repr_and_str(name):
    build, expected = SERIES_CASES[name]
    s = build()
    assert repr(s) == expected
    assert str(s) == expected


def test_empty_series_repr():
    assert repr(vdf({"e": []})["e"]) == "Series([], Name: e, dtype: float64)"


def test_series_to_string_drops_footer():
    # pandas Series.to_string omits the Name/dtype footer
    assert vdf({"open": [1.0, 2.0]})["open"].to_string() == "0    1.0\n1    2.0"


def test_series_has_no_repr_html():
    # pandas defines _repr_html_ only on DataFrame; a Series falls back to text
    assert not hasattr(vdf({"a": [1.0]})["a"], "_repr_html_")


# --- Row repr (== str) -------------------------------------------------------

ROW_CASES = {
    "floats": (
        lambda: vdf({"open": [1.5, 2.0], "close": [3.0, 4.0]}).iloc[0],
        "open     1.5\nclose    3.0\nName: 0, dtype: float64",
    ),
    "mixed_object": (
        lambda: vdf({"a": [1.0], "s": ["x"], "n": [5]}).iloc[0],
        "a    1.0\ns      x\nn      5\nName: 0, dtype: object",
    ),
    "int": (lambda: vdf({"a": [1], "b": [2]}).iloc[0], "a    1\nb    2\nName: 0, dtype: int64"),
    "datetime_label": (
        lambda: vdt(["2020-01-01 09:30", "2020-01-01 09:31"], close=[1.0, 2.0]).iloc[0],
        "close    1.0\nName: 2020-01-01 09:30:00, dtype: float64",
    ),
}


@pytest.mark.parametrize("name", list(ROW_CASES))
def test_row_repr_and_str(name):
    build, expected = ROW_CASES[name]
    row = build()
    assert repr(row) == expected
    assert str(row) == expected


def test_row_to_string_drops_footer():
    assert vdf({"open": [1.5, 2.0], "close": [3.0, 4.0]}).iloc[0].to_string() == "open     1.5\nclose    3.0"


# --- _repr_html_ -------------------------------------------------------------

def test_dataframe_repr_html():
    expected = (
        '<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n'
        "        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n    }\n\n    .dataframe thead th {\n"
        '        text-align: right;\n    }\n</style>\n<table border="1" class="dataframe">\n'
        '  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n'
        "      <th>a</th>\n      <th>b</th>\n    </tr>\n  </thead>\n  <tbody>\n"
        "    <tr>\n      <th>0</th>\n      <td>1.0</td>\n      <td>3</td>\n    </tr>\n"
        "    <tr>\n      <th>1</th>\n      <td>2.0</td>\n      <td>4</td>\n    </tr>\n"
        "  </tbody>\n</table>\n</div>"
    )
    assert vdf({"a": [1.0, 2.0], "b": [3, 4]})._repr_html_() == expected


def test_dataframe_repr_html_truncated_has_caption():
    html = vdf({"a": [float(i) for i in range(100)]})._repr_html_()
    assert "<p>100 rows × 1 columns</p>" in html
    assert html.count("<tr>") == 11  # 5 head + ellipsis + 5 tail
