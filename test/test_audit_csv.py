"""Systematic audit — T8 CSV I/O census (read_csv / to_csv), reviews CG-7.

read_csv is the primary ingestion boundary. Most of it is sound (float/str/NA,
parse_dates, round-trip of value columns); the finding is that an integer column
with a gap is demoted to float64 (legacy numpy behaviour) instead of volas's
native int64+validity — inconsistent with volas's OWN constructor.

Cell IDs:  T8.read_csv/<param> · T8.to_csv.roundtrip
"""

from __future__ import annotations


import volas

_CSV = "a,b,c,t\n1,1.5,x,2021-01-01\n2,,y,2021-01-02\n,3.5,z,2021-01-03\n"


def _write(tmp_path, text=_CSV):
    p = tmp_path / "audit.csv"
    p.write_text(text)
    return str(p)


def test_read_csv_dtypes_and_na(tmp_path):
    df = volas.read_csv(_write(tmp_path))
    assert list(df.columns) == ["a", "b", "c", "t"]
    assert df["b"].dtype == "float64"                 # float column with a gap
    assert df["b"].isna().to_list() == [False, True, False]
    assert df["c"].dtype == "str"                     # string column (never object, C3)
    assert df["c"].to_list() == ["x", "y", "z"]


def test_read_csv_parse_dates(tmp_path):
    df = volas.read_csv(_write(tmp_path), parse_dates=["t"])
    assert str(df["t"].dtype).startswith("datetime64")


def test_to_csv_roundtrip_preserves_values(tmp_path):
    df = volas.read_csv(_write(tmp_path))
    out = tmp_path / "rt.csv"
    df.to_csv(str(out), index=False)                  # index=False -> no Unnamed:0 (pandas-consistent)
    back = volas.read_csv(str(out))
    assert list(back.columns) == ["a", "b", "c", "t"]
    assert back["b"].isna().to_list() == [False, True, False]


# F35 (findings-ledger): read_csv demotes an int column with a gap to float64
# (legacy numpy), while volas's own constructor keeps int64 + native NA
# (DataFrame({'a':[1,None,3]}).dtype == 'int64'). The ingestion boundary should
# follow volas's native-NA model (C2), not legacy float demotion. FIXED.
def test_read_csv_int_with_gap_keeps_int(tmp_path):
    df = volas.read_csv(_write(tmp_path))
    assert df["a"].dtype == "int64"                   # native-NA int, not legacy float
    assert df["a"].isna().to_list() == [False, False, True]


# F38 (review NF-A, P1): to_csv has no RFC-4180 quoting — a string with a comma /
# quote / newline is written raw, corrupting the file (volas's own read_csv then
# can't parse it back). Silent data corruption (C4). FIXED: RFC-4180 quoting.
def test_to_csv_quotes_special_chars(tmp_path):
    df = volas.DataFrame({"s": ["a,b", 'q"x', "l1\nl2"], "v": [1.0, 2.0, 3.0]})
    p = tmp_path / "q.csv"
    df.to_csv(str(p), index=False)
    back = volas.read_csv(str(p))                     # must round-trip (RFC-4180)
    assert back["s"].to_list() == ["a,b", 'q"x', "l1\nl2"]


# F46 (review NF-B; decision: mangle like pandas): a CSV with a duplicate header
# must be disambiguated (a, a.1, b), not read as duplicate columns. owner ruling:
# error is unacceptable (the user can't fix the file). FIXED: mangles a.1.
def test_read_csv_duplicate_header_mangles(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("a,a,b\n1,2,3\n")
    assert list(volas.read_csv(str(p)).columns) == ["a", "a.1", "b"]


def test_read_csv_dup_header_collision_with_existing_mangled(tmp_path):
    # the mangle must dodge an EXISTING a.1 column: a, a.1, a -> a, a.1, a.2
    p = tmp_path / "tricky.csv"
    p.write_text("a,a.1,a\n1,2,3\n")
    assert list(volas.read_csv(str(p)).columns) == ["a", "a.1", "a.2"]
