"""Keyed tzinfo inputs resolve the process's canonical ZoneInfo(key) rules."""

import datetime as dt
from io import BytesIO
from pathlib import Path
import zoneinfo

import pytest

import volas

from .test_tz_database import local_tzif, run_isolated_case, write_zone


def _case_foreign_key(directory, key, explicit_tz):
    foreign = zoneinfo.ZoneInfo.from_file(
        BytesIO(Path(directory, "foreign.tzif").read_bytes()), key=key)
    canonical = zoneinfo.ZoneInfo(key)
    if explicit_tz:
        got = volas.Timestamp("2024-01-01 12:00", tz=foreign)
        expected = dt.datetime(2024, 1, 1, 12, tzinfo=canonical)
    else:
        value = dt.datetime(2024, 1, 1, 12, tzinfo=foreign)
        got = volas.Timestamp(value)
        expected = value.astimezone(canonical)
    # An aware input retains its absolute instant; its key chooses display
    # rules. A naive input with explicit tz is localized in those same rules.
    assert got.to_pydatetime() == expected
    assert got.utcoffset() == expected.utcoffset()
    assert str(got) == expected.replace(tzinfo=None).isoformat(sep=" ")


@pytest.mark.parametrize("key", ["UTC", "America/New_York"])
@pytest.mark.parametrize("explicit_tz", [False, True])
def test_foreign_keyed_objects_use_the_canonical_process_rules(tmp_path, key, explicit_tz):
    write_zone(tmp_path, key, local_tzif(key))
    (tmp_path / "foreign.tzif").write_bytes(local_tzif("Etc/GMT-3"))
    run_isolated_case("_case_foreign_key", str(tmp_path), key, explicit_tz,
                      tzpath=[tmp_path], module=__name__)


def test_utc_objects_remain_compatible_in_one_index():
    values = [
        dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 1, 2, tzinfo=zoneinfo.ZoneInfo("UTC")),
    ]
    frame = volas.DataFrame({"close": [1.0, 2.0]}, index=values)
    assert frame.iloc[0].name.value == volas.Timestamp(values[0]).value
    assert frame.iloc[1].name.value == volas.Timestamp(values[1]).value
    assert frame.to_pandas().index.tz is dt.timezone.utc


def _case_keyed_utc(directory, explicit_tz):
    current = zoneinfo.ZoneInfo("UTC")
    value = dt.datetime(2024, 1, 1, 12, tzinfo=current)
    if explicit_tz:
        got = volas.Timestamp("2024-01-01 12:00", tz=current)
    else:
        got = volas.Timestamp(value)
    assert got.utcoffset() == dt.timedelta(hours=3)
    assert got.to_pydatetime() == value
    assert got.tzinfo is current
    # UTC as an explicit string retains its fixed-zero spelling. A tzinfo
    # object whose key happens to be UTC must still use the source's rules.
    assert volas.Timestamp("2024-01-01 12:00", tz="UTC").utcoffset() == dt.timedelta(0)
    Path(directory, "UTC").write_bytes(Path(directory, "after.tzif").read_bytes())
    assert zoneinfo.ZoneInfo("UTC") is current
    later = volas.Timestamp(value)
    assert later.utcoffset() == dt.timedelta(hours=3)
    assert later.tzinfo is current


@pytest.mark.parametrize("explicit_tz", [False, True])
def test_keyed_utc_does_not_discard_nonzero_first_use_rules(tmp_path, explicit_tz):
    write_zone(tmp_path, "UTC", local_tzif("Etc/GMT-3"))
    (tmp_path / "after.tzif").write_bytes(local_tzif("Etc/GMT+4"))
    run_isolated_case("_case_keyed_utc", str(tmp_path), explicit_tz,
                      tzpath=[tmp_path], module=__name__)
