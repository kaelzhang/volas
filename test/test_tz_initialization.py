"""Failed first loads do not publish a key or prevent a later retry."""

import datetime as dt
from pathlib import Path
import zoneinfo

import pytest

import volas

from .test_tz_database import local_tzif, run_isolated_case, write_zone


def _case_failed_first_load_can_retry(directory):
    directory = Path(directory)
    key = "Local/Retry"
    with pytest.raises(ValueError):
        volas.Timestamp("2024-01-01", tz=key)
    write_zone(directory, key, (directory / "valid.tzif").read_bytes(), "2026b")
    first = volas.Timestamp("2024-01-01", tz=key)
    assert first.utcoffset() == dt.timedelta(hours=3)
    assert first.tzinfo is zoneinfo.ZoneInfo(key)
    (directory / key).unlink()
    assert volas.Timestamp("2024-01-01", tz=key) == first
    assert volas.tz_database(key) == {"source": str(directory), "version": "2026b"}


@pytest.mark.parametrize("invalid", [False, True], ids=["missing", "malformed"])
def test_failed_first_load_can_retry(tmp_path, invalid):
    if invalid:
        write_zone(tmp_path, "Local/Retry", b"invalid TZif")
    (tmp_path / "valid.tzif").write_bytes(local_tzif("Etc/GMT-3"))
    run_isolated_case("_case_failed_first_load_can_retry", str(tmp_path),
                      tzpath=[tmp_path], module=__name__)
