"""Interpreter data is bound once per key and process, like ZoneInfo(key).

Every modified source is configured before a child interpreter starts. No test
clears an initialized rules cache. Fixtures copy existing local bytes only.
"""

from __future__ import annotations

import datetime as dt
import importlib.resources
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zoneinfo

import numpy as np
import pandas as pd
import pytest

import volas


NS = 1_000_000_000
UTC = dt.timezone.utc
EPOCH = dt.datetime(1970, 1, 1, tzinfo=UTC)
ORIGINAL_TZPATH = tuple(zoneinfo.TZPATH)


def local_tzif(key):
    """Read the same local resource that a fresh ZoneInfo(key) resolves."""
    for directory in ORIGINAL_TZPATH:
        path = Path(directory, key)
        if path.is_file():
            return path.read_bytes()
    package, _, resource = key.rpartition("/")
    return importlib.resources.files("tzdata.zoneinfo" + (
        "." + package.replace("/", ".") if package else ""
    )).joinpath(resource).read_bytes()


def oracle(key, data=None):
    return zoneinfo.ZoneInfo.from_file(io.BytesIO(local_tzif(key) if data is None else data), key=key)


def utc_datetime(ns):
    seconds, nanoseconds = divmod(int(ns), NS)
    return EPOCH + dt.timedelta(seconds=seconds, microseconds=nanoseconds // 1000)


def epoch_ns(text):
    return int(np.datetime64(text, "ns").astype("int64"))


def assert_facts(got, ns, expected_zone):
    expected = utc_datetime(ns).astimezone(expected_zone)
    assert got.value == ns
    assert got.utcoffset() == expected.utcoffset()
    assert got.dst() == expected.dst()
    assert got.strftime("%Z") == expected.tzname()
    assert str(got)[:19] == expected.replace(tzinfo=None).isoformat(sep=" ")[:19]
    assert got.microsecond == expected.microsecond
    assert got.nanosecond == ns % 1000


def write_zone(directory, key, data, version=None):
    target = directory / key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if version is not None:
        (directory / "tzdata.zi").write_text(f"# version {version}\n")


def run_isolated_case(case, *args, tzpath, module=__name__):
    """Keep cwd/LLVM_PROFILE_FILE so coverage includes child execution."""
    env = dict(os.environ)
    env["PYTHONTZPATH"] = os.pathsep.join(map(str, tzpath))
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    code = """
import importlib, json, sys
module = importlib.import_module(sys.argv[1])
getattr(module, sys.argv[2])(*json.loads(sys.argv[3]))
"""
    result = subprocess.run([sys.executable, "-c", code, module, case, json.dumps(args)],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _frame(ns, key):
    frame = volas.DataFrame({"close": [1.0], "volume": [1.0]},
                           index=np.array([ns], dtype="datetime64[ns]"))
    return frame.tz_localize("UTC").tz_convert(key)


def _assert_pandas_parity(frame, key):
    # pandas owns its own keyed transition cache. Require the result it
    # constructs with ZoneInfo(key), including that library's lifecycle.
    expected = pd.DatetimeIndex(frame.index).tz_localize("UTC").tz_convert(zoneinfo.ZoneInfo(key))
    actual = frame.to_pandas().index
    pd.testing.assert_index_equal(actual, expected)
    assert list(map(str, actual)) == list(map(str, expected))


@pytest.mark.parametrize("key,when", [
    ("America/Vancouver", "2026-12-01T12:00:00"),
    ("Canada/Pacific", "2026-12-01T12:00:00"),
    ("Europe/Chisinau", "2027-03-28T00:30:00"),
    ("Europe/Tiraspol", "2027-03-28T00:30:00"),
    ("America/Tijuana", "1965-07-01T12:00:00"),
    ("Mexico/BajaNorte", "1965-07-01T12:00:00"),
])
def test_updated_local_database_facts_and_exports(key, when):
    ns = epoch_ns(when)
    expected_zone = oracle(key)
    got = volas.Timestamp(ns, tz=key)
    assert_facts(got, ns, expected_zone)
    assert str(got) == got.to_pydatetime().replace(tzinfo=None).isoformat(sep=" ")
    assert got.to_pydatetime() == utc_datetime(ns).astimezone(expected_zone)
    assert got.tzinfo is zoneinfo.ZoneInfo(key)


def _case_first_resolution(key, ns, expected_hex, provenance):
    got = volas.Timestamp(ns, tz=key)
    expected = oracle(key, bytes.fromhex(expected_hex))
    assert_facts(got, ns, expected)
    assert got.to_pydatetime() == utc_datetime(ns).astimezone(expected)
    assert got.tzinfo is zoneinfo.ZoneInfo(key)
    assert volas.tz_database(key) == provenance


@pytest.mark.parametrize("first_exists", [True, False])
def test_tzpath_order_and_exact_provenance(tmp_path, first_exists):
    key = "America/New_York"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    east, west = local_tzif("Etc/GMT-3"), local_tzif("Etc/GMT+4")
    if first_exists:
        write_zone(first, key, east, "2026b")
    write_zone(second, key, west, "2025b")
    run_isolated_case("_case_first_resolution", key, epoch_ns("2024-01-01T12:00:00"),
                      (east if first_exists else west).hex(), {
                          "source": str(first if first_exists else second),
                          "version": "2026b" if first_exists else "2025b",
                      }, tzpath=[first, second])


def _case_provenance(key, expected):
    assert volas.tz_database(key) == expected


@pytest.mark.parametrize("metadata", [None, b"# no version here\n", b"# version \n", b"\xff"])
def test_unknown_directory_version_is_none(tmp_path, metadata):
    key = "Asia/Shanghai"
    write_zone(tmp_path, key, local_tzif(key))
    if metadata is not None:
        (tmp_path / "tzdata.zi").write_bytes(metadata)
    run_isolated_case("_case_provenance", key, {"source": str(tmp_path), "version": None},
                      tzpath=[tmp_path])


def _case_default_provenance(directory):
    assert volas.tz_database() == {"source": directory, "version": "2026b"}


def test_database_default_key_reports_the_utc_source(tmp_path):
    write_zone(tmp_path, "UTC", local_tzif("UTC"), "2026b")
    run_isolated_case("_case_default_provenance", str(tmp_path), tzpath=[tmp_path])


def test_new_local_zone_keys_need_no_compiled_registry(tmp_path):
    key = "Local/Trading_Session"
    data = local_tzif("America/New_York")
    write_zone(tmp_path, key, data, "2026b")
    run_isolated_case("_case_first_resolution", key, epoch_ns("2115-07-01T12:00:00"),
                      data.hex(), {"source": str(tmp_path), "version": "2026b"}, tzpath=[tmp_path])


def _case_first_use_lifecycle(directory, first_user):
    directory = Path(directory)
    key = "America/New_York"
    before = (directory / key).read_bytes()
    ns = epoch_ns("2024-01-01T15:00:00")
    if first_user == "frame":
        _frame(ns, key)
    elif first_user == "database":
        volas.tz_database(key)
    else:
        volas.Timestamp(ns, tz=key)
    old = volas.Timestamp(ns, tz=key)
    frame = _frame(ns, key)
    live = frame.cumulate("1d")
    canonical = zoneinfo.ZoneInfo(key)
    assert old.tzinfo is canonical
    # The first rules live for this process. Neither library's cache is reset.
    write_zone(directory, key, (directory / "after.tzif").read_bytes(), "2026b")
    fresh = volas.Timestamp(ns, tz=key)
    later_frame = _frame(ns, key)
    expected = oracle(key, before)
    for stamp in (old, fresh, frame.iloc[0].name, later_frame.iloc[0].name):
        assert_facts(stamp, ns, expected)
        assert stamp.tzinfo is canonical
        assert stamp.to_pydatetime() == utc_datetime(ns).astimezone(expected)
    assert volas.tz_database(key) == {"source": str(directory), "version": "2025b"}
    _assert_pandas_parity(frame, key)
    _assert_pandas_parity(later_frame, key)
    assert_facts(volas.Timestamp(old.to_pydatetime()), ns, expected)
    imported = volas.DataFrame.from_pandas(frame.to_pandas())
    assert_facts(imported.iloc[0].name, ns, expected)
    assert int(imported.cumulate("1d").index.view("i8")[0]) == epoch_ns("2024-01-01T05:00:00")

    next_ns = epoch_ns("2024-01-01T17:00:00")
    bar = volas.DataFrame({"close": [2.0], "volume": [1.0]},
                         index=np.array([next_ns], dtype="datetime64[ns]"))
    live.append(bar.tz_localize("UTC"))
    assert live.shape[0] == 1
    assert live["volume"].to_list() == [2.0]
    assert int(live.index.view("i8")[0]) == epoch_ns("2024-01-01T05:00:00")
    _assert_pandas_parity(live, key)


@pytest.mark.parametrize("first_user", ["scalar", "frame", "database"])
def test_first_use_keeps_rules_after_source_changes(tmp_path, first_user):
    write_zone(tmp_path, "America/New_York", local_tzif("America/New_York"), "2025b")
    (tmp_path / "after.tzif").write_bytes(local_tzif("Asia/Shanghai"))
    run_isolated_case("_case_first_use_lifecycle", str(tmp_path), first_user, tzpath=[tmp_path])


def test_changed_source_is_loaded_by_a_new_process(tmp_path):
    key = "Local/Restarted"
    ns = epoch_ns("2024-01-01T15:00:00")
    for original, version in (("America/New_York", "2025b"), ("Asia/Shanghai", "2026b")):
        data = local_tzif(original)
        write_zone(tmp_path, key, data, version)
        run_isolated_case("_case_first_resolution", key, ns, data.hex(),
                          {"source": str(tmp_path), "version": version}, tzpath=[tmp_path])


def _case_removed_source(directory):
    directory = Path(directory)
    key = "Local/Retained_Session"
    data = (directory / key).read_bytes()
    ns = epoch_ns("2115-07-01T12:00:00")
    old = volas.Timestamp(ns, tz=key)
    frame = _frame(ns, key)
    canonical = zoneinfo.ZoneInfo(key)
    # Include pandas' first use before removal; its independent cache remains
    # untouched too. A later export is compared with normal pandas construction.
    _assert_pandas_parity(frame, key)
    (directory / key).unlink()
    later = volas.Timestamp(ns, tz=key)
    for stamp in (old, later):
        assert_facts(stamp, ns, oracle(key, data))
        assert stamp.tzinfo is canonical
        assert stamp.to_pydatetime() == utc_datetime(ns).astimezone(canonical)
    assert zoneinfo.ZoneInfo(key) is canonical
    _assert_pandas_parity(frame, key)
    _assert_pandas_parity(_frame(ns, key), key)


def test_bound_key_remains_available_after_its_source_is_removed(tmp_path):
    write_zone(tmp_path, "Local/Retained_Session", local_tzif("America/New_York"))
    run_isolated_case("_case_removed_source", str(tmp_path), tzpath=[tmp_path])


def _case_first_pandas_export(directory):
    directory = Path(directory)
    key = "Local/First_Export"
    ns = epoch_ns("2024-01-01T15:00:00")
    frame = _frame(ns, key)
    canonical = zoneinfo.ZoneInfo(key)
    write_zone(directory, key, (directory / "after.tzif").read_bytes())
    assert frame.iloc[0].name.tzinfo is canonical
    assert volas.Timestamp(ns, tz=key).tzinfo is canonical
    # Enter pandas only after the update. Require its normal ZoneInfo(key)
    # result, not stronger snapshot export semantics.
    _assert_pandas_parity(frame, key)


def test_pandas_first_export_uses_the_same_cached_zoneinfo_key(tmp_path):
    write_zone(tmp_path, "Local/First_Export", local_tzif("America/New_York"))
    (tmp_path / "after.tzif").write_bytes(local_tzif("Asia/Shanghai"))
    run_isolated_case("_case_first_pandas_export", str(tmp_path), tzpath=[tmp_path])


def _case_many_loaded_keys(directory):
    directory = Path(directory)
    key = "Local/First"
    ns = epoch_ns("2024-01-01T12:00:00")
    first = volas.Timestamp(ns, tz=key)
    canonical = first.tzinfo
    for i in range(80):
        volas.Timestamp(ns, tz=f"Local/Other_{i}")
    write_zone(directory, key, (directory / "after.tzif").read_bytes())
    later = volas.Timestamp(ns, tz=key)
    assert later.utcoffset() == first.utcoffset()
    assert later.tzinfo is canonical


def test_first_use_rules_are_not_evicted_by_other_keys(tmp_path):
    data = local_tzif("Etc/GMT-3")
    write_zone(tmp_path, "Local/First", data)
    for i in range(80):
        write_zone(tmp_path, f"Local/Other_{i}", data)
    (tmp_path / "after.tzif").write_bytes(local_tzif("Etc/GMT+4"))
    run_isolated_case("_case_many_loaded_keys", str(tmp_path), tzpath=[tmp_path])


def _case_concurrent_first_use(directory):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    import volas._timezone as adapter

    key = "Local/Concurrent"
    ns = epoch_ns("2024-01-01T12:00:00")
    original = adapter._resolve
    count = 0
    count_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def counted(name):
        nonlocal count
        with count_lock:
            count += 1
        data = original(name)
        # Every caller observes a cold key before any caller can publish it.
        barrier.wait(timeout=10)
        return data

    adapter._resolve = counted

    def load(_):
        return volas.Timestamp(ns, tz=key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(load, range(8)))
    assert count == 8, f"expected eight racing cold reads, got {count}"
    canonical = zoneinfo.ZoneInfo(key)
    for value in results:
        assert value.tzinfo is canonical
        assert value.utcoffset() == dt.timedelta(hours=3)
        assert value.value == ns

    def forbidden(_):
        raise AssertionError("a bound key must not resolve its source again")

    adapter._resolve = forbidden
    Path(directory, key).unlink()
    Path(directory, "tzdata.zi").write_text("# version changed\n")
    assert volas.tz_database(key) == {"source": directory, "version": "2026b"}
    assert volas.Timestamp(ns, tz=key).tzinfo is canonical
    assert _frame(ns, key).iloc[0].name.tzinfo is canonical


def test_concurrent_first_use_publishes_shared_rules_and_warm_uses_are_memory_only(tmp_path):
    write_zone(tmp_path, "Local/Concurrent", local_tzif("Etc/GMT-3"), "2026b")
    run_isolated_case("_case_concurrent_first_use", str(tmp_path), tzpath=[tmp_path])


def _case_malformed_first():
    with pytest.raises(ValueError):
        zoneinfo.ZoneInfo("America/New_York")
    with pytest.raises(ValueError):
        volas.Timestamp("2024-01-01", tz="America/New_York")


def test_malformed_first_tzif_does_not_fall_through(tmp_path):
    key = "America/New_York"
    first, second = tmp_path / "first", tmp_path / "second"
    write_zone(first, key, b"invalid TZif")
    write_zone(second, key, local_tzif(key))
    run_isolated_case("_case_malformed_first", tzpath=[first, second])


def _local_package(root, key, data):
    """Create an importlib.resources package from existing local TZif bytes."""
    package = root / "tzdata"
    package.mkdir()
    (package / "__init__.py").write_text('IANA_VERSION = "2026b"\n')
    leaf = package / "zoneinfo"
    leaf.mkdir()
    (leaf / "__init__.py").write_text("")
    for part in key.split("/")[:-1]:
        leaf /= part
        leaf.mkdir(exist_ok=True)
        (leaf / "__init__.py").write_text("")
    (leaf / key.split("/")[-1]).write_bytes(data)


@pytest.mark.parametrize("tzpath_mode", ["empty", "missing", "hit"])
def test_python_tzpath_and_tzdata_package_resolution(tmp_path, tzpath_mode):
    key = "America/New_York"
    _local_package(tmp_path, key, local_tzif("Etc/GMT-3"))
    directory = tmp_path / "zone-directory"
    directory.mkdir()
    directory_hit = tzpath_mode == "hit"
    if directory_hit:
        write_zone(directory, key, local_tzif("Etc/GMT+4"), "2025b")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), os.getcwd()])
    env["PYTHONTZPATH"] = "" if tzpath_mode == "empty" else str(directory)
    code = """
import datetime as dt, json, zoneinfo, volas
key = 'America/New_York'
stamp = volas.Timestamp('2024-01-01T12:00:00Z').tz_convert(key)
want = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc).astimezone(zoneinfo.ZoneInfo(key))
assert stamp.utcoffset() == want.utcoffset()
assert stamp.to_pydatetime() == want
print(json.dumps(volas.tz_database(key)))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                            capture_output=True, check=True, timeout=30)
    assert json.loads(result.stdout) == {
        "source": str(directory) if directory_hit else "tzdata",
        "version": "2025b" if directory_hit else "2026b",
    }
