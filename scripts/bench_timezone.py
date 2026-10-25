#!/usr/bin/env python3
"""Release-build timezone benchmark with construction outside every timing.

Run this same script with isolated base and candidate interpreters, then pass
the base JSON to --base for the before/after table. The cases measure one
million localizations, conversions, and hourly period buckets in ordinary and
footer years. Both ordered and shuffled localization queries are measured.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

import numpy as np

import volas
import volas_rs


NS = 1_000_000_000
CASES = (("America/New_York", 2024), ("America/New_York", 2115),
         ("Asia/Shanghai", 2024), ("Asia/Shanghai", 2115))


def jobs(rows):
    out = []
    for zone, year in CASES:
        start = int(np.datetime64(f"{year}-01-01", "ns").astype("int64"))
        values = start + np.arange(rows, dtype="int64") * NS
        offset = int(dt.datetime(year, 1, 1, tzinfo=ZoneInfo(zone)).utcoffset().total_seconds()) * NS
        for order in ("ordered", "shuffled"):
            wall = values.copy()
            if order == "shuffled":
                np.random.default_rng(20261013).shuffle(wall)
            frame = volas.DataFrame({"close": np.ones(rows)}, index=wall.view("datetime64[ns]"))
            utc = frame.tz_localize("UTC")
            out.append((f"localize/{zone}/{year}/{order}",
                        lambda f=frame, z=zone: f.tz_localize(z), wall - offset))
            out.append((f"convert/{zone}/{year}/{order}",
                        lambda f=utc, z=zone: f.tz_convert(z), wall))
            if order == "ordered":
                localized = frame.tz_localize(zone)
                expected = np.unique(values // (3600 * NS)) * (3600 * NS) - offset
                out.append((f"period_start/{zone}/{year}/{order}",
                            lambda f=localized: f.cumulate("1h"), expected))
        # A second period case crosses spring DST: this input is UTC and has
        # no nonexistent wall clocks. Hour-aligned modern offsets preserve the
        # independent expected UTC grid even across the missing local hour.
        start = int(np.datetime64(f"{year}-03-05", "ns").astype("int64"))
        instants = start + np.arange(rows, dtype="int64") * NS
        crossing = volas.DataFrame({"close": np.ones(rows)}, index=instants.view("datetime64[ns]"))
        crossing = crossing.tz_localize("UTC").tz_convert(zone)
        expected = np.unique(instants // (3600 * NS)) * (3600 * NS)
        out.append((f"period_start/{zone}/{year}/transition",
                    lambda f=crossing: f.cumulate("1h"), expected))
    return out


def measure(rows, repeat, operation):
    cases = [job for job in jobs(rows) if operation == "all" or job[0].startswith(operation + "/")]
    for _, action, expected in cases:
        for _ in range(2):
            result = action()
            np.testing.assert_array_equal(result.index.view("i8"), expected)
    timings = {name: [] for name, _, _ in cases}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        # Rotate the first measured case each round to avoid always giving one
        # zone the same position in the warm-up/thermal progression.
        for round_ in range(repeat):
            rotation = round_ % len(cases)
            rotated = cases[rotation:] + cases[:rotation]
            for name, action, _ in rotated:
                started = time.perf_counter_ns()
                result = action()
                timings[name].append((time.perf_counter_ns() - started) / 1_000_000)
                del result
    finally:
        if gc_was_enabled:
            gc.enable()
    metrics = {}
    for name, samples in timings.items():
        median = statistics.median(samples)
        metrics[name] = {
            "median_ms": median,
            "mad_ms": statistics.median(abs(sample - median) for sample in samples),
            "samples_ms": samples,
        }
    return {
        "methodology": "timezone-million-v1",
        "rows": rows,
        "repeat": repeat,
        "operation": operation,
        "warmups": 2,
        "package": str(Path(volas.__file__).resolve()),
        "extension": str(Path(getattr(volas_rs, "volas_rs", volas_rs).__file__).resolve()),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--operation", choices=("all", "localize", "convert", "period_start"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.rows > 2_000_000:
        parser.error("--rows must be between 1 and 2000000 (within the January fixture)")
    if args.repeat < 3:
        parser.error("--repeat must be at least 3")
    current = measure(args.rows, args.repeat, args.operation)
    args.output.write_text(json.dumps(current, indent=2) + "\n")
    base = json.loads(args.base.read_text()) if args.base else None
    if base:
        if (base["methodology"], base["rows"]) != (
            current["methodology"], current["rows"]
        ):
            raise AssertionError()
        print("| Case | Base ms | Current ms | Current/base |")
        print("|---|---:|---:|---:|")
    else:
        print("| Case | Median ms | MAD ms |")
        print("|---|---:|---:|")
    for name, metric in current["metrics"].items():
        median = metric["median_ms"]
        if base:
            before = base["metrics"][name]["median_ms"]
            print(f"| {name} | {before:.3f} | {median:.3f} | {median / before:.3f} |")
        else:
            print(f"| {name} | {median:.3f} | {metric['mad_ms']:.3f} |")


if __name__ == "__main__":
    main()
