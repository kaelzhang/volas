"""Every local TZif transition, including the footer, is a ZoneInfo fact.

CPython's independent TZif reader enumerates exact transitions; its C ZoneInfo
implementation supplies offset, DST, abbreviation and wall-time oracles. No
transition table, expected result, or footer parser comes from volas.
"""

from __future__ import annotations

import datetime as dt
import io
import zoneinfo
from zoneinfo import _common, _zoneinfo

import numpy as np
import pytest

import volas

from .test_tz_database import NS, UTC, assert_facts, local_tzif, oracle, utc_datetime


MIN_NS = np.iinfo(np.int64).min + 1
MAX_NS = np.iinfo(np.int64).max
ZONES = sorted(zoneinfo.available_timezones())
BOUNDARY_DELTAS = (-NS, -1, 0, 1, NS)
EPOCH_NAIVE = dt.datetime(1970, 1, 1)


def transition_seconds(data):
    """Explicit transitions plus all footer recurrences on the ns axis."""
    _, explicit, _, _, _, footer = _common.load_data(io.BytesIO(data))
    transitions = set(explicit)
    if footer:
        rule = _zoneinfo._parse_tz_str(footer.decode())
        if hasattr(rule, "transitions"):
            last = explicit[-1] if explicit else MIN_NS // NS - 1
            for year in range(1677, 2264):
                start, end = rule.transitions(year)
                start -= int(rule.std.utcoff.total_seconds())
                end -= int(rule.dst.utcoff.total_seconds())
                transitions.update(t for t in (start, end) if t > last)
    return sorted(t for t in transitions if MIN_NS <= t * NS <= MAX_NS)


def wall_resolution(zone, wall_ns):
    """Classify by both folds and exact UTC round trips, retaining sub-us ns."""
    wall = utc_datetime(wall_ns).replace(tzinfo=None)
    candidates = set()
    for fold in (0, 1):
        candidate = wall.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if candidate.astimezone(zone).replace(tzinfo=None) == wall:
            delta = candidate.replace(tzinfo=None) - EPOCH_NAIVE
            candidates.add((delta.days * 86400 + delta.seconds) * NS
                           + delta.microseconds * 1000 + wall_ns % 1000)
    if not candidates:
        return "gap", None
    if len(candidates) == 2:
        return "fold", None
    return "unique", candidates.pop()


@pytest.mark.parametrize("key", ZONES)
def test_every_transition_utc_facts_and_ns_boundaries(key):
    data = local_tzif(key)
    expected_zone = oracle(key, data)
    probes = {MIN_NS, MIN_NS + NS, MAX_NS - NS, MAX_NS, -1, 0, 1}
    for transition in transition_seconds(data):
        probes.update(transition * NS + delta for delta in BOUNDARY_DELTAS
                      if MIN_NS <= transition * NS + delta <= MAX_NS)
    # One frame binds the bytes once; row names carry those same rules to each
    # scalar assertion and exercise the actual retained-rule export path.
    values = np.array(sorted(probes), dtype="int64")
    frame = volas.DataFrame({"close": np.ones(len(values))}, index=values.view("datetime64[ns]"))
    frame = frame.tz_localize("UTC").tz_convert(key)
    for i, ns in enumerate(map(int, values)):
        assert_facts(frame.iloc[i].name, ns, expected_zone)


@pytest.mark.parametrize("key", ZONES)
def test_every_transition_gap_fold_boundaries(key):
    data = local_tzif(key)
    expected_zone = oracle(key, data)
    walls = set()
    for transition in transition_seconds(data):
        before = int(utc_datetime((transition - 1) * NS).astimezone(expected_zone)
                     .utcoffset().total_seconds())
        after = int(utc_datetime(transition * NS).astimezone(expected_zone)
                    .utcoffset().total_seconds())
        # Metadata-only transitions still have a unique local boundary.
        a, b = sorted(((transition + before) * NS, (transition + after) * NS))
        walls.add((a + b) // 2)
        for boundary in (a, b):
            walls.update(boundary + delta for delta in BOUNDARY_DELTAS)
    unique_walls, unique_instants = [], []
    for wall_ns in sorted(walls):
        if not MIN_NS <= wall_ns <= MAX_NS:
            continue
        kind, expected = wall_resolution(expected_zone, wall_ns)
        if kind == "unique":
            unique_walls.append(wall_ns)
            unique_instants.append(expected)
        else:
            with pytest.raises(ValueError, match=f"DST {kind}"):
                volas.Timestamp(wall_ns).tz_localize(key)
    if unique_walls:
        frame = volas.DataFrame({"close": np.ones(len(unique_walls))},
                               index=np.array(unique_walls, dtype="datetime64[ns]"))
        assert frame.tz_localize(key).index.view("i8").tolist() == unique_instants
