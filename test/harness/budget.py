"""Run budget of the differential harness, read once from the environment.

``VOLAS_HARNESS_PROFILE``   ``short`` (default; what ``make test`` runs) or ``long``.
``VOLAS_HARNESS_EXAMPLES``  op sequences per property test; overrides the profile.
``VOLAS_HARNESS_STEPS``     max ops per sequence; overrides the profile.
``VOLAS_HARNESS_SECONDS``   CPU-seconds cap per property test (0 = examples only). With
                            a cap the test keeps drawing fresh sequences until either
                            the example count or the CPU budget is spent.
``VOLAS_HARNESS_ONLY``      a regex: the inventory schedules sweep only the directives
                            it matches (to bisect / replay one family).

The short profile is derandomized (a fixed example stream, so ``make test`` is
reproducible); the long profile explores. A failure prints Hypothesis's
``@reproduce_failure`` blob, which replays the exact op trace.
"""

from __future__ import annotations

import os
import time

from hypothesis import HealthCheck, Phase, settings
from hypothesis.stateful import run_state_machine_as_test

_PROFILES = {
    # (examples, steps)
    'short': (6, 20),
    'long': (300, 60),
}

PROFILE = os.environ.get('VOLAS_HARNESS_PROFILE', 'short')
if PROFILE not in _PROFILES:  # pragma: no cover - configuration error
    raise ValueError(f'VOLAS_HARNESS_PROFILE must be one of {sorted(_PROFILES)}, got {PROFILE!r}')
EXAMPLES = int(os.environ.get('VOLAS_HARNESS_EXAMPLES', _PROFILES[PROFILE][0]))
STEPS = int(os.environ.get('VOLAS_HARNESS_STEPS', _PROFILES[PROFILE][1]))
SECONDS = float(os.environ.get('VOLAS_HARNESS_SECONDS', '0'))
ONLY = os.environ.get('VOLAS_HARNESS_ONLY')

# A batch is the unit of work between two CPU-budget checks in a capped run.
_BATCH = 25


def _settings(examples: int, derandomize: bool) -> settings:
    # Deadlines are off: the oracle rebuilds frames on every step and the host may
    # be loaded. No example database: a failure prints its reproduction blob.
    return settings(
        max_examples=max(1, examples),
        stateful_step_count=STEPS,
        deadline=None,
        derandomize=derandomize,
        suppress_health_check=list(HealthCheck),
        phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
        database=None,
        print_blob=True,
    )


def run_machine(machine: type, scale: float = 1.0) -> None:
    from .machine_base import STATS

    STATS.clear()
    try:
        _run_machine(machine, scale)
    finally:
        if os.environ.get('VOLAS_HARNESS_STATS'):
            print(f'\n{machine.__name__} ops:', dict(STATS.most_common()))


def _run_machine(machine: type, scale: float) -> None:
    """Run one stateful property test under the configured budget.

    ``scale`` weights this machine's share of the example budget (a costly machine
    takes fewer sequences). Without a CPU cap this is a single Hypothesis run; with
    one, independent batches run until the examples or the CPU-seconds are spent.
    """
    total = max(1, int(EXAMPLES * scale))
    if SECONDS <= 0:
        derandomize = PROFILE == 'short'
        run_state_machine_as_test(machine, settings=_settings(total, derandomize))
        return
    start = time.process_time()
    done = 0
    while done < total and time.process_time() - start < SECONDS:
        batch = min(_BATCH, total - done)
        run_state_machine_as_test(machine, settings=_settings(batch, False))
        done += batch

