"""Stateful differential property tests: random op sequences on plain, tf and
windowed frames, checked against the ledger oracles after every op.

Budget: ``VOLAS_HARNESS_PROFILE`` / ``_EXAMPLES`` / ``_STEPS`` / ``_SECONDS`` (see
``harness/budget.py``); ``make test`` runs the short derandomized profile.
"""

from .harness.budget import run_machine
from .harness.machine_plain import PlainMachine
from .harness.machine_tf import TfMachine
from .harness.machine_window import TfWindowedMachine, WindowedMachine


def test_plain_frames_match_the_ledger():
    run_machine(PlainMachine)


def test_tf_frames_match_the_reaggregated_ledger():
    run_machine(TfMachine)


def test_windowed_frames_match_the_unbounded_history():
    run_machine(WindowedMachine)


def test_windowed_tf_frames_match_the_unbounded_history():
    run_machine(TfWindowedMachine)
