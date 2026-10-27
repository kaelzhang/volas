"""Captured benchmark calls and their process-CPU measurement protocol."""

import os
import sys
import time


def allocator_settings():
    """Limit magazines for affected allocation classes to reduce migration noise.

    These are measurement-worker inputs, applied before interpreter startup;
    they do not change release builds or the installed product.
    """
    if sys.platform == "darwin":
        return {"MallocMaxMagazines": "1", "MallocMaxMediumMagazines": "1"}
    return {}


def environment():
    return {**os.environ, **allocator_settings()}


class Capture:
    def __init__(self):
        self.extra_info = {}
        self.fn = None
        self.setup = None
        self.args = ()
        self.kwargs = {}
        self.single_call = False
        self.warmup_rounds = 5

    def __call__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def pedantic(self, fn, args=(), kwargs=None, setup=None, warmup_rounds=0, **options):
        self.fn = fn
        self.setup = setup
        self.args = args
        self.kwargs = kwargs or {}
        self.single_call = True
        self.warmup_rounds = warmup_rounds


def prepare(cap):
    """Resolve arguments outside the clocks, as pytest-benchmark pedantic does."""
    if cap.setup:
        arguments = cap.setup()
        if arguments:
            if cap.args or cap.kwargs:
                raise TypeError("Can't use `args` or `kwargs` if `setup` returns the arguments.")
            return arguments
    return cap.args, cap.kwargs


def warmup(cap):
    """Use the workload's declared warmup; discard each result as the wall runner does."""
    for _ in range(cap.warmup_rounds):
        args, kwargs = prepare(cap)
        cap.fn(*args, **kwargs)


def measure(cap, reps):
    if cap.single_call:
        wall = cpu = 0
        previous = None
        for _ in range(reps):
            a, k = prepare(cap)
            c = time.process_time_ns()
            w = time.perf_counter_ns()
            result = cap.fn(*a, **k)
            elapsed_wall = time.perf_counter_ns() - w
            elapsed_cpu = time.process_time_ns() - c
            # The real pedantic runner retains the previous result through the
            # next call's end clocks, replacing it only outside the timed region.
            previous = result
            del result
            wall += elapsed_wall
            cpu += elapsed_cpu
        del previous
    else:
        a, k = prepare(cap)
        c = time.process_time_ns()
        w = time.perf_counter_ns()
        for _ in range(reps):
            cap.fn(*a, **k)
        wall = time.perf_counter_ns() - w
        cpu = time.process_time_ns() - c
    return {"wall": wall / reps, "cpu": cpu / reps}
