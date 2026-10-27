"""Captured benchmark calls and their process-CPU measurement protocol."""

import os
import sys
import time


def allocator_settings():
    """Keep fixture pages and result allocations in the same macOS magazine.

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

    def __call__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def pedantic(self, fn, args=(), kwargs=None, setup=None, **options):
        self.fn = fn
        self.setup = setup
        self.args = args
        self.kwargs = kwargs or {}
        self.single_call = True


def measure(cap, reps):
    if cap.single_call:
        wall = cpu = 0
        for _ in range(reps):
            a, k = cap.setup() if cap.setup else (cap.args, cap.kwargs)
            c = time.process_time_ns()
            w = time.perf_counter_ns()
            result = cap.fn(*a, **k)
            elapsed_wall = time.perf_counter_ns() - w
            elapsed_cpu = time.process_time_ns() - c
            # Match pytest-benchmark pedantic(iterations=1): return a live
            # result past both clocks, then release it before the next call.
            del result
            wall += elapsed_wall
            cpu += elapsed_cpu
    else:
        c = time.process_time_ns()
        w = time.perf_counter_ns()
        for _ in range(reps):
            cap.fn(*cap.args, **cap.kwargs)
        wall = time.perf_counter_ns() - w
        cpu = time.process_time_ns() - c
    return {"wall": wall / reps, "cpu": cpu / reps}
