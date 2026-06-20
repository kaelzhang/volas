#!/usr/bin/env python3
"""Hard cap on source-file length — run by ``make lint``.

A file that grows past ``MAX_LINES`` must be split by concern into focused
modules; this gate turns that into a build failure so an overly-long file (like
the old 1.6k-line ``series.rs`` / 1.4k-line ``frame_methods.rs``) can never creep
back. It checks **code only** — Rust (``.rs``) and Python (``.py``) sources — and
deliberately ignores:

  * documentation (``.md``) and data fixtures (``.csv`` / ``.json``),
  * hand-maintained type stubs (``.pyi``, an API mirror that grows with the API),
  * build / run artifacts (``target/``, ``dist/``, ``.benchmarks/``, …).

Run directly with ``python scripts/check_line_count.py``; exits non-zero (and
lists the offenders, largest first) when any code file exceeds the cap.
"""

from __future__ import annotations

import os
import pathlib
import sys

# The single knob. 1000 lines is the canonical "this file is doing too much"
# threshold; it catches the bloat we just split apart while leaving cohesive core
# files (the struct + helpers, the indicator kernels) comfortably under.
MAX_LINES = 1000

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE_EXTS = {".rs", ".py"}
# Directories that hold build / run artifacts, vendored code, or caches — never
# first-party source, so never line-count-gated.
EXCLUDE_DIRS = {
    "target", "dist", "build", ".benchmarks", "htmlcov", "__pycache__",
    "node_modules", ".git", ".eggs", ".pytest_cache", ".ruff_cache",
}


def code_files() -> "list[pathlib.Path]":
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune excluded dirs in place so we never descend into artifacts.
        dirnames[:] = [
            d for d in dirnames if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")
        ]
        for fn in filenames:
            if pathlib.Path(fn).suffix in CODE_EXTS:
                found.append(pathlib.Path(dirpath) / fn)
    return found


def line_count(path: pathlib.Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def main() -> int:
    over = [
        (line_count(p), p.relative_to(ROOT))
        for p in code_files()
    ]
    over = [(n, rel) for n, rel in over if n > MAX_LINES]
    if over:
        print(f"line-count gate FAILED — {len(over)} file(s) over {MAX_LINES} lines:")
        for n, rel in sorted(over, reverse=True):
            print(f"  {n:>5}  {rel}")
        print(
            f"\nSplit each by concern into focused modules so every code file is "
            f"<= {MAX_LINES} lines."
        )
        return 1
    print(f"line-count gate OK — every code file is <= {MAX_LINES} lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
