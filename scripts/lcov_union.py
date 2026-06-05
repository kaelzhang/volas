#!/usr/bin/env python3
"""Union two LCOV reports at the per-line level and print a coverage table.

`make coverage` exercises the same Rust source twice: once compiled into the
`cargo test` binaries, once into the maturin `.so` driven by the pytest suite.
`llvm-cov` cannot merge the two — the test build and the cdylib have different
coverage-mapping hashes, so it treats each as a separate instantiation and
reports their *average*. That unfairly penalises a file that is thoroughly
covered by one suite but not re-covered by the other (e.g. `exec.rs` is ~98 %
from `cargo test` yet shows ~70 % "combined").

This computes the real union instead: a source line is covered iff it ran in
EITHER suite (the max of the two hit counts). The line universe is the union of
the two inputs, matching `cargo-llvm-cov`'s own accounting (inline `#[cfg(test)]`
lines are counted, as that tool already does).

Usage: lcov_union.py CARGO.lcov PYTEST.lcov [UNION_OUT.lcov]
If UNION_OUT is given, the merged (max-count) LCOV is written there too, so it
can be rendered to HTML with `genhtml`.
Exit code is non-zero if either input is missing or empty.
"""
import os
import sys

EXCLUDE = "volas-python"  # the pyo3 glue: pytest-only, reported behaviourally
MARK = "crates/"          # only workspace-library sources are in scope


def normalize(path):
    """Repo-relative key, e.g. 'volas-directive/src/exec.rs'."""
    i = path.find(MARK)
    return path[i + len(MARK):] if i >= 0 else path


def excluded_lines(name):
    """Line numbers a source opts out of coverage with the standard LCOV markers
    (`// LCOV_EXCL_LINE` on a line, or a `// LCOV_EXCL_START` … `// LCOV_EXCL_STOP`
    block). Reserved for genuinely-unreachable defensive code — `unreachable!()`,
    invariant `panic!()` guards, and test-scaffolding match arms that only fire on a
    bug — which cannot be executed without failing."""
    excl = set()
    try:
        with open(os.path.join(MARK, name)) as fh:
            in_block = False
            for i, line in enumerate(fh, 1):
                if "LCOV_EXCL_START" in line:
                    in_block = True
                if in_block or "LCOV_EXCL_LINE" in line:
                    excl.add(i)
                if "LCOV_EXCL_STOP" in line:
                    in_block = False
    except OSError:
        pass
    return excl


def parse(path):
    """{normalized file -> {line -> max hit count}} for workspace sources."""
    files = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("SF:"):
                raw = line[3:].strip()
                cur = normalize(raw) if (MARK in raw and EXCLUDE not in raw) else None
                if cur is not None:
                    files.setdefault(cur, {})
            elif line.startswith("DA:") and cur is not None:
                num, _, cnt = line[3:].strip().partition(",")
                num = int(num)
                cnt = int(cnt.split(",", 1)[0])  # drop any checksum field
                files[cur][num] = max(files[cur].get(num, 0), cnt)
            elif line.startswith("end_of_record"):
                cur = None
    return files


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    cargo = parse(sys.argv[1])
    pytest = parse(sys.argv[2])
    if not cargo or not pytest:
        sys.exit(f"error: empty coverage ({len(cargo)} cargo / {len(pytest)} pytest files)")

    names = sorted(set(cargo) | set(pytest))
    # merged[name] = {line -> unioned (max) hit count}, minus lines that opt out of
    # coverage via LCOV_EXCL markers (unreachable defensive code).
    merged = {
        name: {
            ln: max(cargo.get(name, {}).get(ln, 0), pytest.get(name, {}).get(ln, 0))
            for ln in set(cargo.get(name, {})) | set(pytest.get(name, {}))
            if ln not in excluded_lines(name)
        }
        for name in names
    }

    width = max(len(n) for n in names)
    tot_found = tot_hit = 0
    print(f"{'FILE':<{width}}  COVER    HIT/FOUND")
    print("-" * (width + 20))
    for name in names:
        lines = merged[name]
        found = len(lines)
        hit = sum(1 for c in lines.values() if c > 0)
        tot_found += found
        tot_hit += hit
        pct = 100.0 * hit / found if found else 100.0
        print(f"{name:<{width}}  {pct:6.2f}%  {hit:5d}/{found:<5d}")
    print("-" * (width + 20))
    pct = 100.0 * tot_hit / tot_found if tot_found else 0.0
    print(f"{'TOTAL (cargo ∪ pytest)':<{width}}  {pct:6.2f}%  {tot_hit:5d}/{tot_found:<5d}")

    # List the lines covered by NEITHER suite, to drive the push to 100 %.
    gaps = [(n, sorted(ln for ln, c in merged[n].items() if c == 0)) for n in names]
    gaps = [(n, m) for n, m in gaps if m]
    if gaps:
        print("\nUncovered by either suite:")
        for name, miss in gaps:
            print(f"  {name}: {', '.join(map(str, miss))}")

    # Optionally emit the merged LCOV (for `genhtml`).
    if len(sys.argv) == 4:
        with open(sys.argv[3], "w") as out:
            for name in names:
                out.write(f"SF:{MARK}{name}\n")
                for ln in sorted(merged[name]):
                    out.write(f"DA:{ln},{merged[name][ln]}\n")
                found = len(merged[name])
                hit = sum(1 for c in merged[name].values() if c > 0)
                out.write(f"LF:{found}\nLH:{hit}\nend_of_record\n")


if __name__ == "__main__":
    main()
