#!/usr/bin/env python3
"""Bump the workspace version in Cargo.toml (SemVer, no leading 'v').

    bump_version.py {major|minor|patch}          rewrite [workspace.package].version, print the new version
    bump_version.py --next {major|minor|patch}   print the next version only; do NOT edit

Only the ``version`` line inside the ``[workspace.package]`` table is touched. The
printed version is the single line written to stdout, so ``make bump`` can capture
it for the git tag.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

CARGO = Path(__file__).resolve().parent.parent / "Cargo.toml"
KINDS = ("major", "minor", "patch")


def current_version(text: str) -> str:
    return tomllib.loads(text)["workspace"]["package"]["version"]


def bump(version: str, kind: str) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        sys.exit(f"version '{version}' is not MAJOR.MINOR.PATCH")
    major, minor, patch = (int(p) for p in parts)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def rewrite(text: str, new: str) -> str:
    """Replace the version line inside the [workspace.package] table only."""
    lines = text.splitlines(keepends=True)
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == "[workspace.package]"
            continue
        if in_section and re.match(r"\s*version\s*=", line):
            lines[i] = re.sub(r'(version\s*=\s*")[^"]*(")', rf"\g<1>{new}\g<2>", line, count=1)
            return "".join(lines)
    sys.exit("could not find the [workspace.package] version line in Cargo.toml")


def main(argv: list[str]) -> None:
    args = argv[1:]
    dry = bool(args) and args[0] == "--next"
    if dry:
        args = args[1:]
    if len(args) != 1 or args[0] not in KINDS:
        sys.exit(f"usage: bump_version.py [--next] {{{'|'.join(KINDS)}}}")
    text = CARGO.read_text()
    new = bump(current_version(text), args[0])
    if not dry:
        CARGO.write_text(rewrite(text, new))
        # Cargo.lock records the workspace members' versions too; the release
        # builds pass --locked (reproducible builds), so a bump that leaves the
        # lock stale fails every wheel job ("cannot update the lock file").
        # `cargo update --workspace` rewrites ONLY the workspace-member entries.
        subprocess.run(
            ["cargo", "update", "--workspace", "--quiet"],
            cwd=CARGO.parent, check=True,
        )
    print(new)


if __name__ == "__main__":
    main(sys.argv)
