#!/usr/bin/env python3
"""Run a developer build with owned inputs and a durable environment record."""

import argparse
from pathlib import Path

from build_config import exec_child


RUST_CHECKS = {
    "checked": "-C debug-assertions -C overflow-checks",
    "coverage": "-Cinstrument-coverage",
}
ALLOCATOR_CHECKS = {"MallocPreScribble": "1", "MALLOC_PERTURB_": "170"}


def shell_inputs(settings, rust_checks):
    """Accept only inputs whose value this shell boundary can substantiate.

    Shell expansion erases assignment provenance. Never promote arbitrary flags
    through --set; the owner constructs the two supported checking profiles.
    Python owners still construct their own explicit inputs via build_config.
    """
    owned = {}
    for item in settings:
        key, sep, value = item.partition("=")
        if not sep or key in owned or not (
            key in ALLOCATOR_CHECKS and value == ALLOCATOR_CHECKS[key]
            or key == "LLVM_PROFILE_FILE" and rust_checks == "coverage"
            and Path(value).is_absolute() and value.endswith("-%m-%p.profraw")
        ):
            raise ValueError(f"unbound owned input: {key}")
        owned[key] = value
    if rust_checks:
        owned["RUSTFLAGS"] = RUST_CHECKS[rust_checks]
    return owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--rust-checks", choices=RUST_CHECKS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a build command is required")
    try:
        owned = shell_inputs(args.set, args.rust_checks)
    except ValueError as error:
        parser.error(str(error))
    exec_child(command, owned=owned, record=args.record)


if __name__ == "__main__":
    main()
