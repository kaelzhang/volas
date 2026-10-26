#!/usr/bin/env python3
"""Run a developer build with owned inputs and a durable environment record."""

import argparse
from pathlib import Path

from build_config import exec_child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a build command is required")
    owned = dict(item.split("=", 1) for item in args.set)
    exec_child(command, owned=owned, record=args.record)


if __name__ == "__main__":
    main()
