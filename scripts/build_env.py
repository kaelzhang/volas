#!/usr/bin/env python3
"""Run a developer build with owned inputs and a durable environment record."""

import argparse
import json
import os
from pathlib import Path
import uuid

from build_config import build_environment, environment_record


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
    env = dict(build_environment(), **owned)
    record = args.record or Path("target/build-environments") / f"{uuid.uuid4().hex}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"argv": command,
                                  "environment": environment_record(owned=owned)},
                                 indent=2) + "\n")
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
