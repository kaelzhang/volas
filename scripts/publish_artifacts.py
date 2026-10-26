#!/usr/bin/env python3
"""Upload exactly one build's verified delivery manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import sys

if __package__:
    from . import build_config as processes
else:
    import build_config as processes


def artifacts(manifest):
    data = json.loads(manifest.read_text())
    if data["schema"] != 1 or not data["artifacts"]:
        raise ValueError("invalid delivery manifest")
    paths = []
    for item in data["artifacts"]:
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(manifest.resolve().parent):
            raise ValueError("delivery artifact outside build directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("delivery artifact hash mismatch")
        paths.append(str(path))
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate delivery artifact")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    paths = artifacts(args.manifest)
    processes.run_child(
        [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--config-file",
            str(Path.home() / ".pypirc"),
            "-r",
            "pypi",
            "--verbose",
            *paths,
        ],
        env=dict(os.environ),
        check=True,
    )
