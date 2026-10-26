#!/usr/bin/env python3
"""Own installed build identities, their uninstall lifetime and coverage snapshots."""

import argparse
import base64
import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

MANIFEST = "volas-build.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(site, data):
    """Register the sidecar as distribution-owned, so pip replacement removes it.

    This includes unwrapped `maturin develop`: no special wrapper or import-time
    repair is needed to invalidate a release identity on non-release installs.
    """
    site = Path(site)
    records = list(site.glob("volas-*.dist-info/RECORD"))
    if len(records) != 1:
        raise ValueError(f"expected one installed volas RECORD: {records}")
    path = site / MANIFEST
    path.write_text(json.dumps(data, indent=2) + "\n")
    with records[0].open(newline="") as f:
        rows = [row for row in csv.reader(f) if row[0] != MANIFEST]
    encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        .rstrip(b"=")
        .decode()
    )
    rows.append([MANIFEST, "sha256=" + encoded, str(path.stat().st_size)])
    with records[0].open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def native_path():
    module = importlib.import_module("volas_rs")
    if not module.__file__.endswith((".so", ".pyd")):
        module = importlib.import_module("volas_rs.volas_rs")
    return Path(module.__file__)


def verify(data):
    if digest(data["wheel"]) != data["wheel_sha256"]:
        raise ValueError("release wheel hash mismatch")
    if digest(native_path()) != data["native_sha256"]:
        raise ValueError("installed native library hash mismatch")


def save(destination):
    """Coverage validates before replacing, and protects an independent copy."""
    path = Path(sysconfig.get_path("purelib")) / MANIFEST
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    verify(data)
    Path(destination).write_bytes(path.read_bytes())
    return data["wheel"]


def install(data):
    if digest(data["wheel"]) != data["wheel_sha256"]:
        raise ValueError("release wheel hash mismatch")
    boundary = [sys.executable, str(Path(__file__).with_name("build_env.py")), "--"]
    subprocess.run(
        [*boundary, sys.executable, "-m", "pip", "install", data["wheel"]],
        check=True,
    )
    # Resolve missing dependencies first, keeping compatible installed versions.
    # Reinstall only this wheel so debug/coverage bytes cannot survive pip's
    # same-version shortcut, without forcing unrelated dependency replacements.
    subprocess.run(
        [
            *boundary,
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            data["wheel"],
        ],
        check=True,
    )
    verify(data)
    publish(Path(sysconfig.get_path("purelib")), data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["save", "install", "restore"])
    ap.add_argument("snapshot", type=Path)
    args = ap.parse_args()
    if args.action == "save":
        wheel = save(args.snapshot)
        if wheel:
            print(wheel)
    else:
        install(json.loads(args.snapshot.read_text()))
