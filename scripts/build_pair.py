#!/usr/bin/env python3
"""Validate installed release identities before either performance driver runs."""

import argparse
import json
from pathlib import Path

from build_release import verify_library


def manifest(site):
    site = Path(site).resolve()
    data = json.loads((site / "volas-build.json").read_text())
    if data["schema"] != 2 or data["mode"] not in ("normal", "use"):
        raise ValueError("performance requires a final normal/use build")
    config = data['identity']['configuration']
    if not {'flags', 'release', 'features', 'tools'} <= config.keys() or not isinstance(config['flags'], list):
        raise ValueError('missing effective build configuration')
    native = list(site.rglob("volas_rs*.so")) + list(site.rglob("volas_rs*.pyd"))
    if len(native) != 1:
        raise ValueError("missing or ambiguous native library")
    verify_library(native[0], site, data["native_sha256"])
    if data["mode"] == "use" and not data.get("profile_sha256"):
        raise ValueError("PGO build missing profile identity")
    return data


def validate(base, head, comparison="regression"):
    if comparison not in ("regression", "optimization"):
        raise ValueError("unknown performance comparison mode")
    b, h = manifest(base), manifest(head)
    bi, hi = b["identity"].copy(), h["identity"].copy()
    bs, hs = bi.pop("source"), hi.pop("source")
    if bi != hi:
        raise ValueError("build provenance mismatch")
    if comparison == "optimization":
        if bs != hs or (b["mode"], h["mode"]) != ("normal", "use"):
            raise ValueError("optimization comparison requires same-source normal/PGO")
    elif b["mode"] != h["mode"]:
        raise ValueError("regression comparison requires matching build modes")
    return {"comparison": comparison, "base": b, "head": h}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument(
        "--comparison", choices=["regression", "optimization"], default="regression"
    )
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = validate(args.base, args.head, args.comparison)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"validated {result['base']['mode']}/{result['head']['mode']} ({args.comparison})"
    )
