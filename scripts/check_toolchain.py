#!/usr/bin/env python3
"""Provisioning values and active-tool checks for existing CI entry points."""

import argparse
from pathlib import Path
import subprocess
import sys

from toolchain import RUST, MATURIN, COMPONENTS


def check(rust_only=False):
    rust = subprocess.check_output(["rustc", "-Vv"], text=True)
    fields = dict(line.split(": ", 1) for line in rust.splitlines() if ": " in line)
    if fields.get("release") != RUST:
        raise ValueError(f"requires rustc {RUST}: {rust}")
    cargo = subprocess.check_output(["cargo", "--version"], text=True).strip()
    if cargo.split()[1] != RUST:
        raise ValueError(f"requires cargo {RUST}: {cargo}")
    maturin = ""
    if not rust_only:
        maturin = subprocess.check_output(
            [sys.executable, "-m", "maturin", "--version"],
            text=True,
        ).strip()
        if maturin != f"maturin {MATURIN}":
            raise ValueError(f"requires maturin {MATURIN}: {maturin}")
    sysroot = Path(
        subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    )
    llvm = sysroot / "lib/rustlib" / fields["host"] / "bin"
    for name in ("llvm-profdata", "llvm-cov", "llvm-objdump"):
        path = llvm / name
        if not path.is_file():
            raise ValueError(
                f"install llvm-tools-preview for Rust {RUST}: missing {path}"
            )
        version = subprocess.check_output([str(path), "--version"], text=True).strip()
        if fields["LLVM version"] not in version:
            raise ValueError(f"LLVM version mismatch: {path}: {version}")
        print(f"{path}: {version}")
    print(f"rustc {RUST}; {cargo}; {maturin}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--versions",
        action="store_true",
        help="GitHub step outputs from the build owner",
    )
    ap.add_argument(
        "--rust-only", action="store_true", help="check Rust and its LLVM tools"
    )
    ap.add_argument("--value", choices=["rust", "maturin", "components"])
    args = ap.parse_args()
    values = {"rust": RUST, "maturin": MATURIN, "components": ",".join(COMPONENTS)}
    if args.value:
        print(values[args.value])
        return
    if args.versions:
        print("\n".join(f"{key}={value}" for key, value in values.items()))
    else:
        check(args.rust_only)


if __name__ == "__main__":
    main()
