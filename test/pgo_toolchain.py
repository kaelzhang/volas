"""Real local compiler/LLVM tools and clean subprocess environments for tooling tests."""

import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOST = (
    subprocess.check_output(["rustc", "-vV"], text=True)
    .split("host: ")[1]
    .splitlines()[0]
)
SYSROOT = Path(
    subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
)
LLVM = SYSROOT / "lib/rustlib" / HOST / "bin"


def clean_env(**extra):
    env = dict(os.environ, CARGO_NET_OFFLINE="true")
    for key in ("LDFLAGS", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        env.pop(key, None)
    return dict(env, **extra)


def extension_link_flags(target):
    """Mirror the extension ABI required by the actual compilation target."""
    return (
        ["-Clink-arg=-undefined", "-Clink-arg=dynamic_lookup"]
        if target.endswith("apple-darwin")
        else []
    )
