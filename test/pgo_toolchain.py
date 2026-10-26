"""Real local compiler/LLVM tools and clean subprocess environments for tooling tests."""

from pathlib import Path
import subprocess

from scripts.build_config import build_environment

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
    return dict(build_environment(), CARGO_NET_OFFLINE="true", **extra)


def extension_link_flags(target):
    """Mirror the extension ABI required by the actual compilation target."""
    return (
        ["-Clink-arg=-undefined", "-Clink-arg=dynamic_lookup"]
        if target.endswith("apple-darwin")
        else []
    )
