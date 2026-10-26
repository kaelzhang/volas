"""Read the repository's standard Rust and Python build-tool pins."""

from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]
RUST = TOOLCHAIN["channel"]
COMPONENTS = TOOLCHAIN["components"]
BUILD_REQUIRES = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"][
    "requires"
]
if len(BUILD_REQUIRES) != 1 or not re.fullmatch(
    r"maturin==\d+\.\d+\.\d+", BUILD_REQUIRES[0]
):
    raise ValueError("build-system.requires must pin exactly one maturin version")
MATURIN = BUILD_REQUIRES[0].split("==")[1]
