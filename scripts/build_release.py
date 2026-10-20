#!/usr/bin/env python3
"""Single release-wheel owner. PGO is opt-in; profiles are source/ABI specific."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import shlex
import tempfile
import subprocess
import sys
import sysconfig
import time
import zipfile

DRIVER = Path(__file__).resolve().parent.parent
RUST = "1.96.0"
MATURIN = "1.13.3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_environment(env):
    bad = [
        k
        for k in env
        if k
        in {
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "MATURIN_PGO",
            "CARGO_BUILD_RUSTFLAGS",
        }
        or (k.startswith("CARGO_TARGET_") and k.endswith("_RUSTFLAGS"))
        or k.startswith("CARGO_PROFILE_RELEASE_")
    ]
    if bad:
        raise ValueError(f"ambient build overrides rejected: {sorted(bad)}")


def flag_environment(flags):
    """Cargo's encoded format preserves whitespace inside individual arguments."""
    return dict(os.environ, CARGO_ENCODED_RUSTFLAGS="\x1f".join(flags))


def verify_library(path, site, expected):
    path, site = Path(path).resolve(), Path(site).resolve()
    if not path.is_relative_to(site) or digest(path) != expected:
        raise ValueError(f"training library identity mismatch: {path}")


def raw_files(raw):
    files = sorted(Path(raw).glob("*.profraw"))
    if not files or any(p.stat().st_size == 0 for p in files):
        raise ValueError("empty or missing raw profiles")
    return files


def validate_profile(profile, identity, provenance, profdata):
    if provenance["identity"] != identity:
        raise ValueError("profile provenance mismatch")
    if not Path(profile).is_file() or not Path(profile).stat().st_size:
        raise ValueError("empty or missing profile")
    if digest(profile) != provenance["profile_sha256"]:
        raise ValueError("profile hash mismatch")
    if profdata:
        subprocess.run(
            [str(profdata), "show", str(profile)], check=True, capture_output=True
        )


def output(cmd, **kwargs):
    return subprocess.check_output(cmd, text=True, **kwargs).strip()


def source_hash(root):
    paths = [
        root / n
        for n in ("Cargo.toml", "Cargo.lock", "pyproject.toml", ".cargo/config.toml")
    ]
    paths += sorted((root / "crates").rglob("*.rs")) + sorted(
        (root / "crates").rglob("Cargo.toml")
    )
    paths += sorted((root / "volas").glob("*.py"))
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def identity(root, target):
    rust = output(["rustc", "-Vv"])
    if f"release: {RUST}\n" not in rust + "\n":
        raise ValueError(f"requires rustc {RUST}: {rust}")
    if output([sys.executable, "-m", "maturin", "--version"]) != f"maturin {MATURIN}":
        raise ValueError(f"requires maturin {MATURIN}")
    host = next(
        line.split(": ", 1)[1] for line in rust.splitlines() if line.startswith("host:")
    )
    llvm = (
        Path(output(["rustc", "--print", "sysroot"]))
        / "lib/rustlib"
        / host
        / "bin/llvm-profdata"
    )
    if not llvm.is_file():
        raise ValueError("install llvm-tools-preview for the pinned Rust toolchain")
    training = [
        DRIVER / "scripts/build_release.py",
        DRIVER / "scripts/count_indicators.py",
        DRIVER / "scripts/pgo_train.py",
        DRIVER / "test/bench_workloads.py",
        DRIVER / "test/pgo_inventory.json",
        DRIVER / "scripts/pgo-requirements.txt",
        DRIVER / "test/bench_tf_live.py",
        DRIVER / "test/bench_first_cache.py",
    ]
    return {
        "source": source_hash(root),
        "rustc": rust,
        "cargo": output(["cargo", "-V"]),
        "target": target,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "maturin": MATURIN,
        "llvm": output([str(llvm), "--version"]),
        "llvm_sha256": digest(llvm),
        "training": {str(p.relative_to(DRIVER)): digest(p) for p in training},
        "dependencies": output(
            [
                sys.executable,
                "-c",
                'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ("numpy","pyarrow","tzdata")} ,sort_keys=True))',
            ]
        ),
        "platform_env": {
            k: os.environ.get(k)
            for k in (
                "MACOSX_DEPLOYMENT_TARGET",
                "SDKROOT",
                "CC",
                "CARGO_BUILD_TARGET",
                "RUSTUP_TOOLCHAIN",
            )
        },
    }, llvm


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run(cmd, log, root, env=None, timeout=None):
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with Path(log).open("w") as f:
        f.write(json.dumps([str(x) for x in cmd]) + "\n")
        f.flush()
        subprocess.run(
            [str(x) for x in cmd],
            cwd=root,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.monotonic() - start
    write(
        Path(log).with_suffix(".metrics.json"),
        {
            "wall_s": elapsed,
            "cpu_s": after.ru_utime
            + after.ru_stime
            - before.ru_utime
            - before.ru_stime,
            "children_maxrss": after.ru_maxrss,
        },
    )
    print(f"{Path(log).name}: {elapsed:.2f}s", flush=True)


def install(wheel, site, root, log):
    if site.exists():
        raise ValueError(f"site must be fresh: {site}")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            site,
            wheel,
        ],
        log,
        root,
    )
    libs = list(site.rglob("volas_rs*.so")) + list(site.rglob("volas_rs*.pyd"))
    if len(libs) != 1:
        raise ValueError(f"expected one native library: {libs}")
    return libs[0]


def resolved_flags(root, target, flags, log):
    """Ask Cargo to resolve target/cfg config, before maturin adds link flags.

    The explicit triple marker precedes cfg-target flags in Cargo's argv. A tiny
    native-independent probe records that tail; no second ISA configuration lives
    in this script. Keep the complete resolver invocation for auditing.
    """
    with tempfile.TemporaryDirectory(prefix="volas-flags-") as tmp:
        probe = Path(tmp)
        (probe / "src").mkdir()
        (probe / "Cargo.toml").write_text(
            '[package]\nname="volas-flags-probe"\nversion="0.0.0"\nedition="2021"\n[workspace]\n'
        )
        (probe / "src/main.rs").write_text("fn main() {}\n")
        marker = "--cfg=volas_flags_marker"
        config = f"target.{target}.rustflags={json.dumps([marker, *flags])}"
        run(
            [
                "cargo",
                "build",
                "--manifest-path",
                probe / "Cargo.toml",
                "--target",
                target,
                "--target-dir",
                probe / "target",
                "--config",
                config,
                "-v",
            ],
            log,
            root,
        )
        lines = Path(log).read_text().splitlines()
        command = next(
            line.strip()[9:-1]
            for line in lines
            if "Running `" in line and "--crate-name volas_flags_probe" in line
        )
        argv = shlex.split(command)
        return argv[argv.index(marker) + 1 :]


def validate_argv(log, mode, target):
    commands = [
        shlex.split(line.strip()[9:-1])
        for line in Path(log).read_text().splitlines()
        if line.strip().startswith("Running `") and "--crate-name" in line
    ]
    target_commands = [
        c for c in commands if "--target" in c and c[c.index("--target") + 1] == target
    ]
    if not any("volas_rs" in c for c in target_commands):
        raise ValueError("missing actual cdylib rustc argv")
    for c in commands:
        feedback = [v for v in c if "profile-generate=" in v or "profile-use=" in v]
        is_target = c in target_commands
        if not is_target and feedback:
            raise ValueError("host build unexpectedly instrumented")
        if is_target and mode in ("generate", "use") and not feedback:
            raise ValueError("target crate missing PGO flags")
        if (
            is_target
            and target.startswith("x86_64")
            and "-Ctarget-feature=+fma" not in c
        ):
            raise ValueError("target crate missing configured FMA")
        if "volas_rs" in c and target.endswith("apple-darwin"):
            if "link-arg=dynamic_lookup" not in c or "link-arg=-undefined" not in c:
                raise ValueError("missing macOS extension link flags")


def build(root, artifacts, mode, target, ident, profile=None):
    stage = artifacts / mode
    stage.mkdir(parents=True, exist_ok=False)
    flags = []
    if mode == "generate":
        (stage / "raw").mkdir()
        flags = [f"-Cprofile-generate={stage / 'raw'}"]
    if mode == "use":
        flags = [f"-Cprofile-use={profile}", "-Cllvm-args=-pgo-warn-missing-function"]
    cmd = [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "--release",
        "--locked",
        "--target",
        target,
        "--interpreter",
        sys.executable,
        "--target-dir",
        stage / "target",
        "--out",
        stage / "wheels",
        "-v",
    ]
    resolved = resolved_flags(root, target, flags, stage / "flags.log")
    env = flag_environment(resolved)
    run(cmd, stage / "build.log", root, env)
    build_log = (stage / "build.log").read_text()
    expected = (
        "profile-generate="
        if mode == "generate"
        else "profile-use="
        if mode == "use"
        else None
    )
    if expected and expected not in build_log:
        raise ValueError("maturin masked PGO flags; inspect actual rustc argv")
    validate_argv(stage / "build.log", mode, target)
    if (
        "hash mismatch" in build_log.lower()
        or "control flow change" in build_log.lower()
    ):
        raise ValueError("stale profile / CFG mismatch; see build log")
    wheels = list((stage / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as z:
        if not any(n.endswith(".pyi") for n in z.namelist()):
            raise ValueError("wheel missing stubs")
    lib = install(wheel, stage / "site", root, stage / "install.log")
    manifest = {
        "schema": 1,
        "flags": sorted(
            set(
                f
                for f in resolved
                if "profile-generate=" not in f
                and "profile-use=" not in f
                and "pgo-warn-missing-function" not in f
            )
        ),
        "mode": mode,
        "identity": ident,
        "wheel": str(wheel),
        "wheel_sha256": digest(wheel),
        "native": str(lib),
        "native_sha256": digest(lib),
    }
    write(stage / "site/volas-build.json", manifest)
    write(stage / "build.json", manifest)
    return stage, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path.cwd())
    ap.add_argument(
        "--mode", choices=["normal", "generate", "use", "pgo"], default="normal"
    )
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--target")
    ap.add_argument("--profile", type=Path)
    ap.add_argument(
        "--generated",
        type=Path,
        help="reuse a verified instrumented stage; training always starts fresh",
    )
    ap.add_argument(
        "--install",
        action="store_true",
        help="install final wheel into this interpreter",
    )
    ap.add_argument("--sdist", action="store_true")
    args = ap.parse_args()
    check_environment(os.environ)
    root, artifacts = args.source.resolve(), args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    target = args.target or next(
        line.split(": ")[1]
        for line in output(["rustc", "-Vv"]).splitlines()
        if line.startswith("host:")
    )
    ident, llvm = identity(root, target)
    write(artifacts / "identity.json", ident)
    profile = args.profile.resolve() if args.profile else artifacts / "profile.profdata"
    if args.mode in ("generate", "pgo"):
        if args.generated:
            if args.mode != "pgo":
                raise ValueError("--generated requires pgo mode")
            stage = args.generated.resolve()
            manifest = json.loads((stage / "build.json").read_text())
            prior, current = manifest["identity"].copy(), ident.copy()
            prior.pop("training")
            current.pop("training")
            if prior != current or manifest["mode"] != "generate":
                raise ValueError("generated build provenance mismatch")
            verify_library(
                Path(manifest["native"]), stage / "site", manifest["native_sha256"]
            )
            validate_argv(stage / "build.log", "generate", target)
            write(artifacts / "generated.json", manifest)
        else:
            stage, manifest = build(root, artifacts, "generate", target, ident)
        if args.mode == "generate":
            return
        raw = artifacts / "training-raw"
        raw.mkdir()
        env = dict(os.environ, LLVM_PROFILE_FILE=str(raw / "train-%m-%p.profraw"))
        run(
            [
                sys.executable,
                DRIVER / "scripts/pgo_train.py",
                "--site",
                stage / "site",
                "--native-sha256",
                manifest["native_sha256"],
                "--output",
                artifacts / "training.json",
                "--identity",
                artifacts / "identity.json",
            ],
            artifacts / "training.log",
            artifacts,
            env,
            620,
        )
        if identity(root, target)[0] != ident:
            raise ValueError("provenance changed during training")
        raws = raw_files(raw)
        run(
            [llvm, "merge", "--failure-mode=any", "-o", profile, *raws],
            artifacts / "merge.log",
            artifacts,
        )
        run(
            [llvm, "show", "--all-functions", "--counts", profile],
            artifacts / "profile-show.log",
            artifacts,
        )
        write(
            profile.with_suffix(".json"),
            {
                "identity": ident,
                "profile_sha256": digest(profile),
                "raw": {p.name: digest(p) for p in raws},
            },
        )
    if args.mode in ("use", "pgo"):
        if identity(root, target)[0] != ident:
            raise ValueError("provenance changed before use")
        validate_profile(
            profile, ident, json.loads(profile.with_suffix(".json").read_text()), llvm
        )
        stage, manifest = build(root, artifacts, "use", target, ident, profile)
        manifest["profile_sha256"] = digest(profile)
    else:
        stage, manifest = build(root, artifacts, "normal", target, ident)
    # A fresh process proves both library identity and absence of instrumentation.
    probe = artifacts / "final-probe"
    probe.mkdir()
    env = dict(
        os.environ,
        PYTHONPATH=str(stage / "site"),
        LLVM_PROFILE_FILE=str(probe / "%m-%p.profraw"),
    )
    run(
        [
            sys.executable,
            "-c",
            'import volas; assert len(volas.DataFrame({"close":[1.,2.,3.]}).exec("ma:2")) == 3',
        ],
        artifacts / "smoke.log",
        artifacts,
        env,
    )
    if list(probe.iterdir()):
        raise ValueError("final wheel is instrumented")
    write(stage / "site/volas-build.json", manifest)
    write(artifacts / "final.json", manifest)
    if args.install:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                manifest["wheel"],
            ],
            artifacts / "install-final.log",
            artifacts,
        )
        installed_manifest = Path(sysconfig.get_path("purelib")) / "volas-build.json"
        write(installed_manifest, manifest)
    if args.sdist:
        run(
            [sys.executable, "-m", "maturin", "sdist", "--out", artifacts / "sdist"],
            artifacts / "sdist.log",
            root,
        )


if __name__ == "__main__":
    main()
