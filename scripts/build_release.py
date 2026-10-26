#!/usr/bin/env python3
"""Single release-wheel owner. PGO is opt-in; profiles are source/ABI specific."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
from pathlib import Path
import sys
import sysconfig
import tempfile
import time
import tomllib
import zipfile

DRIVER = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The driver is also used against source archives that do not contain this tooling.
sys.path.insert(0, str(DRIVER))
from scripts.build_config import (  # noqa: E402
    check_environment as check_environment,
    build_environment,
    run_child,
    child_output,
    environment_record,
    configuration,
    resolved_flags,
    tool_environment,
    trace_environment,
    validate_argv,
    validate_flags,
)
from scripts.pgo_diagnostics import audit, coverage  # noqa: E402
from scripts import build_install, pgo_dependencies  # noqa: E402
from scripts.toolchain import RUST, MATURIN  # noqa: E402


def flag_environment(flags):
    """Cargo's encoded format preserves whitespace inside individual arguments."""
    return dict(build_environment(), CARGO_ENCODED_RUSTFLAGS="\x1f".join(flags))


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
        result = run_child(
            [str(profdata), "show", "--all-functions", "--counts", str(profile)],
            check=True,
            capture_output=True,
            text=True,
        )
        if coverage(result.stdout) != provenance["coverage"]:
            raise ValueError("profile training coverage mismatch")


def output(cmd, **kwargs):
    return child_output(cmd, text=True, **kwargs).strip()


def source_hash(root):
    paths = [
        root / n
        for n in ("Cargo.toml", "Cargo.lock", "pyproject.toml", ".cargo/config.toml")
    ]
    if (root / "rust-toolchain.toml").is_file():
        paths.append(root / "rust-toolchain.toml")
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
    build_environment()
    with tempfile.TemporaryDirectory(prefix="volas-identity-") as temp:
        config = configuration(root, target, Path(temp) / "flags.log")
    rust = config["tools"]["rustc"]["version"]
    if f"release: {RUST}\n" not in rust + "\n":
        raise ValueError(f"requires rustc {RUST}: {rust}")
    if output([sys.executable, "-m", "maturin", "--version"]) != f"maturin {MATURIN}":
        raise ValueError(f"requires maturin {MATURIN}")
    host = next(
        line.split(": ", 1)[1] for line in rust.splitlines() if line.startswith("host:")
    )
    llvm = (
        Path(output([config["tools"]["rustc"]["path"], "--print", "sysroot"]))
        / "lib/rustlib"
        / host
        / "bin/llvm-profdata"
    )
    if not llvm.is_file():
        raise ValueError("install llvm-tools-preview for the pinned Rust toolchain")
    training = [
        DRIVER / "scripts/build_release.py",
        DRIVER / "scripts/build_config.py",
        DRIVER / "scripts/build_env.py",
        DRIVER / "scripts/toolchain.py",
        DRIVER / "rust-toolchain.toml",
        DRIVER / "scripts/release_smoke.py",
        DRIVER / "scripts/rustc_record.py",
        DRIVER / "scripts/pgo_diagnostics.py",
        DRIVER / "scripts/pgo_dependencies.py",
        DRIVER / "scripts/pgo_validation.py",
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
        "configuration": config,
        "target": target,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "maturin": MATURIN,
        "llvm": output([str(llvm), "--version"]),
        "llvm_sha256": digest(llvm),
        "training": {str(p.relative_to(DRIVER)): digest(p) for p in training},
        "dependencies": pgo_dependencies.locked(),
    }, llvm


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run(cmd, log, root, env=None, timeout=None):
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    ambient = dict(os.environ, PYTHONOPTIMIZE="0")
    child = (build_environment(ambient, owned={"PYTHONOPTIMIZE": "0"})
             if env is None else dict(env, PYTHONOPTIMIZE="0"))
    with Path(log).open("w") as f:
        f.write(json.dumps([str(x) for x in cmd]) + "\n")
        f.flush()
        run_child(
            [str(x) for x in cmd],
            cwd=root,
            record=Path(log).with_suffix(".environment.json"),
            env=child,
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


def prepare_runtime(root, artifacts):
    """Smoke uses project runtime requirements, independent of caller packages."""
    requirements = tomllib.loads((root / "pyproject.toml").read_text())["project"].get(
        "dependencies", []
    )
    destination = artifacts / "runtime-dependencies"
    destination.mkdir()
    if requirements:
        run(
            [sys.executable, "-m", "pip", "install", "--no-compile",
             "--target", destination, *requirements],
            artifacts / "runtime-dependencies.log",
            root,
        )
    return destination


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
    config = ident["configuration"]
    # Re-resolve to catch configuration changes after identity collection.
    base = resolved_flags(root, target, [], stage / "flags.log")
    if base != config["flags"]:
        raise ValueError("effective configuration changed before build")
    feedback = (
        stage / "raw" if mode == "generate" else profile if mode == "use" else None
    )
    resolved = [*base, *flags]
    validate_flags(
        resolved,
        mode,
        feedback,
        ["llvm-args=-pgo-warn-missing-function"] if mode == "use" else (),
    )
    resolved += [f"-Clinker={config['tools']['linker']['path']}"]
    env = flag_environment(resolved)
    env.update(tool_environment(config))
    env = trace_environment(env, stage / "rustc-argv.jsonl")
    run(cmd, stage / "build.log", root, env)
    build_log = (stage / "build.log").read_text()
    validate_argv(
        stage / "rustc-argv.jsonl", mode, target, feedback, config
    )
    trained = (
        json.loads(profile.with_suffix(".json").read_text())["coverage"]
        if mode == "use"
        else {}
    )
    warnings = audit(build_log, trained)
    write(stage / "profile-warnings.json", warnings)
    wheels = list((stage / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as z:
        if not any(n.endswith(".pyi") for n in z.namelist()):
            raise ValueError("wheel missing stubs")
    lib = install(wheel, stage / "site", root, stage / "install.log")
    manifest = {
        "schema": 2,
        "environment": environment_record(owned={k: v for k, v in env.items()
                                                   if build_environment().get(k) != v}),
        "argv_sha256": digest(stage / "rustc-argv.jsonl"),
        "profile_warnings": warnings,
        "mode": mode,
        "identity": ident,
        "wheel": str(wheel),
        "wheel_sha256": digest(wheel),
        "native": str(lib),
        "native_sha256": digest(lib),
    }
    # Intermediate records are not authorization for performance consumption.
    write(stage / "build.json", manifest)
    return stage, manifest


def install_final(manifest, artifacts):
    run(
        [
            sys.executable,
            DRIVER / "scripts/build_install.py",
            "install",
            artifacts / "final.json",
        ],
        artifacts / "install-final.log",
        artifacts,
    )


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
    ap.add_argument(
        "--dependency-wheels",
        type=Path,
        help="offline wheelhouse for hash-locked training",
    )
    args = ap.parse_args()
    build_environment()
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
            if prior != current or manifest["mode"] != "generate":
                raise ValueError("generated build provenance mismatch")
            verify_library(
                Path(manifest["native"]), stage / "site", manifest["native_sha256"]
            )
            validate_argv(
                stage / "rustc-argv.jsonl",
                "generate",
                target,
                stage / "raw",
                ident["configuration"],
            )
            write(artifacts / "generated.json", manifest)
        else:
            stage, manifest = build(root, artifacts, "generate", target, ident)
        if identity(root, target)[0] != ident:
            raise ValueError("provenance changed during generate build")
        if args.mode == "generate":
            return
        deps = artifacts / "training-dependencies"
        pgo_dependencies.install(
            deps, artifacts / "dependencies.log", args.dependency_wheels
        )
        raw = artifacts / "training-raw"
        raw.mkdir()
        env = dict(
            build_environment(),
            LLVM_PROFILE_FILE=str(raw / "train-%m-%p.profraw"),
            PYTHONPATH=str(deps),
        )
        run(
            [
                sys.executable,
                "-S",
                DRIVER / "scripts/pgo_train.py",
                "--site",
                stage / "site",
                "--native-sha256",
                manifest["native_sha256"],
                "--output",
                artifacts / "training.json",
                "--identity",
                artifacts / "identity.json",
                "--dependencies",
                deps,
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
                "coverage": coverage((artifacts / "profile-show.log").read_text()),
                "training_sha256": digest(artifacts / "training.json"),
                "training_result_sha256": json.loads(
                    (artifacts / "training.json").read_text()
                )["result_sha256"],
                "training_dependencies": json.loads(
                    (artifacts / "training.json").read_text()
                )["dependencies"],
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
        manifest["profile_provenance_sha256"] = digest(profile.with_suffix(".json"))
    else:
        stage, manifest = build(root, artifacts, "normal", target, ident)
    if identity(root, target)[0] != ident:
        raise ValueError("provenance changed during final build")
    runtime = prepare_runtime(root, artifacts)
    # A fresh process proves both library identity and absence of instrumentation.
    probe = artifacts / "final-probe"
    probe.mkdir()
    env = dict(
        build_environment(),
        PYTHONPATH=os.pathsep.join(map(str, [stage / "site", runtime])),
        LLVM_PROFILE_FILE=str(probe / "%m-%p.profraw"),
    )
    run(
        [
            sys.executable,
            "-S",
            DRIVER / "scripts/release_smoke.py",
        ],
        artifacts / "smoke.log",
        artifacts,
        env,
    )
    if list(probe.iterdir()):
        raise ValueError("final wheel is instrumented")
    build_install.publish(stage / "site", manifest)
    write(artifacts / "final.json", manifest)
    if args.install:
        install_final(manifest, artifacts)
    delivery = [Path(manifest["wheel"])]
    if args.sdist:
        run(
            [sys.executable, "-m", "maturin", "sdist", "--out", artifacts / "sdist"],
            artifacts / "sdist.log",
            root,
        )
        sdists = list((artifacts / "sdist").glob("*.tar.gz"))
        if len(sdists) != 1:
            raise ValueError("expected exactly one source distribution")
        delivery += sdists
    write(
        artifacts / "delivery.json",
        {
            "schema": 1,
            "artifacts": [
                {"path": str(p.resolve()), "sha256": digest(p)} for p in delivery
            ],
        },
    )


if __name__ == "__main__":
    main()
