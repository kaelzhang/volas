"""Effective release configuration, Cargo resolution and invocation invariants."""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def out(cmd, **kw):
    return subprocess.check_output([str(x) for x in cmd], text=True, **kw).strip()


def check_environment(env):
    allowed_cargo = {
        "CARGO_HOME",
        "CARGO_NET_OFFLINE",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_JOBS",
        "CARGO_TERM_COLOR",
        "CARGO_TERM_PROGRESS_WHEN",
    }
    exact = {
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "MATURIN_PGO",
        "RUSTC",
        "CARGO",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTDOC",
        "SDKROOT",
        "DEVELOPER_DIR",
        "CC",
        "CXX",
        "LD",
        "AR",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "LIBRARY_PATH",
        "CPATH",
        "RUSTC_BOOTSTRAP",
        "RUSTC_LINKER",
        "PYO3_CONFIG_FILE",
        "PYO3_CROSS",
        "PYO3_CROSS_LIB_DIR",
        "PYO3_CROSS_PYTHON_VERSION",
    }
    bad = [
        k
        for k in env
        if k in exact
        or (
            k.startswith(
                (
                    "CARGO_BUILD_",
                    "CARGO_TARGET_",
                    "CARGO_PROFILE_",
                    "CARGO_ENCODED_",
                    "CARGO_UNSTABLE_",
                    "MATURIN_",
                    "CC_",
                    "CXX_",
                    "CFLAGS_",
                    "LDFLAGS_",
                )
            )
            and k not in allowed_cargo
        )
    ]
    if bad:
        raise ValueError(f"ambient build overrides rejected: {sorted(bad)}")


def check_cargo_config(root):
    """Cargo owns merging; reject settings that our identity cannot represent."""
    paths = [
        p / ".cargo" / name
        for p in (root, *root.parents)
        for name in ("config", "config.toml")
    ]
    home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    paths += [home / "config", home / "config.toml"]
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        data = tomllib.loads(path.read_text())
        for table, values in data.items():
            if table == "target" and all(
                set(v) <= {"rustflags"} for v in values.values()
            ):
                for value in values.values():
                    flags = value.get("rustflags", [])
                    validate_flags(
                        flags.split() if isinstance(flags, str) else flags, "normal"
                    )
                continue
            if table == "build" and set(values) <= {"rustflags", "jobs", "target-dir"}:
                flags = values.get("rustflags", [])
                validate_flags(
                    flags.split() if isinstance(flags, str) else flags, "normal"
                )
                continue
            if table in ("net", "registries", "registry", "source", "http", "term"):
                continue
            raise ValueError(f"unbound Cargo configuration rejected: {path}: {table}")


def tools():
    sysroot = Path(out(["rustc", "--print", "sysroot"]))
    rustc, cargo = sysroot / "bin/rustc", sysroot / "bin/cargo"
    result = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha(path),
            "version": out([path, "-Vv" if name == "rustc" else "-V"]),
        }
        for name, path in [("rustc", rustc), ("cargo", cargo)]
    }
    if sys.platform == "darwin":
        linker = Path(out(["xcrun", "--find", "clang"])).resolve()
        ld = Path(out(["xcrun", "--find", "ld"])).resolve()
        sdk = Path(out(["xcrun", "--show-sdk-path"])).resolve()
        result["linker"] = {
            "path": str(linker),
            "sha256": sha(linker),
            "version": out([linker, "--version"]),
            "ld_path": str(ld),
            "ld_sha256": sha(ld),
        }
        settings = sdk / "SDKSettings.json"
        result["sdk"] = {
            "path": str(sdk),
            "version": out(["xcrun", "--show-sdk-version"]),
            "settings_sha256": sha(settings),
        }
    else:
        linker = Path(shutil.which("cc")).resolve()
        result["linker"] = {
            "path": str(linker),
            "sha256": sha(linker),
            "version": out([linker, "--version"]),
        }
        result["sdk"] = None
    return result


def tool_environment(config):
    t = config["tools"]
    env = dict(os.environ, RUSTC=t["rustc"]["path"], CARGO=t["cargo"]["path"])
    env["PATH"] = str(Path(t["cargo"]["path"]).parent) + os.pathsep + env["PATH"]
    if t["sdk"]:
        env["SDKROOT"] = t["sdk"]["path"]
    return env


def trace_environment(env, trace):
    return dict(
        env,
        RUSTC_WRAPPER=str(Path(__file__).with_name("rustc_record.py")),
        VOLAS_RUSTC_TRACE=str(trace),
    )


def resolved_flags(root, target, flags, log):
    """Read Cargo's target-info probe, whose middle is the complete ordered flags.

    Validate the pinned Cargo query grammar; unknown forms fail rather than losing
    arguments. A concrete-target array merges before cfg arrays. No marker cuts
    off an existing concrete-target prefix.
    """
    check_cargo_config(root)
    with tempfile.TemporaryDirectory(prefix="volas-flags-") as tmp:
        probe = Path(tmp)
        (probe / "src").mkdir()
        (probe / "Cargo.toml").write_text(
            '[package]\nname="volas-flags-probe"\nversion="0.0.0"\nedition="2021"\n[workspace]\n'
        )
        (probe / "src/main.rs").write_text("fn main() {}\n")
        trace = probe / "argv.jsonl"
        compiler_bin = Path(out(["rustc", "--print", "sysroot"])) / "bin"
        env = trace_environment(
            dict(os.environ, RUSTC=str(compiler_bin / "rustc")), trace
        )
        cmd = [
            str(compiler_bin / "cargo"),
            "build",
            "--manifest-path",
            str(probe / "Cargo.toml"),
            "--target",
            target,
            "--target-dir",
            str(probe / "target"),
        ]
        with Path(log).open("w") as f:
            f.write(json.dumps(cmd) + "\n")
            f.flush()
            subprocess.run(
                cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, check=True
            )
        queries = [json.loads(line) for line in trace.read_text().splitlines()]
        queries = [a for a in queries if "--print=file-names" in a and "--target" in a]
        if len(queries) != 1:
            raise ValueError("unsupported Cargo flags query")
        argv = queries[0]
        prefix = ["-", "--crate-name", "___", "--print=file-names"]
        suffix = ["--target", target]
        for kind in ("bin", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"):
            suffix += ["--crate-type", kind]
        suffix += [
            "--print=sysroot",
            "--print=split-debuginfo",
            "--print=crate-name",
            "--print=cfg",
            "-Wwarnings",
        ]
        if argv[1:5] != prefix or argv[-len(suffix) :] != suffix:
            raise ValueError("unsupported Cargo flags query grammar")
        resolved = argv[5 : -len(suffix)]
        Path(log).with_suffix(".argv.json").write_text(
            json.dumps(queries, indent=2) + "\n"
        )
        return [*resolved, *flags]


def codegen_flags(argv):
    """Normalize -C x and -Cx only for validation; preserve stored argv verbatim."""
    result = []
    for n, value in enumerate(argv):
        if value == "-C":
            if n + 1 == len(argv):
                raise ValueError("incomplete codegen flag")
            result.append(argv[n + 1])
        elif value.startswith("-C"):
            result.append(value[2:])
    return result


def validate_flags(flags, mode, feedback=None):
    actual = [
        f for f in codegen_flags(flags) if re.match(r"profile-(generate|use)(?:=|$)", f)
    ]
    expected = [] if mode == "normal" else [f"profile-{mode}={feedback}"]
    if mode not in ("normal", "generate", "use") or actual != expected:
        raise ValueError(
            f"feedback flags mismatch for {mode}: {actual}; expected {expected}"
        )
    if feedback is not None and not Path(feedback).is_absolute():
        raise ValueError("feedback path must be absolute")
    if mode != "normal" and feedback is None:
        raise ValueError("feedback path is required")


def configuration(root, target, log):
    flags = resolved_flags(root, target, [], log)
    validate_flags(flags, "normal")
    forbidden = (
        "linker",
        "link-arg",
        "link-args",
        "llvm-args",
        "lto",
        "codegen-units",
        "opt-level",
    )
    if any(f.split("=", 1)[0] in forbidden for f in codegen_flags(flags)):
        raise ValueError("unbound codegen override rejected")
    manifests = tomllib.loads((root / "Cargo.toml").read_text())
    profile = manifests["profile"]["release"]
    if profile != {"lto": True, "codegen-units": 1, "opt-level": 3}:
        raise ValueError("requires release fat LTO / CGU=1 / opt=3")
    package = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["maturin"]
    if set(package) - {
        "manifest-path",
        "module-name",
        "binding",
        "python-packages",
        "include",
    }:
        raise ValueError("unbound maturin configuration rejected")
    features = out(
        [
            "cargo",
            "tree",
            "--locked",
            "--target",
            target,
            "-p",
            "volas-python",
            "--prefix",
            "none",
            "--format",
            "{p} features={f}",
        ],
        cwd=root,
    ).replace(str(root), "<source>")
    return {
        "flags": flags,
        "release": profile,
        "features": features,
        "tools": tools(),
        "deployment_target": os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
    }


def validate_argv(log, mode, target, feedback=None, config=None, trace=None):
    commands = (
        [json.loads(line) for line in Path(trace).read_text().splitlines()]
        if trace
        else [
            shlex.split(line.strip()[9:-1])
            for line in Path(log).read_text().splitlines()
            if line.strip().startswith("Running `") and "--crate-name" in line
        ]
    )
    commands = [
        a for a in commands if "--crate-name" in a and "--print=file-names" not in a
    ]
    target_commands = [
        a for a in commands if "--target" in a and a[a.index("--target") + 1] == target
    ]
    if not any("volas_rs" in a for a in target_commands):
        raise ValueError("missing actual cdylib rustc argv")
    for a in commands:
        if (
            config
            and Path(a[0]).resolve() != Path(config["tools"]["rustc"]["path"]).resolve()
        ):
            raise ValueError("actual rustc executable mismatch")
        is_target = a in target_commands
        validate_flags(
            a, mode if is_target else "normal", feedback if is_target else None
        )
        if is_target and config:
            # Cargo appends flags as a contiguous sequence, preserving order/duplicates.
            flags = config["flags"]
            if flags and not any(a[n : n + len(flags)] == flags for n in range(len(a))):
                raise ValueError("actual resolved flags mismatch")
            if f"linker={config['tools']['linker']['path']}" not in codegen_flags(a):
                raise ValueError("actual linker mismatch")
        if "volas_rs" in a and target.endswith("apple-darwin"):
            if not {"link-arg=dynamic_lookup", "link-arg=-undefined"} <= set(
                codegen_flags(a)
            ):
                raise ValueError("missing macOS extension link flags")
