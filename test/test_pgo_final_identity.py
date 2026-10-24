"""Only completed checks authorize consumption of real linked release wheels."""

import shutil
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_release as B
from test.pgo_toolchain import ROOT, HOST, clean_env


@pytest.fixture(scope="module", autouse=True)
def isolated_release_environment():
    # Coverage tests call this release owner from an instrumented interpreter.
    # The nested builds are independent releases with their own raw directories.
    with pytest.MonkeyPatch.context() as patch:
        for key in (
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "LDFLAGS",
            "RUSTC_WRAPPER",
            "SDKROOT",
            "LLVM_PROFILE_FILE",
        ):
            patch.delenv(key, raising=False)
        yield


@pytest.fixture(scope="module")
def linked_release(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-identity-source")
    (root / ".cargo").mkdir()
    (root / ".cargo/config.toml").write_text("")
    crate = root / "crates/volas-python"
    (crate / "src").mkdir(parents=True)
    (root / "volas").mkdir()
    (root / "volas/__init__.py").write_text(
        "from volas_rs import value\nclass DataFrame:\n"
        " def __init__(self, data): self.data=data\n"
        ' def exec(self, directive): return [value()]*len(self.data["close"])\n'
    )
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers=["crates/volas-python"]\nresolver="2"\n'
        "[profile.release]\nlto=true\ncodegen-units=1\nopt-level=3\n"
    )
    (crate / "Cargo.toml").write_text(
        '[package]\nname="volas-python"\nversion="0.0.0"\nedition="2021"\n'
        '[lib]\nname="volas_rs"\ncrate-type=["cdylib"]\n'
        '[dependencies]\npyo3={version="=0.25.1",features=["extension-module"]}\n'
    )
    (crate / "src/lib.rs").write_text(
        "use pyo3::prelude::*;\n#[pyfunction]\nfn value()->usize {42}\n"
        "#[pymodule]\nfn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {"
        " m.add_function(wrap_pyfunction!(value,m)?)?; Ok(()) }\n"
    )
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires=["maturin>=1.4,<2"]\nbuild-backend="maturin"\n'
        '[project]\nname="volas"\nversion="0.0.0"\n[tool.maturin]\n'
        'manifest-path="crates/volas-python/Cargo.toml"\n'
        'python-packages=["volas"]\ninclude=["volas_rs.pyi"]\n'
    )
    (root / "volas_rs.pyi").write_text("def value() -> int: ...\n")
    subprocess.run(
        ["cargo", "generate-lockfile", "--offline"],
        cwd=root,
        env=clean_env(),
        check=True,
        capture_output=True,
    )
    ident, llvm = B.identity(root, HOST)
    generated, manifest = B.build(root, root / "training", "generate", HOST, ident)
    raw = generated / "raw"
    subprocess.run(
        [sys.executable, "-c", "import volas_rs; assert volas_rs.value()==42"],
        cwd=root,
        env=clean_env(
            PYTHONPATH=str(generated / "site"),
            LLVM_PROFILE_FILE=str(raw / "%m-%p.profraw"),
        ),
        check=True,
    )
    profile = root / "profile.profdata"
    subprocess.run(
        [str(llvm), "merge", "-o", str(profile), *map(str, raw.glob("*.profraw"))],
        check=True,
    )
    counters = B.coverage(
        B.output([str(llvm), "show", "--all-functions", "--counts", str(profile)])
    )
    B.write(
        profile.with_suffix(".json"),
        {"identity": ident, "profile_sha256": B.digest(profile), "coverage": counters},
    )
    assert (generated / "build.json").is_file()
    templates = {}
    for mode in ("normal", "use"):
        templates[mode] = B.build(
            root,
            root / "templates",
            mode,
            HOST,
            ident,
            profile if mode == "use" else None,
        )[0]
    for stage in templates.values():
        assert (stage / "build.json").exists()
        assert not (stage / "site/volas-build.json").exists()
    return root, ident, llvm, profile, generated, templates


@pytest.mark.parametrize("mode", ["normal", "use"])
@pytest.mark.parametrize("failure", ["smoke", "provenance", "instrumented", "none"])
def test_final_identity_after_checks(
    tmp_path, monkeypatch, linked_release, mode, failure
):
    root, ident, llvm, profile, _, templates = linked_release
    original_identity, original_run = B.identity, B.run
    built = []
    original_build = B.build

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        built.append(result)
        return result

    def identity(*args):
        result = original_identity(*args)
        return (
            ({**result[0], "source": "changed"}, llvm)
            if built and failure == "provenance"
            else result
        )

    def run(cmd, log, *args, **kwargs):
        if Path(log).name == "build.log":
            # Reuse unedited compiler output and the real linked wheel. The
            # production build() still resolves/validates argv, audits profile
            # diagnostics, checks stubs and performs the actual pip install.
            template, stage = templates[mode], Path(log).parent
            shutil.copy2(template / "build.log", log)
            shutil.copy2(template / "rustc-argv.jsonl", stage / "rustc-argv.jsonl")
            shutil.copytree(template / "wheels", stage / "wheels")
        elif Path(log).name == "smoke.log":
            if failure == "smoke":
                raise subprocess.CalledProcessError(1, cmd)
            original_run(cmd, log, *args, **kwargs)
            if failure == "instrumented":
                (tmp_path / "final-probe/leaked.profraw").write_bytes(b"raw")
        else:
            original_run(cmd, log, *args, **kwargs)

    monkeypatch.setattr(B, "build", build)
    monkeypatch.setattr(B, "identity", identity)
    monkeypatch.setattr(B, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_release",
            "--source",
            str(root),
            "--artifacts",
            str(tmp_path),
            "--mode",
            mode,
            "--profile",
            str(profile),
        ],
    )
    if failure == "smoke":
        with pytest.raises(subprocess.CalledProcessError):
            B.main()
    elif failure != "none":
        with pytest.raises(
            ValueError, match="provenance changed|final wheel is instrumented"
        ):
            B.main()
    else:
        B.main()
    site = tmp_path / mode / "site"
    assert (tmp_path / mode / "build.json").is_file()
    assert (site / "volas-build.json").exists() == (failure == "none")
    assert (tmp_path / "final.json").exists() == (failure == "none")
    if failure == "none":
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_pair.py"), str(site), str(site)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return
    # CPU's actual CLI must fail before it reads the measurement manifest.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/perf_cpu.py"),
            str(site),
            str(site),
            str(tmp_path / "cpu"),
            str(tmp_path / "NO_MEASUREMENT"),
        ],
        capture_output=True,
        text=True,
        env=clean_env(),
    )
    assert result.returncode != 0 and "volas-build.json" in result.stderr
    assert "NO_MEASUREMENT" not in result.stderr
    # Exercise the real BASE_SITE shell and pair guard. Its head build reuses
    # this already-linked artifact; no measurement is authorized by this test.
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        + """import pathlib,runpy,shutil,sys
args=sys.argv[1:]
if args and args[0].endswith('build_release.py'):
 out=pathlib.Path(args[args.index('--artifacts')+1])/'normal/site'
 shutil.copytree("""
        + repr(str(site))
        + """,out)
elif args and args[0].endswith('perf_pairs.py'):
 raise RuntimeError('MEASUREMENT_STARTED')
else:
 sys.argv=[sys.argv[0],*args]
 if args[0]=='-c': sys.argv=['-c',*args[2:]]; exec(args[1])
 else: sys.argv=args; sys.path.insert(0,str(pathlib.Path(args[0]).parent)); runpy.run_path(args[0],run_name='__main__')
"""
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/perf_ab.sh"), "HEAD"],
        cwd=ROOT,
        env=clean_env(VOLAS_PYTHON=str(python), BASE_SITE=str(site)),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "volas-build.json" in result.stderr
    assert "MEASUREMENT_STARTED" not in result.stderr


def test_generated_build_is_intermediate_only(linked_release):
    generated = linked_release[4]
    assert not (generated / "site/volas-build.json").exists()


@pytest.mark.parametrize("mode", ["normal", "generate", "use"])
@pytest.mark.parametrize(
    "flag", ["-Cinstrument-coverage", "--codegen=instrument_coverage=yes"]
)
def test_release_flags_reject_coverage(mode, flag, tmp_path):
    feedback = tmp_path / "feedback" if mode != "normal" else None
    flags = [flag] + ([f"-Cprofile-{mode}={feedback}"] if feedback else [])
    with pytest.raises(ValueError, match="coverage instrumentation"):
        B.validate_flags(flags, mode, feedback)
