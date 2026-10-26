"""A tools-only venv must smoke and install a real release wheel successfully."""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, HOST, clean_env


@pytest.fixture(scope="module")
def release_wheel(tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("runtime-wheel")
    env = clean_env()
    for key in ("LLVM_PROFILE_FILE", "RUSTC_WRAPPER", "SDKROOT", "CARGO_TARGET_DIR"):
        env.pop(key, None)
    # Cache a real normal wheel at the compiler boundary before final smoke.
    code = (
        "from pathlib import Path; from scripts import build_release as B; "
        f"root=Path({str(ROOT)!r}); out=Path({str(artifacts)!r}); "
        f"ident,_=B.identity(root,{HOST!r}); "
        f"B.build(root,out,'normal',{HOST!r},ident)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    return json.loads((artifacts / "normal/build.json").read_text())


@pytest.mark.parametrize("entry", ["owner", "install"])
def test_clean_venv_prepares_runtime_before_smoke_and_install(
    tmp_path, release_wheel, entry
):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin/python"
    env = clean_env(PYTHONPATH=str(ROOT))
    for key in ("LLVM_PROFILE_FILE", "RUSTC_WRAPPER", "SDKROOT", "CARGO_TARGET_DIR"):
        env.pop(key, None)
    # Only the pinned build tool is supplied, with no inherited site-packages.
    site = Path(subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip())
    from maturin.__main__ import get_maturin_path

    (venv / "bin/maturin").symlink_to(get_maturin_path())
    maturin = importlib.metadata.distribution("maturin")
    for name in ("maturin", f"maturin-{maturin.version}.dist-info"):
        (site / name).symlink_to(maturin.locate_file(name), target_is_directory=True)
    subprocess.run(
        [str(python), "-c", "import importlib.util as u; "
         "assert u.find_spec('numpy') is None; assert u.find_spec('tzdata') is None"],
        cwd=tmp_path, env=env, check=True,
    )
    data = tmp_path / "wheel.json"
    data.write_text(json.dumps(release_wheel))
    if entry == "owner":
        # Replay the actual wheel at the build boundary, avoiding another full
        # compilation. Keep identity checks, pip staging, real native smoke,
        # identity publication and --install on their production paths.
        script = tmp_path / "owner.py"
        script.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "from scripts import build_release as B\n"
            f"manifest=json.loads(Path({str(data)!r}).read_text())\n"
            "def build(root, artifacts, mode, target, ident):\n"
            " stage=artifacts/mode; stage.mkdir()\n"
            " native=B.install(Path(manifest['wheel']),stage/'site',root,stage/'install.log')\n"
            " return stage,dict(manifest,native=str(native),identity=ident)\n"
            "B.build=build\nB.main()\n"
        )
        command = [str(python), str(script), "--source", str(ROOT),
                   "--artifacts", str(tmp_path / "final"), "--install"]
    else:
        command = [str(python), str(ROOT / "scripts/build_install.py"),
                   "install", str(data)]
    result = subprocess.run(command, cwd=ROOT if entry == "owner" else tmp_path, env=env, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    # Check dependencies and actual numeric/missing behavior from outside source.
    subprocess.run([str(python), "-m", "pip", "check"], cwd=tmp_path, env=env,
                   check=True, capture_output=True, text=True)
    result = subprocess.run(
        [str(python), "-c", "import runpy,numpy,tzdata,sys; from pathlib import Path; "
         "assert Path(numpy.__file__).is_relative_to(sys.prefix); "
         "assert Path(tzdata.__file__).is_relative_to(sys.prefix); "
         f"runpy.run_path({str(ROOT / 'scripts/release_smoke.py')!r})"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_install_preserves_satisfied_runtime_dependencies(tmp_path, release_wheel):
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True
    )
    python = venv / "bin/python"
    env = clean_env()
    probe = [str(python), "-c", "import numpy; print(numpy.__version__,numpy.__file__)"]
    before = subprocess.check_output(probe, env=env, text=True)
    data = tmp_path / "wheel.json"
    data.write_text(json.dumps(release_wheel))
    result = subprocess.run(
        [str(python), str(ROOT / "scripts/build_install.py"), "install", str(data)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert subprocess.check_output(probe, env=env, text=True) == before
