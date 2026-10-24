"""PGO asm tooling contracts."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_final_asm_parser_rejects_empty_output():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from pgo_asm import parse

    with pytest.raises(ValueError, match="empty disassembly"):
        parse("00000001 <missing>:\n")


def test_assembly_inventory_symbol_missing_is_an_error():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from asm_symbols import parse, require_symbol

    functions = parse("00000001 <_other>:\n00000001: ret\n")
    with pytest.raises(ValueError, match="missing assembly symbol"):
        require_symbol(functions, "probe_ma")


@pytest.mark.parametrize("metric", ["cfg_edges", "calls"])
def test_final_report_distinguishes_addresses_branches_and_calls(
    tmp_path, monkeypatch, metric
):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import pgo_asm

    site = tmp_path / "site"
    site.mkdir()
    (site / "volas_rs.so").touch()
    report = tmp_path / "report"
    assembly = """00001000 <volas_core::frame::execute>:
1000: adr x0, 0x100c
1004: cbz x0, 0x1010
1008: blr x8
100c: b 0x1010
1010: ret
"""

    def output(command):
        if command == ["rustc", "-Vv"]:
            return "host: aarch64-apple-darwin"
        if command == ["rustc", "--print", "sysroot"]:
            return "/toolchain"
        return assembly

    monkeypatch.setattr(pgo_asm, "output", output)
    monkeypatch.setattr(pgo_asm, "manifest", lambda _: {"mode": "use"})
    monkeypatch.setattr(
        sys, "argv", ["pgo_asm", "--site", str(site), "--output", str(report)]
    )
    with pytest.raises(ValueError, match="missing inventory symbols"):
        pgo_asm.main()
    function = json.loads((report / "report.json").read_text())["functions"][
        "volas_core::frame::execute"
    ]
    expected = {"cfg_edges": [[1, 4], [3, 4]], "calls": ["blr x8"]}
    assert function[metric] == expected[metric]


@pytest.mark.parametrize(
    "opcode,operands",
    [
        ("tbz", "w0, #0x0, 0x1010"),
        ("tbnz", "w8, #0x2, 0x1010"),
        ("cbz", "w0, 0x1010"),
        ("cbnz", "x8, 0x1010"),
        ("b.eq", "0x1010"),
        ("b", "0x1010"),
        ("bl", "0x1010"),
    ],
)
def test_opcode_target_uses_address_operand(tmp_path, monkeypatch, opcode, operands):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import pgo_asm

    site = tmp_path / "site"
    site.mkdir()
    (site / "volas_rs.so").touch()
    asm = f"00001000 <volas_core::frame::execute>:\n1000: {opcode} {operands}\n1010: ret\n"

    def output(cmd):
        if cmd == ["rustc", "-Vv"]:
            return "host: aarch64-apple-darwin"
        if cmd == ["rustc", "--print", "sysroot"]:
            return "/toolchain"
        return asm

    monkeypatch.setattr(pgo_asm, "output", output)
    monkeypatch.setattr(pgo_asm, "manifest", lambda _: {"mode": "use"})
    monkeypatch.setattr(
        sys, "argv", ["pgo_asm", "--site", str(site), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="missing inventory symbols"):
        pgo_asm.main()
    f = json.loads((tmp_path / "out/report.json").read_text())["functions"][
        "volas_core::frame::execute"
    ]
    assert f["cfg_edges"] == [[0, 1]]
    assert "@instruction1" in f["normalized"][0]


@pytest.mark.parametrize("changed_target", [False, True])
def test_real_hot_shell_rejects_empty_or_reads_built_target(tmp_path, changed_target):
    import os
    import shutil
    import subprocess
    import sys

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for p in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(p, scripts / p.name)
    shutil.copy2(ROOT / "scripts/hot_path_asm.sh", scripts)
    (scripts / "hot_asm_inventory.txt").write_text("volas-core|execute|max|test\n")
    (scripts / "hot_asm_baseline.txt").write_text("# arch: arm64)\nexecute max 1\n")
    target = tmp_path / "target"
    deps = target / "release/deps"
    deps.mkdir(parents=True)
    (deps / "volas_core-old.s").write_text(
        "_execute17h123456E:\n.cfi_startproc\n"
        + ("\tret\n" if changed_target else "")
        + ".cfi_endproc\n"
    )
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "git").write_text(f'#!/bin/sh\necho "{tmp_path}"\n')
    instructions = "\tret\n" if changed_target else ""
    (bin_ / "cargo").write_text(f"""#!{sys.executable}
import os,pathlib,sys
p=pathlib.Path(os.environ['CARGO_TARGET_DIR'])/'aarch64-apple-darwin/release/deps'
p.mkdir(parents=True,exist_ok=True)
(p/'volas_core-built.s').write_text('_execute17h123456E:\\n.cfi_startproc\\n\\tnop\\n\\tret\\n.cfi_endproc\\n')
for a in sys.argv:
 if a.startswith('asm='):
  pathlib.Path(a[4:]).write_text('_execute17h123456E:\\n.cfi_startproc\\n' + {instructions!r} + '.cfi_endproc\\n')
""")
    for p in bin_.iterdir():
        p.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bin_) + os.pathsep + os.environ["PATH"],
        VOLAS_PYTHON=sys.executable,
        CARGO_TARGET_DIR=str(target),
    )
    env.pop("LDFLAGS", None)
    env.pop("RUSTFLAGS", None)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    if changed_target:
        env["CARGO_BUILD_TARGET"] = "aarch64-apple-darwin"
    result = subprocess.run(
        ["bash", str(scripts / "hot_path_asm.sh"), "check"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    if not changed_target:
        assert "empty" in result.stderr or "instruction" in result.stderr


@pytest.mark.parametrize("gate", ["asm_diff.sh", "hot_path_asm.sh"])
@pytest.mark.parametrize("table", ["build", "target.aarch64-apple-darwin"])
def test_normal_asm_rejects_cargo_feedback_before_compiler(tmp_path, gate, table):
    import os
    import shutil
    import subprocess
    import sys

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for p in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(p, scripts / p.name)
    shutil.copy2(ROOT / "scripts" / gate, scripts)
    (scripts / "hot_asm_inventory.txt").write_text("volas-core|execute|max|test\n")
    (tmp_path / ".cargo").mkdir()
    (tmp_path / ".cargo/config.toml").write_text(
        f'[{table}]\nrustflags=["-Cprofile-use=/tmp/profile.profdata"]\n'
    )
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    marker = tmp_path / "compiler-started"
    (bin_ / "git").write_text(f'#!/bin/sh\necho "{tmp_path}"\n')
    (bin_ / "cargo").write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(2)\n"
    )
    for path in bin_.iterdir():
        path.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bin_) + os.pathsep + os.environ["PATH"],
        VOLAS_PYTHON=sys.executable,
        CARGO_TARGET_DIR=str(tmp_path / "target"),
    )
    for key in ("LDFLAGS", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(scripts / gate), "check"], env=env, capture_output=True, text=True
    )
    assert not marker.exists(), "normal asm started Cargo with feedback flags"
    assert result.returncode != 0 and "feedback" in result.stderr
