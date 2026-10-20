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
