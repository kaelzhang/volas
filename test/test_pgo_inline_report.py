"""Final linked reports locate reviewed source carriers for inlined inventory."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pgo_asm  # noqa: E402
from build_release import source_hash  # noqa: E402


@pytest.mark.parametrize("suffix", ["", "::h0123456789abcdef"])
def test_final_report_covers_inlined_inventory_without_zero_counts(tmp_path, monkeypatch, suffix):
    site = tmp_path / "site"
    site.mkdir()
    (site / "volas_rs.so").touch()
    source = source_hash(ROOT)
    monkeypatch.setattr(pgo_asm, "manifest", lambda _: {"identity": {"source": source}})
    monkeypatch.setattr(pgo_asm, "disassemble", lambda *a: """00001000 <volas_rs::frame_tf::PyDataFrame::fold_bars>:
1000: bl 0x2000
1004: ret
00002000 <volas_directive::exec::execute_node>:
2000: ret
""".replace(">:", suffix + ">:"))
    inventory = tmp_path / "hot_asm_inventory.txt"
    inventory.write_text("core|fold_forming_row|max|fold\ndirective|execute|-|eval\n")
    monkeypatch.setattr(pgo_asm, "__file__", str(tmp_path / "pgo_asm.py"))
    monkeypatch.setattr(sys, "argv", ["pgo_asm", "--site", str(site), "--output", str(tmp_path / "report"), "--source", str(ROOT)])
    pgo_asm.main()
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert set(report["inline_sites"]) == {"fold_forming_row", "execute"}
    assert report["standalone_missing_symbols"] == ["fold_forming_row", "execute"]
    assert report["missing_symbols"] == []
    assert report["status"] == "review-required"
    assert all(f["non_nop"] > 0 for f in report["functions"].values())
    assert "volas_rs::frame_tf::PyDataFrame::fold_bars" + suffix in report["functions"]
    assert all("source" in evidence for evidence in report["inline_sites"].values())


def test_inline_source_mismatch_fails_before_reporting(tmp_path, monkeypatch):
    monkeypatch.setattr(pgo_asm, "manifest", lambda _: {"identity": {"source": "wrong"}})
    monkeypatch.setattr(sys, "argv", ["pgo_asm", "--site", str(tmp_path), "--output", str(tmp_path / "out"), "--source", str(ROOT)])
    with pytest.raises(ValueError, match="source.*mismatch"):
        pgo_asm.main()


def test_inline_mapping_rejects_unrelated_or_similarly_named_containers():
    evidence = pgo_asm.inline_carriers(ROOT, {
        "volas_core::dataframe::refold_forming_row": {},
        "other::exec::execute_node": {},
        "volas_rs::frame_tf::PyDataFrame::fold_bars_other": {},
    })
    assert evidence == {}


def test_unknown_missing_symbol_cannot_claim_complete_comparison(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "volas_rs.so").touch()
    monkeypatch.setattr(pgo_asm, "manifest", lambda _: {"identity": {"source": source_hash(ROOT)}})
    monkeypatch.setattr(pgo_asm, "disassemble", lambda *a: "00001000 <other::function>:\n1000: ret\n")
    (tmp_path / "hot_asm_inventory.txt").write_text("core|execute|-|eval\n")
    monkeypatch.setattr(pgo_asm, "__file__", str(tmp_path / "pgo_asm.py"))
    monkeypatch.setattr(sys, "argv", ["pgo_asm", "--site", str(site), "--output", str(tmp_path / "report"), "--source", str(ROOT)])
    with pytest.raises(ValueError, match="missing inventory symbols"):
        pgo_asm.main()
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["missing_symbols"] == ["execute"]
    assert report["comparison_complete"] is False


def test_incomplete_base_report_cannot_claim_complete_comparison(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "volas_rs.so").touch()
    monkeypatch.setattr(pgo_asm, "manifest", lambda _: {"identity": {"source": source_hash(ROOT)}})
    monkeypatch.setattr(pgo_asm, "disassemble", lambda *a: """00001000 <volas_rs::frame_tf::PyDataFrame::fold_bars>:
1000: ret
00002000 <volas_directive::exec::execute_node>:
2000: ret
""")
    (tmp_path / "hot_asm_inventory.txt").write_text("core|fold_forming_row|max|fold\ndirective|execute|-|eval\n")
    monkeypatch.setattr(pgo_asm, "__file__", str(tmp_path / "pgo_asm.py"))
    base = tmp_path / "base"
    base.mkdir()
    missing = ["fold_forming_row", "execute"]
    (base / "report.json").write_text(json.dumps({
        "functions": {}, "missing_symbols": missing, "comparison_complete": False,
    }))
    monkeypatch.setattr(sys, "argv", ["pgo_asm", "--site", str(site), "--output", str(tmp_path / "report"), "--source", str(ROOT), "--base-report", str(base)])
    with pytest.raises(ValueError, match="base report missing inventory symbols"):
        pgo_asm.main()
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["missing_symbols"] == []
    assert report["base_missing_symbols"] == missing
    assert report["comparison_complete"] is False
