"""Import identity must survive stub/GOT/PLT layout changes in linked libraries."""

import re

import pytest

from test.pgo_linked_imports import FIXTURES, TARGETS, SYMBOL, linked_dump
from test.pgo_toolchain import HOST
from test.test_pgo_final_asm import final_report


def compare(tmp_path, monkeypatch, texts):
    base, before = final_report(tmp_path / "base", monkeypatch, texts[0], inventory=("execute",))
    _, after = final_report(tmp_path / "head", monkeypatch, texts[1], base, inventory=("execute",))
    return before, after


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("variant", ["target", "moved"])
def test_captured_linked_import_identity(tmp_path, monkeypatch, target, variant):
    texts = [(FIXTURES / f"{target}-{v}.txt").read_text() for v in ("base", variant)]
    check_imports(tmp_path, monkeypatch, texts, target, variant)


def check_imports(tmp_path, monkeypatch, texts, target, variant):
    # Independently check the real linker put the changed import in the same slot,
    # and that the movement test actually moves both the stub/PLT and GOT slots.
    apple = "apple" in target
    name = "_foreign" if apple else "foreign"

    def slots(text, symbol):
        pattern = (r"^0x([0-9a-f]+)\s+\d+\s+" + symbol + r"$" if apple else
                   r"^([0-9a-f]+)\s+R_\S+(?:GLOB_DAT|JUMP_SLOT)\s+" + symbol + r"$")
        return re.findall(pattern, text, re.M)

    old = slots(texts[0], name)
    new = slots(texts[1], name if variant == "moved" else name.replace("foreign", "changed"))
    assert len(old) >= 2 and len(old) == len(new)
    assert (old != new) == (variant == "moved")
    _, head = compare(tmp_path, monkeypatch, texts)
    assert head["changed"] == ([SYMBOL] if variant == "target" else [])
    function = head["functions"][SYMBOL]
    imported = name if variant == "moved" else name.replace("foreign", "changed")
    assert "@import(" + imported + ")" in function["calls"][0]
    assert sum("@import(" + imported + ")" in i for i in function["normalized"]) >= 2
    assert function["unresolved_references"] == []
    assert head["comparison_complete"]


@pytest.mark.parametrize("variant", ["target", "moved"])
def test_native_linked_import_identity(tmp_path, monkeypatch, variant):
    texts = [linked_dump(tmp_path / v, HOST, v) for v in ("base", variant)]
    check_imports(tmp_path / "reports", monkeypatch, texts, HOST, variant)


def test_unresolved_linked_slots_cannot_compare_equal(tmp_path, monkeypatch):
    # Genuine LLVM dump without dynamic metadata, not an invented symbol table.
    text = linked_dump(tmp_path / "linked", HOST, metadata=False)
    _, head = compare(tmp_path, monkeypatch, [text, text])
    assert head["changed"] == [SYMBOL]
    assert not head["comparison_complete"]
    unresolved = head["functions"][SYMBOL]["unresolved_references"]
    assert unresolved and any("@unresolved(import" in i for i in unresolved)


def test_native_report_reads_llvm_binding_metadata(tmp_path, monkeypatch):
    base = None
    for variant in ("base", "moved", "target"):
        p = tmp_path / variant
        linked_dump(p, HOST, variant)
        directory, report = final_report(
            p / "entry", monkeypatch, None, base, inventory=("execute",),
            library=p / "linked.so",
        )
        if base:
            assert report["changed"] == ([SYMBOL] if variant == "target" else [])
        assert report["comparison_complete"]
        assert "@import(" in report["functions"][SYMBOL]["calls"][0]
        base = directory if base is None else base


def test_chained_bind_identity_and_addend(tmp_path, monkeypatch):
    # Linked with dyld chained fixups and a non-C-ABI import name. Its data slot
    # has no indirect-symbol entry; only LLVM --dyld-info supplies the binding.
    texts = [(FIXTURES / f"chained-{addend}.txt").read_text() for addend in (8, 16)]
    _, head = compare(tmp_path, monkeypatch, texts)
    assert head["changed"] == [SYMBOL]
    normalized = head["functions"][SYMBOL]["normalized"]
    assert any("@import(foreign,+16)" in i for i in normalized)
    assert head["comparison_complete"]


@pytest.mark.parametrize("target", ["aarch64-apple-darwin", "aarch64-linux-gnu"])
def test_unpaired_import_page_cannot_compare_equal(tmp_path, monkeypatch, target):
    texts = [(FIXTURES / f"page-{target}-{v}.txt").read_text() for v in ("base", "target")]
    if HOST == target:
        texts = [linked_dump(tmp_path / v, target, v, page_boundary=True) for v in ("base", "target")]
    # Real ADRP followed by a branch to the load: the reference owner has no
    # reaching page fact at that block entry. The page's first GOT slot is not
    # proof of which import the later load reaches.
    _, head = compare(tmp_path / "different", monkeypatch, texts)
    assert head["changed"] == [SYMBOL]
    assert not head["comparison_complete"]
    assert any("@unresolved(import-page" in i for i in head["functions"][SYMBOL]["normalized"])
    _, identical = compare(tmp_path / "identical", monkeypatch, [texts[0], texts[0]])
    assert identical["changed"] == [SYMBOL] and not identical["comparison_complete"]
