"""Assembly gates must use the same compiler pin as the release owner."""

from test.pgo_toolchain import ROOT


def test_asm_versions_have_no_independent_literals():
    for script in ("asm_diff.sh", "hot_path_asm.sh"):
        text = (ROOT / "scripts" / script).read_text()
        assert "1.96.0" not in text
        assert "check_toolchain.py --rust-only" in text
