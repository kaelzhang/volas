"""DC-2: the indicator total in README / INDICATORS.md is derived, not hand-kept.

scripts/count_indicators.py computes the count from the Rust source of truth (the
COMMANDS / CANDLE_PATTERNS consts + the directive validator). This test runs its
--check mode, so adding or removing an indicator that changes the count fails CI
until both docs are updated — the numbers can never silently drift again."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_docs_cite_the_programmatic_indicator_count():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "count_indicators.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
