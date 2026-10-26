"""PGO perf tooling contracts."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_pair_rejects_mixed_modes_and_cross_source_adoption(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_pair

    from test.test_pgo_entrypoints import site
    import json

    base = site(tmp_path / "base")
    head = site(tmp_path / "head", mode="use")

    def change(path, **changes):
        p = path / "volas-build.json"
        data = json.loads(p.read_text())
        for key, value in changes.items():
            if key == "source":
                data["identity"]["source"] = value
            else:
                data[key] = value
        p.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="matching build modes"):
        build_pair.validate(base, head)
    assert (
        build_pair.validate(base, head, "optimization")["comparison"] == "optimization"
    )
    change(head, source="different")
    with pytest.raises(ValueError, match="same-source"):
        build_pair.validate(base, head, "optimization")
    change(head, mode="normal")
    assert build_pair.validate(base, head)["comparison"] == "regression"
    change(base, mode="use", profile_sha256="base-profile")
    change(head, mode="use")
    assert build_pair.validate(base, head)["comparison"] == "regression"


def test_consistent_slower_is_retained_even_below_wall_gate():
    from scripts.perf_pairs import summarize

    def row(value):
        return {
            "benchmarks": [
                {
                    "name": "test_coverage[volas-ma:30]",
                    "group": "g",
                    "params": {"candidate": "volas", "indicator": "ma:30"},
                    "stats": {"median": value, "mean": value},
                }
            ]
        }

    _, head = summarize([row(1)] * 6, [row(1.04)] * 6)
    assert len(head["paired"]["consistent_slower"]) == 1
