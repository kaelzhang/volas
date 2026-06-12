#!/usr/bin/env python3
"""Benchmark leader tracking + systematic comparison report.

Every full `make benchmark` run is compared against the all-time best run (the
"performance leader", `.benchmarks/LEADER.json`), and a report is written next
to the run's archive. If the new run beats the leader, it BECOMES the leader.

Cross-run comparability: raw timings from different sessions/machines are not
comparable (thermal state alone moves everything 30%+). All scoring therefore
uses WITHIN-RUN normalized ratios — each volas measurement divided by its
reference twin measured in the same session (`talib` for the indicator
sections, `pandas` for the api section) — so machine speed cancels and only
volas's relative standing is compared across runs (the same insight perf_gate
uses for commit gating).

LEADER.json records, beyond the identity of the leading run, the durable
machine-independent fingerprint plus the forward-looking work queues:
  - per-item normalized ratios (the comparison basis for future runs)
  - top-10 optimization candidates (worst volas/talib, batch + after-append)
  - the after-append degraders (volas append slower than its own batch — the
    fixed-overhead QA list)
  - aggregate counts and geomeans

Usage:  python scripts/bench_leader.py <new-run>/benchmark.json
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import re
import sys

BENCH_ROOT = pathlib.Path(".benchmarks")
LEADER = BENCH_ROOT / "LEADER.json"

# (section regex, volas suffix, reference suffix)
_SECTIONS = [
    (r"test_coverage\[(.+)-volas\]$", "test_coverage[{key}-talib]", "batch"),
    (r"test_coverage_after_append\[(.+)-volas\]$",
     "test_coverage_after_append[{key}-talib]", "after_append"),
    (r"test_calc\[(.+)-volas\]$", "test_calc[{key}-talib]", "calc"),
    (r"test_append\[(.+)-volas\]$", "test_append[{key}-talib]", "append"),
]


def _means(path: pathlib.Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {b["name"]: b["stats"]["mean"] for b in data["benchmarks"]}


def normalized_ratios(path: pathlib.Path) -> dict[str, float]:
    """{'<section>:<indicator>': volas_mean / reference_mean} for one run."""
    means = _means(path)
    out: dict[str, float] = {}
    for pattern, ref_tpl, section in _SECTIONS:
        rx = re.compile(pattern)
        for name, mean in means.items():
            m = rx.match(name)
            if not m:
                continue
            ref = means.get(ref_tpl.format(key=m.group(1)))
            if ref and ref > 0:
                out[f"{section}:{m.group(1)}"] = mean / ref
    return out


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def summarize(path: pathlib.Path, meta: dict) -> dict:
    ratios = normalized_ratios(path)
    means = _means(path)
    batch = {k.split(":", 1)[1]: v for k, v in ratios.items() if k.startswith("batch:")}
    app = {k.split(":", 1)[1]: v for k, v in ratios.items() if k.startswith("after_append:")}
    # volas's own append/batch overhead factor (the QA degrader list)
    degraders = {}
    for name, mean in means.items():
        m = re.match(r"test_coverage_after_append\[(.+)-volas\]$", name)
        if m:
            b = means.get(f"test_coverage[{m.group(1)}-volas]")
            if b and mean > b:
                degraders[m.group(1)] = round(mean / b, 3)
    top = lambda d, n=10: sorted(d.items(), key=lambda kv: -kv[1])[:n]
    return {
        "meta": meta,
        "ratios": {k: round(v, 4) for k, v in sorted(ratios.items())},
        "aggregates": {
            "geomean_vs_reference": round(_geomean(list(ratios.values())), 4),
            "batch_slower_than_talib": sum(1 for v in batch.values() if v > 1),
            "after_append_slower_than_talib": sum(1 for v in app.values() if v > 1),
            "append_degraders": len(degraders),
        },
        "top10_batch_candidates": top(batch),
        "top10_after_append_candidates": top(app),
        "append_degraders": dict(top(degraders, 20)),
    }


def main(argv: list[str]) -> int:
    new_path = pathlib.Path(argv[1])
    run_dir = new_path.parent
    meta = {"run": run_dir.name, "platform": platform.platform()}
    meta_file = run_dir / "meta.txt"
    if meta_file.exists():
        for line in meta_file.read_text().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()

    new = summarize(new_path, meta)
    report = [f"# Benchmark leader report — {run_dir.name}", ""]

    if not LEADER.exists():
        LEADER.write_text(json.dumps(new, indent=1))
        report += ["No previous leader — **this run is the inaugural leader**."]
        verdict = "LEADER (inaugural)"
    else:
        lead = json.loads(LEADER.read_text())
        # A measurement-protocol change makes cross-run movement methodological,
        # not code-driven — flag it loudly instead of letting it read as a
        # regression/improvement.
        lm = lead["meta"].get("methodology", "v1")
        nm = new["meta"].get("methodology", "v1")
        if lm != nm:
            report += [
                f"> **METHODOLOGY CHANGE**: leader measured under `{lm}`, this run "
                f"under `{nm}` — the relative numbers below compare protocols, not "
                f"code. The verdict re-baselines the leader under the new protocol.",
                "",
            ]
        lr, nr = lead["ratios"], new["ratios"]
        common = sorted(set(lr) & set(nr))
        rel = {k: nr[k] / lr[k] for k in common if lr[k] > 0}
        g = _geomean(list(rel.values()))
        movers = sorted(rel.items(), key=lambda kv: -kv[1])
        report += [
            f"Compared against leader **{lead['meta'].get('run', '?')}** "
            f"on {len(common)} normalized items (volas/reference, machine-independent).",
            "",
            f"- relative geomean (new/leader): **{g:.3f}**  (<1 = better than the leader)",
            f"- improved vs leader: {sum(1 for v in rel.values() if v < 1)} items; "
            f"worse: {sum(1 for v in rel.values() if v > 1)}",
            "",
            "Biggest regressions vs leader:",
            *[f"  - {k}: x{v:.2f}" for k, v in movers[:5] if v > 1.02],
            "Biggest improvements vs leader:",
            *[f"  - {k}: x{v:.2f}" for k, v in movers[::-1][:5] if v < 0.98],
        ]
        if g < 1.0 or lm != nm:
            new["previous_leader"] = {"run": lead["meta"].get("run"), "relative_geomean": round(g, 4)}
            LEADER.write_text(json.dumps(new, indent=1))
            verdict = (
                f"NEW LEADER (re-baselined: methodology {lm} -> {nm}; x{g:.3f} not comparable)"
                if lm != nm
                else f"NEW LEADER (geomean {g:.3f} vs previous)"
            )
        else:
            verdict = f"leader unchanged ({lead['meta'].get('run', '?')}; this run x{g:.3f})"

    agg = new["aggregates"]
    report += [
        "",
        f"**Verdict: {verdict}**",
        "",
        "## This run",
        f"- geomean volas/reference: {agg['geomean_vs_reference']}",
        f"- batch indicators slower than ta-lib: {agg['batch_slower_than_talib']}",
        f"- after-append slower than ta-lib: {agg['after_append_slower_than_talib']}",
        f"- after-append slower than own batch (overhead QA): {agg['append_degraders']}",
        "",
        "## Top-10 optimization candidates (batch, volas/talib)",
        *[f"  - {k}: x{v:.2f}" for k, v in new["top10_batch_candidates"]],
        "",
        "## Top-10 optimization candidates (after-append, volas/talib)",
        *[f"  - {k}: x{v:.2f}" for k, v in new["top10_after_append_candidates"]],
    ]
    text = "\n".join(report) + "\n"
    (run_dir / "leader-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
