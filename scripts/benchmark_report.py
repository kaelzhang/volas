#!/usr/bin/env python3
"""Render a pytest-benchmark JSON run into a self-contained HTML report.

Usage::

    python scripts/benchmark_report.py <benchmark.json> [output.html]

For every benchmark group — one per (category, indicator), where category is
``calc`` (batch) or ``append`` (one new bar) — the report shows a vertical bar
chart of median time (shorter = faster) plus a table with columns
``Mean``, ``Median``, ``OPS``, ``rounds`` and ``Perf``. ``Perf`` expresses
relative speed with the **slowest** candidate as ``1.00x`` and every faster
candidate as ``{slowest_time / this_time}x``.

The output is a single self-contained HTML file (inline CSS + inline SVG, no
external assets) so it can be committed and opened directly. It overwrites any
previous report — only one version is kept.
"""

from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

# Fixed candidate order + colour, so the legend and bar colours are stable across
# every chart in the report.
CANDIDATE_ORDER = ['pandas', 'stock_pandas', 'polars', 'talib', 'volas']
COLORS = {
    'pandas': '#5B8FF9',
    'stock_pandas': '#5AD8A6',
    'polars': '#5D7092',
    'talib': '#F6BD16',
    'volas': '#6F5EF9',
}
# Section order in the report: append first, then batch, then the full coverage.
CATEGORY_ORDER = ['append', 'calc', 'coverage']
CATEGORY_TITLES = {
    'calc': 'Batch indicator computation',
    'append': 'Append one new bar → updated indicator',
    'coverage': 'Full coverage — volas vs TA-Lib',
}
CATEGORY_BLURB = {
    'calc': 'Compute the indicator over the whole series, across every library.',
    'append': ('A new bar arrives. <code>volas</code> / <code>stock_pandas</code> refresh their '
               'cached column incrementally (O(lookback)); the libraries with no indicator cache '
               '(pandas / polars / talib) must recompute the series (O(n)). Every candidate is '
               'measured with the same round count so the <code>rounds</code> column is comparable.'),
    'coverage': ('Every indicator <strong>both volas and TA-Lib implement</strong> (the set the parity '
                 'suite aligns), batch-computed and timed against TA-Lib only — an indicator only one '
                 'of them has is omitted. <code>volas vs TA-Lib</code> &gt; 1.00× means volas is faster.'),
}


def _fmt_time(seconds: float) -> str:
    """Human time with a unit that suits the magnitude."""
    us = seconds * 1e6
    if us < 1.0:
        return f'{us * 1000:.1f} ns'
    if us < 1000.0:
        return f'{us:.2f} µs'
    return f'{us / 1000:.2f} ms'


def _fmt_ops(ops: float) -> str:
    return f'{ops:,.0f}'


def parse(data: dict) -> dict:
    """JSON -> {category: {indicator: [(candidate, stats), ...]}}."""
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for b in data['benchmarks']:
        name = b['name']
        if name.startswith('test_calc'):
            category = 'calc'
        elif name.startswith('test_coverage'):
            category = 'coverage'
        else:
            category = 'append'
        params = b.get('params') or {}
        indicator = params.get('indicator', name)
        candidate = params.get('candidate', name)
        groups[category][indicator].append((candidate, b['stats']))
    return groups


def _bar_chart(entries: list[tuple[str, dict]]) -> str:
    """A vertical-bar SVG of median time (shorter = faster) for one group."""
    rows = sorted(entries, key=lambda e: CANDIDATE_ORDER.index(e[0])
                  if e[0] in CANDIDATE_ORDER else 99)
    medians = [s['median'] for _, s in rows]
    worst = max(medians) if medians else 1.0

    pad_l, pad_r, pad_t, pad_b = 8, 8, 22, 40
    bar_w, gap = 64, 26
    plot_h = 190
    width = pad_l + pad_r + len(rows) * bar_w + max(0, len(rows) - 1) * gap
    height = pad_t + plot_h + pad_b

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" class="chart">']
    # baseline
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
                 f'y2="{pad_t + plot_h}" stroke="#d0d3da" stroke-width="1"/>')
    x = pad_l
    for cand, stats in rows:
        med = stats['median']
        h = max(2.0, (med / worst) * plot_h) if worst > 0 else 2.0
        y = pad_t + plot_h - h
        color = COLORS.get(cand, '#999')
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" '
                     f'rx="3" fill="{color}"><title>{html.escape(cand)}: '
                     f'{_fmt_time(med)}</title></rect>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                     f'class="bar-val">{_fmt_time(med)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{pad_t + plot_h + 16:.1f}" '
                     f'text-anchor="middle" class="bar-lbl">{html.escape(cand)}</text>')
        x += bar_w + gap
    parts.append('</svg>')
    return ''.join(parts)


def _table(entries: list[tuple[str, dict]]) -> str:
    """Mean / Median / OPS / rounds / Perf table (Perf: slowest = 1.00x)."""
    rows = sorted(entries, key=lambda e: e[1]['median'])  # fastest first
    worst = max((s['median'] for _, s in rows), default=1.0)
    out = ['<table class="stats"><thead><tr>'
           '<th>Candidate</th><th>Mean</th><th>Median</th>'
           '<th>OPS</th><th>rounds</th><th>Perf</th></tr></thead><tbody>']
    for cand, s in rows:
        perf = worst / s['median'] if s['median'] > 0 else 1.0
        dot = (f'<span class="dot" style="background:{COLORS.get(cand, "#999")}"></span>')
        out.append(
            f'<tr><td>{dot}{html.escape(cand)}</td>'
            f'<td>{_fmt_time(s["mean"])}</td>'
            f'<td>{_fmt_time(s["median"])}</td>'
            f'<td>{_fmt_ops(s["ops"])}</td>'
            f'<td>{int(s["rounds"])}</td>'
            f'<td class="perf">{perf:.2f}×</td></tr>'
        )
    out.append('</tbody></table>')
    return ''.join(out)


def _coverage_section(by_indicator: dict) -> str:
    """A single volas-vs-TA-Lib table over the whole coverage set, ordered by the
    ``volas vs TA-Lib`` speedup from largest to smallest; a win/loss summary on top."""
    rows = []
    for ind, entries in by_indicator.items():
        d = dict(entries)
        if 'volas' not in d or 'talib' not in d:
            continue
        v, t = d['volas']['median'], d['talib']['median']
        rows.append((ind, v, t, (t / v) if v > 0 else 0.0))
    rows.sort(key=lambda r: r[3], reverse=True)  # descending: largest speedup first
    wins = sum(1 for *_, s in rows if s >= 1.0)
    body = []
    for ind, v, t, s in rows:
        cls = 'win' if s >= 1.0 else 'loss'
        body.append(
            f'<tr><td class="ind-name">{html.escape(ind)}</td>'
            f'<td>{_fmt_time(v)}</td><td>{_fmt_time(t)}</td>'
            f'<td class="perf {cls}">{s:.2f}×</td></tr>'
        )
    summary = (f'<p class="blurb">volas beats TA-Lib on <strong>{wins} / {len(rows)}</strong> '
               f'covered indicators.</p>')
    return (f'{summary}<table class="stats cov"><thead><tr>'
            f'<th>Indicator</th><th>volas</th><th>TA-Lib</th><th>volas vs TA-Lib</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table>')


def _legend() -> str:
    items = ''.join(
        f'<span class="leg"><span class="dot" style="background:{COLORS[c]}"></span>{c}</span>'
        for c in CANDIDATE_ORDER
    )
    return f'<div class="legend">{items}</div>'


def render(data: dict) -> str:
    groups = parse(data)
    machine = data.get('machine_info', {})
    when = data.get('datetime', '')
    cpu = machine.get('cpu', {}).get('brand_raw') or machine.get('processor', '')
    pyver = machine.get('python_version', '')

    sections = []
    for category in CATEGORY_ORDER:
        if category not in groups:
            continue
        blurb = f'<p class="blurb">{CATEGORY_BLURB[category]}</p>'
        if category == 'coverage':
            inner = _coverage_section(groups[category])
        else:
            inner = ''.join(
                f'<section class="ind"><h3>{html.escape(indicator)}</h3>'
                f'<div class="grid">{_bar_chart(groups[category][indicator])}'
                f'{_table(groups[category][indicator])}</div></section>'
                for indicator in sorted(groups[category])
            )
        sections.append(
            f'<section class="cat"><h2>{CATEGORY_TITLES[category]}</h2>{blurb}{inner}</section>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>volas benchmark report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 0; color: #1c1e26; background: #fbfbfd; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 28px 20px 64px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  .meta {{ color: #6a7280; font-size: 13px; margin-bottom: 18px; }}
  .meta code {{ background: #eef0f4; padding: 1px 5px; border-radius: 4px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 14px 0 26px; }}
  .leg, .stats td:first-child {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
  h2 {{ font-size: 19px; margin: 34px 0 2px; padding-top: 14px; border-top: 1px solid #e6e8ec; }}
  .note {{ color: #6a7280; background: #f4f6f9; border-left: 3px solid #c9ced8;
          padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 0 0 26px; font-size: 13px; }}
  .note code {{ background: #e7eaef; padding: 1px 4px; border-radius: 4px; }}
  .blurb {{ color: #6a7280; margin: 2px 0 18px; }}
  .blurb code {{ background: #eef0f4; padding: 1px 4px; border-radius: 4px; }}
  .ind {{ margin: 0 0 26px; }}
  .ind h3 {{ font-size: 15px; margin: 0 0 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .grid {{ display: grid; grid-template-columns: minmax(320px, auto) 1fr; gap: 22px; align-items: start; }}
  @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .chart {{ overflow: visible; }}
  .bar-val {{ font-size: 10.5px; fill: #4a4f5a; font-variant-numeric: tabular-nums; }}
  .bar-lbl {{ font-size: 10.5px; fill: #6a7280; }}
  table.stats {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
  .stats th, .stats td {{ text-align: right; padding: 5px 10px; border-bottom: 1px solid #eceef2; }}
  .stats th:first-child, .stats td:first-child {{ text-align: left; }}
  .stats th {{ color: #6a7280; font-weight: 600; font-size: 12px; }}
  .stats .perf {{ font-weight: 700; }}
  .stats.cov td.ind-name {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }}
  .stats .perf.win {{ color: #1f9d57; }}
  .stats .perf.loss {{ color: #d4493f; }}
  footer {{ margin-top: 40px; color: #9aa0aa; font-size: 12px; }}
</style></head>
<body><div class="wrap">
  <h1>volas benchmark report</h1>
  <div class="meta">
    OHLCV technical-indicator computation across libraries ·
    {html.escape(str(when))} · {html.escape(str(cpu))} · Python {html.escape(str(pyver))}
    <br>Lower time is better; <code>Perf</code> shows the slowest candidate as
    <code>1.00×</code> and each faster candidate as <code>{{slowest ÷ this}}×</code>.
  </div>
  {_legend()}
  {''.join(sections)}
  <footer>Generated by <code>scripts/benchmark_report.py</code> from pytest-benchmark output.</footer>
</div></body></html>
"""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path('benchmark-report.html')
    data = json.loads(src.read_text())
    out.write_text(render(data))
    n = len(data.get('benchmarks', []))
    print(f'benchmark report: {n} measurements -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
