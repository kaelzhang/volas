#!/usr/bin/env python3
"""Render a pytest-benchmark JSON run into a self-contained HTML report.

Usage::

    python scripts/benchmark_report.py <benchmark.json> [output.html]

For charted benchmark groups — one per (category, indicator), where category is
``calc`` (batch), ``append`` (one new bar), or core ``api`` — the report shows a
vertical bar chart of median time (shorter = faster) plus a table with columns
``Mean``, ``Median``, ``OPS``, ``rounds`` and ``Perf``. Full coverage is a single
table with one row per TA-Lib-backed indicator and additional ratio columns for
configured generated lengths and cached append refresh.

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
# Section order in the report: keep the existing chart sections first, then coverage.
CATEGORY_ORDER = ['append', 'api', 'calc', 'coverage']
CATEGORY_TITLES = {
    'calc': 'Batch indicator computation',
    'append': 'Append one new bar → updated indicator',
    'api': 'Core DataFrame API (construct / slice / mask / assign / copy)',
    'coverage': 'Full coverage — volas vs TA-Lib',
}
CATEGORY_BLURB = {
    'calc': 'Compute the indicator over the whole series, across every library.',
    'append': ('A new bar arrives. <code>volas</code> / <code>stock_pandas</code> refresh their '
               'cached column incrementally (O(lookback)); the libraries with no indicator cache '
               '(pandas / polars / talib) must recompute the series (O(n)). Every candidate is '
               'measured with the same round count so the <code>rounds</code> column is comparable.'),
    'api': ('The data-handling plumbing a live system runs around every indicator call — frame '
            'construction, column access, row slicing, boolean masking, column assignment, copy — '
            'timed against pandas / polars. Not indicator math; the surrounding core APIs.'),
    'coverage': ('Every indicator <strong>both volas and TA-Lib implement</strong> (the set the parity '
                 'suite aligns), one row per indicator. The default <code>volas vs TA-Lib</code> '
                 'column is the Tencent fixture; optional generated lengths and cached append '
                 'refresh appear as additional ratio columns. Values &gt; 1.00× mean volas is faster.'),
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
        elif name.startswith('test_coverage_extended'):
            category = 'coverage_extended'
        elif name.startswith('test_coverage_after_append'):
            category = 'coverage_after_append'
        elif name.startswith('test_coverage'):
            category = 'coverage'
        elif name.startswith('test_api'):
            category = 'api'
        else:
            category = 'append'
        params = b.get('params') or {}
        indicator = params.get('indicator') or params.get('op') or name
        if category == 'coverage_extended':
            indicator = (indicator, params.get('length'))
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


def _coverage_ratio(entries: list[tuple[str, dict]]) -> tuple[float, float, float] | None:
    d = dict(entries)
    if 'volas' not in d or 'talib' not in d:
        return None
    # `min` (the fastest of many rounds) is the most reproducible statistic: it
    # filters the OS-scheduling / CPU-frequency-scaling noise that makes a
    # borderline indicator's `median` flip the win count between runs.
    v, t = d['volas']['min'], d['talib']['min']
    return v, t, (t / v) if v > 0 else 0.0


def _coverage_section(groups: dict) -> str:
    """A single volas-vs-TA-Lib table over the whole coverage set, ordered by the
    ``volas vs TA-Lib`` speedup from largest to smallest; a win/loss summary on top."""
    by_indicator = groups.get('coverage', {})
    by_length = groups.get('coverage_extended', {})
    after_append = groups.get('coverage_after_append', {})
    lengths = sorted({length for _, length in by_length if length is not None})
    rows = []
    for ind, entries in by_indicator.items():
        base = _coverage_ratio(entries)
        if base is None:
            continue
        ext = {
            length: _coverage_ratio(by_length.get((ind, length), []))
            for length in lengths
        }
        rows.append((ind, *base, ext, _coverage_ratio(after_append.get(ind, []))))
    rows.sort(key=lambda r: r[3], reverse=True)  # descending: largest speedup first
    def verdict(s: float) -> str:
        if s > 1.0:
            return 'win'
        if s < 1.0:
            return 'loss'
        return 'tie'

    def perf_cell(metric: tuple[float, float, float] | None) -> str:
        if metric is None:
            return '<td class="perf missing">n/a</td>'
        score = metric[2]
        return f'<td class="perf {verdict(score)}">{score:.2f}×</td>'

    # The headline is deliberately based only on the default Tencent fixture
    # ratio (`volas vs TA-Lib`). Generated lengths and cached append refresh
    # columns are diagnostics and must not change the top-line coverage count.
    wins = sum(1 for _, _, _, s, _, _ in rows if verdict(s) == 'win')
    ties = sum(1 for _, _, _, s, _, _ in rows if verdict(s) == 'tie')
    losses = len(rows) - wins - ties
    extra_heads = ''.join(
        f'<th>volas vs TA-Lib ({html.escape(str(length))})</th>'
        for length in lengths
    )
    body = []
    for ind, v, t, s, ext, append_metric in rows:
        extra_cells = ''.join(perf_cell(ext[length]) for length in lengths)
        body.append(
            f'<tr><td class="ind-name">{html.escape(ind)}</td>'
            f'<td>{_fmt_time(v)}</td><td>{_fmt_time(t)}</td>'
            f'{perf_cell(append_metric)}'
            f'<td class="perf {verdict(s)}">{s:.2f}×</td>'
            f'{extra_cells}</tr>'
        )
    summary = (f'<p class="blurb">volas beats TA-Lib on <strong>{wins} / {len(rows)}</strong> '
               f'covered indicators by the default ratio '
               f'({ties} exactly even, {losses} slower).</p>')
    return (f'{summary}<table class="stats cov"><thead><tr>'
            f'<th>Indicator</th><th>volas</th><th>TA-Lib</th>'
            f'<th>volas vs TA-Lib (after append)</th><th>volas vs TA-Lib</th>'
            f'{extra_heads}'
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
            inner = _coverage_section(groups)
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
  .site {{ margin: 0 0 22px; padding-bottom: 18px; border-bottom: 1px solid #e6e8ec; }}
  .brand {{ display: flex; align-items: center; justify-content: space-between;
           flex-wrap: wrap; gap: 12px; }}
  h1 {{ font-size: 26px; margin: 0; letter-spacing: -0.01em; }}
  .gh {{ display: inline-flex; align-items: center; gap: 7px; color: #3a3f4b;
        text-decoration: none; font-size: 13px; font-weight: 500;
        border: 1px solid #d4d8e0; border-radius: 7px; padding: 5px 11px;
        background: #f4f6f9; }}
  .gh:hover {{ background: #eceff4; color: #1c1e26; }}
  .gh svg {{ flex: none; }}
  .intro {{ color: #41464f; font-size: 14.5px; line-height: 1.55; margin: 12px 0 0;
           max-width: 760px; }}
  .intro code {{ background: #eef0f4; padding: 1px 5px; border-radius: 4px; }}
  .intro a {{ color: #4845c9; text-decoration: none; }}
  .intro a:hover {{ text-decoration: underline; }}
  h2.rpt {{ font-size: 18px; margin: 0 0 4px; border-top: none; padding-top: 0; }}
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
  .stats .perf.tie {{ color: #8a8a8a; }}
  .stats .perf.missing {{ color: #9aa0aa; font-weight: 400; }}
  footer {{ margin-top: 40px; color: #9aa0aa; font-size: 12px; }}
</style></head>
<body><div class="wrap">
  <header class="site">
    <div class="brand">
      <h1>volas</h1>
      <a class="gh" href="https://github.com/kaelzhang/volas"
         target="_blank" rel="noopener">
        <svg viewBox="0 0 16 16" width="17" height="17" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
        github.com/kaelzhang/volas
      </a>
    </div>
    <p class="intro">A Rust-backed, pandas-shaped <code>DataFrame</code> for live
      OHLCV pipelines: 242 trading indicators, incremental <code>O(lookback)</code>
      refresh on each new bar, and NumPy/Torch-ready output. This page is the
      project's reproducible benchmark report — regenerated from
      <code>make benchmark</code> on each release. See the
      <a href="https://github.com/kaelzhang/volas">repository</a> for docs and
      installation.</p>
  </header>
  <h2 class="rpt">Benchmark report</h2>
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
