import re

from scripts import benchmark_report


def _entry(test_name, indicator, candidate, min_seconds, **params):
    return {
        'name': test_name,
        'params': {'indicator': indicator, 'candidate': candidate, **params},
        'stats': {
            'min': min_seconds,
            'mean': min_seconds,
            'median': min_seconds,
            'ops': 1.0 / min_seconds,
            'rounds': 10,
        },
    }


def test_coverage_report_keeps_extended_metrics_on_one_indicator_row():
    html = benchmark_report.render({
        'datetime': '2026-06-07',
        'machine_info': {'python_version': '3.12'},
        'benchmarks': [
            _entry('test_coverage[roc:10-talib]', 'roc:10', 'talib', 2.0),
            _entry('test_coverage[roc:10-volas]', 'roc:10', 'volas', 1.0),
            _entry('test_coverage_extended[roc:10-20000-talib]', 'roc:10', 'talib', 4.0, length=20000),
            _entry('test_coverage_extended[roc:10-20000-volas]', 'roc:10', 'volas', 2.0, length=20000),
            _entry('test_coverage_after_append[roc:10-talib]', 'roc:10', 'talib', 8.0),
            _entry('test_coverage_after_append[roc:10-volas]', 'roc:10', 'volas', 2.0),
        ],
    })

    assert html.count('<td class="ind-name">roc:10</td>') == 1
    assert '<th>volas vs TA-Lib</th>' in html
    assert '<th>volas vs TA-Lib (20000)</th>' in html
    assert '<th>volas vs TA-Lib (after append)</th>' in html
    # Column order (left to right): after-append, default ratio, extended (20000).
    assert re.search(r'<td class="perf win">4\.00×</td>.*<td class="perf win">2\.00×</td>'
                     r'.*<td class="perf win">2\.00×</td>', html, re.S)


def test_coverage_headline_counts_only_default_ratio_column():
    html = benchmark_report.render({
        'datetime': '2026-06-07',
        'machine_info': {'python_version': '3.12'},
        'benchmarks': [
            _entry('test_coverage[base_fast-talib]', 'base_fast', 'talib', 1.01),
            _entry('test_coverage[base_fast-volas]', 'base_fast', 'volas', 1.0),
            _entry('test_coverage_extended[base_fast-20000-talib]', 'base_fast', 'talib', 1.0, length=20000),
            _entry('test_coverage_extended[base_fast-20000-volas]', 'base_fast', 'volas', 2.0, length=20000),
            _entry('test_coverage_after_append[base_fast-talib]', 'base_fast', 'talib', 1.0),
            _entry('test_coverage_after_append[base_fast-volas]', 'base_fast', 'volas', 2.0),
            _entry('test_coverage[append_fast-talib]', 'append_fast', 'talib', 0.99),
            _entry('test_coverage[append_fast-volas]', 'append_fast', 'volas', 1.0),
            _entry('test_coverage_extended[append_fast-20000-talib]', 'append_fast', 'talib', 2.0, length=20000),
            _entry('test_coverage_extended[append_fast-20000-volas]', 'append_fast', 'volas', 1.0, length=20000),
            _entry('test_coverage_after_append[append_fast-talib]', 'append_fast', 'talib', 2.0),
            _entry('test_coverage_after_append[append_fast-volas]', 'append_fast', 'volas', 1.0),
        ],
    })

    assert html.count('<td class="ind-name">') == 2
    assert 'volas beats TA-Lib on <strong>1 / 2</strong> covered indicators' in html


def test_windowed_section_rescales_to_per_bar_and_sorts_first():
    # A full-stream windowed benchmark is rescaled to per-bar via extra_info.
    def wentry(candidate, total_seconds):
        return {
            'name': f'test_windowed_stream[atr:14-{candidate}]',
            'params': {'indicator': 'atr:14', 'candidate': candidate},
            'extra_info': {'stream_bars': 1000},
            'stats': {
                'min': total_seconds, 'mean': total_seconds, 'median': total_seconds,
                'max': total_seconds, 'stddev': 0.0, 'ops': 1.0 / total_seconds, 'rounds': 3,
            },
        }

    html = benchmark_report.render({
        'datetime': '2026-06-20',
        'machine_info': {'python_version': '3.12'},
        'benchmarks': [
            wentry('volas', 0.020),   # 20us/bar over 1000 bars
            wentry('talib', 0.006),   # 6us/bar
            _entry('test_calc[ma:20-volas]', 'ma:20', 'volas', 1.0),
        ],
    })

    assert 'Windowed live stream' in html
    # 0.020 s / 1000 bars -> 20 µs per bar (NOT 20 ms).
    assert '20.00 µs' in html
    assert '6.00 µs' in html
    # the windowed section renders before the batch (calc) section.
    assert html.index('Windowed live stream') < html.index('Batch indicator computation')
