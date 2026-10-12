"""Deterministic observation schedules over the whole directive inventory.

Where the property machines sample, these sweep: every output line (and a boolean
composite of it) cached together on one frame, driven through a fixed schedule, and
compared with the ledger reference at the end. They are cheap enough for every
``make test`` and are what the historical-defect calibration replays.
"""

from __future__ import annotations

import re

import numpy as np

import volas

from . import budget, build, contracts, observe
from .data import EPOCHS, NS, Schema, make_rows
from .directives import Directive
from .inventory import executable_lines
from .ledger import Ledger, TfLedger, period_start, reference
from .machine_base import triggers_ok

OHLCV = Schema('float64', 'float64', (), 'datetime', None)


def sweep_directives(composites: bool = True, kinds: tuple[str, ...] = ('num', 'bool')) -> list[str]:
    """Every executable line (of the value ``kinds``) in its default form and a
    short-period form, plus (``composites``) a comparison and an arithmetic composite
    of each numeric one — minus open-defect triggers."""
    out: list[str] = []
    for line in executable_lines():
        if line.kind not in kinds:
            continue
        forms = {line.form + line.series_suffix}
        if line.args and line.args[0].kind == 'int':
            lo = int(line.args[0].lo)
            forms.add(f'{line.head}:{max(lo, 3)}{line.series_suffix}')
        for form in sorted(forms):
            out.append(form)
            if composites and line.kind == 'num':
                out.append(f'close>{form}')
                out.append(f'({form})-open')
    out = [d for d in out if triggers_ok(Directive(d, 'num', commands=1))]
    return [d for d in out if re.search(budget.ONLY, d)] if budget.ONLY else out


def rows_of(data_class: str, n: int, seed: int = 11) -> list[dict]:
    """``n`` OHLCV rows: a walk, or a walk with NaN / +inf cells early in the history
    (far behind any window an append refreshes) and one near the end."""
    rows = make_rows(OHLCV, ('walk', seed), n, None)
    if data_class == 'nan':
        rows[6]['close'] = None
        rows[10]['high'] = None
        rows[n // 2]['close'] = None
    elif data_class == 'inf':
        rows[8]['high'] = float('inf')
        rows[n // 2]['high'] = float('inf')
    return rows


def tolerance() -> float:
    return contracts.ULP_DRIFT_TOLERANCE if contracts.avoids('window-recompute-ulp') else 0.0


def compare_all(df: volas.DataFrame, ledger: Ledger, names: dict[str, str]) -> list[str]:
    """Each cached column (``names``: directive -> column) against its reference over
    ``ledger``, on a refreshed copy; returns one line per mismatch."""
    view = df.copy()
    view.fulfill()
    failures = []
    env = ledger.columns
    for directive, column in names.items():
        want, error = reference(ledger, directive, env)
        if error is not None:
            failures.append(f'{directive}: reference raised {error.__name__}')
            continue
        got = observe.column_values_series(view.get_column(column))
        excused = None
        if '>' in directive and tolerance():
            lhs, rhs = directive.split('>', 1)
            ref = ledger.frame(env)
            excused = observe.tie_rows(ref.exec(lhs, dtype='float64'), ref.exec(rhs, dtype='float64'))
        try:
            observe.same_indicator(got, np.asarray(want), directive, tolerance(), excused)
        except AssertionError as e:
            failures.append(str(e))
    return failures


def reach(names: dict[str, str], directives: list[str]) -> list[str]:
    """A schedule that cached too little proves nothing: fail it instead."""
    if len(names) * 4 < len(directives) * 3:
        return [f'only {len(names)} of {len(directives)} directives could be cached']
    return []


def cache_all(df: volas.DataFrame, directives: list[str]) -> dict[str, str]:
    """Cache every directive; returns directive -> the column it was cached under."""
    names = {}
    for d in directives:
        before = set(df.columns)
        try:
            df[d]
        except Exception:  # noqa: BLE001 - not computable here; the reference is not asked
            continue
        added = set(df.columns) - before
        names[d] = added.pop() if added else volas.directive_stringify(d)
    return names


def plain_schedule(data_class: str, mode: str, seed_rows: int = 40, bars: int = 24) -> list[str]:
    """Cache everything on ``seed_rows`` rows, append ``bars`` bars ('one': one at a
    time with a fulfill each; 'read': one at a time, refreshed by column reads; 'batch':
    one multi-row append; 'quiet': one at a time with no read until the end)."""
    rows = rows_of(data_class, seed_rows + bars)
    index = [EPOCHS[1] + i * 60 * NS for i in range(seed_rows + bars)]
    ledger = Ledger(OHLCV)
    ledger.append(rows[:seed_rows], index[:seed_rows])
    df = build.frame(OHLCV, rows[:seed_rows], index[:seed_rows])
    names = cache_all(df, sweep_directives())
    if mode == 'batch':
        df.append(build.frame(OHLCV, rows[seed_rows:], index[seed_rows:]))
    else:
        for i in range(seed_rows, seed_rows + bars):
            df.append(build.frame(OHLCV, rows[i:i + 1], index[i:i + 1]))
            if mode == 'one':
                df.fulfill()
            elif mode == 'read':
                for column in names.values():
                    df[column]
    ledger.append(rows[seed_rows:], index[seed_rows:])
    return reach(names, sweep_directives()) + compare_all(df, ledger, names)


def locality_failures(data_class: str, n: int = 80, cut: int = 12) -> list[str]:
    """Window locality (the oracle for "a non-finite input never outlives the rows
    that read it"): for every line whose output on clean data depends only on its
    last ``lookback + 1`` rows, the full compute must equal a compute over the rows
    from ``cut`` on, once both are past the warm-up — also when a NaN / inf input sits
    before ``cut``."""
    clean = rows_of('walk', n)
    # The same history with a large finite jump early on: a finite-memory line
    # forgets it, a recursive / cumulative one does not.
    jumped = [dict(r) for r in clean]
    for k in ('open', 'high', 'low', 'close'):
        jumped[3][k] = clean[3][k] * 1.5
    jumped[3]['volume'] = clean[3]['volume'] * 7
    dirty = rows_of(data_class, n)
    index = [EPOCHS[1] + i * 60 * NS for i in range(n)]
    frames = {}
    for name, rows in (('clean', clean), ('jumped', jumped), ('dirty', dirty)):
        ledger = Ledger(OHLCV)
        ledger.append(rows, index)
        frames[name] = (ledger.frame(), ledger.frame(start=cut))
    results: dict[str, tuple] = {}
    recursive: set[str] = set()
    for d in sweep_directives(composites=False):
        try:
            lb = volas.directive_lookback(d)
        except Exception:  # noqa: BLE001 - a revision may not know the form
            continue
        start = cut + 2 * lb + 2
        if start >= n:
            continue
        try:
            local = {k: (np.asarray(full[d], float)[start:], np.asarray(tail[d], float)[start - cut:])
                     for k, (full, tail) in frames.items()}
        except Exception:  # noqa: BLE001 - not computable on this data
            continue
        try:
            observe.same_indicator(*local['clean'], f'{d} clean')
            observe.same_indicator(*local['jumped'], f'{d} jumped')
            observe.same_indicator(local['clean'][0], local['jumped'][0], f'{d} forgets the jump')
        except AssertionError:
            # not finite-memory (recursive / cumulative / positional): neither is any
            # other line of the same command (a discrete line can hide the memory)
            recursive.add(d.split(':')[0].split('@')[0].split('.')[0])
            continue
        results[d] = local['dirty']
    failures = []
    for d, (full, tail) in results.items():
        if d.split(':')[0].split('@')[0].split('.')[0] in recursive:
            continue
        try:
            observe.same_indicator(full, tail, f'{d} with {data_class} before row {cut}', tolerance())
        except AssertionError as e:
            failures.append(str(e))
    return failures


def tf_schedule(tf: str, mode: str, cache_when: str, bools: bool = False, bars: int = 40) -> list[str]:
    """A tf frame (``mode``: 'cumulate' from fine history with a forming bar, or
    'construct' from closed bars) folding 1m bars one at a time with a fulfill each;
    directives cached up front or ('forming') after the first fold. Without ``bools``
    only numeric lines are cached: a cached boolean column moves every fold onto the
    batch path (which invalidates all caches), so the in-place fold and the anchored
    forming refresh are exercised only without one."""
    step = {'5m': 5, '15m': 15}[tf]
    start = period_start(EPOCHS[1], tf, None)
    store = TfLedger(OHLCV, tf)
    periods = 30  # closed history longer than the lookbacks, so refreshes resume
    rows = rows_of('walk', periods * step + 2 + bars, seed=5)
    if mode == 'cumulate':
        fine_ts = [start + i * 60 * NS for i in range(periods * step + 2)]
        for t, r in zip(fine_ts, rows):
            store.add(t, r)
        df = build.frame(OHLCV, rows[:len(fine_ts)], fine_ts).cumulate(tf)
        nxt, used = fine_ts[-1] + 60 * NS, len(fine_ts)
    else:
        labels = [start + i * step * 60 * NS for i in range(30)]
        store.seed_index, store.seed_rows = labels, rows[:30]
        df = volas.DataFrame(build.frame(OHLCV, rows[:30], labels), time_frame=tf)
        nxt, used = labels[-1] + step * 60 * NS, 30
    directives = sweep_directives(composites=False, kinds=('num', 'bool') if bools else ('num',))
    names = cache_all(df, directives) if cache_when == 'start' else {}
    for i in range(bars):
        store.add(nxt, rows[used + i])
        df.append(build.frame(OHLCV, rows[used + i:used + i + 1], [nxt]))
        if i == 0 and cache_when == 'forming':
            names = cache_all(df, directives)
        df.fulfill()
        nxt += 60 * NS
    coarse = store.coarse()
    view = df.copy()
    view.fulfill()
    failures = []
    for col in OHLCV.columns:
        try:
            observe.same_data(observe.column_values(view, col),
                              observe.expected_values(coarse.values(col), 'float64'), f'tf {col}')
        except AssertionError as e:
            failures.append(str(e))
    return failures + reach(names, directives) + compare_all(df, coarse, names)
