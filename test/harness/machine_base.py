"""The shared core of the stateful differential machines.

A machine owns the frame under test, the ledger of accepted events, and the cache
model (which directive columns exist, what each was bound against, where its
history starts, and which are stale). After every op it asserts:

* structure (``observe.structure``);
* the cache model: membership, ``is_computed`` per column, and staleness exactly
  as the model predicts (a bulk read raises iff some cached column is stale);
* values, on a refreshed ``copy()``: index and data columns equal the ledger, every
  cached column equals its from-scratch reference.

Every op goes through :meth:`FrameMachine.attempt`: an op that raises must leave the
raw stored state and the refreshed view bit-identical (I8), and must raise exactly
when the contract says it should.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

import volas

from . import contracts, inventory, observe
from .data import EXTREME_SCALES
from .directives import Directive, directives
from .ledger import Ledger, reference


@dataclass(frozen=True)
class Entry:
    """A cached directive column as the model knows it."""

    name: str  # the column name (canonical directive)
    directive: Directive | None
    env: tuple[str, ...]  # the plain columns it was bound against
    start: int  # the ledger row its history starts at


class Unexpected(AssertionError):
    pass


# Ops executed per kind across a run (printed by the tests when VOLAS_HARNESS_STATS=1).
STATS: Counter = Counter()

# Whole-frame readers that never refresh (README "df.fulfill()": they raise while a
# cached column is stale).
BULK_SURFACES = {
    'to_numpy': lambda df: df.to_numpy(dtype='object'),
    'iloc_row': lambda df: df.iloc[-1] if len(df) else df.iloc[0:0],
    'iloc_slice': lambda df: df.iloc[1:],
    'head': lambda df: df.head(3),
    'tail': lambda df: df.tail(3),
    'equals': lambda df: df.equals(df.copy()),
    'to_pandas': lambda df: df.to_pandas(),
    'to_arrow': lambda df: df.to_arrow(),
    'to_csv': lambda df: df.to_csv(),
    'repr': lambda df: repr(df),
    'count': lambda df: df.count(),
    'rolling': lambda df: df.rolling(2).count(),
    'sort_index': lambda df: df.sort_index(),
    'reset_index': lambda df: df.reset_index(),
    'drop_duplicates': lambda df: df.drop_duplicates(),
    'duplicated': lambda df: df.duplicated(),
    'nsmallest': lambda df: df.nsmallest(2, 'close'),
    'astype': lambda df: df.astype({'close': 'float64'}),
}


# The (fast, slow) leading arguments of the two-EMA lines, with their defaults.
_FAST_SLOW = re.compile(r'\b(macd|apo|ppo|dma)(?:\.\w+)?:(\d*)(?:,(\d*))?')
_FAST_SLOW_DEFAULTS = {'macd': (12, 26), 'apo': (12, 26), 'ppo': (12, 26), 'dma': (10, 50)}
_ULTOSC = re.compile(r'\bultosc:(\d*)(?:,(\d*))?(?:,(\d*))?')
# A parenthesized single primary as a series (`@(rsi:5)`, `@(0)`): its parentheses
# are dropped from the cached name.
_PAREN_PRIMARY_SERIES = re.compile(
    r'@\(\s*(?:-?[\d.]+|[a-z_][\w.]*(?::[\w.,\-]*)?(?:@[\w.,]+)?)\s*\)')


def triggers_ok(d: Directive) -> bool:
    """Whether ``d`` steers clear of every open defect's trigger (see contracts)."""
    text = d.text.lower()
    if contracts.avoids('nested-series-name') and _PAREN_PRIMARY_SERIES.search(text):
        return False
    if contracts.avoids('fast-above-slow'):
        for name, fast, slow in _FAST_SLOW.findall(text):
            default_fast, default_slow = _FAST_SLOW_DEFAULTS[name]
            if int(fast or default_fast) > int(slow or default_slow):
                return False
    if contracts.avoids('ultosc-period-order'):
        for a, b, c in _ULTOSC.findall(text):
            periods = [int(a or 7), int(b or 14), int(c or 28)]
            if periods != sorted(periods):
                return False
    families = {m.split('.')[0] for m in re.findall(r'[a-z_][a-z_0-9.]*', text)}
    return not (families & contracts.DIVERGENT_FAMILIES)


class FrameMachine(RuleBasedStateMachine):
    """Subclasses build ``self.df`` / ``self.store`` in an ``@initialize`` and add the
    frame-kind specific ops. ``store.coarse()`` is the ledger of the rows the frame
    shows (the tf fold aggregates fine bars into it)."""

    window: int | None = None
    max_lookback: int = 0

    def __init__(self) -> None:
        super().__init__()
        self.df: volas.DataFrame | None = None
        self.store = None
        self.cached: dict[str, Entry] = {}
        self.stale: set[str] = set()
        self.offset = 0  # ledger rows a slice dropped
        self.snapshots: list[tuple] = []
        self.trace: list[str] = []
        self.column_of: dict[str, str] = {}  # directive_stringify -> the cached column name

    # --- model helpers -------------------------------------------------------------

    def coarse(self) -> Ledger:
        return self.store.coarse()

    def visible_start(self) -> int:
        total = len(self.coarse().rows)
        if self.window is not None:
            return max(self.offset, total - self.window)
        return self.offset

    def env(self) -> tuple[str, ...]:
        return self.coarse().columns

    def plain_columns(self) -> set[str]:
        return set(self.env())

    def tolerance(self, entry: Entry) -> float:
        family = entry.name.split(':')[0].split('.')[0].split('@')[0]
        if family in contracts.NUMERIC_EXCEPTIONS:
            return contracts.NUMERIC_EXCEPTIONS[family]
        tolerance = 0.0
        if self.window is not None and contracts.avoids('windowed-drift'):
            tolerance = contracts.WINDOWED_FINITE_TOLERANCE
        if contracts.avoids('window-recompute-ulp'):
            tolerance = max(tolerance, contracts.ULP_DRIFT_TOLERANCE)
        return tolerance

    def oracle(self, entry: Entry, evaluate: bool = False):
        """The reference values of ``entry`` on the visible rows, or its error class."""
        text = entry.directive.text if entry.directive else entry.name
        values, error = reference(self.coarse(), text, entry.env, entry.start, evaluate)
        if error is not None:
            return None, error
        return values[self.visible_start() - entry.start:], None

    def excused(self, entry: Entry) -> np.ndarray | None:
        """Rows a comparison may legitimately flip on (operands tie within the
        documented ULP drift) while 'window-recompute-ulp' is open."""
        d = entry.directive
        if d is None or d.operands is None or self.tolerance(entry) == 0:
            return None
        ref = self.coarse().frame(entry.env, entry.start)
        try:
            lhs, rhs = (np.asarray(ref.exec(x), float) for x in d.operands)
        except Exception:
            return None
        return observe.tie_rows(lhs, rhs)[self.visible_start() - entry.start:]

    def log(self, text: str) -> None:
        self.trace.append(text)
        STATS[text.split('(')[0].split(' ')[0].split('[')[0]] += 1

    # While 'sliding-sum-magnitude' is open a frame holds one magnitude: a frame
    # started on the 'extreme' regime stays at that scale, any other never sees it.
    scale: float | None = None

    def pin_scale(self, regime: tuple[str, int]) -> None:
        if regime[0] == 'extreme' and contracts.avoids('sliding-sum-magnitude'):
            self.scale = EXTREME_SCALES[regime[1] % len(EXTREME_SCALES)]

    def steer(self, regime: tuple[str, int]) -> tuple[tuple[str, int], float | None]:
        if not contracts.avoids('sliding-sum-magnitude'):
            return regime, None
        if self.scale is not None:
            return ('extreme', regime[1]), self.scale
        return (('walk', regime[1]) if regime[0] == 'extreme' else regime), None

    # --- the op wrapper (I8) -------------------------------------------------------------

    def attempt(self, op, *, expect_error=None, commit=None, label=''):
        """Run ``op()``. ``expect_error`` is the exception class the contract requires
        (``None`` = must succeed). On success ``commit(result)`` updates the model; on
        a raise the stored state and the refreshed view must be unchanged."""
        before = observe.raw_state(self.df)
        # A refreshed view also covers what raw cells cannot show (carried recursive
        # state, the tf forming bar); it costs a refresh, so only around ops the
        # contract expects to fail.
        before_view = observe.fresh_view(self.df) if expect_error is not None else None
        try:
            result = op()
        except Exception as e:  # noqa: BLE001 - classified below
            changed = observe.diff_state(before, observe.raw_state(self.df))
            if changed:
                raise Unexpected(f'{label}: a rejected op changed the frame ({changed}): {e!r}') from e
            if before_view is not None and observe.fresh_view(self.df) != before_view:
                raise Unexpected(f'{label}: a rejected op changed the refreshed view: {e!r}') from e
            if expect_error is None or not isinstance(e, expect_error):
                raise Unexpected(f'{label}: raised {e!r}, contract expects '
                                 f'{expect_error.__name__ if expect_error else "success"}') from e
            self.log(f'{label} -> raised {type(e).__name__}')
            return None
        if expect_error is not None:
            raise Unexpected(f'{label}: succeeded, contract expects {expect_error.__name__}')
        self.log(label)
        if commit is not None:
            commit(result)
        return result

    # --- shared ops -----------------------------------------------------------------------

    def command_columns(self) -> set[str]:
        """The frame's data columns named like a built-in command usable bare."""
        from .inventory import executable_lines
        bare = {ln.head for ln in executable_lines() if ln.canonical == ln.head and '.' not in ln.head}
        return {c for c in self.df.columns if c in bare and c not in self.cached}

    def _directive_ok(self, d: Directive) -> bool:
        if not triggers_ok(d):
            return False
        if contracts.AMBIGUOUS_BARE_NAME_RAISES and d.bare_names & self.command_columns():
            return False  # ambiguous (a data column named like a command): raises, tested apart
        if contracts.avoids('window-recompute-ulp') and d.kind == 'bool' and d.operands is None \
                and any(op in d.text for op in '&|^~'):
            return False  # a logical combination cannot be excused a tie flip
        if self.window is not None:
            try:
                if volas.directive_lookback(d.text) > self.max_lookback:
                    return False  # the documented margin must cover the directive
            except Exception:
                return False
            if contracts.avoids('windowed-drift'):
                # recursive / cumulative lines restart on the retained rows (the
                # forming refresh's cold path, a write, an undersized history)
                names = {m.split('.')[0] for m in re.findall(r'[a-z_][a-z_0-9.]*', d.text.lower())}
                if names & inventory.unbounded_memory():
                    return False
        return True

    @rule(data=st.data())
    def cache(self, data):
        """``df[d]`` for a generated directive (caches it, or refreshes it if cached)."""
        d = data.draw(directives(tuple(self.plain_numeric())))
        if not self._directive_ok(d):
            return
        try:
            name = volas.directive_stringify(d.text)
        except volas.DirectiveError:
            return
        if name in self.plain_columns() or d.text in self.plain_columns():
            return  # a plain column of that name shadows the directive (get_column covers it)
        column = self.column_of.get(name, name)
        entry = self.cached.get(column) or Entry(column, d, self.env(), self.visible_start())
        if self.window is not None and column not in self.cached:
            # README: a windowed frame's cached indicators equal the unbounded frame's
            # (the unbounded twin's history starts at row 0). A directive first cached
            # after rows were dropped cannot see them; while 'windowed-drift' is open
            # that case is not generated.
            dropped = len(self.coarse().rows) - self.df._physical_height
            if dropped and contracts.avoids('windowed-drift'):
                return
            entry = replace(entry, start=0)
        if column not in self.cached and contracts.avoids('resume-missing-warmup') \
                and self._missing_in_warmup(d.text, entry.start):
            return
        want, expect = self.oracle(entry)
        before = set(self.df.columns)

        def commit(series):
            added = set(self.df.columns) - before
            assert len(added) <= 1, f'df[{d.text!r}] added several columns {added}'
            col = added.pop() if added else column
            if contracts.COLUMN_NAME_IS_STRINGIFY and not contracts.avoids('stringify-column-name'):
                assert col == name, f'df[{d.text!r}] cached as {col!r}, directive_stringify says {name!r}'
            assert col in self.df.columns and self.df.is_computed(col), f'df[{d.text!r}] did not cache'
            got = observe.column_values_series(series)
            observe.same_indicator(got, np.asarray(want), f'df[{d.text!r}]', self.tolerance(entry),
                                   self.excused(entry))
            # I9: the column name is a directive of the same meaning (it is what a
            # refresh re-parses).
            renamed = replace(entry, name=col, directive=replace(d, text=col))
            again, error = self.oracle(renamed)
            assert error is None, f'the cached name {col!r} of {d.text!r} does not re-parse: {error}'
            observe.same_indicator(np.asarray(again), np.asarray(want), f'{col!r} vs {d.text!r}')
            self.column_of[name] = col
            self.cached[col] = replace(entry, name=col)
            self.stale.discard(col)

        self.attempt(lambda: self.df[d.text], expect_error=expect, commit=commit, label=f'df[{d.text!r}]')

    def _missing_in_warmup(self, text: str, start: int) -> bool:
        """Whether an input cell is missing inside the directive's warm-up (the first
        lookback + 2 rows of its history) — or anywhere when the history is no longer
        than that yet."""
        try:
            span = volas.directive_lookback(text) + 2
        except Exception:
            return False
        rows = self.coarse().rows[start:start + span]
        return any(r.get(c) is None or (isinstance(r.get(c), float) and not np.isfinite(r[c]))
                   for r in rows for c in ('open', 'high', 'low', 'close', 'volume'))

    def plain_numeric(self) -> list[str]:
        coarse = self.coarse()
        return [c for c in coarse.columns
                if coarse.dtype(c) in ('float64', 'float32', 'int64', 'int32')] or ['close']

    @precondition(lambda self: self.cached)
    @rule(data=st.data())
    def read_cached(self, data):
        """Auto-refreshing column reads: one name, or a list of names."""
        names = data.draw(st.lists(st.sampled_from(sorted(self.cached)), min_size=1, max_size=3, unique=True))
        single = len(names) == 1 and data.draw(st.booleans())

        def commit(result):
            for n in names:
                got = observe.column_values_series(result if single else result.get_column(n))
                entry = self.cached[n]
                want, error = self.oracle(entry)
                assert error is None, f'{n}: reference raised {error} but the read succeeded'
                observe.same_indicator(got, np.asarray(want), f'read {n}', self.tolerance(entry),
                                       self.excused(entry))
                self.stale.discard(n)

        key = names[0] if single else names
        errors = [self.oracle(self.cached[n])[1] for n in names]
        expect = next((e for e in errors if e), None)
        self.attempt(lambda: self.df[key], expect_error=expect, commit=commit, label=f'df[{key!r}]')

    @rule()
    def fulfill(self):
        errors = [self.oracle(e)[1] for e in self.cached.values()]
        expect = next((e for e in errors if e), None)

        def commit(_):
            self.stale.clear()

        self.attempt(self.df.fulfill, expect_error=expect, commit=commit, label='fulfill')

    @rule(data=st.data())
    def exec_read(self, data):
        """``df.exec(d)``: a pure evaluation, fresh even while the frame is stale."""
        d = data.draw(directives(tuple(self.plain_numeric())))
        if not triggers_ok(d):
            return
        if contracts.avoids('unguarded-stale-read') and d.bare_names & set(self.cached):
            return  # a bare name binding a cached column (stale or not) — the open read path
        if self.window is not None:
            return  # exec over a windowed frame sees the retained rows only (contract open)
        entry = Entry(d.text, d, self.env(), self.visible_start())
        want, error = self.oracle(entry, evaluate=True)

        def commit(values):
            observe.same_indicator(np.asarray(values), np.asarray(want), f'exec({d.text!r})',
                                   self.tolerance(entry), self.excused(entry))

        self.attempt(lambda: self.df.exec(d.text), expect_error=error, commit=commit,
                     label=f'exec({d.text!r})')

    @rule(data=st.data())
    def get_column(self, data):
        """``get_column`` never computes: a plain column reads its data at any time; a
        stale computed column raises (contract), a fresh one reads its values."""
        name = data.draw(st.sampled_from(list(self.df.columns)))
        if name in self.cached and name in self.stale:
            if contracts.avoids('unguarded-stale-read') or not contracts.GET_COLUMN_STALE_RAISES:
                return
            self.attempt(lambda: self.df.get_column(name), expect_error=ValueError,
                         label=f'get_column({name!r}) stale')
            return

        def commit(series):
            got = observe.column_values_series(series)
            if name in self.cached:
                entry = self.cached[name]
                observe.same_indicator(got, np.asarray(self.oracle(entry)[0]), f'get_column({name!r})',
                                       self.tolerance(entry), self.excused(entry))
            else:
                self.check_plain(name, got)

        self.attempt(lambda: self.df.get_column(name), commit=commit, label=f'get_column({name!r})')

    @rule(surface=st.sampled_from(sorted(BULK_SURFACES)))
    def bulk_read(self, surface):
        """A non-refreshing read raises while any cached column is stale (README)."""
        ops = BULK_SURFACES
        if surface == 'rolling' and any(t not in ('float64', 'float32', 'int64', 'int32')
                                        for t in self.df.dtypes.values()):
            return  # rolling is numeric-only
        stale = bool(self.stale) and contracts.BULK_READ_STALE_RAISES
        if stale and self.window is not None and contracts.avoids('windowed-stale-logical-read') and any(
                volas.directive_lookback(n) > len(self.df) for n in self.stale):
            return
        self.attempt(lambda: ops[surface](self.df), expect_error=ValueError if stale else None,
                     label=f'{surface}{" (stale)" if stale else ""}')

    @rule()
    def snapshot(self):
        """Keep a copy (an independent frame) and its model for later ops."""
        if len(self.snapshots) >= 3:
            self.snapshots.pop(0)
        self.snapshots.append(self.save(self.df.copy()))
        self.log('copy')

    @precondition(lambda self: self.snapshots)
    @rule(data=st.data())
    def swap(self, data):
        """Continue on a snapshot (several frames / histories in one run)."""
        i = data.draw(st.integers(0, len(self.snapshots) - 1))
        mine = self.save(self.df)
        self.restore(self.snapshots[i])
        self.snapshots[i] = mine
        self.log(f'swap {i}')

    def save(self, df) -> tuple:
        return (df, self.store.copy(), dict(self.cached), set(self.stale), self.offset)

    def restore(self, snap: tuple) -> None:
        self.df, self.store, self.cached, self.stale, self.offset = (
            snap[0], snap[1].copy(), dict(snap[2]), set(snap[3]), snap[4])

    # --- invariants -----------------------------------------------------------------------

    def check_plain(self, name: str, got: np.ndarray) -> None:
        coarse = self.coarse()
        want = observe.expected_values(coarse.values(name, self.visible_start()), coarse.dtype(name))
        observe.same_data(got, want, f'column {name!r}')

    @invariant()
    def consistent(self):
        if self.df is None:
            return
        df = self.df
        try:
            observe.structure(df)
            cols = set(df.columns)
            assert set(self.cached) <= cols, f'cached {set(self.cached) - cols} missing from {cols}'
            for c in df.columns:
                assert df.is_computed(c) == (c in self.cached), \
                    f'is_computed({c!r}) = {df.is_computed(c)}, model says {c in self.cached}'
            stale = observe.is_stale(df)
            if self.window is not None and not stale:
                self.stale.clear()  # a front-drop refreshes before it compacts
            assert stale == bool(self.stale), f'stale = {stale}, model stale set {sorted(self.stale)}'
            self.check_values()
        except AssertionError as e:
            raise AssertionError(f'{e}\n  op trace: {self.trace[-12:]}') from e

    def check_values(self) -> None:
        view = self.df.copy()
        errors = [self.oracle(e)[1] for e in self.cached.values()]
        try:
            view.fulfill()
        except Exception as e:
            assert any(err is not None and isinstance(e, err) for err in errors), \
                f'fulfill on a copy raised {e!r}; no reference raises'
            return
        coarse = self.coarse()
        start = self.visible_start()
        labels = np.asarray(view.index)
        want_labels = np.asarray(coarse.index[start:], dtype='int64')
        got_labels = labels.astype('datetime64[ns]').astype('int64') if labels.dtype.kind == 'M' else labels
        observe.same_data(np.asarray(got_labels, dtype='int64'), want_labels, 'index labels')
        assert view.tz == coarse.schema.tz, f'tz {view.tz!r} != {coarse.schema.tz!r}'
        for c in coarse.columns:
            if c in self.cached:
                continue
            observe.same_data(observe.column_values(view, c),
                              observe.expected_values(coarse.values(c, start), coarse.dtype(c)),
                              f'column {c!r}')
        for name, entry in self.cached.items():
            want, error = self.oracle(entry)
            assert error is None, f'{name}: reference raised {error.__name__}, frame refreshed fine'
            observe.same_indicator(observe.column_values_series(view.get_column(name)), np.asarray(want),
                                   f'cached {name!r}', self.tolerance(entry), self.excused(entry))
