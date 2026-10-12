"""Grammar-driven directive generation over the derived inventory.

A generated directive is a :class:`Directive`: its text plus what the harness needs
to judge it — the value kind (``num`` / ``bool``), the operand texts of a top-level
comparison (to excuse an exact tie under a documented ULP exception), and the bare
names it reads (which may bind to a frame column of the same name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hypothesis import strategies as st

from . import contracts, inventory

ARITHMETIC = ('+', '-', '*', '/')
LOGICAL = ('&', '|', '^')


@dataclass(frozen=True)
class Directive:
    text: str
    kind: str  # 'num' | 'bool'
    operands: tuple[str, str] | None = None  # (lhs, rhs) of a top-level comparison
    bare_names: frozenset[str] = frozenset()
    commands: int = 0  # how many command nodes (0 = pure column arithmetic)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.text


def _operators() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    ops = inventory.binary_operators()
    arith = tuple(o for o in ops if o in ARITHMETIC)
    logical = tuple(o for o in ops if o in LOGICAL)
    compare = tuple(o for o in ops if o not in ARITHMETIC and o not in LOGICAL)
    return arith, logical, compare


@st.composite
def _arg_token(draw, domain: inventory.ArgDomain) -> str:
    if domain.kind == 'int':
        lo, hi = int(domain.lo), int(domain.hi)
        # Mostly short periods (runs longer than the lookback are the common case),
        # sometimes one longer than any generated history, rarely a huge one.
        value = draw(st.one_of(
            st.integers(lo, max(lo, min(hi, 12))),
            st.integers(lo, hi),
            st.sampled_from([max(lo, min(hi, v)) for v in (lo, lo + 1, 60, 250)]),
        ))
        return str(value)
    if domain.kind == 'choice':
        return str(draw(st.sampled_from(domain.choices)))
    if domain.kind == 'str':
        return draw(st.sampled_from(domain.choices))
    value = draw(st.floats(domain.lo, domain.hi, allow_nan=False, allow_infinity=False))
    return repr(round(value, 3)) if domain.lo <= round(value, 3) <= domain.hi else repr(domain.lo)


_BARE = re.compile(r'(?<![\w.])([A-Za-z_][\w.]*)(?![\w.:@])')


def stored_bare_names(name: str) -> set[str]:
    """The bare names (as spelled) of a cached column's name — the directive text a
    refresh re-reads: every identifier that is not a command with arguments or a series
    (``ma:5``, ``accbands@open``), including the ``@`` operands."""
    return set(_BARE.findall(name))


@st.composite
def command(draw, columns: tuple[str, ...], depth: int = 0) -> Directive:
    """One command line with random in-domain arguments, empty default slots, an
    alias / case spelling, and an optional first-series override (nested at depth 0)."""
    line = draw(st.sampled_from(inventory.executable_lines()))
    head = draw(st.sampled_from((line.head,) + line.aliases))
    if draw(st.integers(0, 9)) == 0:
        head = head.upper()
    required = max((i for i, a in enumerate(line.args) if a.required), default=-1)
    n = draw(st.integers(required + 1, len(line.args))) if line.args else 0
    tokens = []
    for i in range(n):
        if not line.args[i].required and draw(st.integers(0, 3)) == 0:
            tokens.append('')  # keep the default
        else:
            tokens.append(draw(_arg_token(line.args[i])))
    text = head + (':' + ','.join(tokens) if tokens else '')
    bare: frozenset[str] = frozenset()
    nested = 0
    series = line.series_suffix[1:].split(',') if line.series_suffix else []
    if draw(st.integers(0, 4)) == 0:
        if depth == 0 and draw(st.booleans()):
            inner = draw(expression(columns, depth + 1, want='num'))
            first, bare, nested = f'({inner.text})', inner.bare_names, inner.commands
        else:
            first = draw(st.sampled_from(columns))
        series = [first] + series[1:]
    if series:
        text += '@' + ','.join(series)
    if not tokens and not series and line.head == head.lower():
        bare = bare | {head}  # a bare command name, as spelled: may bind to a same-named column
    return Directive(text, line.kind, bare_names=bare, commands=1 + nested)


@st.composite
def _literal(draw) -> Directive:
    edges = [0, 1, 0.5, 2**53 + 1]
    if not contracts.avoids('literal-saturation'):
        edges.append(100000000000000000000)  # beyond i64
    value = draw(st.one_of(
        st.integers(0, 200),
        st.floats(0, 500, allow_nan=False).map(lambda f: round(f, 3)),
        st.sampled_from(edges),
    ))
    return Directive(str(value), 'num')


@st.composite
def expression(draw, columns: tuple[str, ...], depth: int = 0, want: str | None = None) -> Directive:
    """A typed expression: commands, columns and literals combined by the parser's
    own operator table. ``want`` asks for a value kind; ``None`` lets it vary."""
    arith, logical, compare = _operators()
    kind = want or draw(st.sampled_from(['num', 'num', 'bool']))
    budget = draw(st.integers(0, 2)) if depth < 2 else 0
    if kind == 'num':
        if budget == 0:
            leaf = draw(st.integers(0, 5))
            if leaf <= 2:
                return draw(command(columns, depth))
            if leaf <= 4:
                name = draw(st.sampled_from(columns))
                return Directive(name, 'num', bare_names=frozenset({name}))
            return draw(_literal())
        if budget == 1 and draw(st.booleans()):
            inner = draw(expression(columns, depth + 1, 'num'))
            return Directive(f'-({inner.text})', 'num', bare_names=inner.bare_names,
                             commands=inner.commands)
        lhs = draw(expression(columns, depth + 1, 'num'))
        rhs = draw(expression(columns, depth + 1, 'num'))
        op = draw(st.sampled_from(arith))
        return Directive(f'({lhs.text}){op}({rhs.text})', 'num',
                         bare_names=lhs.bare_names | rhs.bare_names,
                         commands=lhs.commands + rhs.commands)
    if budget >= 1 and draw(st.booleans()):
        lhs = draw(expression(columns, depth + 1, 'bool'))
        if draw(st.integers(0, 3)) == 0:
            return Directive(f'~({lhs.text})', 'bool', bare_names=lhs.bare_names,
                             commands=lhs.commands)
        rhs = draw(expression(columns, depth + 1, 'bool'))
        op = draw(st.sampled_from(logical))
        return Directive(f'({lhs.text}){op}({rhs.text})', 'bool',
                         bare_names=lhs.bare_names | rhs.bare_names,
                         commands=lhs.commands + rhs.commands)
    lhs = draw(expression(columns, depth + 1, 'num'))
    rhs = draw(expression(columns, depth + 1, 'num'))
    op = draw(st.sampled_from(compare))
    return Directive(f'({lhs.text}){op}({rhs.text})', 'bool', operands=(lhs.text, rhs.text),
                     bare_names=lhs.bare_names | rhs.bare_names,
                     commands=lhs.commands + rhs.commands)


def directives(columns: tuple[str, ...]) -> st.SearchStrategy[Directive]:
    """Directives worth caching: at least one command (pure column arithmetic is
    cached too, but a command is where incremental state lives)."""
    return st.one_of(
        command(columns),
        command(columns),
        expression(columns),
    ).filter(lambda d: d.commands > 0)
