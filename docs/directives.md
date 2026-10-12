# Directives

## Syntax of `directive`

```
command . sub : args @ series  op  command ...
   |      |     |      |
   |      |     |      └── operand column / sub-expression  (e.g. @open, @(boll))
   |      |     └── comma-separated arguments               (e.g. ma:20, kdj.k:9,3)
   |      └── sub-command                                   (e.g. macd.signal)
   └── indicator name                                       (e.g. ma, macd, boll)
```

### `directive` Example

Here lists several use cases of column names

```py
# The middle band of bollinger bands
#   which is actually a 20-period (default) moving average
df['boll']

# kdj j less than 0
# This returns a series of bool type
df['kdj.j < 0']

# kdj %K cross up kdj %D
df['kdj.k // kdj.d']

# 5-period simple moving average
df['ma:5']

# 10-period simple moving average on (@) open prices
df['ma:10@open']

# A DataFrame of 5-period, 10-period and 30-period ma
df[[
    'ma:5',
    'ma:10',
    'ma:30'
]]

# Which means we use the default values of the first and the second parameters,
# and specify the third parameter (for macd.signal)
df['macd.signal:,,10']

# We must wrap a parameter which is a nested command or directive
df['increase:3@(ma:20@close)']

# volas has a powerful directive parser,
# so we could even write directives like this:
df['''
repeat
    :   5
    @   (
            close > boll.upper
        )
''']
```

### Operators

```
left operator right
```

- `//` — whether `left` **crosses up** through `right` (from below to above),
  which we call a "gold cross": `df['macd // macd.signal']`.
- `\\` — whether `left` **crosses down** through `right`, a "dead cross". In a
  Python string the backslash must be escaped, so we write `'macd \\ macd.signal'`.
- `><` — whether `left` crosses `right`, either up or down.
- `<` `<=` `==` `!=` `>=` `>` — for the same record, the value comparison between
  `left` and `right`, returning a `bool` series.
- arithmetic `+ - * /`, logical `& | ^`, and unary `~` (not) / `-` (negate).

**A directive boolean is missing where it cannot be known.** A comparison whose
operand is missing — e.g. the warm-up rows of `close > ma:20` — is `volas.NA`, not
`False`; a cross is missing while either of its two rows is (so its first row is
too, unless the current row already rules the cross out); and `~ & | ^` follow
Kleene logic: a present `False` decides `&`, a present `True` decides `|`, and
anything else involving a missing truth stays missing (`~` of a missing truth is
missing, never `True`). `increase`, `style` and `repeat` follow the same rule. A
cached boolean keeps its missing rows through `append` and refresh, exactly as a
full recompute gives them. To use such a signal as a row mask, state what an
unknown row means first: `df[df['close > ma:20'].fillna(False)]`.

**Numbers stay exact.** An integer literal is an exact int64
(`volume > 9007199254740993` compares against that integer, not its float64
neighbour), and a comparison orders integers exactly — `2**53 + 1 > 2**53`. The
arithmetic operators, unary `-` and every indicator compute in float64, so an
int64 operand past `±2**53` (whose float64 image is not its value) raises
`OverflowError` instead of computing from a rounded input.

`df[directive]` **caches** the result as a real column (so repeated reads are
free), then auto-refreshes its stale tail on access after an `append`. Use
`df.exec(directive)` to compute a directive as a NumPy array **without**
caching it (see [Usage](dataframe-api.md#usage)).

## Error handling

Directive problems raise typed exceptions. Both subclass `DirectiveError` and the
built-in `ValueError`, so existing `except ValueError` handling keeps working.

```py
from volas import DirectiveSyntaxError, DirectiveValueError

try:
    df['ma:2,3']                 # too many arguments
except DirectiveValueError as e:
    ...                          # unknown command/sub-command, bad arg, bad value

try:
    df['a >']                    # malformed expression
except DirectiveSyntaxError as e:
    ...                          # message carries the line / column of the error
```

A directive may nest at most 128 levels (each parenthesised group, `@(...)` series
group, unary operator and binary operator adds one) and hold at most 4096 nodes.
Real directives stay far below both; anything beyond them raises
`DirectiveSyntaxError` rather than exhausting the stack.
