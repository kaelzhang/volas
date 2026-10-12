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

`df[directive]` **caches** the result as a real column (so repeated reads are
free), then auto-refreshes its stale tail on access after an `append`. Use
`df.exec(directive)` to compute a directive as a NumPy array **without**
caching it (see [Usage](dataframe-api.md#usage)).

### Names: a data column or a built-in command

A bare name in a directive (`close`, `obv`, `change` — no colon, no arguments) reads a
**data column** of the frame or runs the **built-in command** of that name. It never
reads a cached directive column: a new directive (`df[d]`, `df.exec(d)`) computes from
the data columns, so `df['close - obv']` runs the `obv` command even when `obv` is also
cached (and possibly stale after an `append`).

A name that is **both** a data column and a built-in command usable bare is ambiguous
and raises when the directive is bound:

```text
DirectiveValueError: ambiguous name 'change' in 'close * change': it is both a column of this DataFrame and the built-in command 'change'. Write 'change:' to use the command, or rename the column — df.rename(columns={'change': 'change_pct'}) — to use the data.
```

`close * change:` runs the command; renaming the column reads the data. For the same
reason, once a cached directive reads a command through its bare name, a data column of
that name cannot be added (it would change what the cached directive refreshes):

```text
ValueError: cannot add column 'change': the cached directive 'close*change' uses the built-in command 'change' under that name. Rename the new column, or drop the cached directive first — df.drop(['close*change'], axis=1).
```

A plain column read, `df['change']`, is not a directive and returns the column. The
command form of a name a data column holds, `df['change:']`, returns the command's
values without caching them (the column keeps the name).

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
