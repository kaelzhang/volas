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
unknown row means first: `df[df['close > ma:20'].fillna(False)]`. The same holds at
the NumPy boundary: a boolean result holding a missing truth has no bare NumPy form
(a `nan` placeholder would read `True` in `np.where` or indexing), so
`df.exec('close > ma:20')`, `np.asarray(df['close > ma:20'])` and `.to_numpy()`
raise until you pass `na_value=` (e.g. `False`), `dtype='float64'` (missing → `NaN`)
or `dtype='object'` (missing → `volas.NA`) — `df.exec(d, na_value=False)`. Numeric
results export as before (`np.asarray(df.exec('ma:20'), dtype=np.float64)`).

**Numbers stay exact.** An integer literal is an exact int64
(`volume > 9007199254740993` compares against that integer, not its float64
neighbour) — written with zero decimals (`9007199254740993.0`) too — and a
comparison orders integers exactly — `2**53 + 1 > 2**53`. A whole-number literal
past int64 that no float64 holds exactly is a `DirectiveSyntaxError`, never a
rounded value. The
arithmetic operators and every indicator compute in float64, so an int64 operand
past `±2**53` (whose float64 image is not its value) raises `OverflowError` instead
of computing from a rounded input. Unary `-` negates an integer exactly (as an int64:
an integer zero has no sign, so `-(0)` is `0` like the literal `-0`), and a float in
float64 (`-0.0` keeps its sign).

`df[directive]` **caches** the result as a real column (so repeated reads are
free), then auto-refreshes its stale tail on access after an `append`. Use
`df.exec(directive)` to compute a directive as a NumPy array **without**
caching it (see [Usage](dataframe-api.md#usage)).

### Refresh cost

A refresh after an `append` reads only a trailing window of the frame — the
directive's lookback twice over — whatever the frame's height: a recursive indicator
continues its carried state over the new row, a running window sum continues its
sums, and any other finite window is recomputed over its last window. An int64 or
float32 input is converted over that window only, and an `@` expression operand
(`ema:14@(close + open)`, `rsi:14@(ma:5)`) is evaluated over it alone. A recursion or
a row position anywhere in a directive — inside an `@` operand (`rsi:14@(ema:5)`,
`ema:3@(maxindex:5)`) or combined with other values (`close - ma:5 + obv`,
`ema:5 > ma:10`) — carries a state of its own and its last few values, so the
directive continues over that window too, and a slice keeps it cached. Each node also
carries the complete input rows its resume reads: an EMA needs none, RSI needs its
previous complete row. Missing rows, even a long missing run, never make a continuation
read or convert the whole column. Single-row chains step their kernels directly using
reused input scratch and scalar expression evaluation; `ema:5@(ema:3)`,
`close-ma:5+obv` and `close>ema:5` allocate nothing on complete inputs. Declared chains
on bounded frames use the same state for checkpoints and replay after writes or head
compaction; warm-up checkpoints keep raw input rows. CRSI, stochastic momentum,
and composite averages (`apo`, `ppo`, `macdext`, `mavp`, `stoch`, `stochf.d`,
`stochrsi`) carry their constituent states at every MA type. Their append cost
is bounded by their configured windows and the new rows. `ma` at a recursive
MA type continues as that average does. `ttm_squeeze` still recomputes its column.

Cached recursions record a checkpoint every 512 rows during computation, including
the states of finite-window children in a recursive expression. A purely finite-window
expression keeps its terminal continuation states without these history checkpoints.
Rust callers can use `volas_directive::exec::execute_for_cache` for this cache policy;
`execute_with_states_and_anchors` explicitly requests checkpoints for either kind of
expression. Both use the same continuation-state representation.


### Names: a data column or a built-in command

A bare name in a directive (`close`, `obv`, `change` — no colon, no arguments) reads a
**data column** of the frame or runs the **built-in command** of that name. It never
reads a cached directive column: a new directive (`df[d]`, `df.exec(d)`) computes from
the data columns, so `df['close - obv']` runs the `obv` command even when `obv` is also
cached (and possibly stale after an `append`).

A data column owns every spelling of its directive-shaped name. If `atr:14` is a
plain data column (for example, after `df[['close', 'atr:14']]`), `ATR:14`,
`atr:14 + 0` and `ema:2@(atr:14)` all read that data; they never recompute ATR.
This also applies to bare command names: with a data column `change`, both
`close * change` and `close * change:` read that column. Rename or drop the data
column to compute the built-in command on that frame.

Each canonical identity has exactly one physical data label. Construction, column
addition and renaming reject equivalent labels such as `ema:2` and `EMA:02` with
an error naming both owners. Equivalent spellings in getters, expressions and
nested operands always resolve to that same label.

A cached directive retains the data/command bindings chosen when it was cached.
Adding a data column under any equivalent spelling of a computation it uses raises:
`close * CHANGE`, `close * ChAnGe` and `close * change:` all prevent adding either
`change` or `CHANGE`. Drop the cached directive first, then add the data column;
every equivalent command spelling will read that data on subsequent binding:

```text
ValueError: cannot add column 'change': the cached directive 'close*change' uses the built-in command 'change' under that name. Rename the new column, or drop the cached directive first — df.drop(['close*change'], axis=1).
```

`df.exec(d)` remains pure and reads only data columns; it never consumes a cached
column's stale values.

### Cached column names

A cached directive's column is named by `directive_stringify` of it — its canonical
name: whitespace removed, minimal parentheses, every argument written as its value
(`ma:05` is `ma:5`, `hv:10,1W` is `hv:10,1w`) and default arguments / series dropped.
The name does not depend on the spelling: `boll`, `boll:`, `BOLL` and `boll:20@close`
all cache as one column `boll`. The cached column keeps the directive it was bound to
when it was cached, and an append continues exactly that directive — never a re-read of
its name. When an operand binds to a data column with a noncanonical physical label,
that exact label is kept in the stored expression name (`ema:2@(EMA:2)`, for example).
Projections and window declarations use this bound name too.

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
