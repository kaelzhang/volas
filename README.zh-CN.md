[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

# [volas](https://github.com/kaelzhang/volas)

[English](README.md) | 简体中文

> 面向股票 / K 线（OHLCV）时间序列数据的高性能 Rust 列式内核。

**volas** 是一个 Rust 驱动、pandas 风格的 `DataFrame`，专为实时 OHLCV 流水线打造：内置 [**254** 个交易指标](INDICATORS.md)，支持增量 O(lookback) 刷新，输出可直接交给 NumPy / Torch 使用。

它**不是**通用 pandas 替代品，而是一个窄而快、专服务于 K 线 / OHLCV 工作流的 DataFrame：append 一根新 bar，指标列保持缓存，只刷新受影响的尾部。

**volas** 同时也是一个 Rust [crate](crates/volas/README.zh-CN.md)。

```python
from volas import read_csv

df = read_csv("btc_1m.csv")

# 把指标 directive 作为 DataFrame 列缓存。
df["rsi:14"]
df[["macd", "macd.signal", "atr:14"]]

# 在实时循环中：
df.append(new_bar)     # 一行 OHLCV frame
df["rsi:14"]           # 只刷新受影响的尾部，O(lookback)
features = df.to_numpy()
```

- 内置 **254** 个指标，directive 与 TA-Lib 对齐
- `append` 后增量刷新：**O(lookback)**，不是 O(n)
- Rust 内核，运行时无 pandas 依赖
- pandas 风格索引：`.loc` / `.iloc` / `.at` / `read_csv` / `to_numpy`
- 输出可直接交给 NumPy / Torch

```sh
pip install volas
```

在可复现的 benchmark 套件中，**volas** 在大多数实时更新指标场景下都快于 pandas、polars、stock-pandas 和 TA-Lib。

## 为什么选 volas

- **pandas 风格 API** `.loc` / `.iloc` / `.at`、`read_csv`、
  `to_numpy` 和重采样都按熟悉的方式使用——面向 OHLCV 工作流，换一个 import
  代码基本不动。它**不是**通用 pandas 替代品。（参见
  [哪些不在覆盖范围内](PANDAS-DIFFERENCES.md#index-limitations)）
- **在实时 OHLCV 指标计算上很快**，且 benchmark 可复现——具体结果以持续更新的[实时 benchmark 报告](https://volas.ost.ai)为准。
  - 按当前已发布报告的默认口径，在 **139 / 157** 个覆盖指标上胜过 TA-Lib
    ——可通过 `make benchmark` 复现。
  - 在增量更新（每来一根新 bar）计算中，volas 在**所有**指标上都是**所有**类库中最快的——比 TA-Lib 快 **~5×**，比 pandas 最高快约 **~360×**。
- **为实时 tick 而生。**新 bar 只触碰受影响的尾部（`O(lookback)`，不是 `O(n)`）；指标以微秒级刷新，不做整列重算。
- **Rust 在内，NumPy / Torch 在外。**编译型内核，热路径经过**汇编指令级**的性能优化，运行时零 pandas 依赖；`to_numpy()` 直接喂给 NumPy 和 `torch.Tensor` 流水线。

![volas 如何在 append 后只刷新 stale tail](https://volas.ost.ai/animated_gif/after-append-indicator-zh-cn.gif)

### 我为什么要做 volas

这些年我一直在做量化交易信号系统的研究，而很长一段时间里，`pandas` 只是我为了把活儿干完不得不交的「税」。光是读一个几十万根 K 线的 CSV，算上几个指标、清洗、再做归一化——单单这一道数据处理，往往就要 10 到 20 分钟，真正的研究还没开始就先耗掉了。

回测让情况更糟。为了诚实地模拟实盘，你得把历史数据一根一根地喂进去——通常是更细的周期，比如 1 分钟 K 线——每来一根就 append 到 `DataFrame` 后面，再把所有指标在整张表上从头重算一遍，如此反复，去逼近实盘里真实流入的 OHLCV。这里头大量都是纯粹的浪费——同样的列在每一步被推倒重算——拿几年的 1 分钟数据跑下来，单这一块重复劳动就能把一次回测拖上**好几个小时**。我想试的每一个想法、想扫的每一组参数，都要再交一遍这份税——决定研究节奏的，成了工具，而不是思路。

于是我不再缝缝补补，而是把整个底层数据系统从头重写，彻底抛弃 `pandas`。这一注押对了：当年要 10–20 分钟的那一道数据处理，如今几秒钟就跑完；曾经把回测拖得没完没了的逐根重算，如今几乎可以忽略不计。回测本身仍有实打实的活要干——策略逻辑、撮合成交、账目结算——但数据层已经不再是我枯坐干等的那一环。**volas** 就是这套数据层：在最要紧的地方快了数百倍，终于快到可以让工具彻底退到一边。

### 什么时候该用 volas

volas **不是**通用 pandas 替代品。一般的 DataFrame 分析请继续用 pandas 或
polars。它是一个窄而快的 DataFrame，专门服务于这一类场景：
**新的 OHLCV bar 到来后，指标必须立刻刷新**：

| | pandas | polars | TA-Lib | volas |
| --- | :---: | :---: | :---: | :---: |
| pandas 风格索引（`.loc` / `.iloc` / `.at`） | ✅ | ❌ | ❌ | ✅ |
| OHLCV 原生的指标 directive（`df['rsi:14']`） | ❌ | ❌ | ✅ | ✅ |
| 由 frame 自身维护的指标缓存 | ❌ | ❌ | ❌ | ✅ |
| 新 bar 上的增量 `O(lookback)` 刷新 | ❌ | ❌ | ❌ | ✅ |
| Rust 内核、运行时无 pandas | ❌ | ✅ | C | ✅ |
| 导出到 NumPy / Torch | ✅ | ✅ | 数组 | ✅ |

## 目录
- [安装](#安装)
- [快速上手](#快速上手)
- [用法](#用法)
- [有界滚动窗口](#有界滚动窗口)
- [累积与 DatetimeIndex](#累积与-datetimeindex)
- [TimeFrame](#timeframe)
- [directive 的语法](#directive-的语法)
- [索引与选择](#索引与选择)
- [写入与赋值](#写入与赋值)
- [时区](#时区)
- [缺失值（`volas.NA`）](#缺失值volasna)
- [与 pandas 互操作](#与-pandas-互操作)
- [与 Arrow、DLPack 互操作（零拷贝）](#与-arrowdlpack-互操作零拷贝)
- [错误处理](#错误处理)
- [内置指标](#内置指标)
- [许可证](#许可证)
- [面向开发者](#面向开发者)

## 安装

```sh
pip install volas
```

要求 Python >= 3.11。Linux（x86_64 / aarch64）、macOS（x86_64 / arm64）和 Windows（x86_64）均提供预编译 wheel。如需从源码本地构建，请参见
[面向开发者](#面向开发者)。

30 秒内验证安装，然后查看 [`examples/`](examples/)——每个脚本都自包含、运行成功会打印一行 `OK:`：

```sh
pip install volas
python examples/00_install_check.py
python examples/03_live_ohlcv_append.py   # append 一根 bar，只刷新受影响的尾部
```

## 快速上手

```py
from volas import DataFrame

df = DataFrame({
    'open':   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    'high':   [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    'low':    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'close':  [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'volume': [100, 200, 300, 400, 500, 600],
})

# 普通的一列 -> Series
df['close']
# 0    3.0
# 1    4.0
# 2    5.0
# 3    6.0
# 4    7.0
# 5    8.0
# Name: close, dtype: float64

# 一个指标 directive -> Series（`close` 的 2 周期 SMA）
df['ma:2']
# 0   <NA>
# 1    3.5
# 2    4.5
# 3    5.5
# 4    6.5
# 5    7.5
# Name: ma:2, dtype: float64

# 一个布尔 directive -> bool Series，可直接用作行掩码
bullish = df['close > open']
df[bullish]                      # close > open 的那些行组成的 DataFrame

# 一次性多个 directive -> DataFrame
df[['ma:2', 'ma:3', 'close > open']]

# 导出到 NumPy（以及零拷贝地导出到 Arrow / DLPack——见互操作小节）
df['close'].to_numpy()           # 1 维 ndarray
df.to_numpy()                    # 2 维 ndarray（行 x 列）
```

## 用法

```py
from volas import (
    DataFrame, Series, read_csv, to_datetime, TimeFrame, Timestamp,
)
```

下面按 volas 的公共接口顺序说明：先介绍 `DataFrame` 类，再介绍它的实例方法、静态方法、其他类和顶层包函数；最后列出其余**与 pandas 兼容**、行为相同的 API。（从 `volas` 导入的顶层名字，例如 `read_csv`，书写时不带 `volas.` 前缀。）

### DataFrame(data, columns=None, time_frame=None, cumulators=None)

`DataFrame` 提供**与 pandas 兼容的 API**。如果你熟悉 `pandas.DataFrame`，上手 volas 基本不需要重新学习。不同之处在于：volas 由 Rust 内核驱动，运行时不依赖
pandas。

```py
df = read_csv('stock.csv')
```

可以用 `[]`，也就是 **pandas 式索引**（Python 中的 `__getitem__`），取出更低维的切片。除了用 `colname`（`DataFrame` 的列名）索引，也可以用 `directive` 索引。

```py
df[directive]                  # 返回一个 Series

df[[directive0, directive1]]   # 返回一个 DataFrame
```

下面是 `[directive]` 最基本的用法：

```py
df = DataFrame({
    'open' : ...,
    'high' : ...,
    'low'  : ...,
    'close': [5, 6, 7, 8, 9]
})

df['ma:2']

# 0   <NA>
# 1    5.5
# 2    6.5
# 3    7.5
# 4    8.5
# Name: ma:2, dtype: float64
```

得到的是 `"close"` 列上的 2 周期简单移动平均。

#### 参数

- **data** `dict[str, list | np.ndarray] | DataFrame` 列数据，为以下之一：
  - 一个 **dict**，把每个列名映射到等长的 list 或 NumPy 数组（`float`、`int`、
    `bool`、`datetime64` 或 `string`）；
  - 另一个 volas **`DataFrame`**，此时会被拷贝（如同 `pandas.DataFrame(df)`）。

  构造函数**不接受** `pandas.DataFrame` 或 Arrow 对象——请改用专门的
  [`DataFrame.from_pandas`](#dataframefrom_pandaspdf---dataframe) /
  [`DataFrame.from_arrow`](#dataframefrom_arrowdata---dataframe) 桥接。如果要附加
  [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)，用 `to_datetime` 解析某一列，用 `set_index` 把它提升为索引，再用 `tz_localize` /
  `tz_convert` 打上时区标记。参见[时区](#时区)。
- **columns** `Optional[list[str]] = None` 选择并排列要保留的列，等价于
  `df[[...]]` 的投影。名字不存在会抛 `KeyError`；空 list 或重复名字会被拒绝，缺失列绝不会被静默填充。
- **time_frame** `Optional[str | TimeFrame] = None` 若设置，则把它变成一个在该
  bar 间隔上**支持 tf 累积**的 DataFrame：给定的各行会被视作该 frame 下已经定型的 bar，后续的 `append` 会把更细的 bar 折叠进正在形成中的 bar。需要一个
  `DatetimeIndex`。参见[累积与 DatetimeIndex](#累积与-datetimeindex)。
- **cumulators** `Optional[dict[str, str]] = None` 折叠时使用的逐列聚合器覆盖（例如 `{'amount': 'sum'}`），仅在与 `time_frame` 同时使用时才有意义。默认采用
  OHLCV 语义（`open`=first、`high`=max、`low`=min、`close`=last、`volume`=sum；其他列默认 `last`）。字典的每个**值**为以下之一：
  - `'first'`——桶内第一个值
  - `'last'`——桶内最后一个值
  - `'max'`——最大值
  - `'min'`——最小值
  - `'sum'`——求和
- **window** `Optional[int] = None` 把它变成一个**有界滚动窗口** frame，只暴露最后 `window` 行（参见[有界滚动窗口](#有界滚动窗口)）。需要配合 `max_lookback`。
- **max_lookback** `Optional[int | list[str]] = None` **与 `window` 同时必填**（且只在有 `window` 时有效）：隐藏历史的 margin（`window + max_lookback`），使缓存指标在自动丢弃旧行前后保持正确——递归类指标（EMA/Wilder/ATR/RSI/MACD）**逐位一致（bit-exact）**，有限窗口类（ma/wma/trima/stddev/…）与无界 frame 在浮点容差内一致（~1e-13）。传一个 **int** 直接声明你会用到的最大指标 lookback，或传一个**指标 directive 列表**从它们中最大的 lookback 推导（例如 `['atr:14', 'ma:50']` → margin 49），省去手算复合指标的预热长度。每个列表项必须是指标 directive（`'ma:50'`），裸名 / 笔误（`'ma50'`）会被拒绝。margin 给得过小会**静默破坏**该保证。

### 有界滚动窗口

传入 `window=` 可把 frame 限制为最后 `window` 行。这正是实盘 / 神经网络输入所需的形态：你可以永远地 `append` bar，但内存始终**有界**——frame 在累积到足够行数后会透明地丢弃旧行，同时保留一段隐藏的 `max_lookback` 行 margin，使缓存指标在每次丢弃前后保持**一致**——递归类指标（EMA/ATR/RSI/…）与无界 frame 逐位一致，有限窗口类（ma/wma/…）在浮点容差内一致（~1e-13）。

```py
# 一个有界的 30 行窗口；margin 大小由你声明的指标推导而来。
wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])

for bar in feed:            # 可以永远运行；内存不增长
    wf.append(bar)          # 把一根 1m bar 折叠进正在形成中的 15m bar
    wf.fulfill()            # 刷新缓存的 atr:14 尾部（O(lookback)）
    if wf.ready:            # 已预热：30 行都有完整的历史
        x = wf[['close', 'atr:14']].to_numpy('float32')   # 30×2 的特征窗口
```

所有面向行的接口——`len`、`shape`、`index`、索引（`[]` / `.iloc` / `.loc` / `.iat` / `.at`）、`head` / `tail`、聚合、`to_numpy`、`to_csv`、`to_pandas`、`repr`——都**只**显示这 `window` 行；margin 永远不可见。

#### `ready`——窗口是否已预热？

`ready`（一个 property）在 frame 累积到 `window + max_lookback` 行后变为 `True`——此时每一个可见的 `window` 行背后都有完整的指标历史，缓存指标端到端有效。预热阶段它为 `False`、可见行仍在填充；请用它来 gate 你的推理 / 训练。

```py
wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
wf.ready          # False——目前还不足 30 + 14 行
...               # append 直到预热完成
wf.ready          # True ——每个可见行都已有有效的指标历史
```

**无界** frame（没传 `window=`）没有预热契约，**恒为 `True`**。

#### `fill_into(out, columns=None)`——零分配特征导出

`fill_into` 把窗口的值**就地**写进**你自己**持有的 NumPy 数组，每次调用零分配。它是实盘推理循环的「导出」一半：配上一块预分配 buffer，`append → fulfill → fill_into → 推理` 的循环每根 bar 零分配（不同于 `to_numpy` 每次都新建一整块矩阵）。

契约：

- **只导出已缓存的列，且按其规范化（canonical）的 directive 名匹配**。directive 列必须**先 materialize**——访问它一次并把列名读回来（`name = wf['atr:14'].name`），因为缓存的 directive 是以规范化形式存储的（例如 `'MA: 5'` → `'ma:5'`），**不一定等于你传入的字符串**，而 `columns=` 按存储的确切列名匹配。（`max_lookback=['atr:14']` 只是给 margin 定大小、并不会创建该列，且 `fill_into` 导出已缓存的值、从不现场计算。）
- **`out`** 必须是 `float32` **或** `float64` 二维数组，形状**严格等于** `(len(df), k)`，其中 `k` 是导出的列数（尊重 strides，非连续或 Fortran-order 视图也可）。形状或 dtype 不符会报错。
- **`columns`** 按顺序选择要导出的列（默认全部列）。**字符串列没有浮点含义、会被拒绝**——在 `columns=` 里只列出数值列以排除它。
- **缺失 / NA 单元格写为 `NaN`**。
- `append` 之后缓存指标是脏的，所以**先 `fulfill()` 再 `fill_into()`**——带着未刷新的 directive 列导出会报错。

```py
import numpy as np

wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
atr = wf['atr:14'].name               # 先 materialize 这个 directive 一次，并把列名读回来：缓存的
                                      # directive 以规范化形式存储（这里是 'atr:14'，但例如 'MA: 5'
                                      # → 'ma:5'），不一定等于你传入的字符串；max_lookback 也只是
                                      # 给 margin 定大小，并不会创建该列
COLS = ['open', 'high', 'low', 'close', atr]    # k = 5 个特征

# 整个运行期复用同一块 buffer——形状 (window, k)，即模型的输入张量。
buf = np.empty((30, len(COLS)), dtype=np.float32)

for bar in feed:
    wf.append(bar)
    wf.fulfill()                      # 导出前刷新 atr 的脏尾部
    if not wf.ready:
        continue                      # 预热中：buf 还不是 (30, k)，指标也未生效
    wf.fill_into(buf, columns=COLS)   # 把 30×5 窗口零分配写进 buf
    prediction = model(buf)           # 每根 bar 都喂给模型同一块内存
```

> **形状提示。**形状是按 `len(df)` 严格匹配的，而 `len(df)` 在**预热期会增长**（1、2、……直到 `window`），之后才稳定在 `window`。像上面那样用 `wf.ready` gate 即可绕开：当它为 `True` 时 `len(df) == window`，固定的 `(window, k)` buffer 必然适配。若要在预热期导出，就把 `out` 调成当前的 `len(df)`。

`fill_into` 并不限于 windowed frame——在普通 frame 上它同样导出整个逻辑视图；只是有界窗口下「复用一块 buffer」最有价值。

### DataFrame.from_pandas(pdf) -> DataFrame

**volas 独有的静态方法**——从 `pandas.DataFrame`（`pdf`）构造 volas `DataFrame`，是 `df.to_pandas()` 的逆操作。pandas 仅在此处惰性 import（所以 `import volas` 保持 pandas-free）。nullable 列保留其 dtype + `volas.NA`，`DatetimeIndex`（含带时区）可往返。参见[与 pandas 互操作](#与-pandas-互操作)。

```py
df = DataFrame.from_pandas(pandas_df)   # pandas.DataFrame -> volas DataFrame
```

### DataFrame.from_arrow(data) -> DataFrame

**volas 独有的静态方法**——从任何实现 Arrow **C-Stream** 协议（`__arrow_c_stream__`）的对象构造 `DataFrame`：`pyarrow.Table` / `RecordBatch` / `RecordBatchReader`、polars `DataFrame` 等。dtype 匹配处借用数据缓冲区（否则拷贝该列），多 chunk 来源会被拼接，结果带一个全新的 `RangeIndex`。

- **data** Arrow 来源——任何实现 `__arrow_c_stream__` 的对象。

```py
df = DataFrame.from_arrow(pa_table)        # pyarrow.Table     -> DataFrame
df = DataFrame.from_arrow(polars_df)       # polars.DataFrame  -> DataFrame
```

> `DataFrame(data=...)` **构造函数**不接受 Arrow（它只接受 `dict` 或另一个 `DataFrame`）；从 Arrow 对象构造请用 `from_arrow`。

### df.exec(directive: str) -> np.ndarray

求值给定的 directive，并返回其值的 NumPy ndarray。这是一次纯粹的**无状态**求值——绝不修改 frame（不创建列，也不读写任何缓存）。

```py
# 只计算，不触碰 frame
df.exec('ma:20')
```

`df[directive]` 与 `df.exec(directive)` 的区别在于
- `df[directive]` 会把结果创建为列并缓存（append 之后增量刷新），而 `df.exec(directive)` 每次都重新计算且不改动 frame；需要缓存后的 ndarray 时，用 `df[directive].to_numpy()`
- `df[directive]` 还能接受其他索引目标（列名、列表、布尔掩码、切片），而 `df.exec(directive)` 只接受合法的 **volas** directive 字符串
- `df[directive]` 返回 `Series` 或 `DataFrame` 对象，而 `df.exec(directive)` 返回 [`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

### df.get_column(key: str) -> Series

按 `key` 直接取列，返回一个 `Series`，并且**绝不触发计算**。`df[key]` 会把未知
key 当作指标 directive 解析并执行；`get_column` 只读取已经存在的列，不存在就抛
`KeyError`。当列名来自外部数据（CSV 表头、用户输入、配置）时，请优先使用它，避免像 `"ma:5"` 这样恰好长得像 directive 的列名悄悄触发计算。

```py
df = DataFrame({
    'open' : ...,
    'high' : ...,
    'low'  : ...,
    'close': [5, 6, 7, 8, 9]
})

df.get_column('close')
# 0    5
# 1    6
# 2    7
# 3    8
# 4    9
# Name: close, dtype: int64
```

### df.append(other: DataFrame | Row | dict) -> DataFrame

把 `other` 的行就地追加到调用者末尾，返回同一个 `DataFrame`，并尽可能把 `DatetimeIndex`
应用到新追加的行上。如果原 `DataFrame` 必须保持不变，请先 `copy()`。

`other` 可以是 `DataFrame`、`Row`，或一个**标量 bar dict**——把一根 bar 写成
`{列名: 值}`，并把该 bar 的时间戳放在 key == 索引名的位置（`RangeIndex` 会自增）。
dict 形态是更快的实时路径——它把 bar 直接组进帧，不构造每根 bar 的 1 行 `DataFrame`：

```python
df.append({'time_key': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
```

它是**严格**的：每一个数据列都必须提供（缺列会报错——这点与 `DataFrame` / `Row` 不同，后者会把缺失列 NA 补齐），出现未知 key 也报错。缓存的指标列无需提供——它们会自动补齐并刷新。

如果调用者是一个**带 tf**的 DataFrame（用 `time_frame` 构建的，或者 `cumulate`
的结果），`append` 会把每一根更细的 bar **折叠**进正在形成中的 bar，而不是新增一行——参见
[实时累积](#实时累积--一个-tf-aware-dataframe)。

`append` 是惰性的：它不会重算新行上的指标列。这些列会保持过期（stale），直到对指标**列**的读取刷新它们，或调用 `df.fulfill()`（见下文）。

### df.cumulate(time_frame: TimeFrame | str, cumulators: dict | None = None) -> DataFrame

把 `DataFrame` 累积（重采样）到更粗的 `time_frame`，返回一个新的 `DataFrame`。需要一个 `DatetimeIndex`。

- **time_frame** `TimeFrame | str` 目标 bar 间隔，例如 `TimeFrame.m5` 或 `'5m'`。参见 [TimeFrame](#timeframe)。
- **cumulators?** `dict[str, str] | None = None` 逐列聚合器覆盖（例如
  `{'amount': 'sum'}`）。默认采用 OHLCV 语义（`open`=first、`high`=max、`low`=min、
  `close`=last、`volume`=sum；其他列默认 `last`）。字典的每个**值**为以下之一：
  - `'first'`——桶内第一个值
  - `'last'`——桶内最后一个值
  - `'max'`——最大值
  - `'min'`——最小值
  - `'sum'`——求和

```py
# 从 1 分钟 K 线到 5 分钟 K 线
five_minute = one_minute.cumulate('5m')
fifteen_minute = one_minute.cumulate('15m')

five_minute.append(new_candle_1m)
# 5 分钟 DataFrame append 一分钟蜡烛之后，会自动聚合为 5 分钟

fifteen_minute.append(new_candle_1m)
# 这样我们可以很方便地用 1 分钟数据，来生成 5 分钟和 15 分钟的测试数据集
```

详见[累积与 DatetimeIndex](#累积与-datetimeindex)。

### df.fulfill() -> None

就地批量刷新每个已缓存指标列过期的尾部（每列 `O(lookback + 新行数)`，不是
`O(n)` 整列重算），返回 `None`。

由于 `append` 是惰性的，缓存有两种方式变回最新：

- **读取指标列**——`df['ma:20']` 或 `df[['ma:20', 'rsi:14']]`——在访问时只自动刷新这些列过期的尾部，所以列读取总是最新且廉价。单列与多列形式行为完全一致。
- **其他所有读取**——`to_numpy()`、`.iloc` / `.loc` / `.at`、归约（`sum` / `mean` / `max` / `describe` / …）、`to_csv`、`repr` 等——**不会**自动刷新；当 frame 处于过期状态时它会**报错**，提示你先调用 `fulfill()`。这是刻意为之：半更新的 frame 失败即报错，而不是静默返回过期值；而且重算（有界）的成本时机由你掌控——这对延迟敏感的实时路径很重要。

```py
df['ma:20']              # 缓存并读取 20 周期 SMA（最新）
df.append(new_bar)       # 惰性：新行的 ma:20 现在过期
df['ma:20']              # 列读取只自动刷新尾部（又变最新）

df.append(new_bar)       # 再次过期
df.fulfill()             # 批量刷新每个已缓存列的尾部
df.to_numpy()            # 现在已刷新（过期时批量读取会直接报错）
```

### df.is_computed(name: str) -> bool

判断列 `name` 是不是 **directive（计算）列**——即由 directive 派生、由 `fulfill` 刷新的列（例如 `df['rsi:14']`），而不是你每根 bar 自己提供的原始数据列。`name` 不是列时抛 `KeyError`。

用它来区分这两类列，例如只导出原始列：

```py
df['ma:5']                                              # 物化一个 directive 列
df.is_computed('ma:5')                                  # True
df.is_computed('close')                                 # False
raw = [c for c in df.columns if not df.is_computed(c)]  # ['open', 'high', 'low', 'close', 'volume']
```

### df.to_numpy(dtype=None, na_value=...) -> np.ndarray

把 DataFrame 导出为二维 NumPy 数组（行 × 列）。行为与 pandas 一致，但有一处刻意的保护：对含缺失值的 frame 指定**整型** `dtype` 会**抛错**，而不是悄悄写入垃圾值（NumPy 无法在整型数组里存 `NA`）——传 `na_value` 可改为填充。

- **dtype** `str | None`——可选的导出转换。`None`（默认）给出按 dtype 的诚实表示；否则：
  - `'object'`（或 `'O'`）——无损的二维「带类型单元格」数组（数字 / `str` /
    `Timestamp` / `volas.NA`）；唯一能保留 `str` 或 datetime 列的 dtype。
  - `'int64'`、`'int32'`、`'int16'`、`'int8'`（以及无符号 `'uint*'`）——**精确**的
    `i64` 通道（大整数与 datetime 的 epoch-ns 不经 `float` 往返而存活）。对含缺失值的
    frame，除非给了 `na_value`，否则**抛错**。
  - `'float64'`、`'float32'`、`'float16'`——（有损的）`float` 通道：缺失单元格为
    `NaN`，超过 2⁵³ ns 的 datetime 会量化。
  - `'bool'`——布尔。
  - `'datetime64[ns]'`——datetime 纳秒；`NaT` 单元格被保留。
  - `str` 列拒绝一切数值/时间 dtype——请用 `'object'`。
- **na_value** `Any`——替换每个缺失单元格的值；默认是 NA 模型表示（`NaN` / `NaT` /
  `volas.NA`）。

```py
df = DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})

df.to_numpy()                 # -> float64 二维数组
df.to_numpy(dtype='int64')    # 精确 int64 转换（稠密 frame）
df.to_numpy(dtype='object')   # 带类型的单元格——无损（数字 / str / Timestamp / volas.NA）

# 整型 dtype 遇缺失值会抛错——除非 na_value 填充
DataFrame({'a': [1, None]}).to_numpy(dtype='int64')               # ValueError
DataFrame({'a': [1, None]}).to_numpy(dtype='int64', na_value=0)   # -> [[1], [0]]  (int64)
```

注意：

- 默认（不传 `dtype`）是诚实表示：全数值/bool 的 frame 是 `float64` 矩阵（缺失 →
  `NaN`），含 `str` 或混合 dtype 的 frame 是带类型单元格的 `object` 矩阵，datetime
  frame 是 `datetime64[ns]`。
- `dtype='object'` 始终无损——每个单元格保留各自的类型值（数字、`str`、`Timestamp`
  或 `volas.NA`）。
- `str` 列没有数值含义，因此任何数值/时间 dtype 都会抛错——用 `dtype='object'` 保留字符串。
- datetime 列**豁免**整型缺失值的抛错规则：在 `dtype='int64'` 下，`NaT` 导出为其精确的
  epoch-ns 哨兵值（datetime 永不经由 `float` 往返）；给了 `na_value` 时它覆盖该哨兵值。
- `to_numpy()` 是一次批量读取；它不会自动刷新陈旧的指标列，若有陈旧列会抛错——先调用
  [`df.fulfill()`](#dffulfill---none)。

需要零拷贝交给 Arrow / DLPack 消费方时，参见
[与 Arrow、DLPack 互操作](#与-arrowdlpack-互操作零拷贝)。

### df.to_arrow() -> pyarrow.Table

**volas 独有**——把 DataFrame 导出为 `pyarrow.Table`，在 dtype 匹配处**零拷贝**：数值 / 字符串 / datetime 列缓冲区与 Arrow 共享，`bool` 与空值位图按需重打包。需要
`pyarrow`（仅在此处惰性 import）。它是 volas Arrow **C-Stream** 桥接的便捷写法：任何
Arrow 消费方都能经标准 `__arrow_c_stream__` PyCapsule 协议直接读取本 frame，无需调用
`to_arrow()`，也无需让 volas 依赖 pyarrow。

```py
import pyarrow as pa
tbl = df.to_arrow()        # -> pyarrow.Table（共享列缓冲区）
tbl = pa.table(df)         # 等价，经 __arrow_c_stream__ 协议
pdf = pl.from_dataframe(df)  # polars 经同一协议读取
```

返回一个 `pyarrow.Table`。完整零拷贝契约与 DLPack 导出见
[与 Arrow、DLPack 互操作](#与-arrowdlpack-互操作零拷贝)。

### df.to_pandas(dtype_backend='numpy') -> pandas.DataFrame

导出为 `pandas.DataFrame`（pandas 仅在此处惰性 import——它不是运行时依赖）。
`DatetimeIndex` 可往返；反向桥接是 [`DataFrame.from_pandas`](#dataframefrom_pandaspdf---dataframe)。

- **dtype_backend?** `str = 'numpy'` 缺失值如何带入 pandas：
  - `'numpy'`——生态兼容性最好的形式：带缺失值的 int / bool 列变成 `float64` /
    `object` + `NaN`（与 `pandas.Int64.to_numpy()` 一致）。
  - `'numpy_nullable'`——忠实、无损的 masked 往返：int / bool / str 列保持
    `Int64` / `boolean` / `string`，空洞为 `pandas.NA`。

```py
pdf = df.to_pandas()                                # 'numpy' 后端（基于 NaN）
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # 无损 masked Int64 / boolean / string
```

往返细节见[与 pandas 互操作](#与-pandas-互操作)。

### df.to_csv(path=None, ...) -> str | None

把 frame 写成 CSV——pandas `to_csv` 的一个子集。给了 `path` 就写文件并返回 `None`；`path=None` 则把 CSV 作为 `str` 返回。

- **path?** `str | os.PathLike | None = None` 输出文件；`None` 时返回字符串。
- **sep?** `str = ','` 字段分隔符。
- **index?** `bool = True` 是否把行索引写成第一列。
- **header?** `bool = True` 是否写列名表头行。
- **na_rep?** `str = ''` 缺失值写成的占位符。
- **columns?** `list[str] | None = None` 要写出的列及其顺序；`None`（默认）写出**所有**列。
- **float_format?** `str | None = None` printf 风格的浮点格式，例如 `'%.2f'`。

默认情况下，`to_csv` 会写出 **frame 持有的每一列——包括 directive（计算）列**。像 `ma:3` 这样的 directive 列一旦物化就是一个真实的列（计入 `df.columns`），因此会和原始 OHLCV 列一起被导出；它的预热期（warm-up）行渲染成空的 `na_rep`。想精确控制写哪些列，传 `columns=`——比如只导出原始列。和所有批量读取一样，`to_csv` 也要求 frame 处于刷新态：`append` 之后请先调用 `fulfill()`，否则在缓存列过期时会直接报错。

```py
from volas import DataFrame

df = DataFrame({
    'open':   [1.0, 2, 3, 4, 5],
    'high':   [2.0, 3, 4, 5, 6],
    'low':    [0.5, 1, 2, 3, 4],
    'close':  [1.5, 2.5, 3.5, 4.5, 5.5],
    'volume': [10, 20, 30, 40, 50.0],
})
df['ma:3']                                # 物化一个 directive（计算）列
df.fulfill()                              # 批量导出前先刷新缓存

# 默认：每一列都会写出——包含 ma:3 这个 directive 列：
df.to_csv(index=False)
# 'open,high,low,close,volume,ma:3\n1.0,2.0,0.5,1.5,10.0,\n...,2.5\n...'
#                            ^^^^ directive 列在内；预热期行为空

# 传 columns= 精确选择写哪些列——这里只写原始列：
df.to_csv(index=False, columns=['open', 'high', 'low', 'close', 'volume'])
```

### Series

`df[col]` 和 `df[directive]` 返回一个 `Series`：一条具名的一维列，API 与
pandas 兼容，包括算术 / 比较 / 逻辑运算符、`.sum()` / `.mean()` / `.std()` / …、
`.shift()` / `.diff()` / `.fillna()`、`.iloc` / `.loc`、`.to_numpy()` /
`.to_list()`。完整列表见
[其余与 pandas 一致的 API](#其余与-pandas-一致的-api)。`Series` 没有公开的构造函数，只能通过索引 `DataFrame` 得到。

```py
s = df['close']
s.name                 # 'close'
(s - s.shift(1)).mean()
df['ma:5 > ma:20']     # directive 也返回 Series（这里是 bool）
```

除 pandas 兼容 API 外，`Series` 还把 TA-Lib 的 15 个 **Math Transform** 函数暴露为方法——`acos` `asin` `atan` `ceil` `cos` `cosh` `exp` `floor` `ln` `log10` `sin`
`sinh` `sqrt` `tan` `tanh`：

```py
df['close'].ln()
df['high'].sqrt()
```

`datetime64[ns]` Series 暴露 pandas 风格的 `.dt` 访问器：日历分量（`year`
`month` `day` `hour` `minute` `second` `microsecond` `nanosecond` `quarter`
`dayofweek` `dayofyear` `days_in_month`）、日历谓词（`is_month_start` …
`is_year_end`、`is_leap_year`）、名称（`day_name()` / `month_name()`）、格式化（`strftime(fmt)`）、bar 对齐（`floor(freq)` / `ceil(freq)` / `round(freq)` /
`normalize()`）以及 `isocalendar()`。某个元素缺失时，各分量都返回 `NA`：

```py
t = volas.to_datetime(df['time'])
t.dt.hour                  # int64 Series，0..23
t.dt.dayofweek             # 周一=0 .. 周日=6
t.dt.floor('15min')        # 对齐到 15 分钟 bar
```

### Series.from_arrow(data, name=None) -> Series

**volas 独有的静态方法**——从任何实现 Arrow **C-Data** 数组协议（`__arrow_c_array__`）的对象构造 `Series`：`pyarrow.Array`、polars `Series` 等。dtype 匹配处借用数据缓冲区（否则拷贝），结果带一个全新的 `RangeIndex`。

- **data** Arrow 来源——任何实现 `__arrow_c_array__` 的对象。
- **name?** `str | None = None` 结果 `Series` 的名字。

```py
s = Series.from_arrow(pa_array, name='close')   # pyarrow.Array -> Series
```

### series.to_numpy(dtype=None, na_value=...) -> np.ndarray

把列的值导出为一维 NumPy 数组——`pandas.Series.to_numpy` 语义：

- **dtype** `str | None`——可选的导出转换。`None`（默认）给出列的原生表示；否则可为任何 `numpy.ndarray.astype` 接受的 NumPy dtype 字符串，常用取值为：
  - `'int64'`、`'int32'`、`'int16'`、`'int8'`（以及无符号 `'uint64'`、`'uint32'`、
    `'uint16'`、`'uint8'`）——整型。对含缺失值的列，除非给了 `na_value`，否则**抛错**
    （NA 没有整型表示）。
  - `'float64'`、`'float32'`、`'float16'`——浮点；缺失单元格为 `NaN`。
  - `'bool'`——布尔。
  - `'datetime64[ns]'`——datetime 纳秒；缺失单元格为 `NaT`。
  - `'object'`（或 `'O'`）——Python 对象，每个单元格保留各自的类型值（无损）。
- **na_value** `Any`——替换每个缺失单元格的值。默认是 NA 模型表示（float 导出为 `NaN`，
  object 数组为 `None`）。给了显式整型 `dtype` 时，值保持**精确**（大整数不经 `float64`
  漏斗），空洞变为 `na_value`。

```py
series = DataFrame({'qty': [1, None, 3]})['qty']    # 带缺失值的 int64

series.to_numpy()                          # -> array([ 1., nan,  3.])   （float64；缺失的 int -> NaN）
series.to_numpy(dtype='int64', na_value=0) # -> array([1, 0, 3])         （int64；填充 NA，保留 dtype）
series.to_numpy(na_value=-1)               # -> array([ 1., -1.,  3.])   （默认 float 导出，NA -> -1）

# 不给 na_value 时，整型 dtype 遇缺失值会抛错
series.to_numpy(dtype='int64')
# ValueError: cannot convert a column with missing values to integer NumPy dtype 'int64' ...
```

注意：

- 默认（不传 `dtype`、不传 `na_value`）是按 dtype 的导出：缺失的 int / bool / datetime
  单元格折叠为 `NaN` / `NaT`（NumPy 没有 `NA`），稠密列保留原生 dtype。float `NaN` 是带内值，因此 float 列转整型 dtype 时只要存在 `NaN` 同样会抛错（请传 `na_value`）。
- 与 pandas 一致，`na_value` 只改变缺失单元格——不给显式 `dtype` 时，带 NA 的 int 列仍导出为 `float64`（默认），`na_value` 只是填充那些 `NaN` 槽。
- 若需要**无损**保留原生 dtype 与缺失位置（不填充、不折叠 float），用 Arrow 路径（`series.to_arrow()` 携带空值位图）或 `series.to_pandas(dtype_backend='numpy_nullable')`；单独取 NA 掩码用 `series.isna().to_numpy()`。

### series.to_arrow() -> pyarrow.Array

**volas 独有**——把列导出为 `pyarrow.Array`，在 dtype 匹配处**零拷贝**（数值 / 字符串
/ datetime 缓冲区与 Arrow 共享，`bool` 与空值位图按需重打包）。需要 `pyarrow`（惰性
import）。它是 volas Arrow **C-Data** 桥接的便捷写法：任何 Arrow 消费方都能经标准
`__arrow_c_array__` PyCapsule 协议直接读取本 series。

```py
import pyarrow as pa
arr = series.to_arrow()    # -> pyarrow.Array（共享缓冲区）
arr = pa.array(series)     # 等价，经 __arrow_c_array__ 协议
```

返回一个 `pyarrow.Array`。本列还可经 DLPack 零拷贝导出到 NumPy / PyTorch / JAX
（`np.from_dlpack(series)`）——见[与 Arrow、DLPack 互操作](#与-arrowdlpack-互操作零拷贝)。

### Row

`df.iloc[i]` 和 `df.loc[label]` 返回一个 `Row`：一条单独记录，`.name` 是它的索引标签。`Row` **没有公开构造函数**（`Row(...)` 会抛
`TypeError: No constructor defined for Row`）；只能通过索引 frame 获得，也可以把它传给 `df.append`。

```py
row = df.iloc[-1]      # 最新 bar
row.name               # 它的索引标签（DatetimeIndex 下是一个 Timestamp）
row.to_dict()          # {列: 值}
row.to_numpy()         # 数值单元格组成的 1 维 ndarray
```

### 实时累积——一个 tf-aware DataFrame

在**实时**流式场景中，给 DataFrame 指定一个 `time_frame`，再把更细的 bar
`append` 进去，而不是每个 tick 都重新累积整段历史。`df.cumulate(tf)` 会返回这样的
frame（正在形成的周期保持 live）；也可以用 `DataFrame(data, time_frame=...,
cumulators=...)` 直接构建（给定的各行会被视作该 frame 下已经定型的 bar；需要
DatetimeIndex）。

在一个带 tf 的 frame 上：

- **df.append(bar)** 把 bar 折叠进去：落在当前未收口周期内的 bar 会**更新正在形成的最后一行**（`df.iloc[-1]`）；落在新周期内的 bar 会滚动成新行；同一时间戳的形成中 bar 再次到来时，会更新原 bar，而不是重复计入。
- **df.iloc[-1]** 是当前（仍未收口的）周期——live bar。
- **df[directive]** / **df.exec(directive)** 在包含形成中那一行的累积 frame 上计算指标——惰性、读时计算：一次 `append` 只把它们标记为过期，下一次读取只重算尾部。
- **df.cumulate(target)** 必须是源 frame 的整数倍（例如 `5m→15m`，而非
  `5m→7m`；一周或 3 天的 bar 不会嵌套进月 / 年）；同一个 frame 是一次 `copy()`。

```py
df = history.cumulate('5m')   # 带 tf 的 5m frame（history 更细，例如 1m）
for bar in stream:            # 每个 `bar` 是一个更细的 DataFrame
    df.append(bar)            # 折叠进正在形成的 5m bar
    df.iloc[-1]               # live 的、仍在形成中的 bar
    df['macd']               # 在累积 frame 上计算的指标
```

详见[累积与 DatetimeIndex](#累积与-datetimeindex)。

### read_csv(path, sep=',', header=True, parse_dates=None, index_col=None, na_values=None, keep_default_na=True, tz=None, date_unit=None) -> DataFrame

顶层函数，把 CSV 文件读成 `DataFrame`，并逐列推断 dtype；它是一个快速的
pandas 子集 CSV 读取器。

- **path** `str | os.PathLike` CSV 文件路径——字符串或任意 `os.PathLike`
  （例如 `pathlib.Path`）。
- **sep?** `str = ','` 字段分隔符（单个字符）；`delimiter` 是一个被接受的别名。
- **header?** `bool = True` `True`（或省略）把第一行当作表头；`False` / `None`
  表示无表头（列被命名为 `'0'`…`'n-1'`）。
- **parse_dates?** `list[str] | None = None` 要解析成 datetime 列的列名。
- **index_col?** `str | int | None = None` 要移入行索引的列名或整数位置；在
  `parse_dates` *之后*应用，所以指定一个已解析的日期列会产生一个 `DatetimeIndex`。
- **na_values?** `str | list[str] | None = None` 额外的缺失值标记。
- **keep_default_na?** `bool = True` 同时把默认的 NA 标记也当作缺失。
- **tz?** `str | None = None` `index_col` datetime 的时区：一个*naive*日期字符串按
  `tz` 读入（存为 UTC，索引被打上标记）。日期列应通过 `index_col` 传入，并且*不要*
  同时把它列在 `parse_dates` 里。参见[时区](#时区)。接受以下之一：
  - 一个固定 UTC 偏移，例如 `'+08:00'` / `'-05:00'`
  - 一个 IANA 时区名称，例如 `'America/New_York'` / `'Asia/Shanghai'` / `'UTC'`
- **date_unit?** `str | None = None` 把 `index_col` 当作此单位下的 epoch 整数读取（绝对 UTC；此时 `tz` 只设置显示时区）。为以下之一：
  - `'s'`——秒
  - `'ms'`——毫秒
  - `'us'`——微秒
  - `'ns'`——纳秒

```py
from volas import read_csv

df = read_csv('klines.csv')                        # RangeIndex
df = read_csv('klines.csv',
              parse_dates=['time_key'],            # 解析成 datetime
              index_col='time_key')                # -> DatetimeIndex
df = read_csv('data.tsv', sep='\t', header=False,  # 无表头 -> '0'..'n-1'
              na_values=['NA', 'null'])
```

### to_datetime(obj, unit='ns', format=None) -> Series

顶层函数，把 epoch 数字或 datetime 字符串转换成 datetime `Series`，行为对齐
`pandas.to_datetime`。`obj` 可以是 `Series`、一维 NumPy 数组或 list。
**缺失**输入（float `NaN`，或 int 列里的 `volas.NA`）会变成 `NaT`，和 `pd.to_datetime` 一样。

- **obj** 要转换的值：数值 epoch、datetime 字符串，或已经是 datetime 的
  `Series`（原样返回）。
- **unit?** `str = 'ns'` 用于**数值**输入的 epoch 单位（亚单位的小数部分会被保留，和
  `pd.to_datetime` 一样）。为以下之一：
  - `'s'`——秒
  - `'ms'`——毫秒
  - `'us'`——微秒
  - `'ns'`——纳秒（默认）
- **format?** `str | None = None` 用于**字符串**输入的显式 datetime 格式（pandas 的
  `format=`，例如 `'%Y-%m-%d %H:%M:%S'`）——更快且无歧义；对数值输入忽略。任何
  `strftime`/`strptime` 指令字符串；`None` 时自动推断。

Naive 字符串按 UTC 解析，带偏移（`…+08:00`）的字符串表示绝对瞬时。若要把结果索引
*显示*在某个时区，请先设为索引，再用 `tz_localize` / `tz_convert` 打上时区标记（参见
[时区](#时区)）。

```py
from volas import to_datetime

# 把 epoch 秒列解析成 datetime，再设为索引
df['time'] = to_datetime(df['time'], unit='s')
df = df.set_index('time')                       # -> DatetimeIndex
df = df.tz_localize('America/New_York')         # 打上显示时区标记（见“时区”）
```

如果想做一次就地、**截断式** cast（NumPy / pandas 的 `astype` 惯用法），请改用 `df.astype({'time': 'datetime64[s]'})`。

### directive_stringify(directive: str) -> str

得到 `directive` 的规范全名，也就是 volas 实际缓存时使用的列名。命令名会小写化，默认参数 / series 会被省略，以节省空间。

```py
from volas import directive_stringify

directive_stringify('kdj.j')
# 'kdj.j'

directive_stringify('kdj.j:9,3,2,100@high,close,close')
# 'kdj.j:,,2,100@,close'

# 命令名大小写不敏感，并规范化为小写
directive_stringify('MACD:12,26')
# 'macd'
```

### directive_lookback(directive: str) -> int

得到 `directive` 的回看（lookback）周期，即指标产出有效结果之前所需的最少先前数据点数量。

```py
from volas import directive_lookback

directive_lookback('ma:20')
# 19

directive_lookback('boll')
# 19（默认周期 20）

# 复合 directive：lookback 会在嵌套表达式间累加。
# repeat:5 需要额外 4 个点，boll.upper（周期 20）需要 19 -> 23
directive_lookback('repeat:5@(close > boll.upper)')
# 23
```

### 其余与 pandas 一致的 API

下面列出的接口都与对应的 `pandas` 行为一致——如果你在 pandas 里会用它，在 volas 里用法也相同，除了列表之后特别说明的
[NA 模型差异](#已知的-pandas-差异volasna-模型)。

```py
# --- DataFrame：元数据 ----------------------------------------------------
df.columns / df.shape / len(df) / df.dtypes      # dtypes -> dict
df.index                          # 行标签，作为一个 NumPy 数组
col in df ; for col in df         # 成员判断 / 迭代列名
df.tz / df.tz_localize(tz) / df.tz_convert(tz)   # DatetimeIndex 的 tz；见“时区”

# --- DataFrame：选择 ------------------------------------------------------
df[col]                           # -> Series
df[[col, ...]]                    # -> DataFrame
df[bool_mask]                     # -> DataFrame（过滤行；mask = Series | ndarray）
df.iloc[...] / df.loc[...] / df.at[label, col] / df.iat[i, j]
df.head(n=5) / df.tail(n=5)

# --- DataFrame：重塑与 dtype ----------------------------------------------
df.drop([label, ...], axis=0)     # 按标签删行（axis=1 -> 删列）
df.dropna(how='any') / df.sort_index(ascending=True) / df.reset_index(drop=False)
df.rename({old: new}) / df.astype({col: dtype}) / df.set_index(col)
df.astype({col: 'datetime64[s]'})  # 数值 epoch -> datetime（单位 s|ms|us|ns；截断式）
df.copy() / df.equals(other) / df.to_csv(path=None, ...)   # to_numpy：见其专属小节（dtype、na_value）

# --- DataFrame：写入 ------------------------------------------------------
df[col] = scalar | array | Series          # 增加 / 替换一列（按位置）
df.loc[mask, col] = value ; df.iloc[i, j] = value ; df.at[label, col] = value

# --- Series ---------------------------------------------------------------
s.name / s.dtype / len(s) / s.tz / s.index
s.to_list()                       # to_numpy 有 NA 注意点 -> 见其专属小节（dtype、na_value）
s.iloc[...] / s.loc[...]
s + s, s - 1, -s, ...             # 逐元素算术
s > 0, s == t, s != t, ...        # 比较 -> bool Series
s & t, s | t, ~s, s ^ t           # 逻辑 -> bool Series
s.sum() / s.mean() / s.min() / s.max() / s.std() / s.var() / s.median()   # 跳过缺失
s.shift(n=1) / s.diff(n=1) / s.fillna(v) / s.ffill() / s.bfill()           # 见“缺失值”：NA 保持 dtype
s.isna() / s.notna() / s.dropna() / s.equals(t)
```

#### 窗口运算（`rolling` / `expanding` / `ewm`）——仅为兼容性而存在

> **这套接口是为了让 pandas 里的研究 / 标注代码可以原样迁移。它不是计算指标的
> 推荐方式，也不应该进入实盘交易系统**：窗口结果只是普通 Series，不进入 directive
> 缓存，也**不会**被 `append()` / `fulfill()` 增量刷新；每来一根新 bar，都要完整
> `O(n)` 重算一次。请优先使用等价 directive（`df['ma:20']`、`df['median:30']`、
> `df['stddev:20']`、…）：同样的内核，还能享受缓存和逐 bar `O(lookback)` 刷新。

```py
s.rolling(window, min_periods=None, center=False)   # int 窗口；min_periods 默认等于 window
s.expanding(min_periods=1)
s.ewm(com=|span=|halflife=|alpha=, min_periods=0, adjust=True, ignore_na=False)
                                                    # 只能选择一种衰减写法
# Rolling / Expanding（pandas 语义：跳过 NA，由 min_periods 控制有效窗口）：
.count() .nunique()                                 # -> int64 Series（原生 NA）
.sum() .mean() .median() .min() .max()
.var(ddof=1) .std(ddof=1) .sem(ddof=1) .skew() .kurt()
.quantile(q, interpolation='linear') .rank(method='average', ascending=True, pct=False)
.first() .last()                                    # 保留 dtype
.corr(other) .cov(other, ddof=1)

# Ewm:
.mean() .sum() .var(bias=False) .std(bias=False) .corr(other) .cov(other, bias=False)
```

`center=True` 会把每个窗口标在中心位置，因此会读取相对于标签而言的**未来** bar。这正是标注流程（labeling pass）需要的，也正是实盘信号绝不能做的；volas 在这里只为前者提供支持。

基于时间的窗口（`rolling('5min')` / `timedelta`）是刻意不支持的。多时间框计算请维护
**两个带 tf 的 DataFrame**（见[累积](#累积与-datetimeindex)），并把每根 bar 都
`append` 到两者上；这才是受支持的逐 bar `O(lookback)` 设计。用窗口算术模拟更粗时间框，会在每根 bar 上重算全部内容。

未提供的 pandas 成员（因为与 volas 模型冲突）：`apply` / `agg` / `pipe`
（每个窗口执行任意 Python）、`win_type`、`step`、`on`、`closed`、`method`、
`ewm(times=...)`、`ewm.online()`——`append()` + directive 已经覆盖了流式用例。

#### 已知的 pandas 差异（`volas.NA` 模型）

少数 API **有意**偏离 pandas，因为 volas 原生存储缺失值为
[`volas.NA`](#缺失值volasna)（没有 `object` dtype，也不会静默提升到 float）：

- **`shift` / `diff` / `fillna` 及同类方法**保持列的 dtype——缺失值是
  `volas.NA`，而不是把一个 int/bool/str 列提升到 float/object。
- **比较**（`==` `!=` `<` `<=` `>` `>=`）返回*非 nullable* 的 bool 掩码：缺失值比较为 `False`（而 `!=` 比较为 `True`），遵循 IEEE / NumPy——而非 pandas-nullable
  的三值 `NA`。这样掩码不含 `NA`，`df[mask]` 和赋值都保持全定义。
- **存储保留 dtype。** pandas 会把带缺失值的 int/bool 列上转为 `float64` / `object`，而 volas 把它保持为 `int64` / `boolean`、空洞为 `volas.NA`——因此 `to_list()` 返回精确的 int 与 `volas.NA`。numpy *导出*（`to_numpy()`）则完全遵循 pandas 3.0：缺失单元格默认变为 `NaN` / `NaT`，整型 `dtype=` 遇缺失值会抛错，`na_value=` 可填充——见上文 `df.to_numpy` / `series.to_numpy` 专属小节。

完整背景——volas 的类型系统为何如此设计、pandas 的问题在哪里、迁移时有哪些坑——参见 [volas vs pandas ——类型系统](PANDAS-DIFFERENCES.md)。

pandas 风格的索引和写入细节分别见：
[索引与选择](#索引与选择)和
[写入与赋值](#写入与赋值)。

## 累积与 DatetimeIndex

假设有一个 csv 文件，里面是某只股票的 1 分钟 K 线数据：

```py
csv = read_csv(csv_path)

print(csv)
```

```
                   date   open   high    low  close    volume
0   2020-01-01 00:00:00  329.4  331.6  327.6  328.8  14202519
1   2020-01-01 00:01:00  330.0  332.0  328.0  331.0  13953191
2   2020-01-01 00:02:00  332.8  332.8  328.4  331.0  10339120
3   2020-01-01 00:03:00  332.0  334.2  330.2  331.0   9904468
4   2020-01-01 00:04:00  329.6  330.2  324.9  324.9  13947162
5   2020-01-01 00:04:00  329.6  330.2  324.8  324.8  13947163    <- an update of
                                                                    2020-01-01 00:04:00
...
19  2020-01-01 00:19:00  327.0  327.2  322.0  323.0  15086985
```

> 注意：同一时间戳的重复记录不会重复累积。除最新一条外，其余都会被丢弃。

读取同一个 csv，并把 `date` 列解析为 `DatetimeIndex`：

```py
df = read_csv(
    csv_path,
    parse_dates=['date'],
    index_col='date'
)

print(df)
```

```
                      open   high    low  close    volume
2020-01-01 00:00:00  329.4  331.6  327.6  328.8  14202519
2020-01-01 00:01:00  330.0  332.0  328.0  331.0  13953191
...
2020-01-01 00:19:00  327.0  327.2  322.0  323.0  15086985
```

此时，这个 data frame 已经带上了
[`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)。

但它还不是 5 分钟 K 线；需要显式累积：

```py
df_5m = df.cumulate('5m')

print(df_5m)
```

现在得到的是 5 分钟 K 线：

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
2020-01-01 00:05:00  325.0  327.8  316.2  322.0  82176419.0
2020-01-01 00:10:00  323.0  327.8  314.6  327.6  74409815.0
2020-01-01 00:15:00  330.0  335.2  322.0  323.0  82452902.0
```

`cumulate` 默认采用 OHLCV 语义：`open`=first、`high`=max、`low`=min、
`close`=last、`volume`=sum；**其他列默认回退为 `last`**。可以用 `cumulators=`
覆盖某列的聚合器。最常见的场景是非 OHLCV 列本应求和，却会默认成 `last`，例如成交额（`amount`）列：

```py
df.cumulate('1h', cumulators={'amount': 'sum'})
```

支持的聚合器包括 `first`、`max`、`min`、`last` 和 `sum`。

`time_frame` 可以是字符串标签，也可以是 `TimeFrame` 常量；完整列表见
[TimeFrame](#timeframe)。

#### bar 标签是周期起点

每个时间框都落在一个**固定网格**上。被累积出来的 bar 会以所在周期的网格**起点**
作为标签，即使第一根原始 bar 是在周期中途才到。一根 `15m` bar 如果第一笔 tick 在
`09:07`，它的标签仍然是 `09:00`，绝不会是 `09:07`；因此 volas 的 bar 会与交易所 K 线以及 pandas `resample`（`label='left'`）完全对齐。

各时间框的网格原点如下：日内时间框锚定在该索引所属（带时区的）交易日**午夜**——
`15m` bar 起于 `:00`/`:15`/`:30`/`:45`，`4h` bar 起于 `00:00`/`04:00`/…；`1d` 起于午夜；`1w` 起于**周一**；`3d` 使用从 Unix 纪元开始的连续网格；`1M` / `1y` 起于日历月 / 年。如果夏令时切换移除或重复了某个周期边界，标签会解析到该周期最早的真实瞬时。

在**实时**流式场景中，不应该每个 tick 都重新累积整段历史。更自然的做法是让当前
5 分钟 bar 保持*正在形成*的状态，并随着更细 bar 到来而更新。**带 tf 的 DataFrame**
正是为此设计：它仍然是普通 DataFrame（读列、跑 directive、切片都一样），只是
`append` 会把更细 bar **折叠**进正在形成的 bar，而不是新增一行。用
`df.cumulate('5m')` 或 `DataFrame(data, time_frame='5m')` 构造后，实时循环只剩下：

| 步骤                           | 调用                      |
| ------------------------------ | ------------------------- |
| 创建一个 `5m` frame            | `cum = df.cumulate('5m')` |
| 喂入下一根更细的 bar           | `cum.append(bar)`         |
| 读取当前正在形成的 bar         | `cum.iloc[-1]`            |
| 在该 frame 上读取指标          | `cum['macd']`             |

#### 看着正在形成的 bar 长大

从上面的 1 分钟 `df` 出发，一根一根构建 5 分钟 frame。先用 `00:00` bar 播种，再折叠进 `00:01`。两者都落在同一个 `00:00`–`00:05` 窗口内，所以 frame 仍然只有**一**
行，也就是那根正在形成的 bar；它已经被更新（`high` 涨到 `332.0`，`close` 到
`331.0`，`volume` 已求和）：

```py
cum = df.iloc[0:1].cumulate('5m')   # 用 00:00 bar 播种 5m frame
cum.append(df.iloc[1:2])            # 折叠进 00:01（同一个 5m 窗口）

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  332.0  327.6  331.0  28155710.0
```

再折叠进 `00:02`、`00:03` 和 `00:04`，这个窗口就填满了。那一行正在形成的 bar 此时就是
**已完成**的第一根 5 分钟 bar，和之前一次性 `df.cumulate('5m')` 打印出来的第一行完全一致：

```py
for i in range(2, 5):
    cum.append(df.iloc[i:i + 1])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
```

现在折叠进 `00:05`。它会开启**下一个**窗口，于是 `00:00` bar 定型，一根新的、正在形成的 bar 开始；frame 增长到两行，`cum.iloc[-1]` 就是那根新的、仍在形成的
`00:05` bar：

```py
cum.append(df.iloc[5:6])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0   <- finalized
2020-01-01 00:05:00  325.0  327.8  324.8  327.6  10448427.0   <- still forming
```

下面两个性质让它适合实时数据流：

- **指标是惰性的，并且读取时保持最新。** `append` 不做任何重算，只把依赖的
  directive 列标记为过期（它们的有效行游标落后于 frame 高度）。真正的重算发生在你
  **读取** `cum['ema:9']`（或任何 directive）时：只刷新过期尾部，即
  `O(lookback)`，不是整列重算；计算对象包含正在形成的那一行，结果与一次性「先累积再计算」逐位相同。（`to_numpy()` 这类批量读取不会自动刷新；请先调用 `cum.fulfill()`，或者直接读取一次 directive。）
- **重发的 bar 不会重复计入。**如果折叠一根时间戳已经出现过的 bar，volas 会**更新**
  对应周期，而不是继续累加——这就是本节开头展示的同一条去重规则，适合会修订最近一根
  bar 的交易所。

API 概览见[实时累积](#实时累积--一个-tf-aware-dataframe)。

## TimeFrame

`TimeFrame` 表示一个 bar 间隔。凡是 volas 需要重采样的地方都接受它：`df.cumulate`、
`DataFrame` 的 `time_frame` 参数，以及 `hv` 指标。它既可以写成 `TimeFrame` 常量，也可以写成等价的**字符串标签**。没有 `TimeFrame(...)` 构造函数；请使用下面的常量或标签字符串。

```py
TimeFrame.m5            # 5 分钟 frame
'5m'                    # 等价的标签字符串，各处都接受

df.cumulate(TimeFrame.m5)     # 与 df.cumulate('5m') 相同
```

支持的时间框（常量 ⇄ 标签）：

| 常量 | 标签 | 对齐 |
| --- | --- | --- |
| `TimeFrame.s1` | `'1s'` | 民用秒。 |
| `TimeFrame.m1` | `'1m'` | 民用分钟。 |
| `TimeFrame.m3` | `'3m'` | 小时内分钟桶，起于 `00`、`03`、`06`、... |
| `TimeFrame.m5` | `'5m'` | 小时内分钟桶，起于 `00`、`05`、`10`、... |
| `TimeFrame.m15` | `'15m'` | 小时内分钟桶，起于 `00`、`15`、`30`、`45`。 |
| `TimeFrame.m30` | `'30m'` | 小时内分钟桶，起于 `00` 和 `30`。 |
| `TimeFrame.H1` | `'1h'` | 民用小时。 |
| `TimeFrame.H2` | `'2h'` | 日内小时桶，起于 `00`、`02`、`04`、... |
| `TimeFrame.H4` | `'4h'` | 日内小时桶，起于 `00`、`04`、`08`、... |
| `TimeFrame.H6` | `'6h'` | 日内小时桶，起于 `00`、`06`、`12`、`18`。 |
| `TimeFrame.H8` | `'8h'` | 日内小时桶，起于 `00`、`08`、`16`。 |
| `TimeFrame.H12` | `'12h'` | 日内小时桶，起于 `00` 和 `12`。 |
| `TimeFrame.D1` | `'1d'` | frame 所在时区下的民用日。 |
| `TimeFrame.D3` | `'3d'` | 锚定 Unix 纪元的连续 3 天桶；不在月边界重置。 |
| `TimeFrame.W1` | `'1w'` | 周一起始的连续周，可跨越月边界。 |
| `TimeFrame.M1` | `'1M'` | frame 所在时区下的民用日历月。 |
| `TimeFrame.Y1` | `'1y'` | frame 所在时区下的民用日历年。 |

每个桶都按 **frame 所在时区的本地挂钟时间**对齐，但存储仍保持 UTC：日内小时桶（`2h`/`4h`/`6h`/`8h`/`12h`）从本地 `00` 开始，按本地小时步进；`3d` 从该时区下的
Unix 纪元日开始，计算连续 3 个本地民用日的桶（不在月边界重置）；`1w` 按本地民用时间从周一开始。因此日 / 周 bar 会跟随本地交易日，而具名时区会让小时桶具备 DST 意识。

## `directive` 的语法

```
command . sub : args @ series  op  command ...
   |      |     |      |
   |      |     |      └── 操作数列 / 子表达式  (例如 @open, @(boll))
   |      |     └── 逗号分隔的参数              (例如 ma:20, kdj.k:9,3)
   |      └── 子命令                            (例如 macd.signal)
   └── 指标名                                   (例如 ma, macd, boll)
```

#### `directive` 示例

下面是几种常见写法：

```py
# 布林带中轨
#   实际上就是默认 20 周期移动平均
df['boll']

# kdj j 小于 0，返回 bool Series
df['kdj.j < 0']

# kdj %K 上穿 kdj %D
df['kdj.k // kdj.d']

# 5 周期简单移动平均
df['ma:5']

# 基于 open 价的 10 周期简单移动平均
df['ma:10@open']

# 返回包含 5、10、30 周期 ma 的 DataFrame
df[[
    'ma:5',
    'ma:10',
    'ma:30'
]]

# 保留第一、第二个参数默认值，只覆盖第三个参数（macd.signal）
df['macd.signal:,,10']

# 参数如果是嵌套命令或 directive，需要用括号包起来
df['increase:3@(ma:20@close)']

# directive 解析器允许换行和空白，因此也可以这样写：
df['''
repeat
    :   5
    @   (
            close > boll.upper
        )
''']
```

#### 运算符

```
left operator right
```

- `//` —— `left` 是否**上穿** `right`（从下方穿到上方），也就是「金叉」：
  `df['macd // macd.signal']`。
- `\\` —— `left` 是否**下穿** `right`，即「死叉」。在 Python 字符串里反斜杠必须转义，所以我们写 `'macd \\ macd.signal'`。
- `><` —— `left` 是否穿越 `right`，向上或向下都算。
- `<` `<=` `==` `!=` `>=` `>` ——对同一条记录比较 `left` 和 `right` 的值，返回一个 `bool` series。
- 算术 `+ - * /`、逻辑 `& | ^`，以及一元 `~`（非）/ `-`（取负）。

`df[directive]` 会把结果**缓存**成真正的列，因此重复读取不再重复计算；`append`
之后再次访问时，它会自动刷新过期尾部。若只想把 directive 计算成 NumPy 数组而**不**
缓存，请用 `df.exec(directive)`（见[用法](#用法)）。

## 索引与选择

volas 提供 pandas 兼容的标签 / 位置访问子集。行索引可以是 range、`DatetimeIndex`、整数索引，或**字符串索引**。

```py
df.iloc[2]          # 按位置取一个 Row（row.name 是它的索引标签）
df.iloc[10:]        # 按位置切出一个 DataFrame
df.loc[label]       # 按索引标签取一个 Row
df.loc[lo:hi]       # 闭区间标签切片（字符串索引按字典序）
df.at[label, col]   # 按标签 + 列取一个标量
df.iat[i, j]        # 按位置取一个标量
df.index            # 行标签，作为一个 NumPy 数组
```

字符串（代码）索引：在字符串列上 `set_index`，再按代码查找：

```py
df = DataFrame({'sym': ['aa', 'bb', 'cc'], 'px': [1.0, 2.0, 3.0]}).set_index('sym')
df.loc['bb']           # 键为 'bb' 的那一行
df.loc['aa':'bb']      # 闭区间、字典序的切片
df.at['cc', 'px']      # 3.0
df.drop(['bb'])        # 按字符串标签删除
```

### 与 pandas 的差异（vs pandas）

volas 表面上是 pandas 风格，但它的**类型系统有意不同**，而且差异不只在索引上：缺失值保留原 dtype，没有 `object` dtype，返回值的方法仍返回 `Series`，有损转换会报错，而不是静默退化。**完整对比见
[volas vs pandas ——类型系统](PANDAS-DIFFERENCES.md)：为什么 volas 这样设计、
pandas 类型系统的问题在哪里，以及迁移时需要注意什么。**

具体到索引，volas 只支持**单层**、单一同质标签类型。相对于 pandas，volas **不**支持：

- **`MultiIndex`**（分层 / 多层索引），无论在行*还是*列上——列是一个由唯一字符串名组成的扁平列表。
- **任意标签 dtype**——索引只能是 range、datetime（`datetime64[ns]`）、整数或字符串之一。没有 float、categorical、interval、period、timedelta，也没有混合类型
  `object` 索引。
- **索引代数**——重索引（reindex）、索引集合运算（并 / 交），以及在合并 frame 时的自动按索引对齐。
- **重复标签**的查找（标签访问假定标签唯一）。

如果你的工作流依赖以上能力，请继续使用 pandas；volas 面向的是 K 线数据里常见的单层 OHLCV 索引形态。

## 写入与赋值

可以赋整列，也可以写入位置 / 标签 / 布尔选择（底层采用写时复制，copy-on-write）。
Series 赋值是**按位置**的：按行序写入，而不是按索引对齐。

```py
df['signal'] = 0.0                      # 增加 / 替换一列（scalar | array | Series）
df.iat[3, 0] = 99.0                     # 按位置赋一个单元格
df.at[label, 'close'] = 99.0            # 按标签 + 列赋一个单元格
df.iloc[10:20, 0] = 0.0                 # 一段列切片
df.loc[df['close'] > df['open'], 'signal'] = 1.0   # 掩码列赋值
```

把小数写入整数列会**报错**：int dtype 会被保留，有损写入会失败，而不是静默扩宽为
float（见[与 pandas 的差异](PANDAS-DIFFERENCES.md)；写入 `volas.NA` / `None` 会保持 int dtype，并把该单元格标为缺失）。如果写入已缓存的 directive 列，该列会失去缓存状态，因此后续 `fulfill()` 绝不会静默覆盖你的修改。

## 时区

底层存储始终是 **UTC 纪元纳秒**——这是 crypto、美股、港股和 A 股等市场共存时，用来按绝对瞬时对齐的统一轴。`DatetimeIndex` 还会携带一个**frame 级时区**，决定这些瞬时如何显示、裸字符串标签如何匹配，以及 `cumulate` 如何对齐日线及更粗周期。时区可以是
**固定偏移**（如 `'+08:00'`，成本低，适合 crypto / A 股 / 港股），也可以是**具名
IANA 时区**（如 `'America/New_York'`，通过 `chrono-tz` 支持 DST，适合美股 / 欧洲）。默认是 UTC。

完整流程如下：先用 `to_datetime` 解析一列，构建 `DatetimeIndex`；再用 `set_index`
提升为索引；最后用 `tz_localize`（把 naive 挂钟*重新解释为*某个时区，瞬时会移动）或
`tz_convert`（保持瞬时不变，只改显示时区）打上时区标记。下面的例子表示美国交易所在
2021-01-04 当地 09:30 开盘，原始数据是一个 naive 本地字符串：

```py
from volas import DataFrame, to_datetime, Timestamp

# 把 naive 的 't' 字符串解析成 UTC 瞬时并设为索引，再用 tz_localize
# 把这个挂钟解释为*纽约本地时间*。瞬时以 UTC 存储（14:30Z），索引按纽约时间显示和匹配。
df = DataFrame({'t': ['2021-01-04 09:30:00'], 'close': [100.0]})
df['t'] = to_datetime(df['t'])
df = df.set_index('t').tz_localize('America/New_York')
df.tz       # 'America/New_York'
df.index    # ['2021-01-04T14:30:00.000000000']  （裸 .index 是 UTC，与 pandas .values 一致）

# 正是 tz 让裸本地字符串能匹配到正确那一行：它会按 df.tz 解析。
df.at['2021-01-04 09:30:00', 'close']   # 100.0

# Timestamp 是带类型、可跨时区比较的标签。同一瞬时在上海是
# 22:30+08:00；无论 df.tz 是什么，它都能匹配：
ts = Timestamp('2021-01-04 22:30:00', tz='+08:00')   # == 纽约 09:30
df.at[ts, 'close']                       # 100.0
ts.value                                 # 它的 UTC 纪元纳秒（int）
ts.tz                                    # '+08:00'

# 整数 epoch：to_datetime(unit=...) 读取单位。epoch 是*绝对*时间：
# 先锚定为 UTC，再按显示需要转换时区。1609770600000 ms == 14:30Z。
e = DataFrame({'t': [1609770600000], 'close': [100.0]})
e['t'] = to_datetime(e['t'], unit='ms')
e.set_index('t').tz_localize('UTC').tz_convert('America/New_York').index
# ['2021-01-04T14:30:00.000000000']

# 带偏移的字符串本身已经表示绝对瞬时；to_datetime 会解析这个偏移：
o = DataFrame({'t': ['2021-01-04T09:30:00+08:00'], 'close': [1.0]})
o['t'] = to_datetime(o['t'])
o.set_index('t').index
# ['2021-01-04T01:30:00.000000000']  （09:30+08:00 == 01:30Z）
```

一个 frame 的时间轴只有两种状态（pandas 模型）：**naive**（未锚定的挂钟，
`df.tz is None`）或 **tz-aware**（已锚定，`df.tz` 指向时区，`'UTC'` 也算）。
`tz_localize` 用来锚定 naive 轴（移动瞬时以匹配该时区的挂钟）；`tz_convert` 用来把
aware 轴改写到另一个时区（瞬时不变）。两者都会拒绝错误状态，因为转换未锚定挂钟或重新锚定已锚定轴，都可能静默移动真实瞬时：

```py
naive = df                                   # df.tz is None
aware = naive.tz_localize('America/New_York')   # 锚定：瞬时移动，挂钟保持
aware.tz_convert('+08:00')                   # 转换：瞬时保持，挂钟移动
naive.tz_convert('+08:00')                   # TypeError ——先用 tz_localize 锚定
aware.tz_localize('UTC')                     # TypeError ——已锚定；请用 tz_convert
```

`cumulate` 到日线（或更粗）时，会把桶对齐到 frame 的本地交易日；如果使用具名时区，这个对齐具备 DST 意识。裸 `.index` 的 NumPy 导出仍保持 UTC（与 pandas `.values`
一致）。

## 缺失值（`volas.NA`）

`volas.NA` 是唯一的缺失值标记，而且**每一种 dtype 都支持它**。关键是：缺失值
**绝不会改变列的 dtype**。

| dtype | 缺失值的存储方式 | 元素访问 | 控制台显示 |
|---|---|---|---|
| `float64` / `float32` | `NaN`，带内（in-band） | `np.float64(nan)` | `<NA>` |
| `int64` / `int32` / `bool` / `str` | 有效性掩码（dtype 保持不变） | `volas.NA` | `<NA>` |
| `datetime64[ns]` | `NaT` | `np.datetime64('NaT')` | `<NA>` |

无论底层怎么存，**控制台都统一打印 `<NA>`**：缺失值只有一个展示符号，不随 dtype
变化（float `NaN`、datetime `NaT`、int / bool / str 的空洞，显示完全一致；
`to_string(na_rep=...)` 可以覆盖它）。元素访问和 `to_numpy` 仍然保留 dtype 语义（float 空洞读回为 `np.nan`），因此与 NumPy / pandas 互操作不会丢信息。

这与 pandas 自身的方向（[PDEP-16]）一致，也意味着 volas **没有 `object` dtype**：带空洞的 `int` / `bool` / `str` 列仍然是 `int` / `bool` / `str`，而 pandas 3.0 会把它提升为 `float64` / `object`。

```py
import volas
s = volas.DataFrame({'a': [1, None, 3]})['a']
s.dtype                  # 'int64'        （pandas 会给 float64）
s[1]                     # <NA>           （s[1] 是 volas.NA；float 空洞仍是 np.nan）
s.sum()                  # np.int64(4)    归约会跳过 NA
s.fillna(0).to_list()    # [1, 0, 3]
s.isna().to_numpy()      # [False, True, False]
print(s)                 # 缺失单元格打印为 <NA>

# shift / diff 保持 int dtype（pandas 会提升到 float）；空缺位置是 NA：
volas.DataFrame({'a': [10, 20, 30]})['a'].shift(1).to_list()   # [<NA>, 10, 20]
```

- **产生 NA** ——构造函数 list 中的 `None`（或 `volas.NA`）、`shift` / `diff` 的空缺，以及 `where` / `mask` 的默认填充。
- **消费 NA** ——归约（`sum` / `mean` / `min` / …）和 `count` 跳过它；算术传播它（`x ∘ NA = NA`）；`~` / `&` / `|` / `^` 用 Kleene 三值逻辑（`NA & False = False`、
  `NA | True = True`）；`cumsum` / `abs` / `round` / `clip` / 索引把它带过去；
  `isna` / `notna` / `dropna` / `fillna` / `ffill` / `bfill` 在每一种 dtype 上都有效。
- **比较**以 IEEE / NumPy 的方式处理缺失值：涉及 NA 的 `==`、`<`、`<=`、`>`、
  `>=` 都是 `False`，而 `!=` 是 `True`。因此布尔掩码永远是纯 `bool`，可直接用于
  `df[mask]`。注意 `!=` 这个例外：`s != value` 会*包含*缺失行。

[PDEP-16]: https://github.com/pandas-dev/pandas/pull/58988

## 与 pandas 互操作

pandas **不是**运行时依赖；这些桥接只在被调用时才惰性 import pandas，所以
`import volas` 仍然不需要 pandas。

```py
from volas import DataFrame

df = DataFrame.from_pandas(pandas_df)  # numeric / bool / str / datetime 原生；带时区 DatetimeIndex 可无损往返；
                                   # nullable Int64 / boolean / string 列会读回为 int / bool / str + volas.NA
pdf = df.to_pandas()               # -> pandas.DataFrame（'numpy' 后端：带 NA 的 int/bool 列变成 float64 + NaN）
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # 忠实的 masked Int64 / boolean（一次无损的 NA 往返）
df.to_csv('out.csv', index=True)   # pandas to_csv 的一个子集；path=None 时返回一个 str
```

`to_pandas` 的 **dtype_backend**（`'numpy'` 对比无损的 `'numpy_nullable'`）决定缺失值如何带入 pandas——逐值说明见
[`df.to_pandas`](#dfto_pandasdtype_backendnumpy---pandasdataframe)。

## 与 Arrow、DLPack 互操作（零拷贝）

volas 的每个 dtype 列各持有一块连续缓冲区（字符串列也是 Arrow 原生布局），因此跨到 Arrow 与 DLPack 的消费方时**无需拷贝**——消费方借用同一段字节，由 volas 负责保活。

```py
import pyarrow as pa, numpy as np

# Arrow C-Data / C-Stream —— pyarrow、polars 等通过标准 PyCapsule 协议
#（__arrow_c_array__ / __arrow_c_stream__）直接读取 volas。
pa.array(df['close'])              # Series   -> pyarrow.Array（共享缓冲区）
pa.table(df)                       # DataFrame -> pyarrow.Table（单个 RecordBatch）
df['close'].to_arrow()             # pa.array(...) 的便捷写法

Series.from_arrow(pa_array, name='close')   # Arrow array -> Series（dtype 匹配时借用）
DataFrame.from_arrow(pa_table)              # Arrow table -> DataFrame

# DLPack —— NumPy / PyTorch / JAX 借用一个稠密的数值（或 bool）列。
np.from_dlpack(df['close'])        # 零拷贝 ndarray 视图
```

高层入口——[`df.to_arrow`](#dfto_arrow---pyarrowtable) /
[`DataFrame.from_arrow`](#dataframefrom_arrowdata---dataframe)、
[`series.to_arrow`](#seriesto_arrow---pyarrowarray) /
[`Series.from_arrow`](#seriesfrom_arrowdata-namenone---series)——在 Usage 下有说明。它们建立在以下**标准协议方法**之上，Arrow / 数组消费方会自动调用它们（所以你很少需要自己调用）：

- **`Series.__arrow_c_array__`**——Arrow C-Data 数组协议；返回 `(schema, array)`
  PyCapsule 二元组，让 `pa.array(s)` / `pl.Series(s)` 读取一列。
- **`Series.__arrow_c_schema__`**——只含 schema 的那一半（列的 Arrow dtype）。
- **`DataFrame.__arrow_c_stream__`**——Arrow C-Stream 协议；frame 作为单个
  `RecordBatch`，让 `pa.table(df)` / `pl.from_dataframe(df)` 读取一个 frame。
- **`Series.__dlpack__` / `Series.__dlpack_device__`**——DLPack 协议，让
  `np.from_dlpack(s)` / `torch.from_dlpack(s)` 借用一个稠密的数值 / bool 列。借来的视图是**只读**的（带版本号 DLPack 的标志位），写穿它无法绕过写时复制而改坏帧；无法接收该标志位的 1.0 之前的旧消费方，则改为得到一份独立副本。
  `np.from_dlpack(s, copy=True)` 返回一份独立的**可写**副本。非 CPU 设备或 stream 会被拒绝（`BufferError`）；含缺失值的 int/bool 列（DLPack 无空值掩码）或 str/datetime 列会抛错。

**零拷贝契约。**在双向上，volas 原生的数值（`int32` / `int64` / `float32` / `float64`）、字符串、纳秒 datetime 列的*数据*缓冲区被共享（无拷贝）。小的重打包有：

- **`bool`**——volas 每个值存一个字节，Arrow 存一个*位*；
- **空值位图**（≤ `n/8` 字节）——包括导出时把缺失 `float`（带内 `NaN`）扫描成真正的
  Arrow null 的那一遍；
- 仅在**导入时**：32 位偏移的 `Utf8`（加宽到 64 位偏移）或 `string_view`（物化为连续字节）、粗于纳秒的时间戳（重新缩放）、`Decimal128` / `Decimal256`（→ `f64`，超过约 15 位**有损**
  ——精确价格请上游保持字符串）、窄/无符号整数（→ `i64`；超过 `i64::MAX` 的 `UInt64` 无法无损表示，会报错）、字典 / 类别（categorical）列（解码为其取值）、无类型的 `Null` 列（→ 全 NA 的 `f64` 列）、`Date32` / `Date64`（→ 纳秒 datetime）。其余 Arrow 类型会抛清晰的错误。

**边界上的 NA。**每一个缺失值——int、bool、**以及 float**——都跨越为真正的 Arrow null
（缺失的 float 即带内 `NaN`，导出时扫描进空值位图），因此下游的 `is_null` / `null_count`
能一致地看见它们。DLPack 与整型 `to_numpy` 没有空值通道，遇缺失值会抛错——请传
`na_value=`，或用 Arrow 路径（它无损携带空值位图）：

```py
df['qty'].to_numpy(dtype='int64')                # 任一值为 NA 时抛错（与 pandas 对齐）
df['qty'].to_numpy(dtype='int64', na_value=0)    # 或用 na_value 填充空洞
pa.array(df['qty'])                              # 无损：保留 int64 + 空值位图
```

**时区。** Arrow 表没有*索引*，因此帧的索引（及其时区）不会被携带：datetime 列以裸 UTC
纳秒跨越（绝对时刻精确，只丢「用哪个时区显示」的标签），`from_arrow` 返回全新
`RangeIndex`。导入后用 `tz_localize` / `tz_convert` 重新打上时区。

## 错误处理

directive 出错时会抛出有类型的异常。两者都继承自 `DirectiveError` 和内置
`ValueError`，因此已有的 `except ValueError` 处理仍然有效。

```py
from volas import DirectiveSyntaxError, DirectiveValueError

try:
    df['ma:2,3']                 # 参数过多
except DirectiveValueError as e:
    ...                          # 未知命令/子命令、坏参数、坏取值

try:
    df['a >']                    # 格式错误的表达式
except DirectiveSyntaxError as e:
    ...                          # 消息里带有错误的行 / 列
```

## 内置指标

完整 directive 参考见 [INDICATORS.md](INDICATORS.md)，其中覆盖 Volas 独有指标、内置统计命令，以及与 TA-Lib 兼容的 directive。

# 参与与反馈

欢迎提 issue、指标请求、benchmark 挑战和 PR——参见
[CONTRIBUTING.md](CONTRIBUTING.md)，并通过 [issue](https://github.com/kaelzhang/volas/issues/new/choose) 选择合适的模板提交。最有价值的反馈是关于 API 设计和 [benchmark 方法论](docs/benchmark-faq.md)的。

如果你在 Python 里搭建实时 OHLCV / 技术指标流水线，欢迎 star 本仓库，关注新增指标、benchmark 结果与发布。

# 许可证

[MIT](LICENSE)

# 面向开发者

开发者说明、本地构建命令、依赖分组和 benchmark 报告指引都在
[DEVELOPMENT.md](DEVELOPMENT.md) 中。
