[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

# [![volas](docs/social.png)](https://github.com/kaelzhang/volas)

[English](README.md) | 简体中文

> 面向股票 / K 线（OHLCV）时间序列数据的高性能 Rust 列式内核。

**volas** 是一个 Rust 驱动、pandas 风格的 `DataFrame`，专为实时 OHLCV 流水线打造：内置 [**254** 个交易指标](INDICATORS.md)，支持增量 O(lookback) 刷新，输出可直接交给 NumPy / Torch 使用。

它**不是**通用 pandas 替代品，而是一个窄而快、专服务于 K 线 / OHLCV 工作流的 DataFrame：append 一根新 bar，指标列保持缓存，只刷新受影响的尾部。

**volas** 同时也是一个 Rust [crate](crates/volas/README.zh-CN.md)。

索引在构造及行选择后统一受检：datetime 标签必须严格递增且不能包含 NaT，整数和字符串标签必须唯一。Rust 数据模型通过受检构造器和只读 Column / Series 访问器使用，详见 [Rust 数据模型](crates/volas/README.zh-CN.md#数据模型)。

```python
from volas import read_csv

df = read_csv("aapl_1m.csv")

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

Rust 写入会在发布前检查完整选择：`Column::scatter` 检查位置和值数量，`DataFrame::append` 在提交前检查 schema、值和一份索引接纳计划；`OpenPeriod::push` 复用相同的受检追加。已聚合行通过 `check_folded_row(...).land()` 发布，原始追加及聚合提交 helper 为私有。`Series::from_column` 和 `DataFrame::from_columns` 根据接纳的形状构建受检位置索引；NumPy 与 DLPack 导出由 Column owner 完成一次类型分派。


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
- [累积与 DatetimeIndex](docs/zh-CN/cumulation.md#累积与-datetimeindex)
- [TimeFrame](docs/zh-CN/cumulation.md#timeframe)
- [directive 的语法](#directive-的语法)
- [索引与选择](#索引与选择)
- [写入与赋值](#写入与赋值)
- [时区](#时区)
- [缺失值（`volas.NA`）](#缺失值volasna)
- [与 pandas 互操作](docs/zh-CN/interop.md#与-pandas-互操作)
  - [datetime 列是 UTC 瞬时](docs/zh-CN/interop.md#datetime-列是-utc-瞬时)
- [与 Arrow、DLPack 互操作（零拷贝）](docs/zh-CN/interop.md#与-arrowdlpack-互操作零拷贝)
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

[![在 Colab 中试用 quickstart](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaelzhang/volas/blob/main/examples/notebooks/volas_quickstart.ipynb)
[![在 GitHub Codespaces 中打开 notebook](https://img.shields.io/badge/Codespaces-Open%20notebook-181717?logo=githubcodespaces&logoColor=white)](https://codespaces.new/kaelzhang/volas)

```sh
pip install volas
python examples/00_install_check.py
python examples/03_live_ohlcv_append.py   # append 一根 bar，只刷新受影响的尾部
```

更多文档：[TA-Lib 迁移](docs/migration-talib.md)、[pandas 迁移](docs/migration-pandas.md)、[directive 速查](docs/directive-cheatsheet.md)、[什么时候不该用 volas](docs/when-not-to-use.md)。

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
    `bool`、`datetime64` 或 `string`）；NumPy **masked array** 中被屏蔽的单元格成为 `volas.NA`；
  - 另一个 volas **`DataFrame`**，此时会被拷贝（如同 `pandas.DataFrame(df)`）。

  构造函数**不接受** `pandas.DataFrame` 或 Arrow 对象——请改用专门的 [`DataFrame.from_pandas`](docs/zh-CN/interop.md#dataframefrom_pandaspdf---dataframe) / [`DataFrame.from_arrow`](docs/zh-CN/interop.md#dataframefrom_arrowdata-copyfalse---dataframe) 桥接。如果要附加 [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)，用 `to_datetime` 解析某一列，用 `set_index` 把它提升为索引，再用 `tz_localize` / `tz_convert` 打上时区标记。参见[时区](#时区)。
- **index** `Optional[list | np.ndarray] = None` 每行一个标签；datetime 标签必须严格递增且不含 NaT，整数和字符串标签必须唯一但不限制顺序。无效标签在构造、`set_index`、`read_csv(index_col=...)` 或 `from_pandas` 的最早入口抛出 `ValueError`。
- **columns** `Optional[list[str]] = None` 选择并排列要保留的列。DataFrame 源与 `df[[...]]` 共用投影事务：有效指令在首次请求时计算，结果保留为普通数据列；任一条目无效时源不变，异常也与投影相同。dict 源只接受已存在的键，缺失键抛 `KeyError`。空 list 保留零列；重复身份会被拒绝，缺失列绝不会被静默填充。
- **time_frame** `Optional[str | TimeFrame] = None` 若设置，则把它变成一个在该 bar 间隔上**支持 tf 累积**的 DataFrame：给定的各行是该 frame 下的 bar，每行以所在周期的**起点**为标签，最后一行是**正在形成**的 bar——之后 `append` 进来、落在它所在周期内的更细 bar 会折叠进这一行（这一行不能已经包含那根 bar）。
  需要一个 `DatetimeIndex`。参见[实时累积](#实时累积--一个-tf-aware-dataframe)。
- **cumulators** `Optional[dict[str, str]] = None` 折叠时使用的逐列聚合器覆盖（例如 `{'amount': 'sum'}`），仅在与 `time_frame` 同时使用时才有意义。默认采用
  OHLCV 语义（`open`=first、`high`=max、`low`=min、`close`=last、`volume`=sum；其他列默认 `last`）。字典的每个**值**为以下之一：
  - `'first'`——桶内第一个值
  - `'last'`——桶内最后一个值
  - `'max'`——最大值
  - `'min'`——最小值
  - `'sum'`——求和

  缺失值从不被跳过：只要桶内任一输入值缺失（含 `NaN`、`±inf`），该列的聚合结果（`'first'` / `'last'` / `'max'` / `'min'` / `'sum'` 皆然）即为缺失（全缺失的桶也是缺失；有限输入相加溢出的 IEEE 中间结果仍可为 `±inf`），实时折叠与批量 `cumulate` 结果一致。`datetime` 列不支持 `'sum'`，`bool` / `str` 列只支持 `'first'` / `'last'`；不支持的组合在构造 frame 时即被拒绝。
- **window** `Optional[int] = None` 把它变成一个**有界滚动窗口** frame，只暴露最后 `window` 行（参见[有界滚动窗口](#有界滚动窗口)）。需要配合 `max_lookback`。
- **max_lookback** `Optional[int | list[str]] = None` **与 `window` 同时必填**（且只在有 `window` 时有效）：可见行之前保留的隐藏 margin 行数，以及该 frame 能精确计算的 directive（参见[有界窗口能计算什么](#有界窗口能计算什么)）。传一个**指标 directive 列表**即为**声明**它们：margin 取它们 lookback 的最大值（例如 `['atr:14', 'ma:50']` → margin 49），每个声明的 directive 在构造时基于全部给定行计算并缓存——滑动累加类或递归类指标（`ma`、EMA、Wilder 平滑、累积线等）只能以这种方式计算。传一个 **int** 则只给出 margin，不声明任何 directive。每个列表项必须是指标 directive（`'ma:50'`），裸名 / 笔误（`'ma50'`）和空列表会被拒绝；列名与 frame 中某个数据列相同的 directive（数据列 `macd` 旁的 `'macd:'`）同样会被拒绝：请先重命名该数据列再声明。

### 有界滚动窗口

传入 `window=` 可把 frame 限制为最后 `window` 行。这正是实盘 / 神经网络输入所需的形态：你可以永远地 `append` bar，但内存始终**有界**——frame 保留最后 `window + max_lookback` 行（且至少 `max_lookback + 2` 行，使重发的最后一行能从它前一行的状态续算）；当保留行数达到其两倍时，把头部丢回到这个数量，摊还每根 bar O(1)。它计算的每个指标都与喂入相同行的无界 frame **逐位一致**，不论经历多少次丢弃、重发、缺失值或写入。

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

`ready`（一个 property）只有在每个声明的 directive 在每个可见行上都已度过预热期时才为 `True`，预热按它读取的完整（非缺失）行计数：每个声明列的第一个值位于第一个可见行或之前。它在每次 append 时以 O(1) 维护；读取新鲜值仍是读取方的职责（先 `fulfill()`）。不声明任何 directive 的 frame（int 形式的 `max_lookback`）恒为 `True`。请用它来 gate 你的推理 / 训练。

```py
wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
wf.ready          # False——第一个可见行上还没有 atr:14 的值
...               # append 直到预热完成
wf.ready          # True ——每个可见行都已有 atr:14 的值
```

**无界** frame（没传 `window=`）没有预热契约，**恒为 `True`**。

#### 有界窗口能计算什么

有界窗口只持有最近的行，因此只在能精确计算的地方计算指标。每个 directive 属于三种**历史类别**之一（由命令表统一定义）：**第 1 类**只依赖一个尾部窗口、没有滑动累加器（`roc`、`llv`、`maxindex`、`willr`、`psy`、`close > open` 等）；**第 2 类**依赖尾部窗口上的滑动累加器（`ma`、`wma`、`boll`、`stddev`、`cci`、`mfi`、`linearreg` 等）；**第 3 类**依赖之前的每一行——递归、累积线或无界计数（EMA、Wilder 平滑、`obv`、`increase` 等）。一个 directive 取其各组成部分中最高的类别。

- **声明的 directive**（`max_lookback=[...]`）在构造时基于完整历史计算并缓存。递归类 directive 的**检查点（checkpoint）**与追加续算使用同一份状态，保存每个有状态节点的内核状态、最后完整输入行及输出环。检查点随丢弃头部而前移；写入后从检查点重放整个 directive。缺失行输出 NaN，递归状态停留在上一完整行；预热期间则保存原始输入行。声明的链式表达式（`'ema:5@(rsi:5)'`、`'ema:3 - ema:5'`、`'close > ema:20'`、`'ema:2@(maxindex:2)'`、`'close-ma:5+obv'`）在追加、压缩和可见行写入后仍与完整历史精确一致，带 `time_frame` 的窗口也一样。`crsi`、`stoch_momentum` 以及选择任意均线类型的复合均线命令也保存各阶段的续算状态，可直接声明在窗口 frame 上。
- **构造之后才请求的 directive**（`wf[d]`、`wf[[...]]`、`wf.exec(d)`）只有在它属于第 1 类、且每行之前读取的行数不超过 `max_lookback` 时才会计算。第 2 类和第 3 类必须声明；构造之后再请求会报错，并给出两条正确的路径（对已经声明的 directive，`exec` 会提示读取它的列）：

```text
ValueError: cannot compute 'ema:10' on this windowed frame: each of its values depends on every row before it, and the frame keeps only its last 50 rows. Declare it when constructing the frame — DataFrame(data, window=30, max_lookback=[..., 'ema:10']) — and read its column df['ema:10'], or compute it on an unbounded frame.
```

把一个 frame 复制进窗口（`DataFrame(df, window=..., max_lookback=...)`）时，它携带的缓存列只有在窗口能精确保持时才会保留（在 margin 之内，且不是没有携带状态能覆盖的递归）；否则构造报错。

整列赋值（`wf['x'] = values`，标量、数组或与可见行对齐的 Series）只写入可见行：隐藏 margin 保留它持有的历史——即该列自己的值，精确转换为新值的 dtype（无法精确容纳的值，包括 `float32` 舍入，会抛出 `TypeError`），新列则为缺失值。

由另一个 frame 的尾部构造的窗口（`DataFrame(df.iloc[-50:], window=..., ...)`）会从该尾部携带的状态继续其中的递归列：append 时精确。它不持有尾部之前的历史，因此一旦写入可见行，就无法重算该列，读取时报错。

#### `fill_into(out, columns=None)`——零分配特征导出

`fill_into` 把窗口的值**就地**写进**你自己**持有的 NumPy 数组，每次调用零分配。它是实盘推理循环的「导出」一半：配上一块预分配 buffer，`append → fulfill → fill_into → 推理` 的循环每根 bar 零分配（不同于 `to_numpy` 每次都新建一整块矩阵）。

契约：

- **只导出已缓存的列，且按其规范化（canonical）的 directive 名匹配**。directive 列必须**先 materialize**——访问它一次并把列名读回来（`name = wf['atr:14'].name`），因为缓存的 directive 是以规范化形式存储的（例如 `'MA: 5'` → `'ma:5'`），**不一定等于你传入的字符串**，而 `columns=` 按存储的确切列名匹配。（在 `max_lookback=['atr:14']` 中声明的 directive 会在构造时缓存；`fill_into` 导出已缓存的值、从不现场计算。）
- **`out`** 必须是 `float32` **或** `float64` 二维数组，形状**严格等于** `(len(df), k)`，其中 `k` 是导出的列数（尊重 strides，非连续或 Fortran-order 视图也可）。形状或 dtype 不符会报错。
- **`columns`** 按顺序选择要导出的列（默认全部列）。**字符串列没有浮点含义、会被拒绝**——在 `columns=` 里只列出数值列以排除它。
- **缺失 / NA 单元格写为 `NaN`**。
- `append` 之后缓存指标是脏的，所以**先 `fulfill()` 再 `fill_into()`**——带着未刷新的 directive 列导出会报错。

```py
import numpy as np

wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
atr = wf['atr:14'].name               # 把声明的 directive 的列名读回来：缓存的 directive 以
                                      # 规范化形式存储（这里是 'atr:14'，但例如 'MA: 5' → 'ma:5'），
                                      # 不一定等于你传入的字符串
COLS = ['open', 'high', 'low', 'close', atr]    # k = 5 个特征

# 整个运行期复用同一块 buffer——形状 (window, k)，即模型的输入张量。
buf = np.empty((30, len(COLS)), dtype=np.float32)

for bar in feed:
    wf.append(bar)
    wf.fulfill()                      # 导出前刷新 atr 的脏尾部
    if not wf.ready or len(wf) < 30:
        continue                      # 预热中：指标未生效 / buf 还不是 (30, k)
    wf.fill_into(buf, columns=COLS)   # 把 30×5 窗口零分配写进 buf
    prediction = model(buf)           # 每根 bar 都喂给模型同一块内存
```

> **形状提示。**形状是按 `len(df)` 严格匹配的，而 `len(df)` 在**预热期会增长**（1、2、……直到 `window`），之后才稳定在 `window`。`ready` 只说明指标，因此上面的循环同时等待 `len(wf) == window`，此后固定的 `(window, k)` buffer 必然适配。若要在预热期导出，就把 `out` 调成当前的 `len(df)`。

`fill_into` 并不限于 windowed frame——在普通 frame 上它同样导出整个逻辑视图；只是有界窗口下「复用一块 buffer」最有价值。

[pandas 与 Arrow 构造入口](docs/zh-CN/interop.md#dataframefrom_pandaspdf---dataframe)。

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
`KeyError`。当列名来自外部数据（CSV 表头、用户输入、配置）时，请优先使用它，避免像 `"ma:5"` 这样恰好长得像 directive 的列名悄悄触发计算。它属于列读取：`append` 之后变为过期（stale）的缓存指标列会先被刷新再返回，与 `df[key]` 一致。

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

如果调用者是一个**带 tf**的 DataFrame（用 `time_frame` 构建的，或者 `cumulate` 的结果），`append` 会把每一根 bar **折叠**进当前未收口的周期：该周期的 bar 保存在一个小 DataFrame 中，与其中某根 bar 时间戳相同的 bar 会替换它，周期的累积结果会**替换最后一行**；落在之后周期的 bar 则开启新的一行。
参见[实时累积](#实时累积--一个-tf-aware-dataframe)。

`append` 是惰性的：它不会重算新行上的指标列。这些列会保持过期（stale），直到对指标**列**的读取刷新它们，或调用 `df.fulfill()`（见下文）。

### df.cumulate(time_frame: TimeFrame | str, cumulators: dict | None = None) -> DataFrame

把 `DataFrame` 累积（重采样）到更粗的 `time_frame`，返回一个新的 `DataFrame`。需要一个 `DatetimeIndex`。若目标周期与源 frame 相同，则比较每个数据列的有效聚合规则：相同就复制数据列并保留未收口周期的细 bar，不同则在变更前报错；省略覆盖项仍采用 OHLCV 默认规则。缓存指标列不携带到结果。

- **time_frame** `TimeFrame | str` 目标 bar 间隔，例如 `TimeFrame.m5` 或 `'5m'`。参见 [TimeFrame](docs/zh-CN/cumulation.md#timeframe)。
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

详见[累积与 DatetimeIndex](docs/zh-CN/cumulation.md#累积与-datetimeindex)。

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

[DataFrame / Series / Row 导出与互操作](docs/zh-CN/interop.md#dfto_numpydtypenone-na_value---npndarray)。

显式 NumPy 导出在解析共同目标 dtype 前保留每列的源表示和时间单位，逐列转换有效槽并在目标中填充缺失值；规则和示例见[互操作指南](docs/zh-CN/interop.md#numpy-逐列目标转换)。

### 实时累积——一个 tf-aware DataFrame

在**实时**流式场景中，给 DataFrame 指定一个 `time_frame`，再把更细的 bar `append` 进去，而不是每个 tick 都重新累积整段历史。
`df.cumulate(tf)` 会返回这样的 frame（正在形成的周期保持 live）；也可以用 `DataFrame(data, time_frame=..., cumulators=...)` 直接构建（需要 DatetimeIndex）。

frame 会把**未收口周期**的各根 bar 单独保存在一个小 DataFrame 中，它的最后一行就是这些 bar 的累积结果。在一个带 tf 的 frame 上：

- **df.append(bar)** 把 bar 加入未收口周期，并用该周期的累积结果**替换正在形成的最后一行**（`df.iloc[-1]`）。与周期内已有 bar **时间戳相同**的 bar 会替换它，因此重发的 bar 绝不会重复计入。落在之后周期的 bar 会开启新的一行。
- 因此，一根与 frame **自身**时间框相同、以周期起点为时间戳的 bar（例如交易所在 `15m` K 线形成期间反复推送的快照）会**整行替换最后一行**。
- **构造时的最后一行就是正在形成的 bar。** `DataFrame(data, time_frame=...)` 把它作为未收口周期的第一根 bar，所以落在它所在周期内的更细 bar 会折叠进这一行，而不会再开一行同标签的新行。
- **df.iloc[-1]** 是当前（仍未收口的）周期——live bar。
- **df[directive]** / **df.exec(directive)** 在包含形成中那一行的累积 frame 上计算指标——惰性、读时计算：一次 `append` 只把它们标记为过期，下一次读取只重算尾部。
- **df.cumulate(target)** 必须是源 frame 的整数倍（例如 `5m→15m`，而非 `5m→7m`；一周或 3 天的 bar 不会嵌套进月 / 年）；相同周期按现有数据列比较有效聚合规则：相同则复制并保留未收口周期的细 bar，不同则在变更前报错；缓存指标列被丢弃。

```py
df = history.cumulate('5m')   # 带 tf 的 5m frame（history 更细，例如 1m）
for bar in stream:            # 每个 `bar` 是一个更细的 DataFrame
    df.append(bar)            # 折叠进正在形成的 5m bar
    df.iloc[-1]               # live 的、仍在形成中的 bar
    df['macd']               # 在累积 frame 上计算的指标
```

volas 无法得知你的各行数据是怎样产生的，因此依赖下面两条约定，并且**不做**校验：

- **bar 以周期起点为标签**：`17:00`–`17:14:59` 的 `15m` bar 标为 `17:00`，绝不是周期终点。以终点时间标注的数据需要先自行平移。
- **构造时的最后一行不能已经包含之后要 append 进来的更细 bar。** 在 `17:09:30` 取到的 `15m` 快照，再 append `17:05` 的 1m bar，这一分钟就会被计入两次。如果要用更细的 bar 继续一根尚未收口的 bar，请用 `cumulate` 从更细的历史数据构建 frame——这样它的未收口周期恰好保存着这些更细的 bar。

详见[累积与 DatetimeIndex](docs/zh-CN/cumulation.md#累积与-datetimeindex)。

### read_csv(path, sep=',', header=True, parse_dates=None, index_col=None, na_values=None, keep_default_na=True, tz=None, date_unit=None) -> DataFrame

顶层函数，把 CSV 文件读成 `DataFrame`，并逐列推断 dtype；它是一个快速的
pandas 子集 CSV 读取器。

- **path** `str | os.PathLike` CSV 文件路径——字符串或任意 `os.PathLike`
  （例如 `pathlib.Path`）。
- **sep?** `str = ','` 字段分隔符（单个字符）；`delimiter` 是一个被接受的别名。
- **header?** `bool = True` `True`（或省略）把第一行当作表头；`False` / `None`
  表示无表头（列被命名为 `'0'`…`'n-1'`）。
- **parse_dates?** `list[str] | None = None` 要解析成 datetime 列的列名，按其文本逐个解析（全部缺失的列就是全 `NaT`）。
- **index_col?** `str | int | None = None` 要移入行索引的列名或整数位置；在
  `parse_dates` *之后*应用，所以指定一个已解析的日期列会产生一个 `DatetimeIndex`。
- **na_values?** `str | list[str] | None = None` 额外的缺失值标记。
- **keep_default_na?** `bool = True` 同时把默认的 NA 标记也当作缺失。
- **tz?** `str | None = None` `index_col` datetime 的时区：一个*naive*日期字符串按 `tz` 读入（存为 UTC，索引被标记为 `tz`，`'UTC'` 也不例外，因此索引是 tz-aware 的）。该列也可以同时列在 `parse_dates` 里。参见[时区](#时区)。接受以下之一：
  - 一个固定 UTC 偏移，例如 `'+08:00'` / `'-05:00'`
  - 一个 IANA 时区名称，例如 `'America/New_York'` / `'Asia/Shanghai'` / `'UTC'`
- **date_unit?** `str | None = None` 把 `index_col` 当作此单位下的 epoch 数值、按其文本精确读取（绝对 UTC，带小数的 epoch 保留到最近的纳秒；此时 `tz` 只设置显示时区）。为以下之一：
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
  `strftime`/`strptime` 指令字符串；`None` 时自动推断。解析出的每个字段都会生效：偏移量
  （`%z` / `%:z`）使值成为绝对时刻，与不带 `format` 的带偏移字符串完全一致；格式未给出的
  时间字段取零。`%Z`（时区*名字*）抛出 `ValueError`——请改用数字偏移量。

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

得到 `directive` 的规范全名，也就是 volas 实际缓存时使用的列名。命令名会小写化，每个参数写成它的取值，默认参数 / series 会被省略，以节省空间。同一指令的所有写法只对应一个名字。

```py
from volas import directive_stringify

directive_stringify('kdj.j')
# 'kdj.j'

directive_stringify('kdj.j:9,3,2,100@high,close,close')
# 'kdj.j:,,2,100@,close'

# 命令名大小写不敏感，并规范化为小写；默认参数被省略
directive_stringify('MACD:12,26')
# 'macd'

# 参数写成它的取值
directive_stringify('ma:05')
# 'ma:5'
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

下面列出的接口沿用对应的 `pandas` 用法；需要遵守[索引与选择](#索引与选择)中的严格标签约束，以及列表之后说明的 [NA 模型差异](#已知的-pandas-差异volasna-模型)。

```py
# --- DataFrame：元数据 ----------------------------------------------------
df.columns / df.shape / len(df) / df.dtypes      # dtypes -> dict
df.index                          # 行标签，作为一个 NumPy 数组（tz-aware 索引导出为 UTC 瞬时）
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
**两个带 tf 的 DataFrame**（见[累积](docs/zh-CN/cumulation.md#累积与-datetimeindex)），并把每根 bar 都
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
df.index            # 行标签，作为一个 NumPy 数组（tz-aware 索引导出为 UTC 瞬时）
```

字符串（代码）索引：在字符串列上 `set_index`，再按代码查找：

```py
df = DataFrame({'sym': ['aa', 'bb', 'cc'], 'px': [1.0, 2.0, 3.0]}).set_index('sym')
df.loc['bb']           # 键为 'bb' 的那一行
df.loc['aa':'bb']      # 闭区间、字典序的切片
df.at['cc', 'px']      # 3.0
df.drop(['bb'])        # 按字符串标签删除
```

标签切片按索引标签的顺序进行，与 pandas 一致。整数和字符串标签只要求唯一，因此可以是任意顺序：

- **递增的标签**：选出所有满足 `lo <= label <= hi` 的行；边界不必是索引中的标签；
- **递减的标签**：选出所有满足 `lo >= label >= hi` 的行；
- **其他任何顺序**：按精确标签定位每个边界，选出两个位置之间的行。边界若不是索引中的标签，会抛出 `KeyError`，因为按值比较会选中（在 `df.loc[lo:hi, col] = v` 中则是覆盖）所请求标签之外的行。

```py
df = DataFrame({'x': [20, 0, 10]}, index=[2, 0, 1])
df.loc[0:1]            # 标签为 0 和 1 的两行
df.loc[0:1, 'x'] = 99  # 只写这两行：x == [20, 99, 99]
df.loc[0:5]            # KeyError：5 不是这个无序索引中的标签
```

datetime 索引在所有构造入口都必须严格递增且不能含 NaT；重复、倒序或缺失的标签立即抛出 `ValueError`，不能先构造无效索引再靠 `Timestamp` 边界或排序修复。日期字符串可作为部分日期边界，按完整日期范围选择行。边界的标签类型与索引不符（整数索引配字符串边界、字符串索引配整数边界）时抛出 `TypeError`。

DataFrame 与 Series 的派生结果也遵守同一规则：重复选取同一行会抛出 `ValueError`；逆序切片、`sort_index(ascending=False)` 或按值排序若导致 datetime 标签不再严格递增，也会抛错，源对象保持不变。整数和字符串索引允许倒序及任意顺序，只要标签仍然唯一。

### 与 pandas 的差异（vs pandas）

volas 表面上是 pandas 风格，但它的**类型系统有意不同**，而且差异不只在索引上：缺失值保留原 dtype，没有 `object` dtype，返回值的方法仍返回 `Series`，有损转换会报错，而不是静默退化。**完整对比见
[volas vs pandas ——类型系统](PANDAS-DIFFERENCES.md)：为什么 volas 这样设计、
pandas 类型系统的问题在哪里，以及迁移时需要注意什么。**

具体到索引，volas 只支持**单层**、单一同质标签类型。相对于 pandas，volas **不**支持：

- **`MultiIndex`**（分层 / 多层索引），无论在行*还是*列上——列是一个由唯一字符串名组成的扁平列表。
- **任意标签 dtype**——索引只能是 range、datetime（`datetime64[ns]`）、整数或字符串之一。没有 float、categorical、interval、period、timedelta，也没有混合类型
  `object` 索引。
- **索引代数**——重索引（reindex）、索引集合运算（并 / 交），以及在合并 frame 时的自动按索引对齐。
- **重复标签**——不支持构造或派生重复标签索引，标签访问始终对应唯一一行。

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

`volas.tz_database()` 返回指定 key 首次使用时的数据来源和 IANA 版本；未使用的 key 会在查询时初始化。同一进程内，每个具名时区在首次初始化时确定规则，后续对象共享同一份规则。多主机部署应在 lockfile 中固定 `tzdata`，在首次使用前设置 `PYTHONTZPATH=""`，更新数据后重启进程；详见[时区数据与部署](docs/cumulation.md#reproducible-timezone-deployment)。

底层存储始终是 **UTC 纪元纳秒**——这是美股、港股和 A 股等股票市场共存时，用来按绝对瞬时对齐的统一轴。`DatetimeIndex` 还会携带一个**frame 级时区**，决定这些瞬时如何显示、裸字符串标签如何匹配，以及 `cumulate` 如何对齐日线及更粗周期。时区可以是**固定偏移**（如 `'+08:00'`，成本低，适合 A 股 / 港股），也可以是**具名 IANA 时区**（如 `'America/New_York'`，由 volas 读取当前 Python 的 TZif 数据并支持 DST，适合美股 / 欧洲）。默认是 UTC。

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

`cumulate` 到日线（或更粗）时，会把桶对齐到 frame 的本地交易日；如果使用具名时区，这个对齐具备 DST 意识。裸 `.index` 的 NumPy 导出仍保持 UTC（与 pandas `.values` 一致，NumPy 没有时区槽位）；行的 `.name` 与单元格是 aware `Timestamp`。读回的标签（如 `df.index[-1]`）可以原样用于 `append`：`np.datetime64` / 整数 ns 按帧自身的时区读作 epoch 瞬时。用 `index=df.index` 构造的帧是 naive 的（UTC 瞬时），不会与 aware 帧对齐，需要先恢复时区：`mask.tz_localize('UTC').tz_convert(zone)`。

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

[pandas、Arrow 与 DLPack 互操作](docs/zh-CN/interop.md#与-pandas-互操作)。

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

一个 directive 最多嵌套 128 层（每个括号组、`@(...)` series 组、一元运算符和二元运算符各算一层），最多包含 4096 个节点。真实的 directive 远低于这两个上限；超出时抛出 `DirectiveSyntaxError`，而不会耗尽栈。

Python 选择会检查布尔 Series 的索引对齐，并读取 NumPy MaskedArray 的有效性掩码；相等检查比较索引标签，拒绝混合无时区与有时区的 datetime 索引。构造器的 `columns=` 与 `df[[...]]` 共用指令投影事务；显式 NumPy dtype 的 `na_value` 在目标类型中填充。参见[选择规则](docs/pandas-compatibility.md#indexing--selection)与[互操作](docs/interop.md)。

## 内置指标
完整 directive 参考见 [INDICATORS.md](INDICATORS.md)，其中覆盖 Volas 独有指标、内置统计命令，以及与 TA-Lib 兼容的 directive。

Rust 调用方通过 `parse` 或 `bind_frame` 获得经过校验、不可变更内部表示的 `volas_directive::Ast`。通过 `node()` 与 `continuation()` 查看它；命令参数提供只读的 `values()`、`spec()` 与 `bare()` 访问器。执行、lookback 与续算 API 都接受 `&Ast`。binder 在绑定时一次性准备续算计划，包括其中的有状态子树及其隐藏输入列。`stringify` 接受 `&Ast` 或宽松的 `&Form`，使用同一个规范渲染器。底层 Rust 分组 `Column` 归约的分组迭代器须可重复遍历（`Clone`）；密集选择复用输入有限性事实，稀疏选择不会扫描整列。

# 参与与反馈

欢迎提 issue、指标请求、benchmark 挑战和 PR——参见
[CONTRIBUTING.md](CONTRIBUTING.md)，并通过 [issue](https://github.com/kaelzhang/volas/issues/new/choose) 选择合适的模板提交。最有价值的反馈是关于 API 设计和 [benchmark 方法论](docs/benchmark-faq.md)的。

如果你在 Python 里搭建实时 OHLCV / 技术指标流水线，欢迎 star 本仓库，关注新增指标、benchmark 结果与发布。

# 许可证

[MIT](LICENSE)

# 面向开发者
开发者说明、本地构建命令、依赖分组和 benchmark 报告指引都在
[DEVELOPMENT.md](DEVELOPMENT.md) 中。

Rust 归约接纳保存不可变位置切片或拥有型 Range，不能依赖任意 Clone 迭代器。`RowGroups::new` / `from_offsets` 一次验证分组数量、非空及每个位置；不可变凭证经 `reduce` / `reduce_with_spare` 可供同一已接纳高度的各列复用。

底层缺失占位增长仅供已接纳的 frame 写入内部使用；公开列扩展经 `Column::append` 追加同 dtype 的缺失列。
