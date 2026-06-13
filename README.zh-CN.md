[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

# [volas](https://github.com/kaelzhang/volas)

[English](README.md) | 简体中文

> 面向股票 / K 线（OHLCV）时间序列数据的高性能、Rust 内核列式引擎。

**volas** 是一个 Rust 内核、pandas 形状的 `DataFrame`，专为实时 OHLCV 流水线打造：[**242** 个交易指标](INDICATORS.md)、增量 O(lookback) 刷新，以及可直接喂给 NumPy / Torch 的输出。

在我们可复现的 benchmark 套件上，**volas** 在绝大多数实时更新（live-update）指标场景中都快于 pandas、polars、stock-pandas 和 TA-Lib。

## 为什么选 volas

- **pandas 的无缝替代。** 同样的 `.loc` / `.iloc` / `.at`、`read_csv`、
  `to_numpy` 和重采样——改个 import，代码照旧。（参见
  [哪些不在覆盖范围内](PANDAS-DIFFERENCES.md#index-limitations)）
- **同领域最快。** 在几乎每一个指标上都快于 pandas、polars 和 TA-Lib——
  以始终最新的 [实时 benchmark 报告](https://volas.ost.ai) 为准。
  - 在已发布报告的默认口径下，于 **137 / 158** 个覆盖指标上胜过 TA-Lib
    ——可通过 `make benchmark` 复现。
  - 每来一根新 bar 增量刷新指标——比 TA-Lib 快约 **~5×**，比 pandas
    最高快约 **~360×**。
- **为实时 tick 而生。** 一根新 bar 只触碰受影响的尾部
  （`O(lookback)`，而非 `O(n)`）；指标在微秒级刷新，绝不整列重算。
- **Rust 在内，NumPy / Torch 在外。** 编译型内核，运行时零 pandas 依赖；
  `to_numpy()` 直接喂给 NumPy 和 `torch.Tensor` 流水线。

### 什么时候该用 volas

volas **不是** 通用的 pandas 替代品——做普通的 dataframe 分析，请继续用
pandas 或 polars。它是一个窄而快的 DataFrame，专门服务于这样一个场景：
**一根新的 OHLCV bar 到来，指标必须立刻刷新**：

| | pandas | polars | TA-Lib | volas |
| --- | :---: | :---: | :---: | :---: |
| pandas 形状的索引（`.loc` / `.iloc` / `.at`） | ✅ | ❌ | ❌ | ✅ |
| OHLCV 原生的指标 directive（`df['rsi:14']`） | ❌ | ❌ | ✅ | ✅ |
| 由 frame 自身持有的指标缓存 | ❌ | ❌ | ❌ | ✅ |
| 新 bar 上的增量 `O(lookback)` 刷新 | ❌ | ❌ | ❌ | ✅ |
| Rust 内核、运行时无 pandas | ❌ | ✅ | C | ✅ |
| 导出到 NumPy / Torch | ✅ | ✅ | 数组 | ✅ |

## 目录
- [安装](#安装)
- [快速上手](#快速上手)
- [用法](#用法)
- [累积与 DatetimeIndex](#累积与-datetimeindex)
- [TimeFrame](#timeframe)
- [directive 的语法](#directive-的语法)
- [索引与选择](#索引与选择)
- [写入与赋值](#写入与赋值)
- [时区](#时区)
- [缺失值（`volas.NA`）](#缺失值volasna)
- [与 pandas 互操作](#与-pandas-互操作)
- [错误处理](#错误处理)
- [内置指标](#内置指标)
- [许可证](#许可证)
- [面向开发者](#面向开发者)

## 安装

```sh
pip install volas
```

要求 Python >= 3.11。已为 Linux（x86_64 / aarch64）、macOS（x86_64 / arm64）
和 Windows（x86_64）发布预编译 wheel。如需从源码本地构建，参见
[面向开发者](#面向开发者)。

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

# 一个布尔 directive -> bool Series，可作为行掩码（row mask）
bullish = df['close > open']
df[bullish]                      # close > open 的那些行组成的 DataFrame

# 一次性多个 directive -> DataFrame
df[['ma:2', 'ma:3', 'close > open']]

# 近乎零拷贝地导出到 NumPy
df['close'].to_numpy()           # 1 维 ndarray
df.to_numpy()                    # 2 维 ndarray（行 x 列）
```

## 用法

```py
from volas import (
    DataFrame, Series, read_csv, to_datetime, TimeFrame, Timestamp,
)
```

下面各小节按 volas 公共接口的顺序展开：先是 `DataFrame` 类，然后是它的实例
方法、静态方法，再到其他类，以及顶层包函数——最后收尾于其余那些**与 pandas
完全一致**、行为和 pandas 一模一样的 API。（从 `volas` 导入的顶层名字，例如
`read_csv`，书写时不带 `volas.` 前缀。）

### DataFrame(data, columns=None, time_frame=None, cumulators=None)

`DataFrame` 拥有**与 pandas 一致的 API**，所以如果你熟悉 `pandas.DataFrame`，
那你已经会用 volas 了。与 pandas 不同的是，volas 由 Rust 内核驱动，运行时不依赖
pandas。

```py
df = read_csv('stock.csv')
```

我们可以用 `[]`——称为 **pandas 索引**（即 python 里的 `__getitem__`）——来选出
更低维的切片。除了用 `colname`（`DataFrame` 的列名）索引外，我们还可以用
`directive` 来索引。

```py
df[directive]                  # 得到一个 Series

df[[directive0, directive1]]   # 得到一个 DataFrame
```

下面用一个例子展示用 `[directive]` 进行的最基本的索引

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

它得到的是列 `"close"` 上的 2 周期简单移动平均。

#### 参数

- **data** `dict[str, list | np.ndarray] | DataFrame` 列数据——一个把每个列名
  映射到等长 list 或 NumPy 数组（float、int、bool、`datetime64` 或字符串）的
  dict——**或者另一个 volas `DataFrame`，此时会被拷贝**（如同
  `pandas.DataFrame(df)`）。要附加一个
  [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)，
  用 `to_datetime` 解析某一列，用 `set_index` 把它提升为索引，再用 `tz_localize`
  / `tz_convert` 打上时区标记。参见 [时区](#时区)。
- **columns** `Optional[list[str]] = None` 选择并排列要保留的列——与 `df[[...]]`
  相同的投影。名字不存在会抛 `KeyError`；空 list 或重复名字会被拒绝，缺失的列
  绝不会被静默填充。
- **time_frame** `Optional[str | TimeFrame] = None` 若设置，则把它变成一个在该
  bar 间隔上**带 tf（累积）**的 DataFrame：给定的各行被视作该 frame 下已经定型
  的 bar，后续的 `append` 会把更细的 bar 折叠进正在形成中的 bar。需要一个
  `DatetimeIndex`。参见 [累积与 DatetimeIndex](#累积与-datetimeindex)。
- **cumulators** `Optional[dict[str, str]] = None` 折叠时使用的逐列聚合器覆盖
  （例如 `{'amount': 'sum'}`）；默认采用 OHLCV 语义（`open`=first、`high`=max、
  `low`=min、`close`=last、`volume`=sum；其余任何列为 `last`）。仅在与
  `time_frame` 同时使用时才有意义。

### df.exec(directive: str, create_column: bool = False) -> np.ndarray

执行给定的 directive，并按 directive 返回一个 numpy ndarray。

```py
df['ma:5']  # 返回一个 Series

df.exec('ma:5', create_column=True)  # 返回一个 numpy ndarray
```

```py
# 这只会计算，而不会在 dataframe 中创建新列
df.exec('ma:20')
```

`df[directive]` 与 `df.exec(directive)` 的区别在于
- 前者会为 `directive` 的结果创建一个新列作为缓存，供之后使用；而
  `df.exec(directive)` 不会，除非我们把参数 `create_column` 传为 `True`
- 前者接受其他 pandas 索引目标，而 `df.exec(directive)` 只接受一个合法的
  **volas** directive 字符串
- 前者返回一个 `Series` 或 `DataFrame` 对象，而后者返回一个
  [`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

### df.get_column(key: str) -> Series

按 `key` 直接取列值，返回一个 `Series`——并且**绝不计算**：与会把未知 key 解析
成指标 directive 并执行的 `df[key]` 不同，`get_column` 只取一个已存在的列，否则
抛 `KeyError`。当列名来自外部数据（CSV 表头、用户输入、配置）时务必用它，这样
一个碰巧长得像 directive 的名字（例如 `"ma:5"`）就绝不会静默触发一次计算。

如果给定的 `key` 是一个别名（alias），它会返回对应原始列的值。如果列没找到，
则抛出 `KeyError`。

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

### df.append(other: DataFrame | Row) -> DataFrame

把 `other`（一个 `DataFrame` 或一个 `Row`）的行就地追加到调用者的末尾，返回同一个
`DataFrame`，并尽可能地把 `DatetimeIndex` 应用到新追加的行上。当原 frame 必须保持
不变时，请先 `copy()`。

如果调用者是一个**带 tf**的 DataFrame（用 `time_frame` 构建的，或者 `cumulate`
的结果），`append` 会改为把每一根更细的 bar **折叠**进正在形成中的 bar，而不是
新增一行——参见
[实时累积](#实时累积--一个-tf-aware-dataframe)。

默认情况下，追加新行不会更新这些新行的指标列；它们会保持过期（stale），直到再次
被读取，或者直到调用 `df.fulfill()`（见下文）。

### df.cumulate(time_frame: TimeFrame | str, cumulators: dict | None = None) -> DataFrame

把数据 frame 累积（重采样）到一个更粗的 `time_frame`，返回一个新的 `DataFrame`。
需要一个 `DatetimeIndex`。

- **time_frame** `TimeFrame | str` 目标 bar 间隔，例如 `TimeFrame.m5` 或 `'5m'`。
  参见 [TimeFrame](#timeframe)。
- **cumulators?** `dict[str, str] | None = None` 逐列聚合器覆盖（例如
  `{'amount': 'sum'}`）；默认采用 OHLCV 语义（`open`=first、`high`=max、
  `low`=min、`close`=last、`volume`=sum；其余任何列为 `last`）。

```py
# 从 1 分钟 K 线到 5 分钟 K 线
five_minute = one_minute.cumulate('5m')
```

详见 [累积与 DatetimeIndex](#累积与-datetimeindex)。

### df.fulfill() -> None

兑现（fulfill）所有指标列。默认情况下，向 `DataFrame` 追加新行不会更新这些新行
的指标。

指标只在访问指标列或调用 `df.fulfill()` 时才更新。访问 `df[directive]` 只会增量
刷新受影响的尾部（`O(lookback)`，而非 `O(n)` 的整列重算）；对于批量读取
（`to_numpy()`、`.iloc`），调用一次 `fulfill()` 即可就地批量刷新每一个已缓存的
directive 列。

```py
df['ma:20']              # 把 20 周期 SMA 缓存成一列
df = df.append(new_bar)  # 新行的 ma:20 是过期的（一个缺失占位）
df.fulfill()             # 只重算每个已缓存列的尾部
df.to_numpy()            # 现在是新鲜的了
```

### df.alias(as_name: str, src_name: str) -> None

定义一个列别名。

- **as_name** `str` 别名
- **src_name** `str` 一个已存在列的名字

```py
# 有些绘图库（例如 `mplfinance`）要求一个首字母大写的 `Open` 列，
# 不过没关系，我们可以建一个别名。
df.alias('Open', 'open')
```

别名在每一处查找列的地方都会被解析，**包括在 directive 内部**，并且在 `drop` /
`copy` / 切片之后依然有效。

```py
df['Open']        # 与 df['open'] 同样的数据
df['ma:5@Open']   # 别名在 directive 内部也会被解析
```

### Series

`df[col]` 和 `df[directive]` 返回一个 `Series`——一个具名的 1 维列，其 API 与
pandas 一致：算术 / 比较 / 逻辑运算符、`.sum()` / `.mean()` / `.std()` / …、
`.shift()` / `.diff()` / `.fillna()`、`.iloc` / `.loc`、`.to_numpy()` /
`.to_list()`。完整列表见
[其余与 pandas 一致的 API](#其余与-pandas-一致的-api)。`Series` 没有公开的
构造函数——`Series` 总是通过索引一个 `DataFrame` 得到。

```py
s = df['close']
s.name                 # 'close'
(s - s.shift(1)).mean()
df['ma:5 > ma:20']     # 一个 directive 同样返回一个 Series（这里是 bool 的）
```

在 pandas 之外，`Series` 还把 TA-Lib 的 15 个 **Math Transform** 函数暴露为方法
——`acos` `asin` `atan` `ceil` `cos` `cosh` `exp` `floor` `ln` `log10` `sin`
`sinh` `sqrt` `tan` `tanh`：

```py
df['close'].ln()
df['high'].sqrt()
```

一个 `datetime64[ns]` Series 暴露 pandas 的 `.dt` 访问器：日历分量（`year`
`month` `day` `hour` `minute` `second` `microsecond` `nanosecond` `quarter`
`dayofweek` `dayofyear` `days_in_month`）、日历谓词（`is_month_start` …
`is_year_end`、`is_leap_year`）、名称（`day_name()` / `month_name()`）、格式化
（`strftime(fmt)`）、bar 对齐（`floor(freq)` / `ceil(freq)` / `round(freq)` /
`normalize()`），以及 `isocalendar()`。某个元素缺失时，每个分量都给出 `NA`：

```py
t = volas.to_datetime(df['time'])
t.dt.hour                  # int64 Series，0..23
t.dt.dayofweek             # 周一=0 .. 周日=6
t.dt.floor('15min')        # 对齐到 15 分钟 bar 的 datetime Series
```

### Row

`df.iloc[i]` 和 `df.loc[label]` 返回一个 `Row`——一条单独的记录，其 `.name` 是它
的索引标签。`Row` **没有公开的构造函数**（`Row(...)` 会抛
`TypeError: No constructor defined for Row`）；你只能通过索引一个 frame 来获得它，
并且可以把它传给 `df.append`。

```py
row = df.iloc[-1]      # 最新的一根 bar
row.name               # 它的索引标签（DatetimeIndex 下是一个 Timestamp）
row.to_dict()          # {列: 值}
row.to_numpy()         # 数值单元格组成的 1 维 ndarray
```

### 实时累积 — 一个 tf-aware DataFrame

对于**实时**流式场景，给一个 DataFrame 一个 `time_frame`，然后把更细的 bar
`append` 进去，而不是每个 tick 都重新累积整个 frame。`df.cumulate(tf)` 返回这样一个
frame（正在形成的周期保持 live），或者用 `DataFrame(data, time_frame=...,
cumulators=...)` 直接构建一个（给定的各行被视作该 frame 下已经定型的 bar；需要一个
DatetimeIndex）。

在一个带 tf 的 frame 上：

- **df.append(bar)** 把 bar 折叠进去：落在当前未收口周期内的一根会**更新正在形成
  的最后一行**（`df.iloc[-1]`）；落在新周期内的一根会翻滚成全新的一行；一根重新
  发来的、形成中的 bar（相同时间戳）是更新而非重复计入。
- **df.iloc[-1]** 是当前（仍未收口的）周期——live bar。
- **df[directive]** / **df.exec(directive)** 在包含形成中那一行的累积 frame 上
  计算指标——惰性的、读时计算：一次 `append` 只把它们标记为过期，下一次读取只重算
  尾部。
- **df.cumulate(target)** 必须是源 frame 的整数倍（例如 `5m→15m`，而非
  `5m→7m`；一周或 3 天的 bar 不会嵌套进月 / 年）；同一个 frame 是一次 `copy()`。

```py
df = history.cumulate('5m')   # 一个带 tf 的 5m frame（history 更细，例如 1m）
for bar in stream:            # 每个 `bar` 是一个更细的 DataFrame
    df.append(bar)            # 折叠进正在形成的 5m bar
    df.iloc[-1]               # live 的、仍在形成中的 bar
    df['macd']               # 在累积 frame 上计算的指标
```

详见 [累积与 DatetimeIndex](#累积与-datetimeindex)。

### read_csv(path, sep=',', header=True, parse_dates=None, index_col=None, na_values=None, keep_default_na=True, tz=None, date_unit=None) -> DataFrame

一个顶层函数，把一个 CSV 文件读成 `DataFrame`，逐列推断 dtype——一个快速的、
pandas 子集的 CSV 读取器。

- **path** `str` CSV 文件路径。
- **sep?** `str = ','` 字段分隔符（单个字符）；`delimiter` 是一个被接受的别名。
- **header?** `bool = True` `True`（或省略）把第一行当作表头；`False` / `None`
  表示无表头（列被命名为 `'0'`…`'n-1'`）。
- **parse_dates?** `list[str] | None = None` 要解析成 datetime 列的列名。
- **index_col?** `str | int | None = None` 要移入行索引的列名或整数位置；在
  `parse_dates` *之后*应用，所以指定一个已解析的日期列会产生一个 `DatetimeIndex`。
- **na_values?** `str | list[str] | None = None` 额外的缺失值标记。
- **keep_default_na?** `bool = True` 同时把默认的 NA 标记也当作缺失。
- **tz?** `str | None = None` `index_col` datetime 的时区：一个*naive*日期字符串
  按 `tz` 读入（存为 UTC，索引被打上标记）。接受固定偏移（`'+08:00'`）或一个 IANA
  名称（`'America/New_York'`）；把日期列通过 `index_col` 传入，并*不要*同时把它列在
  `parse_dates` 里。参见 [时区](#时区)。
- **date_unit?** `str | None = None` 把 `index_col` 当作此单位下的 epoch 整数读取
  （`'s'` / `'ms'` / `'us'` / `'ns'`，绝对 UTC）；此时 `tz` 只设置显示时区。

```py
from volas import read_csv

df = read_csv('klines.csv')                        # RangeIndex
df = read_csv('klines.csv',
              parse_dates=['time_key'],            # 解析成 datetime
              index_col='time_key')                # -> DatetimeIndex
df = read_csv('data.tsv', sep='\t', header=False,  # 无表头 -> '0'..'n-1'
              na_values=['NA', 'null'])
```

### from_pandas(pdf) -> DataFrame

一个顶层函数，把一个 `pandas.DataFrame`（`pdf`）桥接进 volas（而
`df.to_pandas()` 桥接回去）。参见 [与 pandas 互操作](#与-pandas-互操作)。

### to_datetime(obj, unit='ns', format=None) -> Series

一个顶层函数，把 epoch 数字或 datetime 字符串转换成一个 datetime `Series`，镜像
`pandas.to_datetime`。`obj` 可以是一个 `Series`、一个 1 维 NumPy 数组，或一个 list。
一个**缺失**的输入（一个 float `NaN`，或一个 int 列里的 `volas.NA`）会变成 `NaT`，
和 `pd.to_datetime` 一样。

- **obj** 要转换的值——数值 epoch、datetime 字符串，或一个已经是 datetime 的
  `Series`（原样返回）。
- **unit?** `str = 'ns'` 用于**数值**输入的 epoch 单位（`'s'` / `'ms'` / `'us'` /
  `'ns'`）；亚单位的小数部分会被保留，和 `pd.to_datetime` 一样。
- **format?** `str | None = None` 用于**字符串**输入的显式 datetime 格式（pandas
  的 `format=`，例如 `'%Y-%m-%d %H:%M:%S'`）——更快且无歧义；对数值输入忽略。

Naive 字符串按 UTC 解析，带偏移（`…+08:00`）的字符串是绝对的。要把结果索引*显示*
在某个时区里，把它设为索引，并用 `tz_localize` / `tz_convert` 打上时区标记（参见
[时区](#时区)）。

```py
from volas import to_datetime

# 把一个 epoch 秒的列解析成 datetime，再把它设为索引
df['time'] = to_datetime(df['time'], unit='s')
df = df.set_index('time')                       # -> DatetimeIndex
df = df.tz_localize('America/New_York')         # 打上显示时区标记（见“时区”）
```

如果想做一次就地的、**截断式**的 cast（NumPy / pandas 的 `astype` 惯用法），
请改用 `df.astype({'time': 'datetime64[s]'})`。

### directive_stringify(directive: str) -> str

得到一个 `directive` 的规范全名——volas 实际把它缓存所用的列名。命令名被小写化，
默认参数 / series 被丢弃以节省空间。

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

得到一个 `directive` 的回看（lookback）周期——在指标产出一个有效结果之前所需的
最少先前数据点数量。

```py
from volas import directive_lookback

directive_lookback('ma:20')
# 19

directive_lookback('boll')
# 19（默认周期 20）

# 复合 directive：lookback 在嵌套表达式间累加。
# repeat:5 需要额外 4 个点，boll.upper（周期 20）需要 19 -> 23
directive_lookback('repeat:5@(close > boll.upper)')
# 23
```

### 其余与 pandas 一致的 API

下面的一切都和它对应的 `pandas` 行为一致——如果你在 pandas 里会用它，在 volas 里
用法相同，除了列表之后所标注的那些刻意的
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
df.copy() / df.to_numpy(dtype=None) / df.equals(other) / df.to_csv(path=None, ...)

# --- DataFrame：写入 ------------------------------------------------------
df[col] = scalar | array | Series          # 增加 / 替换一列（按位置）
df.loc[mask, col] = value ; df.iloc[i, j] = value ; df.at[label, col] = value

# --- Series ---------------------------------------------------------------
s.name / s.dtype / len(s) / s.tz / s.index
s.to_numpy(dtype=None) / s.to_list()
s.iloc[...] / s.loc[...]
s + s, s - 1, -s, ...             # 逐元素算术
s > 0, s == t, s != t, ...        # 比较 -> bool Series
s & t, s | t, ~s, s ^ t           # 逻辑 -> bool Series
s.sum() / s.mean() / s.min() / s.max() / s.std() / s.var() / s.median()   # 跳过缺失
s.shift(n=1) / s.diff(n=1) / s.fillna(v) / s.ffill() / s.bfill()           # 见“缺失值”：NA 保持 dtype
s.isna() / s.notna() / s.dropna() / s.equals(t)
```

#### 窗口运算（`rolling` / `expanding` / `ewm`）—— 仅为兼容性而存在

> **这套接口的存在，是为了让 pandas 的研究 / 标注代码能原样迁移过来。它
> 不是计算指标的推荐方式，更不应被用在实盘交易系统里**：窗口结果是一个普通
> Series——它不加入 directive 缓存，也**不**会被 `append()` / `fulfill()` 增量
> 刷新；每来一根新 bar 都要付出一次完整的 `O(n)` 重算。请优先用等价的 directive
> （`df['ma:20']`、`df['median:30']`、`df['stddev:20']`、…）：同样的内核，外加缓存
> 和 `O(lookback)` 的逐 bar 刷新。

```py
s.rolling(window, min_periods=None, center=False)   # int 窗口；min_periods 默认等于 window
s.expanding(min_periods=1)
s.ewm(com=|span=|halflife=|alpha=, min_periods=0, adjust=True, ignore_na=False)
                                                    # 恰好一种衰减写法
# Rolling / Expanding（pandas 语义：跳过 NA，min_periods 设门槛）：
.count() .nunique()                                 # -> int64 Series（原生 NA）
.sum() .mean() .median() .min() .max()
.var(ddof=1) .std(ddof=1) .sem(ddof=1) .skew() .kurt()
.quantile(q, interpolation='linear') .rank(method='average', ascending=True, pct=False)
.first() .last()                                    # 保持 dtype
.corr(other) .cov(other, ddof=1)

# Ewm:
.mean() .sum() .var(bias=False) .std(bias=False) .corr(other) .cov(other, bias=False)
```

`center=True` 把每个窗口标在它的中心——它会读取相对于标签的**未来** bar。这正是
一次标注过程（labeling pass）想要的，也正是一个实盘信号绝不能做的；这里为前者
提供支持。

基于时间的窗口（`rolling('5min')` / 一个 `timedelta`）刻意没有实现。对于多时间框
计算，请维护**两个带 tf 的 DataFrame**（见 [累积](#累积与-datetimeindex)），并把
每根 bar `append` 到两者上——这才是受支持的、逐 bar `O(lookback)` 的设计；用窗口
算术去模拟一个更粗的时间框，会在每根 bar 上重算一切。

未提供（与 volas 模型冲突的 pandas 成员）：`apply` / `agg` / `pipe`
（任意 Python-每窗口）、`win_type`、`step`、`on`、`closed`、`method`、
`ewm(times=...)`、`ewm.online()`——`append()` + directive 已经覆盖了流式用例。

#### 已知的 pandas 差异（`volas.NA` 模型）

少数几个 API **刻意**偏离 pandas，因为 volas 把缺失值原生地存储为
[`volas.NA`](#缺失值volasna)（没有 `object` dtype，没有静默的 float 提升）：

- **`shift` / `diff` / `fillna` 及同类**保持列的 dtype——一个缺失值是
  `volas.NA`，而不是把一个 int/bool/str 列提升到 float/object。
- **比较**（`==` `!=` `<` `<=` `>` `>=`）返回一个*非空*的 bool 掩码：一个缺失值
  比较为 `False`（而 `!=` 比较为 `True`），遵循 IEEE / NumPy——而非 pandas-nullable
  的三值 `NA`。这让掩码不含 `NA`，从而 `df[mask]` 和赋值保持完整（total）。
- **`to_numpy()`** 把一个缺失单元格导出为 `NaN`（NumPy 没有 `NA`），所以一个
  int / bool / datetime 列会物化为 `float64` / `NaT`。存储和 `to_list()` 保持
  dtype 与 `volas.NA`。

要看完整图景——volas 的类型系统为什么这样建、pandas 的在哪里崩坏、以及迁移时的
坑——参见 [volas vs pandas —— 类型系统](PANDAS-DIFFERENCES.md)。

pandas 形状的索引和写入细节有它们各自的小节——
[索引与选择](#索引与选择) 和
[写入与赋值](#写入与赋值)。

## 累积与 DatetimeIndex

假设我们有一个 csv 文件，包含某只股票在 1 分钟时间框上的 K 线数据：

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

> 注意：同一时间戳的重复记录不会被累积。除最新的一条外，其余全部被丢弃。

读取同一个 csv，但把 `date` 列解析成一个 `DatetimeIndex`：

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

你想必已经看出来，这个数据 frame 现在有了一个
[`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)。

但它不会变成 5 分钟 K 线，除非我们把它累积：

```py
df_5m = df.cumulate('5m')

print(df_5m)
```

现在我们得到一个 5 分钟 K 线：

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
2020-01-01 00:05:00  325.0  327.8  316.2  322.0  82176419.0
2020-01-01 00:10:00  323.0  327.8  314.6  327.6  74409815.0
2020-01-01 00:15:00  330.0  335.2  322.0  323.0  82452902.0
```

`cumulate` 默认采用 OHLCV 语义——`open`=first、`high`=max、`low`=min、
`close`=last、`volume`=sum——而**其余任何列回退为 `last`**。传 `cumulators=` 来覆盖
某列的聚合器；最常见的情况是一个本应被求和、却会默认成 `last` 的非 OHLCV 列，例如
一个成交额（`amount`）列：

```py
df.cumulate('1h', cumulators={'amount': 'sum'})
```

支持的聚合器有 `first`、`max`、`min`、`last` 和 `sum`。

`time_frame` 可以是一个字符串标签或一个 `TimeFrame` 常量——完整列表见
[TimeFrame](#timeframe)。

#### bar 标签是周期起点

每个时间框都落在一个**固定的网格**上，一根被累积的 bar 以它所在周期的网格**起点**
为标签——即使第一根原始 bar 在周期中途才到来。一根在 `15m` 框上以 `09:07` tick 开盘
的 bar 被标为 `09:00`，绝不会是 `09:07`，所以 volas 的 bar 与交易所 K 线、以及
pandas `resample`（`label='left'`）完全对齐。

各时间框的网格原点：日内时间框锚定在该索引（带时区的）交易日的**午夜**——一根 `15m`
bar 起于 `:00`/`:15`/`:30`/`:45`，一根 `4h` bar 起于 `00:00`/`04:00`/…；`1d` 起于
午夜；`1w` 起于**周一**；`3d` 是从 Unix 纪元开始的连续网格；`1M` / `1y` 起于日历的
月 / 年。如果一次夏令时切换移除或重复了某个周期的边界，标签会解析到该周期最早的
真实瞬时。

对于**实时**流式场景，你不会在每个 tick 上都重新累积整段历史——你会让当前的 5 分钟
bar *正在形成*，并随着每根更细的 bar 到来而更新它。一个**带 tf 的 DataFrame** 做的
正是这件事：它仍然是一个普通的 DataFrame（读列、跑 directive、切片它），只是
`append` 会把每根更细的 bar **折叠**进正在形成的那根 bar，而不是新增一行。你用
`df.cumulate('5m')` 或 `DataFrame(data, time_frame='5m')` 造一个，那么实时循环就只是：

| 步骤                           | 调用                      |
| ------------------------------ | ------------------------- |
| 造一个 `5m` frame              | `cum = df.cumulate('5m')` |
| 喂给它下一根更细的 bar         | `cum.append(bar)`         |
| 读当前正在形成的 bar           | `cum.iloc[-1]`            |
| 在它之上读一个指标             | `cum['macd']`             |

#### 看着正在形成的 bar 长大

从上面的 1 分钟 `df` 出发，一根一根地构建 5 分钟 frame。先用 `00:00` bar 播种，再
折叠进 `00:01`。两者都落在同一个 `00:00`–`00:05` 窗口里，所以 frame 仍然只持有**一**
行——那根正在形成的 bar——现在被更新了（`high` 涨到 `332.0`，`close` 到 `331.0`，
`volume` 求和）：

```py
cum = df.iloc[0:1].cumulate('5m')   # 用 00:00 bar 播种这个 5m frame
cum.append(df.iloc[1:2])            # 折叠进 00:01（同一个 5m 窗口）

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  332.0  327.6  331.0  28155710.0
```

折叠进 `00:02`、`00:03` 和 `00:04`，窗口就填满了。那一行正在形成的行现在就是**已
完成**的第一根 5 分钟 bar——和之前一次性 `df.cumulate('5m')` 打印出来的第一行完全
一致：

```py
for i in range(2, 5):
    cum.append(df.iloc[i:i + 1])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
```

现在折叠进 `00:05`。它开启了**下一个**窗口，于是 `00:00` bar 被定型，一根全新的、
正在形成的 bar 开始；frame 增长到两行，而 `cum.iloc[-1]` 是那根新的、仍在形成的
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

有两个性质让这对实时数据流是安全的：

- **指标是惰性的，且读时新鲜。** `append` 不重算任何东西——它只把依赖的 directive
  列标记为过期（它们的有效行游标现在落后于 frame 高度）。重算发生在你**读取**
  `cum['ema:9']`（或任何 directive）时：只刷新过期的尾部——`O(lookback)`，而非整列
  ——在包含正在形成那一行的 frame 上进行，与一次性「先累积再计算」逐位相同。（像
  `to_numpy()` 这样的批量读取不会自动刷新；先调用 `cum.fulfill()`，或者干脆读一下
  directive。）
- **重发的 bar 不会重复计入。** 折叠一根你已经见过其时间戳的 bar，是**更新**那个
  周期而不是累加到它上面——就是本节开头展示的同一条去重规则——这与那些会修订自己
  最近一根 bar 的交易所相吻合。

API 概览见 [实时累积](#实时累积--一个-tf-aware-dataframe)。

## TimeFrame

一个 `TimeFrame` 命名一个 bar 间隔。在 volas 进行重采样的任何地方它都被接受——
`df.cumulate`、`time_frame` DataFrame 参数，以及 `hv` 指标——既可以是一个 `TimeFrame`
常量，也可以是它等价的**字符串标签**。没有 `TimeFrame(...)` 构造函数——请使用下面的
某个常量或一个标签字符串。

```py
TimeFrame.m5            # 5 分钟框
'5m'                    # 等价的标签字符串，到处也都接受

df.cumulate(TimeFrame.m5)     # 与 df.cumulate('5m') 相同
```

支持的时间框（常量 ⇄ 标签）：

| 常量 | 标签 | 对齐 |
| --- | --- | --- |
| `TimeFrame.s1` | `'1s'` | 民用秒。 |
| `TimeFrame.m1` | `'1m'` | 民用分钟。 |
| `TimeFrame.m3` | `'3m'` | 分钟-时桶，起于 `00`、`03`、`06`、... |
| `TimeFrame.m5` | `'5m'` | 分钟-时桶，起于 `00`、`05`、`10`、... |
| `TimeFrame.m15` | `'15m'` | 分钟-时桶，起于 `00`、`15`、`30`、`45`。 |
| `TimeFrame.m30` | `'30m'` | 分钟-时桶，起于 `00` 和 `30`。 |
| `TimeFrame.H1` | `'1h'` | 民用小时。 |
| `TimeFrame.H2` | `'2h'` | 小时-日桶，起于 `00`、`02`、`04`、... |
| `TimeFrame.H4` | `'4h'` | 小时-日桶，起于 `00`、`04`、`08`、... |
| `TimeFrame.H6` | `'6h'` | 小时-日桶，起于 `00`、`06`、`12`、`18`。 |
| `TimeFrame.H8` | `'8h'` | 小时-日桶，起于 `00`、`08`、`16`。 |
| `TimeFrame.H12` | `'12h'` | 小时-日桶，起于 `00` 和 `12`。 |
| `TimeFrame.D1` | `'1d'` | 框时区下的民用日。 |
| `TimeFrame.D3` | `'3d'` | 锚定到 Unix 纪元的连续 3 天桶；不在月边界处重置。 |
| `TimeFrame.W1` | `'1w'` | 连续的周一起始周，含跨越月边界的连续段。 |
| `TimeFrame.M1` | `'1M'` | 框时区下的民用日历月。 |
| `TimeFrame.Y1` | `'1y'` | 框时区下的民用日历年。 |

每个桶都在**框时区的本地挂钟时间**上对齐，而存储保持 UTC：小时-日时间框
（`2h`/`4h`/`6h`/`8h`/`12h`）起于本地 `00` 并按本地小时步进；`3d` 计算从该时区下的
Unix 纪元日开始的连续 3 个本地民用日的桶（不在月边界处重置）；`1w` 在本地民用时间下
以周一起始。于是一根日 / 周 bar 跟随本地交易日，而一个具名时区让小时桶具备 DST
意识。

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

这里列出几种列名用例

```py
# 布林带的中轨
#   它其实是一个 20 周期（默认）的移动平均
df['boll']

# kdj j 小于 0
# 这返回一个 bool 类型的 series
df['kdj.j < 0']

# kdj %K 上穿 kdj %D
df['kdj.k // kdj.d']

# 5 周期简单移动平均
df['ma:5']

# 在（@）open 价上的 10 周期简单移动平均
df['ma:10@open']

# 一个由 5、10、30 周期 ma 组成的 DataFrame
df[[
    'ma:5',
    'ma:10',
    'ma:30'
]]

# 意思是我们用第一和第二个参数的默认值，
# 只指定第三个参数（给 macd.signal）
df['macd.signal:,,10']

# 当一个参数是嵌套的命令或 directive 时，我们必须把它括起来
df['increase:3@(ma:20@close)']

# volas 有一个强大的 directive 解析器，
# 所以我们甚至可以这样写 directive：
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

- `//` —— `left` 是否**上穿** `right`（从下方穿到上方），我们称之为「金叉」：
  `df['macd // macd.signal']`。
- `\\` —— `left` 是否**下穿** `right`，即「死叉」。在 Python 字符串里反斜杠必须
  转义，所以我们写 `'macd \\ macd.signal'`。
- `><` —— `left` 是否穿越 `right`，向上或向下都算。
- `<` `<=` `==` `!=` `>=` `>` —— 对同一条记录，`left` 与 `right` 之间的值比较，
  返回一个 `bool` series。
- 算术 `+ - * /`、逻辑 `& | ^`，以及一元 `~`（非）/ `-`（取负）。

`df[directive]` 把结果**缓存**成一个真正的列（所以重复读取是免费的），然后在
`append` 之后访问时自动刷新它过期的尾部。用 `df.exec(directive)` 来把一个 directive
计算成一个 NumPy 数组而**不**缓存它（见 [用法](#用法)）。

## 索引与选择

一个与 pandas 一致的、用于标签和位置访问的子集。行索引可以是一个 range、一个
`DatetimeIndex`、一个整数索引，或一个**字符串索引**。

```py
df.iloc[2]          # 按位置取一个 Row（row.name 是它的索引标签）
df.iloc[10:]        # 按位置切出一个 DataFrame
df.loc[label]       # 按索引标签取一个 Row
df.loc[lo:hi]       # 闭区间标签切片（字符串索引按字典序）
df.at[label, col]   # 按标签 + 列取一个标量
df.iat[i, j]        # 按位置取一个标量
df.index            # 行标签，作为一个 NumPy 数组
```

字符串（代码）索引——在一个字符串列上 `set_index`，然后按代码查找：

```py
df = DataFrame({'sym': ['aa', 'bb', 'cc'], 'px': [1.0, 2.0, 3.0]}).set_index('sym')
df.loc['bb']           # 键为 'bb' 的那一行
df.loc['aa':'bb']      # 闭区间、字典序的切片
df.at['cc', 'px']      # 3.0
df.drop(['bb'])        # 按字符串标签删除
```

### 与 pandas 的差异（vs pandas）

volas 在表面上是 pandas 形状的，但它的**类型系统刻意不同**，且不止于索引：缺失值
保持它们的 dtype，没有 `object` dtype，返回值的方法仍是 `Series`，而一次有损转换会
报错而不是静默退化。**完整对比见
[volas vs pandas —— 类型系统](PANDAS-DIFFERENCES.md)——volas 为什么这样建、
pandas 的类型系统在哪里崩坏，以及迁移时的坑。**

索引本身具体来说是一个**单层**、单一同质标签类型。相对于 pandas，volas **不**支持：

- **`MultiIndex`**（分层 / 多层索引），无论在行*还是*列上——列是一个由唯一字符串名
  组成的扁平列表。
- **任意标签 dtype**——一个索引恰好是 range、datetime（`datetime64[ns]`）、整数或
  字符串之一。没有 float、categorical、interval、period、timedelta，或混合类型的
  `object` 索引。
- **索引代数**——重索引（reindex）、索引集合运算（并 / 交），以及在合并 frame 时的
  自动按索引对齐。
- **重复标签**的查找（标签访问假定标签唯一）。

如果你的工作流需要这些中的任何一个，请继续用 pandas；volas 瞄准的是 K 线数据所用的
那种单层、OHLCV 形状的索引。

## 写入与赋值

赋整列，或写入一个位置 / 标签 / 布尔的选择（底层是写时复制，copy-on-write）。Series
赋值是**按位置**的（按行序，而非按索引对齐）。

```py
df['signal'] = 0.0                      # 增加 / 替换一列（scalar | array | Series）
df.iat[3, 0] = 99.0                     # 按位置赋一个单元格
df.at[label, 'close'] = 99.0            # 按标签 + 列赋一个单元格
df.iloc[10:20, 0] = 0.0                 # 一段列切片
df.loc[df['close'] > df['open'], 'signal'] = 1.0   # 掩码列赋值
```

把一个小数值写进一个整数列会**报错**——int dtype 被保留，一次有损写入会报错而不是
静默地拓宽到 float（见
[与 pandas 的差异](PANDAS-DIFFERENCES.md)；写入 `volas.NA` / `None` 会保持 int
dtype 并把该单元格标记为缺失）。写入一个已缓存的 directive 列会去掉它的缓存状态，
所以之后的 `fulfill()` 绝不会静默覆盖你的修改。

## 时区

存储永远是 **UTC 纪元纳秒**——这是 crypto、美股、港股和 A 股各框共存、并在绝对瞬时
上对齐的那条通用轴。一个 `DatetimeIndex` 额外携带一个**逐 frame 的时区**，它支配着
那些瞬时如何渲染、裸字符串标签如何匹配，以及 `cumulate` 如何对齐日及更粗的桶。一个
时区要么是一个**固定偏移**（`'+08:00'`，便宜；crypto / A 股 / 港股），要么是一个
**具名 IANA 时区**（`'America/New_York'`，通过 `chrono-tz` 具备 DST 意识；美股 / 欧
洲）。默认是 UTC。

这里是整幅图景。用 `to_datetime` 解析一列来构建一个 `DatetimeIndex`，用 `set_index`
把它提升为索引，再用 `tz_localize`（把一个 naive 挂钟*重新解释为*那个时区——瞬时会
移动）或 `tz_convert`（保持瞬时，重述时区）打上显示时区标记。一个美国交易所在
2021-01-04 当地 09:30 开盘，以一个 naive 本地字符串持有：

```py
from volas import DataFrame, to_datetime, Timestamp

# 把 naive 的 't' 字符串解析成 UTC 瞬时并设为索引，然后用 tz_localize 把这个挂钟
# 读作 *纽约本地时间*。瞬时存为 UTC（14:30Z），但索引在纽约渲染和匹配。
df = DataFrame({'t': ['2021-01-04 09:30:00'], 'close': [100.0]})
df['t'] = to_datetime(df['t'])
df = df.set_index('t').tz_localize('America/New_York')
df.tz       # 'America/New_York'
df.index    # ['2021-01-04T14:30:00.000000000']  （裸 .index 是 UTC，与 pandas .values 一致）

# tz 正是让一个裸本地字符串匹配到正确那一行的东西——它在 df.tz 里被解析：
df.at['2021-01-04 09:30:00', 'close']   # 100.0

# 一个 Timestamp 是一个有类型的、跨时区的标签。同一个瞬时在上海是
# 22:30+08:00，它仍然匹配，无论 df.tz 是什么：
ts = Timestamp('2021-01-04 22:30:00', tz='+08:00')   # == 纽约 09:30
df.at[ts, 'close']                       # 100.0
ts.value                                 # 它的 UTC 纪元纳秒（int）
ts.tz                                    # '+08:00'

# 整数 epoch：to_datetime(unit=...) 读取单位。一个 epoch 是 *绝对* 的——
# 把它锚为 UTC，再为显示重述时区。1609770600000 ms == 14:30Z：
e = DataFrame({'t': [1609770600000], 'close': [100.0]})
e['t'] = to_datetime(e['t'], unit='ms')
e.set_index('t').tz_localize('UTC').tz_convert('America/New_York').index
# ['2021-01-04T14:30:00.000000000']

# 一个带偏移的字符串本身也已经是绝对的了——to_datetime 解析其偏移：
o = DataFrame({'t': ['2021-01-04T09:30:00+08:00'], 'close': [1.0]})
o['t'] = to_datetime(o['t'])
o.set_index('t').index
# ['2021-01-04T01:30:00.000000000']  （09:30+08:00 == 01:30Z）
```

一个 frame 的时间轴处于两种状态之一（pandas 模型）：**naive**（一个未锚定的挂钟，
`df.tz is None`）或 **tz-aware**（已锚定，`df.tz` 命名时区——`'UTC'` 也算）。
`tz_localize` 锚定一个 naive 轴（瞬时移动以匹配那个时区里的挂钟）；`tz_convert`
把一个 aware 轴在另一个时区里重述（瞬时不变）。两者都拒绝对方的状态——转换一个未锚定
的挂钟、或者重新锚定一个已锚定的轴，都会静默地移动瞬时：

```py
naive = df                                   # df.tz is None
aware = naive.tz_localize('America/New_York')   # 锚定：瞬时移动，挂钟保持
aware.tz_convert('+08:00')                   # 重述：瞬时保持，挂钟移动
naive.tz_convert('+08:00')                   # TypeError —— 先用 tz_localize 锚定
aware.tz_localize('UTC')                     # TypeError —— 已锚定；请用 tz_convert
```

`cumulate` 到一个日（或更粗）的 bar 时，会把桶对齐到框的本地交易日——对一个具名
时区是 DST 感知的——而裸 `.index` 的 numpy 导出仍保持 UTC（与 pandas `.values` 一致）。

## 缺失值（`volas.NA`）

`volas.NA` 是唯一的缺失值标记，而且**每一种 dtype 都支持它**——关键在于，一个缺失值
**绝不改变列的 dtype**：

| dtype | 缺失如何存储 | 元素访问 | 控制台显示 |
|---|---|---|---|
| `float64` / `float32` | `NaN`，带内（in-band） | `np.float64(nan)` | `<NA>` |
| `int64` / `int32` / `bool` / `str` | 一个有效性掩码（dtype 保持） | `volas.NA` | `<NA>` |
| `datetime64[ns]` | `NaT` | `np.datetime64('NaT')` | `<NA>` |

无论底层如何存储，**控制台永远打印 `<NA>`**——对一个缺失值只有一个符号，与 dtype
无关（一个 float `NaN`、一个 datetime `NaT`，以及一个 int / bool / str 的空洞，渲染
得一模一样；`to_string(na_rep=...)` 可覆盖它）。元素访问和 `to_numpy` 仍是 dtype 特定
的（一个 float 空洞读回为 `np.nan`），所以与 numpy / pandas 的互操作是无损的。

这与 pandas 自身的方向（[PDEP-16]）一致，并意味着 volas **没有 `object` dtype**：一个
带空洞的 `int` / `bool` / `str` 列仍然是 `int` / `bool` / `str`，而 pandas 3.0 会把它
提升到 `float64` / `object`。

```py
import volas
s = volas.DataFrame({'a': [1, None, 3]})['a']
s.dtype                  # 'int64'        （pandas 会给 float64）
s[1]                     # <NA>           （s[1] 是 volas.NA；一个 float 空洞仍是 np.nan）
s.sum()                  # np.int64(4)    归约跳过 NA
s.fillna(0).to_list()    # [1, 0, 3]
s.isna().to_numpy()      # [False, True, False]
print(s)                 # 缺失单元格打印为 <NA>

# shift / diff 保持 int dtype（pandas 提升到 float）；空缺是 NA：
volas.DataFrame({'a': [10, 20, 30]})['a'].shift(1).to_list()   # [<NA>, 10, 20]
```

- **产生 NA** —— 构造函数 list 里的 `None`（或 `volas.NA`）、`shift` / `diff` 的
  空缺，以及 `where` / `mask` 的默认填充。
- **消费 NA** —— 归约（`sum` / `mean` / `min` / …）和 `count` 跳过它；算术传播它
  （`x ∘ NA = NA`）；`~` / `&` / `|` / `^` 用 Kleene 三值逻辑（`NA & False = False`、
  `NA | True = True`）；`cumsum` / `abs` / `round` / `clip` / 索引把它带过去；
  `isna` / `notna` / `dropna` / `fillna` / `ffill` / `bfill` 在每一种 dtype 上都有效。
- **比较**以 IEEE / numpy 的方式对待一个缺失值：涉及 NA 的 `==`、`<`、`<=`、`>`、
  `>=` 比较为 `False`，而 `!=` 比较为 `True`——所以一个布尔掩码永远是纯 `bool`，对
  `df[mask]` 是干净的。注意 `!=` 这个例外：`s != value` 因此会*包含*缺失行。

[PDEP-16]: https://github.com/pandas-dev/pandas/pull/58988

## 与 pandas 互操作

pandas **不是**一个运行时依赖；这些桥接惰性地 import 它，只在被调用时——所以
`import volas` 保持无 pandas。

```py
from volas import from_pandas

df = from_pandas(pandas_df)        # numeric / bool / str / datetime 原生；一个（带时区的）DatetimeIndex 往返无损；
                                   # 一个 nullable 的 Int64 / boolean / string 列读回为 int / bool / str + volas.NA
pdf = df.to_pandas()               # -> pandas.DataFrame（'numpy' 后端：一个带 NA 的 int/bool 列变成 float64 + NaN）
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # 忠实的 masked Int64 / boolean（一次无损的 NA 往返）
df.to_csv('out.csv', index=True)   # pandas to_csv 的一个子集；path=None 时返回一个 str
```

## 错误处理

directive 的问题会抛出有类型的异常。两者都同时继承自 `DirectiveError` 和内置的
`ValueError`，所以已有的 `except ValueError` 处理仍然有效。

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

完整的 directive 参考见 [INDICATORS.md](INDICATORS.md)。它覆盖 Volas 独有的指标、
内置的统计命令，以及与 TA-Lib 一致的 directive。

# 许可证

[MIT](LICENSE)

# 面向开发者

开发者说明、本地构建命令、依赖分组，以及 benchmark 报告指引，都在
[DEVELOPMENT.md](DEVELOPMENT.md) 中。
