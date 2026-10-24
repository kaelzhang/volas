# pandas、NumPy、Arrow 与 DLPack 互操作

本指南说明 DataFrame / Series 的构造和导出，以及 Row 的边界表示。选择和写入时会校验布尔 Series 键的索引；MaskedArray 的未知布尔信号会抛错。相等检查包含索引，并拒绝混合无时区与有时区的 datetime 索引。返回[中文 README](../../README.zh-CN.md)。

显式指定 NumPy dtype 时，仅转换有效槽，`na_value` 在目标数组中填充；缺失载荷不触发转换错误或警告，全缺失列也如此。例如 `DataFrame({'qty': [1, None]})['qty'].to_numpy(dtype='float64', na_value=0.5)` 返回 `[1.0, 0.5]`。原生整数直接转换，`dtype='int64', na_value=0` 保留超过 2⁵³ 的整数精度；NumPy 调用失败会把原异常传播给调用方。Frame、Series 和 `exec` 的未定宽文本 dtype（`U` / `str` / `S`）由有效值与填充值共同推断宽度；`U2` 等显式宽度保留用户指定宽度，遵循 NumPy 的截断行为。

### DataFrame.from_pandas(pdf) -> DataFrame

**volas 独有的静态方法**——从 `pandas.DataFrame`（`pdf`）构造 volas `DataFrame`，是 `df.to_pandas()` 的逆操作。pandas 仅在此处惰性 import（所以 `import volas` 保持 pandas-free）。nullable 列保留其 dtype + `volas.NA`，`DatetimeIndex`（含带时区）可往返。索引始终作为索引、保留自己的名字（无名索引保持无名），与列相互独立——与索引同名的列照样作为列保留。参见[与 pandas 互操作](#与-pandas-互操作)。

```py
df = DataFrame.from_pandas(pandas_df)   # pandas.DataFrame -> volas DataFrame
```

### DataFrame.from_arrow(data, copy=False) -> DataFrame

**volas 独有的静态方法**——从任何实现 Arrow **C-Stream** 协议（`__arrow_c_stream__`）的对象构造 `DataFrame`：`pyarrow.Table` / `RecordBatch` / `RecordBatchReader`、polars `DataFrame` 等。默认 `copy=False` 借用匹配的数值与密集 ns-datetime 缓冲区，结果带一个全新的 `RangeIndex`；`copy=True` 创建 owned 快照，多 chunk 来源直接写入最终缓冲区。Arrow 数据不可变；在任何导入对象或视图存活期间，生产方及所有别名都不得再写借用内存。需转换的布局仍会复制。生产方的缓冲区在读取前先校验：格式错误的数组（缓冲区大小不符、offsets 越界或非单调、非空字符串不是合法 UTF-8）抛出 `ValueError`。Arrow 中 null 字符串槽位的字节未作规定，volas 从不读取它们。

- **data** Arrow 来源——任何实现 `__arrow_c_stream__` 的对象。

```py
df = DataFrame.from_arrow(pa_table)        # pyarrow.Table     -> DataFrame
df = DataFrame.from_arrow(polars_df)       # polars.DataFrame  -> DataFrame
```

> `DataFrame(data=...)` **构造函数**不接受 Arrow（它只接受 `dict` 或另一个 `DataFrame`）；从 Arrow 对象构造请用 `from_arrow`。

### df.to_numpy(dtype=None, na_value=...) -> np.ndarray

把 DataFrame 导出为二维 NumPy 数组（行 × 列）。行为与 pandas 一致，但有一处刻意的保护：对含缺失值的 frame 指定**整型**或**布尔** `dtype` 会**抛错**，而不是悄悄写入垃圾值（NumPy 无法在整型或布尔数组里存 `NA`）——传 `na_value` 可改为填充。

- **dtype** `str | None`——可选的导出转换。`None`（默认）给出按 dtype 的诚实表示；否则目标取 NumPy 对该写法的解析结果（`'f8'`、`'d'`、`'<f8'` 都是 `float64`，`'|O'` 是 `object`）：
  - `'object'`（或 `'O'`）——无损的二维「带类型单元格」数组（数字 / `str` /`Timestamp` / `volas.NA`）；唯一能保留 `str` 或 datetime 列的 dtype。
  - `'int64'`、`'int32'`、`'int16'`、`'int8'`（以及无符号 `'uint*'`）——**精确**的`i64` 通道（大整数与 datetime 的 epoch-ns 不经 `float` 往返而存活）。对含缺失值的frame，除非给了 `na_value`，否则**抛错**。
  - `'float64'`、`'float32'`、`'float16'`——（有损的）`float` 通道：缺失单元格为`NaN`，超过 2⁵³ ns 的 datetime 会量化。
  - `'bool'`——布尔：取每个值的真值（`0.5` → `True`）。对含缺失值的 frame，除非给了`na_value`，否则**抛错**。
  - `'datetime64[ns]'`——datetime 纳秒；`NaT` 单元格被保留。
  - 文本 dtype（`'U'` / `'S'`）——格式化每个带类型单元格（`1.5` → `'1.5'`）。
  - `str` 列拒绝一切数值/时间 dtype——请用 `'object'`。
- **na_value** `Any`——替换每个缺失单元格的值；默认是 NA 模型表示（`NaN` / `NaT` /`volas.NA`）。

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

- 默认（不传 `dtype`）是诚实表示：全数值/bool 的 frame 是 `float64` 矩阵（缺失 →`NaN`），含 `str` 或混合 dtype 的 frame 是带类型单元格的 `object` 矩阵，datetimeframe 是 `datetime64[ns]`。
- `dtype='object'` 始终无损——每个单元格保留各自的类型值（数字、`str`、`Timestamp`或 `volas.NA`）。
- `str` 列没有数值含义，因此任何数值/时间 dtype 都会抛错——用 `dtype='object'` 保留字符串。
- datetime 列**豁免**整型缺失值的抛错规则：在 `dtype='int64'` 下，`NaT` 导出为其精确的epoch-ns 哨兵值（datetime 永不经由 `float` 往返）；给了 `na_value` 时它覆盖该哨兵值。
- `to_numpy()` 是一次批量读取；它不会自动刷新陈旧的指标列，若有陈旧列会抛错——先调用[`df.fulfill()`](../../README.zh-CN.md#dffulfill---none)。

需要零拷贝交给 Arrow / DLPack 消费方时，参见[与 Arrow、DLPack 互操作](#与-arrowdlpack-互操作零拷贝)。

### df.to_arrow() -> pyarrow.Table

**volas 独有**——把 DataFrame 导出为 `pyarrow.Table`，在 dtype 匹配处**零拷贝**：数值 / 字符串 / datetime 列缓冲区与 Arrow 共享，`bool` 与空值位图按需重打包。需要`pyarrow`（仅在此处惰性 import）。它是 volas Arrow **C-Stream** 桥接的便捷写法：任何Arrow 消费方都能经标准 `__arrow_c_stream__` PyCapsule 协议直接读取本 frame，无需调用`to_arrow()`，也无需让 volas 依赖 pyarrow。含 NUL 字节的列名无法跨越 Arrow C 接口，会抛出 `ValueError`。

```py
import pyarrow as pa
tbl = df.to_arrow()        # -> pyarrow.Table（共享列缓冲区）
tbl = pa.table(df)         # 等价，经 __arrow_c_stream__ 协议
pdf = pl.from_dataframe(df)  # polars 经同一协议读取
```

返回一个 `pyarrow.Table`。完整零拷贝契约与 DLPack 导出见[与 Arrow、DLPack 互操作](#与-arrowdlpack-互操作零拷贝)。

### df.to_pandas(dtype_backend='numpy') -> pandas.DataFrame

导出为 `pandas.DataFrame`（pandas 仅在此处惰性 import——它不是运行时依赖）。`DatetimeIndex` 可往返；反向桥接是 [`DataFrame.from_pandas`](#dataframefrom_pandaspdf---dataframe)。

- **dtype_backend?** `str = 'numpy'` 缺失值如何带入 pandas：
  - `'numpy'`——生态兼容性最好的形式：带缺失值的 int 列变成 `float64` + `NaN`（与 `pandas.Int64.to_numpy()` 一致）；带缺失值的 bool 列变成 masked `boolean`（`pd.NA`），绝不会是会被当作真值的 `NaN`。
  - `'numpy_nullable'`——忠实、无损的 masked 往返：int / bool / str 列保持`Int64` / `boolean` / `string`，空洞为 `pandas.NA`。

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

`df[col]` 和 `df[directive]` 返回一个 `Series`：一条具名的一维列，API 与pandas 兼容，包括算术 / 比较 / 逻辑运算符、`.sum()` / `.mean()` / `.std()` / …、`.shift()` / `.diff()` / `.fillna()`、`.iloc` / `.loc`、`.to_numpy()` /`.to_list()`。完整列表见[其余与 pandas 一致的 API](../../README.zh-CN.md#其余与-pandas-一致的-api)。`Series` 没有公开的构造函数，只能通过索引 `DataFrame` 得到。

```py
s = df['close']
s.name                 # 'close'
(s - s.shift(1)).mean()
df['ma:5 > ma:20']     # directive 也返回 Series（这里是 bool）
```

除 pandas 兼容 API 外，`Series` 还把 TA-Lib 的 15 个 **Math Transform** 函数暴露为方法——`acos` `asin` `atan` `ceil` `cos` `cosh` `exp` `floor` `ln` `log10` `sin` `sinh` `sqrt` `tan` `tanh`：

```py
df['close'].ln()
df['high'].sqrt()
```

`datetime64[ns]` Series 暴露 pandas 风格的 `.dt` 访问器：日历分量（`year` `month` `day` `hour` `minute` `second` `microsecond` `nanosecond` `quarter` `dayofweek` `dayofyear` `days_in_month`）、日历谓词（`is_month_start` …`is_year_end`、`is_leap_year`）、名称（`day_name()` / `month_name()`）、格式化（`strftime(fmt)`）、bar 对齐（`floor(freq)` / `ceil(freq)` / `round(freq)` /`normalize()`）以及 `isocalendar()`。某个元素缺失时，各分量都返回 `NA`：

```py
t = volas.to_datetime(df['time'])
t.dt.hour                  # int64 Series，0..23
t.dt.dayofweek             # 周一=0 .. 周日=6
t.dt.floor('15min')        # 对齐到 15 分钟 bar
```

### Series.from_arrow(data, name=None, copy=False) -> Series

**volas 独有的静态方法**——从任何实现 Arrow **C-Data** 数组协议（`__arrow_c_array__`）的对象构造 `Series`：`pyarrow.Array`、polars `Series` 等。默认 `copy=False` 借用匹配的数值与密集 ns-datetime 缓冲区，结果带一个全新的 `RangeIndex`；`copy=True` 创建 owned 快照。Arrow 数据不可变；在任何导入对象或视图存活期间，生产方及所有别名都不得再写借用内存。需转换的布局仍会复制。格式错误的数组抛出 `ValueError`，与 [`DataFrame.from_arrow`](#dataframefrom_arrowdata-copyfalse---dataframe) 相同。

- **data** Arrow 来源——任何实现 `__arrow_c_array__` 的对象。
- **name?** `str | None = None` 结果 `Series` 的名字。

```py
s = Series.from_arrow(pa_array, name='close')   # pyarrow.Array -> Series
```

### series.to_numpy(dtype=None, na_value=...) -> np.ndarray

把列的值导出为一维 NumPy 数组——`pandas.Series.to_numpy` 语义：

- **dtype** `str | None`——可选的导出转换。`None`（默认）给出列的原生表示；否则可为任何 `numpy.ndarray.astype` 接受的 NumPy dtype 字符串，常用取值为：
  - `'int64'`、`'int32'`、`'int16'`、`'int8'`（以及无符号 `'uint64'`、`'uint32'`、`'uint16'`、`'uint8'`）——整型。对含缺失值的列，除非给了 `na_value`，否则**抛错**（NA 没有整型表示）。
  - `'float64'`、`'float32'`、`'float16'`——浮点；缺失单元格为 `NaN`。
  - `'bool'`——布尔。对含缺失值的列，除非给了 `na_value`，否则**抛错**（NA 没有布尔表示）。
  - `'datetime64[ns]'`——datetime 纳秒；缺失单元格为 `NaT`。
  - `'object'`（或 `'O'`）——Python 对象，每个单元格保留各自的类型值（无损）。
- **na_value** `Any`——替换每个缺失单元格的值。默认是 NA 模型表示（float 导出为 `NaN`，object 数组为 `None`）。给了显式整型 `dtype` 时，值保持**精确**（大整数不经 `float64`漏斗），空洞变为 `na_value`。

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

- 默认（不传 `dtype`、不传 `na_value`）是按 dtype 的导出：缺失的 int / datetime 单元格折叠为 `NaN` / `NaT`（NumPy 没有 `NA`），稠密列保留原生 dtype。float `NaN` 是带内值，因此 float 列转整型 dtype 时只要存在 `NaN` 同样会抛错（请传 `na_value`）。带缺失值的 **bool** 列没有默认导出，会**抛错**——`nan` / `None` 占位会被 `np.where` 或布尔索引当作 `True` / `False`；请传 `na_value=`（例如 `False`，得到 bool 数组）、`dtype='float64'`（`NaN`）或 `dtype='object'`（`volas.NA`）。`np.asarray(series)` 与 `df.exec(directive, dtype=..., na_value=...)` 遵循同一规则。
- 与 pandas 一致，`na_value` 只改变缺失单元格——不给显式 `dtype` 时，带 NA 的 int 列仍导出为 `float64`（默认），`na_value` 只是填充那些 `NaN` 槽。
- 若需要**无损**保留原生 dtype 与缺失位置（不填充、不折叠 float），用 Arrow 路径（`series.to_arrow()` 携带空值位图）或 `series.to_pandas(dtype_backend='numpy_nullable')`；单独取 NA 掩码用 `series.isna().to_numpy()`。

### series.to_arrow() -> pyarrow.Array

**volas 独有**——把列导出为 `pyarrow.Array`，在 dtype 匹配处**零拷贝**（数值 / 字符串/ datetime 缓冲区与 Arrow 共享，`bool` 与空值位图按需重打包）。需要 `pyarrow`（惰性import）。它是 volas Arrow **C-Data** 桥接的便捷写法：任何 Arrow 消费方都能经标准`__arrow_c_array__` PyCapsule 协议直接读取本 series。

```py
import pyarrow as pa
arr = series.to_arrow()    # -> pyarrow.Array（共享缓冲区）
arr = pa.array(series)     # 等价，经 __arrow_c_array__ 协议
```

返回一个 `pyarrow.Array`。本列还可经 DLPack 零拷贝导出到 NumPy / PyTorch / JAX（`np.from_dlpack(series)`）——见[与 Arrow、DLPack 互操作](#与-arrowdlpack-互操作零拷贝)。

### Row

`df.iloc[i]` 和 `df.loc[label]` 返回一个 `Row`：一条单独记录，`.name` 是它的索引标签。`Row` **没有公开构造函数**（`Row(...)` 会抛`TypeError: No constructor defined for Row`）；只能通过索引 frame 获得，也可以把它传给 `df.append`。

```py
row = df.iloc[-1]      # 最新 bar
row.name               # 它的索引标签（DatetimeIndex 下是一个 Timestamp）
row.to_dict()          # {列: 值}
row.to_numpy()         # 数值单元格组成的 1 维 ndarray
```

## 与 pandas 互操作

pandas **不是**运行时依赖；这些桥接只在被调用时才惰性 import pandas，所以`import volas` 仍然不需要 pandas。

```py
from volas import DataFrame

df = DataFrame.from_pandas(pandas_df)  # numeric / bool / str / datetime 原生；带时区 DatetimeIndex 可无损往返；
                                   # nullable Int64 / boolean / string 列会读回为 int / bool / str + volas.NA
pdf = df.to_pandas()               # -> pandas.DataFrame（'numpy' 后端：带 NA 的 int/bool 列变成 float64 + NaN）
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # 忠实的 masked Int64 / boolean（一次无损的 NA 往返）
df.to_csv('out.csv', index=True)   # pandas to_csv 的一个子集；path=None 时返回一个 str
```

### datetime 列是 UTC 瞬时

datetime **列**存储的是绝对瞬时（UTC 纪元纳秒），本身不带时区——只有 `DatetimeIndex` 携带时区。任意时区（或 naive）的 pandas 列都按其瞬时读入；凡是能表达时区的导出，一律标为 **UTC**：

| 导出 | datetime 列变成 |
|---|---|
| `to_pandas()`（两种后端） | `datetime64[ns, UTC]` |
| `pa.table(df)` / `pa.array(s)` / `to_arrow()` | `timestamp[ns, tz=UTC]` |
| `s.to_list()` / `to_dict()` / `items()`、单元格、`to_numpy(dtype='object')` | `Timestamp(..., tz='UTC')`（`NaT` 为 `volas.NA`） |
| `s.dt.tz` | `'UTC'`（`.dt` 字段读取 UTC 挂钟时间） |
| `to_numpy()` / `__array__` | 相同 UTC 瞬时的 `datetime64[ns]`（NumPy 没有时区） |
| `s.dtype` / `df.dtypes`、`repr`、`to_csv` | 不变：dtype 仍为 `datetime64[ns]`（dtype 名中不含时区），repr / CSV 文本显示不带偏移的 UTC 挂钟时间 |

因此瞬时永远没有歧义，但**源列的时区不会被保存**：pandas 的 `datetime64[ns, Asia/Shanghai]` 列会以 `datetime64[ns, UTC]`（相同瞬时）返回。通过转换恢复时区——在 pandas 中，或在 volas 中经由索引：

```py
pdf = df.to_pandas()
pdf['t'] = pdf['t'].dt.tz_convert('Asia/Shanghai')      # pandas：重新标注时区

df.set_index('t').tz_localize('UTC').tz_convert('Asia/Shanghai')
# volas：列的 UTC 瞬时变成锚定 UTC 的索引，再重新标注时区
```

naive 的 pandas datetime 列同样按 UTC 瞬时读入，因此也会导出为 `datetime64[ns, UTC]`（如有需要，在 pandas 中用 `.dt.tz_localize(None)` 去掉时区）。

`to_pandas` 的 **dtype_backend**（`'numpy'` 对比无损的 `'numpy_nullable'`）决定缺失值如何带入 pandas——逐值说明见[`df.to_pandas`](#dfto_pandasdtype_backendnumpy---pandasdataframe)。

## 与 Arrow、DLPack 互操作（零拷贝）

volas 的每个 dtype 列各持有一块连续缓冲区（字符串列也是 Arrow 原生布局），因此跨到 Arrow 与 DLPack 的消费方时**无需拷贝**——消费方借用同一段字节，由 volas 负责保活。

```py
import pyarrow as pa, numpy as np

# Arrow C-Data / C-Stream —— pyarrow、polars 等通过标准 PyCapsule 协议
#（__arrow_c_array__ / __arrow_c_stream__）直接读取 volas。
pa.array(df['close'])              # Series   -> pyarrow.Array（共享缓冲区）
pa.table(df)                       # DataFrame -> pyarrow.Table（单个 RecordBatch）
df['close'].to_arrow()             # pa.array(...) 的便捷写法

Series.from_arrow(pa_array, name='close')   # 默认借用匹配缓冲区
Series.from_arrow(pa_array, name='close', copy=True)   # owned 快照
DataFrame.from_arrow(pa_table)              # Arrow table -> DataFrame

# DLPack —— NumPy / PyTorch / JAX 借用一个稠密的数值（或 bool）列。
np.from_dlpack(df['close'])        # 零拷贝 ndarray 视图
```

高层入口——[`df.to_arrow`](#dfto_arrow---pyarrowtable) /[`DataFrame.from_arrow`](#dataframefrom_arrowdata-copyfalse---dataframe)、[`series.to_arrow`](#seriesto_arrow---pyarrowarray) /[`Series.from_arrow`](#seriesfrom_arrowdata-namenone-copyfalse---series)——在 Usage 下有说明。它们建立在以下**标准协议方法**之上，Arrow / 数组消费方会自动调用它们（所以你很少需要自己调用）：

- **`Series.__arrow_c_array__`**——Arrow C-Data 数组协议；返回 `(schema, array)`PyCapsule 二元组，让 `pa.array(s)` / `pl.Series(s)` 读取一列。
- **`Series.__arrow_c_schema__`**——只含 schema 的那一半（列的 Arrow dtype）。
- **`DataFrame.__arrow_c_stream__`**——Arrow C-Stream 协议；frame 作为单个`RecordBatch`，让 `pa.table(df)` / `pl.from_dataframe(df)` 读取一个 frame。
- **`Series.__dlpack__` / `Series.__dlpack_device__`**——DLPack 协议，让`np.from_dlpack(s)` / `torch.from_dlpack(s)` 借用一个稠密的数值 / bool 列。借来的视图是**只读**的（带版本号 DLPack 的标志位），写穿它无法绕过写时复制而改坏帧；无法接收该标志位的 1.0 之前的旧消费方，则改为得到一份独立副本。`np.from_dlpack(s, copy=True)` 返回一份独立的**可写**副本。非 CPU 设备或 stream 会被拒绝（`BufferError`）；含缺失值的 int/bool 列（DLPack 无空值掩码）或 str/datetime 列会抛错。

**Arrow 导入契约。**Arrow 数据不可变。`Series.from_arrow(array)`／`DataFrame.from_arrow(table)` 默认 `copy=False`，借用匹配数值与密集 ns-datetime 的缓冲区；生产方及所有别名在任何导入对象或视图存活期间都不得写借用内存。`copy=True` 创建 owned 快照，隔离后续生产方写入对原列、缓存、续算状态或检查点的影响；导入期间不得并发写输入。需转换的布局和多 chunk 列仍可能复制。直接 NumPy 导入保持复制。**零拷贝导出契约**：原生数值、ns-datetime 和字符串数据共享只读缓冲区。其余重打包有：

- **导入时**的每个字符串列（`string` / `large_string`，无论是否切片）——拷贝进 volas 自有缓冲区并在拷贝上校验（offsets 在范围内且单调、每个非空值是合法 UTF-8；null 槽位未作规定的字节被丢弃），因此生产方在导入之后改写自己的缓冲区不会影响 volas 列；
- **`bool`**——volas 每个值存一个字节，Arrow 存一个*位*；
- **空值位图**（≤ `n/8` 字节）——包括导出时把缺失 `float`（带内 `NaN`）扫描成真正的Arrow null 的那一遍；
- 仅在**导入时**：`string_view`（物化为连续字节）、粗于纳秒的时间戳（重新缩放）、`Decimal128` / `Decimal256`（→ `f64`，超过约 15 位**有损**——精确价格请上游保持字符串）、窄/无符号整数（→ `i64`；超过 `i64::MAX` 的 `UInt64` 无法无损表示，会报错）、字典 / 类别（categorical）列（解码为其取值）、无类型的 `Null` 列（→ 全 NA 的 `f64` 列）、`Date32` / `Date64`（→ 纳秒 datetime）。其余 Arrow 类型会抛清晰的错误。

**边界上的 NA。**每一个缺失值——int、bool、**以及 float**——都跨越为真正的 Arrow null（缺失的 float 即带内 `NaN`，导出时扫描进空值位图），因此下游的 `is_null` / `null_count`能一致地看见它们。DLPack 与整型 `to_numpy` 没有空值通道，遇缺失值会抛错——请传`na_value=`，或用 Arrow 路径（它无损携带空值位图）：

```py
df['qty'].to_numpy(dtype='int64')                # 任一值为 NA 时抛错（与 pandas 对齐）
df['qty'].to_numpy(dtype='int64', na_value=0)    # 或用 na_value 填充空洞
pa.array(df['qty'])                              # 无损：保留 int64 + 空值位图
```

**时区。** Arrow 表没有*索引*，因此帧的索引（及其时区）不会被携带，`from_arrow` 返回全新 `RangeIndex`。datetime 列导出为 `timestamp[ns, tz=UTC]`（见 [datetime 列是 UTC 瞬时](#datetime-列是-utc-瞬时)）；导入的任意时区（或无时区）timestamp 列都变成其精确瞬时，时区本身不保留。导入后用 `set_index` + `tz_localize('UTC')` + `tz_convert` 重新标注时区。
