# volas

[English](https://github.com/kaelzhang/volas/blob/main/crates/volas/README.md) | 简体中文

一个 Rust 驱动、OHLCV 形态的 `DataFrame`，面向 K 线 / 行情时间序列，内置技术指标
**directive（指令）** 引擎。

volas 刻意做窄——它不是通用 DataFrame，而是面向实时 OHLCV 流水线：通过指标的
*directive* 名字来计算它（`"ma:20"`、`"macd.signal"`、`"close > open"`），并把 bar
重采样 / 累积到更粗的时间框架。

```toml
[dependencies]
volas = "1"
```

> PyPI 上还有一个同名的 Python 包 `volas`——同一套 Rust 内核，经 PyO3 暴露。参见
> [volas Python 项目(GitHub)](https://github.com/kaelzhang/volas/blob/main/README.zh-CN.md)
> 与 [PyPI 上的 `volas`](https://pypi.org/project/volas/)。它与本 crate 是两个独立的
> 发行物；两边的 directive 词汇完全一致。

## 快速上手

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse};

// 用具名列构造一个 frame（这里只有 `close`）。
let df = DataFrame::new(
    vec!["close".to_string()],
    vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])],
    None,
).unwrap();

// 用 directive 字符串计算指标。`ma:2` 是 2 周期 SMA。
let directive = parse("ma:2").unwrap();
let ma = execute(&df, &directive).unwrap();

assert!(!ma.is_valid(0));             // 预热行是 NA（尚无取值）
assert_eq!(ma.to_f64_vec()[3], 3.5);  // (3.0 + 4.0) / 2
```

## 数据模型

`DataFrame` 是一组等长、具名、带类型的 `Column`，共享同一个行 `Index`。`Column` 是一段
连续的类型化缓冲区；任意单元格都可以独立地为 `NA`（缺失），与其取值无关——参见下文
[缺失值(NA)](#缺失值na)。

```rust
use volas::{Column, DataFrame, DType};

let df = DataFrame::new(
    vec!["open".to_string(), "close".to_string()],
    vec![
        Column::f64(vec![10.0, 11.0, 12.0]),
        Column::f64(vec![10.5, 10.5, 13.0]),
    ],
    None, // None => 默认的 0..n RangeIndex
).unwrap();

assert_eq!(df.height(), 3);
assert_eq!(df.width(), 2);
assert_eq!(df.names(), &["open".to_string(), "close".to_string()]);

let close = df.column("close").unwrap();   // -> &Column
assert_eq!(close.dtype(), DType::F64);
assert_eq!(close.to_f64_vec()[2], 13.0);    // 拥有所有权的 Vec<f64>
assert_eq!(close.as_f64(), Some(&[10.5, 10.5, 13.0][..])); // 借用切片（仅 F64）
```

列构造器：`Column::f64`、`Column::i64`、`Column::bool`、`Column::str`。取值用
`Column::to_f64_vec`（拥有所有权）、`Column::as_f64`（借用，非 `f64` 返回 `None`）、
`Column::is_valid`（逐单元格 NA 检查）、`Column::len`、`Column::dtype`。

### 缺失值(NA)

单元格的缺失与其取值无关。浮点列用 `NaN` 作为 NA 标记；整数 / 布尔 / 字符串列携带一个
显式的有效性掩码（validity mask）。用 `is_valid` 逐单元格检查缺失，用 `null_count` 计数。

```rust
use volas::Column;
use volas::core::Validity;

// 浮点列：NaN 即缺失标记。
let prices = Column::f64(vec![1.0, f64::NAN, 3.0]);
assert_eq!(prices.null_count(), 1);
assert!(prices.is_valid(0));
assert!(!prices.is_valid(1));         // 第 1 格是 NA
assert!(prices.get_f64(1).is_nan());  // 读取 NA 的 f64 得到 NaN

// 整数 / 布尔 / 字符串列用有效性掩码，而非哨兵值。
let volume = Column::i64_with(
    vec![100, 0, 300],
    Validity::from_valid_iter(3, [true, false, true]), // 第 1 格是 NA
);
assert_eq!(volume.null_count(), 1);
assert!(!volume.is_valid(1));
```

指标的预热行也是 NA——例如 2 周期 SMA 在头部有一个 NA 行：

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse};

let df = DataFrame::new(
    vec!["close".to_string()],
    vec![Column::f64(vec![1.0, 2.0, 3.0])],
    None,
).unwrap();
let ma = execute(&df, &parse("ma:2").unwrap()).unwrap();

assert!(!ma.is_valid(0));        // 预热行是 NA
assert_eq!(ma.null_count(), 1);
```

## 通过 directive 计算指标

directive 引擎是计算指标的主要方式：用 `parse` 把 directive 字符串解析成 `Ast`，再
用 `execute` 在 `DataFrame` 上执行得到一个 `Column`。它**统一**覆盖所有内置指标——完整
的 directive 参考见
[INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md)。

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse, stringify};

let n = 40;
let df = DataFrame::new(
    vec!["open".to_string(), "close".to_string()],
    vec![
        Column::f64((1..=n).map(|x| x as f64).collect()),
        Column::f64((1..=n).map(|x| x as f64 + 0.5).collect()),
    ],
    None,
).unwrap();

// 单输出 directive -> 一个 Column。
let rsi = execute(&df, &parse("rsi:14").unwrap()).unwrap();
assert_eq!(rsi.len(), 40);

// 多输出指标：用 sub-command 取它的每一条线。
let macd   = execute(&df, &parse("macd").unwrap()).unwrap();
let signal = execute(&df, &parse("macd.signal").unwrap()).unwrap();
let hist   = execute(&df, &parse("macd.histogram").unwrap()).unwrap();
assert_eq!((macd.len(), signal.len(), hist.len()), (40, 40, 40));

// 比较类 directive -> Bool 列（可用作行掩码）。
let bullish = execute(&df, &parse("close > open").unwrap()).unwrap();
assert_eq!(bullish.len(), 40);

// `@` 后缀指定输入列；规范形式会省略默认输入。
let ast = parse("ma:20@close").unwrap();
assert_eq!(stringify(&ast), "ma:20");
```

directive 的形态是 `name:arg0,arg1,...@col0,col1,...`——`name` 是指标，`:` 后是它的
参数，`@` 后是输入列（各有合理默认，例如 `close`）。多输出指标把每条线暴露为
`name.subcommand`（`macd.signal`、`boll.upper`、`kdj.k`）。所有 directive 的确切参数、
默认值与 sub-command 见
[INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md)。

预热长度（出现第一个有效值之前的行数）无需计算即可获得：

```rust
use volas::directive::{lookback, parse};

let ast = parse("ma:20").unwrap();
assert_eq!(lookback(&ast), 19); // 20 周期 SMA 有 19 个预热行
```

### 原始 kernel（进阶）

directive 引擎是推荐的 API。若想不经 `DataFrame` 直接调用，`compute` 模块以对 slice 的
纯函数（`&[f64] -> Vec<f64>`）暴露这些数值内核；多数用户应优先用 directive——它稳定、
完整，且与 Python 侧完全一致。

## 读取 CSV

```rust,no_run
use volas::{read_csv, ReadCsvOptions};

// 默认：逗号分隔、首行为表头、标准 NA token。
let df = read_csv("ohlcv.csv", &ReadCsvOptions::default()).unwrap();
println!("{} rows x {} cols", df.height(), df.width());
```

`path` 接受任意 `AsRef<Path>`（`&str`、`String`、`PathBuf`）。通过 `ReadCsvOptions`
（`delimiter`、`has_header`、`na_values`、`keep_default_na`）调整解析。

## 时间框架累积（OHLCV 重采样）

把更细的 bar 聚合到更粗的 `TimeFrame`（源必须带 `DatetimeIndex`）。默认 OHLCV 规则：
`open`=first、`high`=max、`low`=min、`close`=last、`volume`=sum。

```rust,no_run
use volas::{Column, DataFrame, TimeFrame};
use volas::time::{cumulate, AggSpec};

// 实际中 `df` 带 1 分钟 DatetimeIndex;这里用一个占位 frame。
let df = DataFrame::new(vec!["close".to_string()], vec![Column::f64(vec![1.0])], None).unwrap();

// 重采样到 5 分钟 bar。
let five_min = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
let _ = five_min;
```

`TimeFrame` 是一个枚举（`Min1`、`Min5`、`Hour1`、`Day1` …），也可用
`TimeFrame::from_label("5m")` 从标签解析。

## 错误处理

可失败的函数返回 `Result<_, VolasError>`（单一错误枚举）——对非法输入不 panic。未知或
格式错误的 directive 在 `parse` 处报错：

```rust
use volas::VolasError;
use volas::directive::parse;

let ok = parse("ma:2");
assert!(ok.is_ok());

let bad: Result<_, VolasError> = parse("definitely_not_an_indicator:5");
assert!(bad.is_err());
```

## crate 结构

- 顶层——数据模型（`DataFrame`、`Series`、`Column`、`Index`、`DType`、`Scalar`、`Tz`、
  `Result`、`VolasError`），以及 `read_csv` 和 `TimeFrame`；
- `directive`——`parse` / `execute` / `stringify` / `lookback`，以及 `Ast`；
- `compute`——数值内核与技术指标（纯函数）；
- `time`——时间框架累积（OHLCV 重采样）；
- `core`——完整的 `volas-core` 表面，用于不常用的类型。

本 crate 是一个轻量门面，把 volas workspace（`volas-core`、`volas-compute`、
`volas-directive`、`volas-time`、`volas-io`）重导出到单一依赖之后。

## 许可证

[MIT](https://github.com/kaelzhang/volas/blob/main/LICENSE)
