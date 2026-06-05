# 设计 Spec：TimeFrame 修正 + 把 Cumulator 合并进 DataFrame

状态：待评审（写代码前过一遍）。语言：尽量说人话、给具体公式与边界。

---

## 0. 目标

1. **修正 TimeFrame 的分桶**：删除无意义的 `minutes`；把 `Week1`、`Day3` 从"按当月第几天取模、每月重置"改成**连续的、按 epoch 锚定**的桶（周线=周一锚定、默认 UTC，跟 Binance/ISO 一致）。
2. **把 Cumulator 合并进 DataFrame**：DataFrame 可声明 `time_frame`，之后 `append` 自动把更细的 K 线折叠进当前周期，并移动指标游标（"正在形成的那根 K 线"就是 `df.iloc[-1]`）。独立的 `Cumulator` / `cum.last` 随之取消。

两部分可分别落地；Part A 不依赖 Part B，先做 A。

---

## 1. 已锁定的决策（来自讨论）

- 折叠时**不区分原始列/指标列**，统一简单折叠；指标的正确性由**游标**控制（游标回退→重算）。分桶**只看 index（时间戳）**。
- 构造带 `time_frame` 时，**已有的行不重新聚合**（视为"已经是该周期"），**不做校验**（校验会拖慢性能；用户若担心不一致，自己 `cumulate()`）。
- `time_frame` 但无 DatetimeIndex → **报错**（无 DatetimeIndex 调 `cumulate` 当前已报错）。
- `df.cumulate(tf)` 的结果**带上 tf**。
- 目标 tf 必须是源 tf 的**合法粗化**（见 §A3），否则报错；**无 tf 的原始数据**可合成任意 tf。
- 目标 tf **≠** 源 tf → 游标归零（指标从头算）；目标 tf **==** 源 tf → 等价 `copy()`（游标/缓存保留）。
- 递归类指标（EMA/RSI/MACD…）在"正在形成那根"上的正确性**依赖 origin 修复增量计算**；在此之前滑动窗口类（MA/WMA…）正确、递归类错。**本 spec 不修这个**。
- `minutes`：**删除**。
- 周线定义：**连续 7 天、周一锚定、默认 UTC**（调研结论：Binance/ISO 主流；Polygon/pandas 默认周日为少数派）。锚定时区与周起始日做成可配置参数（默认 UTC + 周一），为股票/周日场景预留。

---

## 2. Part A：TimeFrame 修正（volas-time，自包含）

### A1. 删除 `minutes`
- 删 `crates/volas-time/src/time_frame.rs` 的 `pub fn minutes()`，及其内部测试断言（`label_roundtrips_and_minutes_for_every_frame` 里关于 minutes 的部分）。
- 删 `crates/volas-python/src/timeframe.rs` 的 `minutes` getter。
- 删 `test/test_cumulation.py::test_time_frame_minutes`。
- 改 `README.md` 第 542 行（`TimeFrame.m5.minutes # 5` 示例行）与第 559 行（`tf.minutes` 说明段）。
- 理由：核心分桶（`unify_tz`）用日历分量，不用 minutes；`hv` 指标走独立的 `tf_to_minutes(字符串)`，也不用它；`minutes` 注释自称 "parity field"，且对月/年是错的占位（月==年==525600）。唯一消费者是这个 getter+测试。

### A2. 修正 `Week1` 与 `Day3` 为连续 epoch 锚定

当前（错）：`Week1 => (d/7)*DAY + mo*MONTH + y*YEAR`、`Day3 => (d/3)*DAY + mo*MONTH + y*YEAR`，`d` 是"当月第几天"，每月重置，把真实的一周/三天在月界劈开。

改为：先得到**连续的"距 epoch 天数"**，再算桶号。

- 在锚定时区下取日历日：用现有 `tz.civil_parts(ns)` 拿 `(y, mo, d)`（DST 正确）。
- 加一个 helper `days_from_civil(y, mo, d) -> i64`（Howard Hinnant 标准算法），`days_from_civil(1970,1,1) == 0`（周四）。
- 周线桶号（周一锚定）：`week = (days + 3).div_euclid(7)`
  - 校验：1969-12-29 周一 = day −3 → `(−3+3)/7 = 0`；1970-01-01 周四 = day 0 → `(0+3)/7 = 0`（同一周）；1970-01-05 周一 = day 4 → `(4+3)/7 = 1`。✅ 周一为界、连续、跨月不重置。
- 3 日桶号（epoch 锚定）：`d3 = days.div_euclid(3)`（day 0 起每 3 天一桶，连续、跨月不重置）。
- `unify_tz` 对这两支返回上述桶号即可（桶号单调、每桶唯一；分组只在同一 tf 内比较，互不干扰）。
- 其余周期**保持不变**（秒/分/时/日逐级嵌入日历日、`Month1`=日历月、`Year1`=日历年，本来就正确）。
- 周起始日参数化（默认周一）：把 `+3` 抽成 `monday_offset`，未来支持周日只需换 offset（周日锚定 offset=+4）。先实现默认周一，参数留接口。
- 锚定时区：`unify_tz` 已带 `tz` 参数（默认 `Tz::Utc`）。无需新增。

> 粗 bar 的 index 标签 = **桶起始时刻**（周线=该周一 00:00；与 Binance kline start time 一致）。emit 粗 bar 时按桶号反算桶起始的 epoch-ns。

### A3. `can_coarsen(source_tf, target_tf)` —— 能否粗化的判定

定义（一句话）：**目标周期的每一条分界线，都必须也是源周期的分界线**（细的能整齐拼成粗的）。实现成一个经单元测试验证的谓词，分两类：

- **定长梯（秒/分/时，含 2h/4h/6h/8h/12h）**：按"每桶毫秒数"整除判定。例：`1m→5m`✅、`5m→15m`✅、`3m→5m`❌、`4h→6h`❌。
- **跨到日及以上**：用嵌套关系（基于 §A2 的连续 epoch 桶）：
  - `任意能整除 1 天的 sub-day → Day1`✅（时/分/秒边界含日界）。
  - `Day1 → Day3`✅（3 日 epoch 网格的界都是日界）。
  - `Day1 → Week1`✅（周界=某些日界）；`Day1 → Month1`✅；`Month1 → Year1`✅；`Day1 → Year1`✅。
  - **关键（反直觉但正确）**：`Week1 → Month1`❌、`Week1 → Year1`❌（ISO 周跨月跨年，线对不齐）；`Day3 → Week1`❌（3∤7）；`Day3 → Month1`❌（3 日网格不对齐月初）；`Day3 → Year1`❌。
- **无 tf 的源（原始数据）**：允许任意目标 tf。
- `source_tf == target_tf`：合法，结果是 `copy()`（见 §B4）。

实现建议：一个 `fn can_coarsen(src: TimeFrame, dst: TimeFrame) -> bool`，用"桶分界线"原则推导的小关系表/规则，配穷举单测（对若干时间戳验证 dst 边界 ⊆ src 边界）。

### A4. Part A 测试
- 周线跨月连续：`2024-01-29..02-04` 落同一周桶（现在会劈成两个）。
- 周一为界：周一 vs 同周周日同桶、与上周日不同桶。
- 3 日 epoch 连续：跨月不重置、桶大小恒为 3 天。
- `can_coarsen` 真值表（含上面所有 ✅/❌）。
- DST：在带 DST 的命名时区下，日界/周界落在本地午夜/周一本地午夜。

---

## 3. Part B：tf-aware DataFrame（合并 Cumulator）

### B1. 数据模型与分层（**放在 binding 层 PyDataFrame，不放 core**）

> **为什么不放 volas-core**：core 的 `DataFrame` 若要折叠就得依赖 `volas-time`（TimeFrame/AggSpec/聚合），而 `volas-time` 已依赖 `volas-core` → **循环依赖**。所以 tf-aware 行为放在 **`volas-python` 的 `PyDataFrame`**（它本来就依赖 core+time+directive），既避免环、又正好对应用户说的"Python 的 DataFrame 增加 time_frame 参数"。core `DataFrame` 不变。

`PyDataFrame`（tf-aware 时）持有：
- `inner: DataFrame` —— **粗粒度帧**（已收盘行 + 末行=正在形成的 bar），带 computed 列与游标（复用现有机制）。
- `time_frame: Option<TimeFrame>` —— None=原始帧（`append` 走旧逻辑，不折叠）。
- `cumulators: AggSpec` —— 每列聚合方式（默认 OHLCV：open=first/high=max/low=min/close=last/volume=sum）。
- `open_buf: Option<DataFrame>` —— **当前开放周期的原始细 bar 缓冲**（复用 `volas_time::Cumulator` 的做法，用于去重/被重发的 forming bar 更新；只存一个周期的细 bar，量很小）。

> 复用：折叠用 `volas_time` 的 `unify`/`aggregate_period`（按需 `pub`），不重写状态机。Rust 的 `volas_time::Cumulator` 作为内部引擎可保留；**Python 层的 `Cumulator` 类取消**（功能并入 DataFrame）。

### B2. 构造 `DataFrame(data, time_frame=None, cumulators=None, ...)`
- `time_frame` 给定：
  - 无 DatetimeIndex → 报错（同 `cumulate` 的守卫）。
  - 已有行**原样作为已收盘的粗 bar**（不重新聚合、不校验）。`open_buf = None`（无正在形成的周期）。
  - `cumulators` 默认 OHLCV，可覆盖。
- `time_frame=None`：与现状完全一致（原始帧）。

### B3. `append(fine)` 算法（tf-aware 时）

对传入的每根细 bar（按时间升序）：
```
key = unify(bar.ts, time_frame, tz)
若 open_buf 为空 或 key == unify(open_buf 末根.ts):   # 同一开放周期
    open_buf.append_or_replace(bar)        # 按 ts 去重/更新（被重发的 forming bar）
    agg = aggregate_period(open_buf, cumulators)   # 重算本周期聚合
    若 inner 末行就是这个开放周期: 原地更新 inner 末行 = agg   # df.iloc[-1] 变
    否则:                                   inner.append(agg)  # 第一次开这个周期
    游标: valid_rows = inner.len - 1         # 末行变脏 → 下次 df[directive] 重算末行
否则:                                        # 跨入新周期（旧的就此定型）
    open_buf = [bar]
    inner.append(aggregate_period(open_buf, cumulators))
    游标: 新末行变脏（valid_rows = inner.len - 1）
```
- **简单折叠**：上面对**所有列**一视同仁地用 `cumulators` 聚合（指标列也跟着被"聚合"一个临时值），但**游标会把末行/新行标脏**，下次读 `df[directive]` 时指标列被正确重算覆盖——所以无需特判指标列（符合你的决策）。
- `time_frame=None` → 走现有 `append`（不折叠）。
- 空 `fine` → 报错（现状）。

### B4. `cumulate(target_tf, cumulators=None)`
- `inner` 无 DatetimeIndex → 报错。
- `can_coarsen(self.time_frame_or_finest, target_tf)` 为假 → 报错（无 tf 源视为最细，任意 tf 允许）。
- `target_tf == self.time_frame` → 等价 `copy()`：克隆 inner（**含 computed 列与游标**）+ open_buf + tf。（core `clone()` 已保留 computed/cursor，所以 copy 保留游标——这就是"同周期=copy"。）
- 否则：把 `inner`（按需 + open_buf 展开的细 bar）按 `target_tf` 重新折叠成新粗帧；结果**带上 `target_tf`** 与 open_buf；**游标归零**（valid_rows=0，指标从头算）。

### B5. 游标规则（汇总）
- 末行（forming）更新 → 游标回退到 `len-1`（只重算末行）。
- 跨周期新增行 → 游标保持，新行变脏（下次访问算新行）。
- 换周期 cumulate → 游标归零。
- 同周期 cumulate / copy → 游标保留。
- 游标 = core `DataFrame.computed[name].valid_rows`，复用现有 `refresh_computed`。

### B6. `copy` / `iloc` / `slice` 等产帧操作
- 必须把 `time_frame` / `cumulators` 透传到新帧；`open_buf` 仅在"保持继续 append"语义的产物上保留（`copy` 保留；`iloc`/`slice` 出的是历史快照，**不保留 open_buf**，即切片结果是只读历史，不再 folding——简单、安全）。
- core `DataFrame` 不变；这些字段在 `PyDataFrame` 层透传。

### B7. `Cumulator` 类的去留（迁移）
- **取消 Python 层 `volas.Cumulator`**（及 `cum.frame`/`cum.last`）。等价用法：
  - `cum.frame` → tf-aware `df` 本身。
  - `cum.last`（正在形成的 bar）→ `df.iloc[-1]`。
  - 当前指标值 → `df[directive]` 的最后一个（正确，且增量）。
- `volas_time::Cumulator`（Rust）保留为内部聚合引擎或重构为被 PyDataFrame 复用的函数。
- `__init__.py` 移除 `Cumulator` 导出；更新 README 的 Cumulator 段。

### B8. 错误与边界
- tf 但无 DatetimeIndex → 报错（构造与 cumulate 都加守卫）。
- 非法粗化 → 报错（`can_coarsen` 假）。
- 空 append → 报错（现状）。
- 用户给的已有数据与 tf 不一致（如把 1m 数据标成 5m）→ **不报错**（决策：不校验）；后果自负，建议 `cumulate()`。

### B9. 去重 / forming-bar 更新语义
- 通过 `open_buf` 按细 bar 时间戳去重/更新：被重发的"当前 forming 细 bar"（如 Binance WS 每秒推送未收盘 kline）→ 替换而非累加（volume 不重复计）。这正是保留 `open_buf` 的原因。
- 文档明确：append 的细 bar 应按时间升序；同 ts = 更新当前 forming 周期内该细 bar。

### B10. 与 origin 的 IIR 修复的关系
- B 的折叠+游标**复用** `df.append + df[directive]/refresh_computed` 这条增量机制。origin 把它对递归指标修对后，tf-aware 帧的递归指标（含 forming 末行）**自动变正确+增量**，本 spec 无需改动。
- 修复前：FIR 正确、IIR 在 forming/新确认行上错（已知，交给 origin）。

### B11. Part B 测试
- 构造 tf 帧 + append 同周期细 bar → 末行原地更新、行数不变、OHLCV 正确、`df.iloc[-1]` 变化。
- append 跨周期 → 新增行、上一行定型。
- forming bar 被重发（同 ts，volume 更新）→ 不重复累加。
- `cumulate` 后带 tf；非法粗化报错；`can_coarsen` 真值表对应的成功/失败。
- 同周期 cumulate == copy()（数据、列、游标一致）。
- tf 无 DatetimeIndex 报错。
- FIR 指标（ma/wma）在 tf 帧上 append 增量 == 一次性；IIR 标 xfail（待 origin）。

---

## 4. 分层与依赖（不引入环）
- Part A 全在 `volas-time`（自包含）。
- Part B 全在 `volas-python`（`PyDataFrame`），复用 `volas-core`（帧/游标）+ `volas-time`（unify/aggregate/can_coarsen）+ `volas-directive`（指标）。
- `volas-core` 不新增对 `volas-time` 的依赖（避免 core→time 环）。

## 5. 落地顺序
1. **A1 删 minutes**（含 README/测试）——独立、最小。
2. **A2/A3 修 Week1/Day3 + can_coarsen + 测试**——独立。
3. **B 合并**（构造参数 → append 折叠+游标 → cumulate → 去 Cumulator 类 → 测试）。
4. origin 修 IIR 后，把 B 测试里的 IIR xfail 转正。

## 6. 留待确认 / 交给 origin
- origin：递归指标的增量刷新正确性（流式状态）。
- 未来扩展（不在本 spec）：股票"日=交易时段"（需交易日历+假日，IB/Alpaca 那种"按收盘日标注"的日线/周线）；周起始日=周日的可配置（接口已留）。
- 调研未逐一核实：Coinbase/OKX/Kraken 的周起始（按 Binance+通用约定取周一/UTC）。
