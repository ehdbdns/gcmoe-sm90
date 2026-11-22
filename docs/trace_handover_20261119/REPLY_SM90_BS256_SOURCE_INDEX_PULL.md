# 【SM90】REPLY_SM90_BS256_SOURCE_INDEX_PULL — 源端索引拉取把 split 的 +0.157% 收到 +0.090%，仍是净慢

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 臂 | 状态 | 每步 MoE（999 步，剔采样步）| vs 本轮 baseline |
|---|---|---:|---:|
| **baseline** | ✅ `exit_code=0` | **35.7853 ms** | — |
| **source_index** | ✅ `exit_code=0` | **35.8175 ms** | **+0.0900%（慢 0.032 ms）** |

| 臂 | 全 1000 步含采样 | vs baseline |
|---|---:|---:|
| baseline | 35.7857 ms | — |
| source_index | 35.8179 ms | **+0.0902%** |

**三档对照（同口径：本轮 999 步、含插桩）**：

| 方案 | 每步 MoE | vs 同轮 baseline |
|---|---:|---:|
| baseline（ECO 默认）| 35.7853 / 35.7994（两轮）| — |
| **串行 split**（上轮）| 35.8554 | **+0.157%** |
| **源端索引拉取**（本轮）| 35.8175 | **+0.090%** |

即：源端索引拉取**把 split 的损失减半**（+0.157% → +0.090%），
**但没有变成收益**。三段拆分的符号全部一致（见 §2）。

## 1. 两臂校验

| 检查 | baseline | source_index |
|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 |
| 两窗口 audit / 八 rank native token | ✅ | ✅ |
| 物理槽 counts / epoch / 预测值 / 拍卖数组 | ✅ | ✅ |
| `eco-layout.json` | ✅ 与历史 `cmp` 相同 | ✅ 同左 |
| launch grid (dispatch, shared, routed, plan) | ✅ (24, 71, 71, 1) | ✅ (24, 71, 71, 1) |
| 历史 KV manifest（启动前逐文件 `cmp`）| ✅ | ✅ |
| `hardware.csv` vs 历史 `ref/eco/hardware.csv` | ✅ `diff -u` 无差异 | ✅ 同左 |

**配置实读**（`run.json` 的 `controls`）：

| 臂 | `GC_MOE_SM90_SOURCE_INDEX_PULL` | `SPLIT_EXCHANGE` | `PLAN_DISPATCH` | stats ms | measured ms |
|---|---|---:|---:|---:|---:|
| baseline | （未设）| （未设）| （未设）| 98.679 | 92.271 |
| source_index | **1** | （未设）| （未设）| 98.885 | 93.534 |

两臂均为逐层 EPLB + ECO fine cost mode 2、slack 11469、M64、stream 开、6 SM dispatch、
SM reuse 关；**上轮的串行 split 与 plan 分区开关两臂都关闭**（按要求保留代码但运行时不启用）。

## 2. 主指标三段拆分（`moe.csv`）

| arm | steps | graph→first ms | first→last ms | last→combine ms | **moe_ms** | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 8.0246 | 25.7162 | 2.0445 | **35.7853** | — |
| source_index | 999 | 8.0401 | 25.7148 | 2.0627 | **35.8175** | **+0.090%** |
| baseline | 1000 | 8.0246 | 25.7166 | 2.0444 | 35.7857 | — |
| source_index | 1000 | 8.0402 | 25.7149 | 2.0629 | 35.8179 | +0.090% |

**拆解**：

- `graph→first` **+15.5 µs**（8.0246 → 8.0401）—— 变慢；
- `first→last` −1.5 µs（25.7162 → 25.7148）—— 略快，噪声级；
- `last→combine` **+18.2 µs**（2.0445 → 2.0627）—— 变慢。

即 **+32.2 µs 的净变化里，首段与末段各占一半，中间的实际计算段没有改善**。
与上轮串行 split（三段全部变慢）相比，本轮至少把 `first→last` 拉回打平。

## 3. 分段耗时（八 rank 均值，µs）

| layer | arm | count exchange | pull token | shared L1+L2 | ECO plan | wait token | GEMM | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 27.8 | 141.3 | 194.7 | 114.4 | 2.0 | 665.6 | 35.0 |
| 0 | source_index | 27.2 | 149.8 | 195.5 | **109.9** | 2.2 | 665.6 | 35.2 |
| 8 | baseline | 25.1 | 173.6 | 195.0 | 119.0 | 2.2 | 665.8 | 45.9 |
| 8 | source_index | **20.2** | 185.2 | 195.6 | 123.6 | 2.1 | 665.8 | 43.6 |
| 20 | baseline | 22.9 | 197.0 | 196.6 | 127.4 | 2.4 | 656.2 | 49.1 |
| 20 | source_index | **19.6** | 204.1 | 197.8 | 126.3 | 2.1 | 655.8 | 47.4 |
| 32 | baseline | 23.5 | 177.8 | 196.2 | 120.1 | 2.5 | 666.7 | 37.6 |
| 32 | source_index | **20.1** | 187.9 | 196.8 | 118.7 | 2.1 | 666.7 | 40.7 |
| 39 | baseline | 22.1 | 155.7 | 194.7 | 111.2 | 2.1 | 662.5 | 32.4 |
| 39 | source_index | **18.6** | 165.2 | 195.4 | 110.4 | 2.1 | 664.7 | 36.1 |

规律：

1. **`count exchange` 每层稳定下降 3.5–4.9 µs**（L0 例外 −0.6）—— 源端写入确实减少了目标侧交换量；
2. **`pull token` 每层上升 8–11 µs** —— 目标 CTA 要按 quota 从源端读连续小组索引，pull 变长；
3. **`ECO plan` 基本持平**（L0 −4.5 µs，其余 −1.1 ~ +4.6 µs）——
   这是本轮相对串行 split 的关键改善（上轮 plan +30~34 µs，因为 exchange 被塞进 plan kernel）；
4. **`GEMM` 不变**（≤2.2 µs 波动）—— 分工布局未变。

**净**：exchange 省 4 µs、pull 涨 10 µs，每层净 +6 µs 量级；40 层摊薄后
约 +0.03 ms/步，与实测 +0.032 ms 一致。

## 4. 最慢 rank 明细

| layer | arm | rank | tokens | rows | M 块 | 活跃槽 | 首块可领 µs | pull 完成 µs | routed 退出 µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 2 | 194.0 | 1552 | 50 | 49 | 195 | 160 | 875.9 |
| 0 | source_index | 2 | 194.0 | 1552 | 50 | 49 | 199 | **175** | 881.1 |
| 8 | baseline | 7 | 198.0 | 1584 | 48 | 40 | 195 | 211 | 890.7 |
| 8 | source_index | 7 | 198.0 | 1584 | 48 | 40 | 193 | **216** | 886.0 |
| 20 | baseline | 3 | 180.0 | 1440 | 49 | 44 | 204 | 181 | 888.8 |
| 20 | source_index | 3 | 180.0 | 1440 | 49 | 44 | 208 | **197** | 889.6 |
| 32 | baseline | 4 | 203.0 | 1624 | 48 | 42 | 200 | 191 | 883.4 |
| 32 | source_index | 4 | 203.0 | 1624 | 48 | 42 | 194 | 191 | 886.7 |
| 39 | baseline | 3 | 208.0 | 1664 | 47 | 43 | 195 | 203 | 875.6 |
| 39 | source_index | 3 | 208.0 | 1664 | 47 | 43 | 196 | **216** | 881.4 |

**五层的最慢 rank 与 baseline 完全同 rank、同 tokens、同 rows、同 M 块、同活跃槽** ——
这是本轮最强的可比性证据（种子与放置完全一致）。
差别集中在 `pull 完成`：L0 +15、L8 +5、L20 +16、L32 0、L39 +13 µs。
即源码拉取的代价直接体现在 pull 尾端，与 §3 的 `pull token` 上升吻合。

## 5. 源端索引复制的三段时序（任务书 §2 要求）

对每个**有实际接收行**的多副本槽，`detail.json` 的 `replica_timing` 给出三个本 rank 设备时间轴上的标记：
`quota_us`（配额发布）、`first_issue_us`（首个 warp 发起激活复制前）、`last_block_ready_us`（最后一个 L1 M 块发布）。

| layer | 最慢 rank | quota_us | first_issue_us | last_block_ready_us | **quota→首笔 issue** | **首笔 issue→末块就绪** | 多副本槽数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 103 | 106 | 169 | **3** | **63** | 8 |
| 8 | 7 | 116 | 119 | 210 | **3** | **91** | 11 |
| 20 | 3 | 125 | 127 | 191 | **2** | **64** | 11 |
| 32 | 4 | 114 | 116 | 186 | **2** | **70** | 13 |
| 39 | 3 | 111 | 114 | 210 | **3** | **96** | 11 |

- **quota→首笔 issue 只有 2–3 µs**：配额一发布，目标侧立刻发出第一笔复制请求，
  说明"源端写 + 目标端读"的握手没有额外延迟。
- **首笔 issue→末块就绪 63–96 µs**：这一段是实际的数据搬运，也是 pull 变长的主体。
- 三个标记都在同一 rank 的同一设备时间轴上，与 Perfetto 轨可直接对齐；
  `first_issue` 含"实际发出指令前少量插桩"（任务书已注明）。
- **零行槽保留**：`replica_timing` 只列出有接收行的槽，空的零行槽在
  `slots_owned` / `pull_slots` 中保留计数，未丢弃。

与 baseline 的对照（同层同 rank）：

| layer | 首块可领 baseline→si | pull 完成 baseline→si | 完整 MoE（八卡均值段和）|
|---|---:|---:|---:|
| 0 | 195→199 (+4) | 160→175 (**+15**) | 1183.0 → 1183.9 (+0.9) |
| 8 | 195→193 (−2) | 211→216 (+5) | 1225.0 → 1227.3 (+2.3) |
| 20 | 204→208 (+4) | 181→197 (**+16**) | 1251.1 → 1252.8 (+1.7) |
| 32 | 200→194 (−6) | 191→191 (0) | 1243.3 → 1245.5 (+2.2) |
| 39 | 195→196 (+1) | 203→216 (**+13**) | 1179.2 → 1182.9 (+3.7) |

**首块可领时刻没有系统性推迟**（−6 ~ +4 µs，无方向），而 **pull 完成稳定推迟**（0 ~ +16 µs）——
即源端索引拉取**不影响 CTA 何时能开工**，只把 pull 阶段的尾部拉长。

## 6. 判读与证据边界

1. **本轮判定**：源端索引拉取是**方向正确但幅度不足**的改动。
   它消除了串行 split 在 plan kernel 里新增的 30–34 µs 等待，把总损失从 +0.157% 压到 +0.090%；
   但 `pull token` 的 +8~11 µs/层抵消了 `count exchange` 的 −4~−5 µs/层。
2. **不确定度**：每臂只跑一次，两轮 baseline 分别是 35.7853 与 35.7994 ms
   （差 0.014 ms，0.04%），**与本轮测得的 0.032 ms 差额同量级**。
   因此 +0.090% **不能**断言为确定性变慢，只能说明"没有可测收益，且方向为负"。
   要给出置信结论需要同轮 A/B/A/B 交替。
3. **跨轮不可比**：本轮与上轮的 baseline 都是含插桩口径，但两次 baseline 之间已有 0.04% 的漂移；
   主表 eco 的 35.600 ms 是**无插桩**口径，与本轮 35.785 不可直接相减。
4. **编译**：本轮 `source/` 有设计端改动（`sm90_fp8_fp4_gc_moe.py`、`sm90_gc_moe.py` layout），
   **编译一次通过**，执行者未做任何修复。

## 7. 交付物

`docs/trace_executor_20261119/sm90_bs256_source_index_results/`（161 MB，74 个文件）：

| 路径 | 内容 |
|---|---|
| `baseline/`、`source_index/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| `moe.csv` | 主指标（两臂 × 999/1000 步）|
| `summary.json` | 每样本 epoch/capture_id/kineto base/分段耗时/最慢 rank/每 rank tokens·M 块·活跃槽/`replica_timing` 三段 |
| `versions.txt`、`state-path.txt` | 版本与现场路径 |

**Perfetto 直链**（`<arm>` ∈ {baseline, source_index}，`<AA>`/`<EP>` =
00/46481、08/46489、20/46501、32/46513、39/46520）：

- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.perfetto.json`（粗粒度）
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.detail.perfetto.json`（absolute/detail）
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.counts.json`
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.detail.json`

## 8. 清理

| 项 | 内容 |
|---|---|
| 作业退出 | 两臂全部结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-source-index-code-run1` 已 `git worktree remove` 并 `prune` |
| 临时输出 | 仅保留 `/tmp/sm90-bs256-eco-QfEs2lbs`（试次 raw/Perfetto/日志，237 MB 级）供复查 |
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |
| 旧现场 | 其余 BS256/512/1024 state 目录均未改动 |
