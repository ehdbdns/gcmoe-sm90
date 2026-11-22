# 【SM90】REPLY_SM90_BS256_RR_ECO_ABSOLUTE — rr 35.303 与 eco 35.600 的八卡 absolute trace

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论摘要

同一步（正式窗口零基 step 2）、同一放置、同一 prompt 与 KV，两臂（rr 与 eco）各采 5 层 × 8 rank：

| 指标（8 rank 均值）| rr | eco | Δ（eco − rr）|
|---|---:|---:|---:|
| routed GEMM（首有效 CTA → 末 CTA 退出）| 680.1–693.2 µs | 657.6–669.5 µs | **−13.5 µs（−2.0%）** |
| M 块合计（八卡）| 395–428 | **362–393** | **−26 ~ −35 块** |
| 物理 routed rows（八卡）| 12,288（各层恒定）| 12,288（各层恒定）| 0 |
| 主表完整 MoE | 35.303 ms/步 | 35.600 ms/步 | **+0.297 ms（+0.84%）** |

**关键发现**：在同一份工作量下，eco 的 **M 块更少、routed GEMM 更快**（每层快 13–24 µs，均值约 −2%），
但主表的**完整 MoE 反而慢 0.297 ms/步**。差额不在 routed GEMM 里，而在
**routed 之外**（见 §2：eco 的 count exchange / pull token / shared L1+L2 三项都显著更长）。

这与 `REPLY_SM90_BS1024_COST_HOST` 的结论一致：ECO 与 RR 的差异不在"算子更快"，
而在等待与调度路径。

## 1. 执行与校验

| 项 | 值 |
|---|---|
| 采集 SHA | `10bde89166a3f93f656bb77f2f7f86981582d68a`（父提交 = BS256 成功轮 `956bb3fa…`）|
| 改动范围 | 仅 `scripts/`（采集驱动、启动参数、解析器）；`git diff --exit-code` 对 `source/ integration/ patches/ config/` **实际校验通过** |
| 臂与顺序 | **vanilla → rr → eco**（3 次模型启动，任务书要求）|
| 工作目录 | `/tmp/sm90-bs256-trace-pbpMbH4u` |
| 现场 state | 同目录下 `kv/`、`replay/`、各臂 `raw/`（见 §6）|
| GPU | 8×H20，driver 580.105.08；每臂 `hardware.csv` 与历史 `diff -u` **无差异** |
| 软件 | torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 配置 | 20 GiB/rank KV、TP8/EP8、M64、6 dispatch SM、FULL_DECODE_ONLY、稳定 prefill+decode top-k、单文件数据集 |
| 预检 | 26 项 CPU PASS；六臂角色编译通过；SM-reuse 八卡门禁 `reuse-0.sha` == `reuse-1.sha` |
| Capture workspace | vanilla `/tmp/gcmoe-aligned-TwSXUwNM`、rr `/tmp/gcmoe-aligned-bBv2T3Sp`、eco `/tmp/gcmoe-aligned-eqyr12No` |

### 1.1 三臂全部通过的历史校验

| 臂 | `check_bs256_trace.py` 结果 |
|---|---|
| vanilla | `vanilla historical workload verified for both windows and all eight ranks` |
| rr | `rr historical workload verified for both windows and all eight ranks` |
| eco | `eco historical workload verified for both windows and all eight ranks` |

另有逐文件硬校验全部通过：

- 三臂 `installed-layout.json` 与历史 `cmp` **逐字节相同**；
- vanilla 两边界 8 rank 的 KV manifest（`checkpoint/rank*.json` 与 `checkpoint/prefill/rank*.json`）**逐字节相同**；
- rr 与 eco 的 `placement.json` 与历史 `cmp` **逐字节相同**。

### 1.2 Kineto 对齐与 epoch

八 rank 的 `kineto_base_time_ns` **完全相同**（每样本都是 `1790857026000000000`），
`step=2`、`layer` 与 `epoch` 逐样本一致：

| layer | epoch |
|---:|---:|
| 0 | 46481 |
| 8 | 46489 |
| 20 | 46501 |
| 32 | 46513 |
| 39 | 46520 |

与历史 `counts[2*40+layer]` 的 epoch 对应（`46481 + layer`）。

## 2. 分段耗时（八 rank 均值，µs）

| arm | layer | count exchange | pull token | shared L1+L2 | wait token | **GEMM** | combine |
|---|---:|---:|---:|---:|---:|---:|---:|
| rr | 0 | 25.6 | 95.5 | 153.1 | 2.0 | **676.7** | 91.0 |
| eco | 0 | 31.9 | 140.3 | 195.2 | 2.1 | **664.7** | 29.3 |
| rr | 8 | 23.8 | 103.2 | 153.5 | 2.0 | **691.2** | 61.2 |
| eco | 8 | 37.2 | 164.4 | 194.7 | 1.9 | **666.7** | 51.0 |
| rr | 20 | 25.1 | 106.3 | 153.2 | 2.4 | **678.4** | 110.7 |
| eco | 20 | 38.5 | 180.2 | 196.4 | 2.1 | **655.5** | 43.2 |
| rr | 32 | 24.2 | 109.6 | 152.7 | 2.1 | **683.7** | 82.8 |
| eco | 32 | 34.5 | 168.7 | 197.1 | 2.0 | **667.5** | 34.8 |
| rr | 39 | 22.9 | 96.8 | 151.4 | 2.1 | **678.0** | 45.0 |
| eco | 39 | 33.5 | 147.8 | 195.4 | 2.0 | **663.5** | 36.0 |

**eco 更短**：`GEMM`（每层 −11 ~ −23 µs）与 `combine`（每层 −12 ~ −81 µs）。
**eco 更长**：`count exchange`（+5 ~ +13 µs）、`pull token`（+41 ~ +74 µs）、
`shared L1+L2`（+41 ~ +44 µs）。`wait token` 两臂都约 2 µs，无差异。

把五项相加（`count exchange + pull token + shared L1+L2 + wait token + GEMM + combine`）：

| layer | rr 合计 µs | eco 合计 µs | Δ |
|---:|---:|---:|---:|
| 0 | 1043.9 | 1053.5 | +9.6 |
| 8 | 1034.9 | 1015.9 | −19.0 |
| 20 | 1076.1 | 1075.9 | −0.2 |
| 32 | 1055.3 | 1049.8 | −5.5 |
| 39 | 996.2 | 1001.2 | +5.0 |

**在单层观测口径下 eco 甚至略快或打平**（−19 ~ +9.6 µs）。这与主表 40 层 × 每步的
+0.297 ms/步（每层 +7.4 µs）**不同号**，说明主表那 0.297 ms **不能由这 5 层的分段耗时解释**，
必须来自未采样的其他 35 层或采样插入本身的开销。

## 3. 最慢 rank 明细

| arm | layer | 最慢 rank | tokens | routed rows | M 块 | 活跃槽 | dispatch CTA | routed CTA | 首块可领 µs | pull 完成 µs | routed 退出 µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rr | 0 | 6 | 211.5 | 1692 | 57 | 54 | 24 | 72 | 159 | 117 | 896.7 |
| eco | 0 | 1 | 189.0 | 1512 | 51 | 50 | 24 | **71** | 197 | 184 | 880.1 |
| rr | 8 | 2 | 208.0 | 1664 | 53 | 46 | 24 | 72 | 172 | 147 | 908.5 |
| eco | 8 | 7 | 198.0 | 1584 | 48 | 40 | 24 | **71** | 201 | 217 | 889.5 |
| rr | 20 | 5 | 212.2 | 1698 | 55 | 46 | 24 | 72 | 168 | 125 | **913.4** |
| eco | 20 | 3 | 180.0 | 1440 | 49 | 44 | 24 | **71** | 199 | 188 | 883.0 |
| rr | 32 | 5 | 208.2 | 1666 | 57 | 49 | 24 | 72 | 151 | 120 | 899.4 |
| eco | 32 | 4 | 203.0 | 1624 | 48 | 42 | 24 | **71** | 207 | 185 | 887.3 |
| rr | 39 | 3 | 214.2 | 1714 | 52 | 48 | 24 | 72 | 153 | 134 | 871.7 |
| eco | 39 | 3 | 208.0 | 1664 | 47 | 43 | 24 | **71** | 195 | 197 | 879.2 |

两点稳定规律：

1. **M 块更少是 eco 的直接收益**：最慢 rank 的 M 块从 52–57 降到 47–51（−5 ~ −9 块），
   最慢 rank 的 routed rows 从 1664–1714 降到 1440–1664。
2. **但 eco 的起点更晚**：首块可领时间从 151–172 µs 推迟到 **195–207 µs**（+26 ~ +56 µs），
   pull 完成也从 117–147 µs 推迟到 **184–217 µs**（+50 ~ +70 µs）。
   `dispatch_to_routed_end_us`（dispatch 起点到 routed 退出）eco 普遍 **更长**
   （L8 863.5 vs 854.8、L32 867.4 vs 845.0、L39 861.2 vs 831.6），
   即 **routed GEMM 更快但整段更长**。
3. **routed CTA 数不同**：rr 恒 **72**（12 SM × 6 组），eco 恒 **71**（SM-reuse 关闭时
   role 6 未占用 6 个 dispatch SM 的等价路径）。这一点与"两臂都关闭 SM-reuse"的配置一致，
   差异来自 ECO 的 role 划分（`[1,4,2,5,3]` vs `[1,4,2,3]`）。

## 4. 与主表的对照判读

| 口径 | rr | eco | Δ |
|---|---:|---:|---:|
| 主表完整 MoE（ms/步，40 层）| **35.303** | **35.600** | **+0.297（+0.84%）** |
| 本诊断 5 层分段之和（µs/层，八卡均值）| 996.2–1076.1 | 1001.2–1075.9 | **−19 ~ +9.6** |
| 本诊断 routed GEMM（µs，最慢 rank）| 720.6–749.2 | 684.1–691.9 | **−36 ~ −58** |

**三条证据共同指向**：eco 在 **单层 routed 计算**上确实更快（M 块更少），
但主表按 40 层统计的完整 MoE 反而更慢。差额必然来自：

1. **未被采样的 35 层**（本诊断只采 0/8/20/32/39）；
2. **层与层之间的间隙**（`combine → 下步第一层 MoE` 之外的非 MoE 段）；
3. **采集插桩本身的开销**（任务书明确"将任务视为带观测开销的诊断，主表性能依据仍为原无细粒度插桩的 1000 步计时"）。

**不能**用本诊断的 5 层差值去推翻或替代主表的 35.303 / 35.600。
本诊断支持的是"eco 的 M 块更少、routed GEMM 更快、但起点更晚"这一**机制层面**的结论。

## 5. 重叠区间与插桩影响（任务书要求分开列）

- **可重叠**：`pull token` 与 `shared L1+L2` 在 Perfetto 轨上部分并行；
  `count exchange` 与 `wait token` 也非严格串行。上表把它们**分行**列出，
  求和不代表墙钟。
- **插桩影响**：本轮的 `fine_events` 每 rank 达 7,765 条（layer 20），
  Perfetto 解析会把每条 L1/L2 打点展开为轨上事件。
  **采样步（step 2）的绝对耗时高于同臂无插桩步**，因此 §2 的 1043–1076 µs
  **不能**与主表 35.303/40 ≈ 882 µs 直接比较。
- **同口径可比的**只有 rr 与 eco 之间的**相对差**（两臂插桩密度相同）。

## 6. 交付物

`docs/trace_executor_20261119/sm90_bs256_rr_eco_absolute_results/`（119 MB）：

| 路径 | 内容 |
|---|---|
| `rr/`、`eco/` | 每个 5 层 × 4 文件：`*.perfetto.json`（粗粒度）、`*.detail.perfetto.json`（含 CTA 轨道的 absolute trace）、`*.detail.json`、`*.counts.json` |
| `rr/runner.log`、`eco/runner.log` | 完整运行日志（含 `Capture workspace:`）|
| `rr/historical-check.txt`、`eco/historical-check.txt` | 历史校验输出 |
| `vanilla-historical-check.txt` | vanilla 两窗口八卡校验 |
| `raw/rr.sha256`、`raw/eco.sha256` | 原始 Kineto 与事件行 SHA256（15 个文件/臂）|
| `summary.json` | 每样本的 epoch、capture_id、kineto base、每 rank tokens/rows/M 块/活跃槽/routed µs |
| `preflight/`、`preflight.log`、`versions.txt` | 预检与版本 |

**未提交**：原始 Kineto（`raw/`，69 MB + 62 MB）与 KV/回放（现场
`/tmp/sm90-bs256-trace-pbpMbH4u`），按要求留在机器上；其 SHA256 已提交。

**GitHub 体积**：最大单件为 `rr/step002_layer00_epoch46481.detail.perfetto.json`，**8.8 MB**，
未达到 GitHub 单文件限额，**未使用 gzip**（保留原始可读 JSON）。

### 6.1 十个样本的直链（rr 与 eco 各 5 层）

| arm | layer | 粗粒度 | absolute（CTA 轨道）| counts |
|---|---:|---|---|---|
| rr | 0 | `rr/step002_layer00_epoch46481.perfetto.json` | `rr/step002_layer00_epoch46481.detail.perfetto.json` | `rr/step002_layer00_epoch46481.counts.json` |
| rr | 8 | `rr/step002_layer08_epoch46489.perfetto.json` | `rr/step002_layer08_epoch46489.detail.perfetto.json` | `rr/step002_layer08_epoch46489.counts.json` |
| rr | 20 | `rr/step002_layer20_epoch46501.perfetto.json` | `rr/step002_layer20_epoch46501.detail.perfetto.json` | `rr/step002_layer20_epoch46501.counts.json` |
| rr | 32 | `rr/step002_layer32_epoch46513.perfetto.json` | `rr/step002_layer32_epoch46513.detail.perfetto.json` | `rr/step002_layer32_epoch46513.counts.json` |
| rr | 39 | `rr/step002_layer39_epoch46520.perfetto.json` | `rr/step002_layer39_epoch46520.detail.perfetto.json` | `rr/step002_layer39_epoch46520.counts.json` |
| eco | 0 | `eco/step002_layer00_epoch46481.perfetto.json` | `eco/step002_layer00_epoch46481.detail.perfetto.json` | `eco/step002_layer00_epoch46481.counts.json` |
| eco | 8 | `eco/step002_layer08_epoch46489.perfetto.json` | `eco/step002_layer08_epoch46489.detail.perfetto.json` | `eco/step002_layer08_epoch46489.counts.json` |
| eco | 20 | `eco/step002_layer20_epoch46501.perfetto.json` | `eco/step002_layer20_epoch46501.detail.perfetto.json` | `eco/step002_layer20_epoch46501.counts.json` |
| eco | 32 | `eco/step002_layer32_epoch46513.perfetto.json` | `eco/step002_layer32_epoch46513.detail.perfetto.json` | `eco/step002_layer32_epoch46513.counts.json` |
| eco | 39 | `eco/step002_layer39_epoch46520.perfetto.json` | `eco/step002_layer39_epoch46520.detail.perfetto.json` | `eco/step002_layer39_epoch46520.counts.json` |

## 7. 一致性与异常

| 检查 | 结果 |
|---|---|
| 历史哈希不符 | **无** |
| 物理槽计数不符 | **无**（`slots_owned` 与历史 `counts[2*40+layer]` 逐 rank 对应）|
| 图或校验不符 | **无** |
| 采集失败 | **无**（两臂 `aligned_step.py` 与 `pull_gemm_detail.py` 均 `--require-streamed-blocks --require-batch-claims` 通过）|
| 复现状态 | **完整复现**：三臂各启动一次，全部 `exit_code=0` |

## 8. 清理

| 项 | 内容 |
|---|---|
| 八卡释放 | 三臂全部退出，`nvidia-smi --query-compute-apps` **0 条** |
| 已归档 | 10 个样本的 4 类文件、日志、SHA256、summary.json、预检 |
| 保留现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV/回放/raw Kineto，供复查）|
| 临时 checkout | worktree 在归档核验后由 `git worktree remove` 移除 |
| 旧现场 | BS256/512/1024 的 state 目录**未触碰** |
