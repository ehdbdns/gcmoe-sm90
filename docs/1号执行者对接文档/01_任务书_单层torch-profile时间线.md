# 交接文档：GC-MoE 单层 torch-profile 时间线重做

> **写这份文档的人（前任）把这件事做错了三次，下面第一、二节先说清他错在哪，
> 然后才是任务、环境、启动方式、要求。新接手的人请**完全不要相信**他产出的任何
> trace 结果文件，自己读代码、自己写抽取脚本。**

---

## 一、前任犯的错误（必须知道，否则会重复）

### 错误 1（最严重）：按「数组下标」而不是「trace 行号」取同一次调用

torch profile trace 里，每个 forward call 产生 **5 个 gc_moe kernel**，顺序固定：

```
dispatch(24×256) → shared(71×256) → routed(71×384) → plan(1×512) → combine(288×384)
```

即**一次调用占 5 个连续 trace 行**，`call_index = row // 5`。

前任把 5 个 role 分别抽出、各自排序，然后**都用同一个下标 39608** 去取。
但 5 个数组的第 39608 项**不是同一次调用**——实测相差 53 秒。
**因此前任生成的所有时间线图、所有 phase 时长表全部作废。**

正确做法：按 trace 行号取，同一 call 的 5 行行号连续且角色顺序固定，可逐 rank 断言。

### 错误 2：漏掉了「GEMM 等 token」这一段

前任把 routed kernel 整段标成 `GEMM`。实际上 routed kernel 启动后，
CTA 还要等本地槽位把 token 拉完才能开始算。这段必须单独成为一段。

### 错误 3：用两次不同运行的数据拼一张图

perfetto 来自一次运行（t19），token/M块/预测表来自另一次运行（t20）。
两份数据的「同一层」根本不是同一层。**任何图、表都必须来自同一次运行。**

### 错误 4：用相关性去「猜」是哪一步哪一层

前任拿 token 数和 GEMM 时长做相关性，猜出 `step 992 / layer 8`。
这是猜的，不是记录的。**不要这么做。**

### 错误 5：改了内核插桩，把架构行为改坏了

前任在 t20 目录里改 role4 的 smem 布局和打点，导致 shared 被排到 routed 之后执行，
与生产环境不符，且污染了 6 轮实验。**不要改内核。用 torch profile 就够。**

---

## 二、警告

**不要相信前任产出的任何 trace 相关文件**，包括但不限于：

- `docs/final_orig_layer/` 下的 `one_layer.png`、`GROUPED.md` 里的所有数值
- `perfetto_from_torch.json`、`one_call_from_torch.perfetto.json`
- `/tmp/k11/*.json`（按 role 数组下标建的索引，方法就是错的）
- `/tmp/rows39608.json`、`/tmp/one_call.json`
- `gcmoe_t20_aligned2/` 整个目录（内核被改坏）

只有 `prof/*/traces/rankN.json` 这些**原始 torch profile trace 文件**是可信的，
因为它们是 profiler 直接导出的。

---

## 三、任务目标

用 **torch profile** 对**某几个层**（不需要全部）做时间线，产出：

1. **一个 perfetto 文件**，8 个 rank，每 rank 分段：
   - `count exchange`（dispatch 内核内，routing + count 交换）
   - `pull token`（dispatch 内核内，拉 token）
   - `shared L1+L2`（role4 内核）
   - `ECO plan`（role5 内核）
   - **`wait token`**（routed 内核起 → 本 rank 最后一个本地槽 pull 完成）
   - **`GEMM`**（上述时刻 → routed 内核止）
   - `combine`（role3 内核）
2. **同一层的各 rank 的 M 块数、token 数**
3. **同一层的 plan 预测耗时 vs 实测耗时**，判断 plan 准不准
4. 推送 perfetto + 说明文档到 GitHub
5. 画一张同数据的 PNG 时间线图（标签居中、字号大、8 rank 行序一致）

### 硬约束（物理事实，用来校验你的结果）

| 关系 | 预期 |
|---|---|
| `shared` 起点 − `dispatch` 起点 | ≈ 0（**几乎同时启动**，实测 −2 ~ +3 µs）|
| `routed` 起点 − `shared` 结束 | **必须 > 0**，实测 +0.9 ~ +1.2 µs（`role2 wait_for=(4,)`）|
| `combine` 起点 − `routed` 结束 | > 0，实测 +2.6 ~ +4.5 µs |
| `count exchange` 止 == `pull token` 起 | 无缝衔接 |
| `wait token` + `GEMM` | == routed kernel 总时长 |

**只要有一项不满足，说明你又取错了调用，立刻停下重来。**

---

## 四、环境

### 源码与实验目录

```
/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/
```

- `source/gc_moe/` 内核源码（**不要改**）
- `integration/deepep_compare.py` vLLM 接入层
- `scripts/real_decode_100_100.py` 实验主脚本
- `scripts/prof_phases.py` 专门做 profile 的脚本（**推荐用它，它已经内置了
  prefill → EPLB 重排 → 再 profile 的正确顺序**）
- `scripts/env.sh` 运行环境变量
- `config/experiment.env` 实验配置
- `scripts/warm_page_cache.sh` 权重预热（**必须用**）
- `prof/final_v2/traces/rank{0..7}.json` 已存在的 torch profile trace（可直接分析）
- `runs/FINAL_V2.json` 与上面 trace 同一次运行的结果 JSON（含 `ir_step` 表、`eco_trace`）

### 模型与数据集

```
/mnt/nas2/models/DeepSeek-V4.1-Flash
/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
.../dsv4-flash_online_input_ids_26-10-08_2.jsonl
```

---

## 五、怎么读 trace（核心技术点）

### 5.1 kernel 识别

torch trace 里 `cat == "kernel"` 且 `name` 含 `gc_moe` 的条目，按 `args.grid` / `args.block` 区分 role：

| grid | block | role |
|---:|---:|---|
| 24 | 256 | dispatch (role1) |
| 71 | **256** | shared (role4) |
| 71 | **384** | routed (role2) |
| 1 | 512 | plan (role5) |
| 288 | 384 | combine (role3) |

**注意 shared 和 routed 的 grid 都是 71，只能靠 block 区分（256 vs 384）。前任在这里也差点出错。**

### 5.2 按行号取同一次调用

```python
# 只扫需要的行范围，别全量扫 2.7GB
rows = [e for e in trace if cat=='kernel' and 'gc_moe' in name]
call_index = row // 5      # 每 5 行一个 call
```

一次调用 = 5 个连续行，角色顺序固定。**断言 8 个 rank 的角色序列一致。**

### 5.3 count exchange / pull token 的切分

`pull_tl[epoch][10]` = dispatch 入口，`[3]` = routing/count 交换结束，`[5]` = 最后一个本地槽拉完。
用**同一次调用内**的差值（与时钟基准无关）：

```
count_exchange_frac = ([3]-[10]) / (dispatch_kernel_duration)
```

然后按比例切分 dispatch kernel span。**不要用绝对值。**

### 5.4 wait token 的计算

```
routed_start        = routed kernel 的 ts
pull_done           = dispatch 入口 ts + ([5]-[10])
wait_token_end      = clamp(pull_done, routed_start, routed_end)
wait_token          = wait_token_end - routed_start
gemm                = routed_end - wait_token_end
```

**关键**：必须保证 `pull_tl` 那一行和 trace 该行是**同一次调用**。
校验方法：`[10]` 换算后与 dispatch kernel 的 ts 相差应 < 几 µs（同一时钟域，
`pull_tl` 是 globaltimer µs，CUPTI 是 ns，需要找偏移；用整次运行的
`trace_ts - pull_tl[10]` 做线性拟合，残差应 < 1 µs）。

---

## 六、怎么启动实验

### 6.1 推荐：直接用现成的 profile 脚本

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn
source scripts/env.sh
export GC_MOE_EPLB=1 GC_MOE_ECO=1
export GC_MOE_SRC=$PWD/source
export PYTHONPATH=$PWD/source:$PWD/patches:$PWD/integration
bash scripts/warm_page_cache.sh          # 必须，否则权重加载 10 分钟
timeout 3600 $GC_ENV/bin/python -u scripts/prof_phases.py \
    --output prof/my_trace --prof-steps 5
```

`prof_phases.py` 已经实现：prefill（不计时）→ EPLB 重排 → 再 warm → 开 profiler →
跑几步 → 导出 trace。**这正是「选重排之后的步数」的正确做法。**

### 6.2 或者用主脚本

```bash
bash scripts/run_arm_eco.sh 1 1 MYLABEL \
    --seqs-per-rank 32 --warm-steps 100 --measure-steps 100 \
    --profile-out prof/my_trace
```

- 参数：`<eplb 0|1> <eco 0|1> <label>`
- `--profile-out` 触发 torch profiler
- 内部顺序：prefill → **EPLB 重排** → flush 5 步 → 开 profiler → measure

### 6.3 关键环境变量

| 变量 | 作用 |
|---|---|
| `GC_MOE_EPLB=1` | 开启 EPLB 放置 |
| `GC_MOE_ECO=1` | 开启 ECO plan（role5）|
| `GC_MOE_SHARED_IN_GC=1` | **必须开**，否则 role4 不存在，shared 段无法测量 |
| `GC_MOE_PULL_TRACE=1` | 打点 `pull_tl`（`[3]`/`[5]`/`[15]` 等）|
| `GC_MOE_ECO_TRACE=1` | 打点 `eco_tl` |
| `GC_MOE_IR_STEP=1` | 记录每 (layer, step) 的 token/M块表 |
| `GC_MOE_CAPACITY=2048` | kernel 编译期容量 |

**`GC_MOE_SHARED_IN_GC=1` 不开的话，role4 内核根本不存在，shared 段会全部是 0。**

---

## 七、获取「M 块数 / token 数 / plan 预测」的正确方法

同一次运行的结果 JSON（`runs/*.json`）里有：

| 字段 | 含义 |
|---|---|
| `ir_step.tables.tokens[layer][step]` | 该层该步、8 个 rank 的 token 数 |
| `ir_step.tables.m_blocks[layer][step]` | 该层该步、8 个 rank 的 M 块数 |
| `ir_step.tables.epoch0` | 测量窗口起始 epoch |
| `eco_trace.ranks[r].pull_rows[epoch%256][24..31]` | plan 导出的预测 T[0..7]（µs）|

**取哪一层**：`epoch = epoch0 + step*40 + layer`，或用 trace 里该 call 的 dispatch kernel
与 `pull_rows` 的 `[10]` 对齐（见 5.4 的拟合方法）。

**取哪一层做图**：任选重排之后的一个中间层（例如 measure 窗口里的第 100 步、layer 20），
**只要 perfetto 和表格用的是同一次运行的同一次调用即可。**

### plan 准不准的判据

```
pred_ratio[r] = T[r] / mean(T)
meas_ratio[r] = measured[r] / mean(measured)
L1 = sum(|pred_ratio[r] - meas_ratio[r]|)
```
L1 越小越准。

---

## 八、必须遵守的工作方式

1. **先读源码，再动手**。不要凭猜测写脚本。
2. **最小规模验证**：只抽 trace 的几行（例如 `row in range(39605,39610)`）就够，
   不要全量扫 2.7 GB × 8 个文件（前任在这上面浪费了大量时间）。
3. **不要长时间 sleep**。轮询时每 20s 一次，实时汇报进度。
4. **改代码后必须确认跑的是新编译产物**（看日志里的编译时间戳）。
5. **启动前核对脚本**（路径、环境变量、参数），别白跑 10 分钟。
6. **启动报错立即停下汇报**，不要反复瞎试。
7. **数据出来必须检查物理合理性**（用第三节的硬约束表）。
8. **CPU 短测通过后再上真实推理**。
9. 权重加载必须开预热（`warm_page_cache.sh`），否则每次多等 10 分钟。
10. **不要陷入完美主义**：抽 1~2 个层就够，不需要全 40 层。
11. **做完直接推送，不要反复征求同意**。

---

## 九、交付物清单

- [ ] `docs/final_orig_layer/<name>.perfetto.json` —— 8 rank × 7 段
- [ ] `docs/final_orig_layer/<name>.png` —— 同一数据的时间线图
- [ ] `docs/final_orig_layer/<name>.md` —— 说明 + 数值表 + 校验结果
- [ ] 文档中给出该层的 `step`/`layer`、8 个 rank 的 token/M块/预测/实测
- [ ] 文档中说明 plan 预测是否准确（L1 值）
- [ ] git push

GitHub 仓库：`/tmp/eco_publish/repo`，remote `github.com/ehdbdns/gcmoe-sm90`
推送命令：

```bash
cd /tmp/eco_publish/repo
git add -A docs/final_orig_layer
git -c user.email=agent@local -c user.name=agent commit -m "..."
timeout 300 git -c http.postBuffer=524288000 push origin HEAD
```

---

## 十、已知可用的中间产物

- `prof/final_v2/traces/rank{0..7}.json`：torch profile 原始 trace（2.7 GB each，可信）
- `runs/FINAL_V2.json`：与上面**同一次运行**的结果（863 MB，含 ir_step 表与 eco_trace）
- 它们对应的配置：BS256、原始耗时函数、CUDA graph、mb-claim、EPLB 重排后

**这些是唯一可信的起点。其余前任的中间文件一律不要用。**
