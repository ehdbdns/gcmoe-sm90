# 最新环境的运行配置说明（2026-11-19）

本目录同时收录了**当前可用的完整源码**（`source/`、`integration/`、`scripts/`、`config/`），
可以开启全部最新实验配置：**MB-CLAIM / PULL-TRACE / ECO-TRACE / IR-STEP / 逐层 EPLB**。

> 兼容性声明：本文的源码对应 commit `sync: latest environment ...`。
> 内核侧的 `ir_step_epoch`（交接文档 §5.1 提到的实验性改动）**已移除**，
> 因为它会让 CuTe DSL 编译失败（`st_global_u32` 那版未验证通过）。
> 目前 `ir_step` 与 `pull_tl` 的 epoch 对齐仍依赖 host 侧 `epoch0`，见 §6。

---

## 1. 一次完整实验的最小命令

```bash
cd <本仓库>
export MODEL_DIR=/path/to/DeepSeek-V4.1-Flash        # 或直接 export MODEL=/path/...
export DATASET_DIR=/path/to/datasets

source config/experiment.env
source scripts/env.sh

# 关键：PYTHONPATH 必须指向本目录，否则跑的是环境里未修改的库
export PYTHONPATH=$PWD/source:$PWD/patches:$PWD/integration

# 可选：模型预热（冷页缓存时必需，否则加载 10 分钟）
export GC_MOE_SKIP_WARM=1 GC_MOE_NO_PREFETCH=1

$GC_ENV/bin/python -u scripts/real_decode_100_100.py \
  --label MYRUN --output runs/MYRUN.json \
  --seqs-per-rank 32 --warm-steps 1000 --measure-steps 1000
```

`SEQS_PER_RANK` 与 BS 的关系（8 卡）：

| BS | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| `--seqs-per-rank` | 4 | 8 | 16 | 32 |

---

## 2. 五个"臂"怎么开

`scripts/run_arm_eco.sh <eplb> <eco> <label>` 的两个参数组合出五臂：

| 臂 | 命令 | 说明 |
|---|---|---|
| vanilla | `0 0` | 无副本、无 plan |
| random+rr | `1 0` + `GC_MOE_SLOT_PLACEMENT=random GC_MOE_EPLB_REARRANGE=0` | 真对照臂 |
| eplb+rr | `1 0` | 有副本，host 轮询分流 |
| eplb+eco | `1 1` | 副本 + 内核 plan（role 5） |
| eco+smr | `1 1` + `GC_MOE_SM90_SM_REUSE=1` | 再让 role 6 复用 dispatch 让出的 6 个 SM |

---

## 3. 开关总表（按用途分组）

### 3.1 放置（EPLB）

| 变量 | 默认 | 含义 |
|---|---|---|
| `GC_MOE_EPLB` | 0 | 开副本放置 |
| `GC_MOE_EPLB_EXTRA_PER_RANK` | 8 | **每 rank 冗余槽数**（384 专家 + 8×8 = 448 物理槽）|
| `GC_MOE_EPLB_REDUNDANCY` | 0.10 | 旧的**比例**写法（被上面覆盖；仅在未设 EXTRA 时生效）|
| `GC_MOE_ECOPLB_PER_LAYER` | 1 | **逐层放置**（DeepInfer 语义）。设 0 会让 40 层共享一张表 → IR 一半 1.1 一半 2.0 |
| `GC_MOE_ECO_PER_LAYER` | 1 | 逐层的 warm-start quota 也各存一份 |
| `GC_MOE_SLOT_PLACEMENT` | — | `random` = 固定种子打散；必须同时 `GC_MOE_EPLB_REARRANGE=0` |
| `GC_MOE_EPLB_NO_REARRANGE` | 0 | 1 = 不做重排（对照臂用）|

### 3.2 分流（ECO plan，role 5）

| 变量 | 默认 | 含义 |
|---|---|---|
| `GC_MOE_ECO` | 0 | 开内核 plan |
| `GC_MOE_ECO_WB` | 0 | **每多一个 M 块的代价**（token 单位）。主线用 **26** |
| `GC_MOE_ECO_WA` | 0 | 每多一个激活槽的代价。主线用 **0** |
| `GC_MOE_ECO_MAX_COPIES` | 8 | EcoMeta 每个专家最多存几个副本；必须 ≥ 实际最大副本数 |
| `GC_MOE_ECO_ROUNDS` | 1 | 稳态下拍卖轮数 |
| `GC_MOE_ECO_COLD_ROUNDS` | 6 | 冷启动轮数 |
| `GC_MOE_ECO_COSTFN` | 0 | 实测查表 cost。**默认关**：8/8 更差（见 `docs/COSTFN_RESULT.md`）|
| `GC_MOE_ECO_BLOCKS_ONLY` | 0 | 只按 M 块计费（实验开关）|
| `GC_MOE_ECO_TOKENS_ONLY` | 0 | 只按 token 计费（实验开关）|

### 3.3 领任务粒度

| 变量 | 默认 | 含义 |
|---|---|---|
| `GC_MOE_MB_CLAIM` | 0 | **1 = 每个 (expert, ntile, M块) 一张票**。BS256 实测 −6.1% ms，token 不变；0 = 历史行为（一个 CTA 吃完整 tile 的所有块）|

### 3.4 打点 / 测量

| 变量 | 默认 | 含义 |
|---|---|---|
| `GC_MOE_PULL_TRACE` | 0 | 写 `pull_tl`：`[10]` dispatch 入口、`[3]` routing 结束、`[5]` 最后的槽拉完、**`[15]` 首个 CTA 领到任务**、`[24..31]` plan 的 T[r] |
| `GC_MOE_ECO_TRACE` | 0 | 写 `eco_tl`（plan 各阶段时间戳）。**关掉就没有 T[r]** |
| `GC_MOE_IR_STEP` | 0 | 内核记录每 `(layer, step)` 的 token / M块 / 激活槽三张表 |
| `GC_MEASURE_IR_STEP` | 1 | driver 侧是否去 dump 上面那张表（须配合上一条）|
| `GC_IRSTEP_STEPS` | 1024 | 表的 step 容量 |
| `GC_MOE_LAYERS` | 40 | 层数；**必须与实际模型一致**，否则 (layer,step) 索引全错 |
| `GC_MOE_FINE_TRACE` | 0 | 逐事件明细日志（每事件一次全局原子，开销大）|
| `GC_MEASURE_FLUSH` | 5 | 重排后丢弃几步再开始计时。**做时间线对齐时要设 0**，否则环被 flush 步占满 |

### 3.5 计算 / 显存

| 变量 | 默认 | 含义 |
|---|---|---|
| `GC_MOE_SM90_DISPATCH_SMS` | 4 | dispatch 用几个 SM（实测 2 更慢，别调）|
| `GC_MOE_SM90_SM_REUSE` | 0 | 1 = role 6 复用 dispatch 的 6 个 SM（BS32–128 有收益，BS256 略负）|
| `GC_MOE_SM90_PULL_TMA` | 1 | token pull 走 TMA（关掉会慢很多）|
| `GC_MOE_SHARED_IN_GC` | 0 | 0 = vLLM 算 shared expert（gc_moe 只 4 个 kernel）；1 = gc_moe role 4 算（5 个 kernel）|
| `GC_MOE_CAPACITY` | 2048 | 编译期容量 |
| `GC_MOE_ECO_MAX_COPIES` | 8 | 见上 |
| `GPU_UTIL` | 0.85 | 8 个 worker 抢显存时用 **0.72**；0.85 会 OOM |
| `MAX_BATCHED` | 2048 | `--max-num-batched-tokens` |

### 3.6 环境 / 加载

| 变量 | 含义 |
|---|---|
| `GC_MOE_SKIP_WARM=1` | 跳过那个只发 advisory 的假预热脚本 |
| `GC_MOE_NO_PREFETCH=1` | **必开**。否则 vLLM 8 线程后台 prefetch 与 worker 抢 NAS 带宽，加载 25 s → 657 s |
| `GC_ENV` | python 环境（`scripts/env.sh` 里设置）|
| `NCCL_IB_HCA` / `NVSHMEM_HCA_LIST` | **每台机器不同**，用 `ibdev2netdev` 查，从 shell 覆盖 |

---

## 4. 三套典型配置

### 4.1 主表（性能结论）

```bash
GC_MOE_EPLB=1 GC_MOE_ECO=1 GC_MOE_MB_CLAIM=1 \
GC_MOE_EPLB_EXTRA_PER_RANK=8 GC_MOE_ECOPLB_PER_LAYER=1 \
GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0 GC_MOE_ECO_MAX_COPIES=8 \
GC_MOE_SM90_PULL_TMA=1 \
--warm-steps 1000 --measure-steps 1000
```

### 4.2 单层时间线（7 段 perfetto）

```bash
# 在上面的基础上再加：
GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 \
GC_MOE_IR_STEP=1 GC_MEASURE_IR_STEP=1 \
GC_MOE_MEASURE_FLUSH=0 \
--warm-steps 200 --measure-steps 2 --profile-out prof/x
```

> `GC_MOE_MEASURE_FLUSH=0` + 少步数是关键：`pull_tl` 只保留**最后 256 个调用**，
> 而一步 = 40 个调用，所以测量窗口必须 ≤ 6 步，否则环里最早的已被覆盖。

### 4.3 只做 CPU 验证（不占卡）

```bash
$GC_ENV/bin/python scripts/smoke_compile_eco.py        # 编译 role cubin，~15 s
$GC_ENV/bin/python eco_ref/test_unit_ref.py            # plan 与 numpy 参考逐位对拍
$GC_ENV/bin/python tests/test_eplb_logic_cpu.py
```

---

## 5. 运行前必查

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 必须全 4 MiB
```

清理残留（**本环境禁止 `rm -f`**，用 kill 处理）：

```bash
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  ppid=$(ps -o ppid= -p $p | tr -d ' '); [ -n "$ppid" ] && kill -9 $ppid; kill -9 $p
done
pkill -9 -f "VLLM::"
```

后台启动必须用 `setsid`（否则会话回收会杀掉）：

```bash
setsid nohup ./scripts/run_xxx.sh > /tmp/outer.log 2>&1 < /dev/null &
```

---

## 6. 已知限制（别踩）

1. **`ir_step` 与 `pull_tl` 的 epoch 不一定落在同一空间**。
   `irstep_begin()` 在 host 侧写 `epoch0 = header[0]`，CUDA-graph 重放下这个读可能是冻结值
   （实测偏 225 个 epoch ≈ 5.6 步）。所以在**同一次 run** 里，
   "7 段耗时" 与 "该层 token/M块" 可能拿不到同一个 call。
   绕过办法：`GC_MOE_MEASURE_FLUSH=0` + 少步数，让两者窗口重叠；
   或按 `docs/trace_handover_20261119/HANDOVER.md` §5.1 的 `ir_step_epoch` 方案
   （**该方案尚未编译验证通过，勿直接启用**）。
2. **profiler 实测只抓到 2 步**（80 个 call），与 `--measure-steps` 不对应。取样前先数 call。
3. **`ts` 和 `dur` 都是 µs**（判定：`routed.ts − shared.ts` ÷ `shared.dur` = 1.005）。
4. `GC_MOE_SHARED_IN_GC=0` 时 gc_moe 只有 **4 个 kernel**（没有 71×256 的 shared），
   按行号解析时要按 `(grid, block)` 自适应。
5. **`GC_MOE_LAYERS` 必须等于真实层数**，否则 `ir_step` 全表索引错位。
