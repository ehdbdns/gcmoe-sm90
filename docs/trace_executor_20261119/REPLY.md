# REPLY — 八卡执行回执（失败：停在步骤 3 末的 epoch 校验）

## 结论

步骤 1、2 **全部通过**（含修复后的 role 4 编译）。步骤 3 采集执行完毕、
八份 torch trace 均已导出，但 **采样步的 epoch 校验失败**，因此 `capture.json`
未生成，无法进入步骤 4 解析。

**失败点**：`integration/deepep_compare.py:2327`

```
RuntimeError: rank N: expected 40 calls, epoch advanced 120
```

8 个 rank 全部同样报错（rank 0..7），数值一致（120）。

## 版本

| 项 | 值 |
|---|---|
| 起始 SHA | `d92e7140f1b4ddadf1be4efa3d0e08d2a2a3e813` |
| 执行 SHA | `d92e7140f1b4ddadf1be4efa3d0e08d2a2a3e813` |
| `cffe483` 为祖先 | 是 |
| 是否做过编译修复 | **否**（`d92e714` 已由设计端修好 `shared_experts=shared`，我未改任何代码）|
| 工作区 | 干净 |

## 步骤 1：通过

```
8 × NVIDIA H20, driver 580.105.08, 97871 MiB/卡，起始全部 4 MiB
torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / capability 9.x × 8
MODEL=/home/admin/local_models/DeepSeek-V4.1-Flash
DATASET=.../dsv4-flash_online_input_ids_26-10-08_1.jsonl
```

## 步骤 2：通过

| 检查 | 结果 |
|---|---|
| `tests/test_aligned_step_cpu.py` | **ALIGNED_STEP_CPU PASS** |
| 常规 `GC_MOE_STEP_PROFILE=0` | `compiled roles [1, 2, 5, 3]` in 9.4s ✅ |
| 插桩 `GC_MOE_STEP_PROFILE=1 SMOKE_ECO_WB=26` | `roles [(1,…),(4,199680,256,71),(2,…),(5,53248,512,1),(3,…)]` → **`compiled roles [1, 4, 2, 5, 3]`** in 12.1s ✅ |

role 4 已出现，与任务书要求一致。

## 步骤 3：采集执行完毕，epoch 校验失败

### 3.1 实际执行

```bash
bash scripts/warm_page_cache.sh "$MODEL"      # advisory only (0.05 s)
# 我另外做了真正的串行预热：48 分片 dd，2m09s
python -u scripts/real_decode_100_100.py --label ALIGNED_EP8   --seqs-per-rank 32 --prefill-lo 256 --prefill-hi 2048   --warm-steps 200 --measure-steps 1 --profile-aligned-step --profile-out "$out/raw"
```

环境变量完全按任务书步骤 1 原文（EPLB=1、ECO=1、WB=26/WA=0、MAX_COPIES=8、
EXTRA_PER_RANK=8、逐层 EPLB、MB_CLAIM=1、PULL_TMA=1、SHARED_IN_GC=1、
SM_REUSE=0、DISPATCH_SMS=6、CAPACITY=BOUND=2048、BLOCK_M=64、
PULL_TRACE=1、ECO_TRACE=1、FINE_TRACE=0、IR_STEP=0、LAYERS=40、
MEASURE_FLUSH=2、ENFORCE_EAGER=0、CUDAGRAPH_MODE=FULL_DECODE_ONLY）。

### 3.2 做了什么、到哪一步

| 阶段 | 状态 |
|---|---|
| 权重加载 | 完成（预热后，8 worker 全部就绪）|
| prefill 256 prompts | 完成 |
| 200 步 warmup | 完成 |
| 逐层 EPLB 重排 | 完成 |
| `prof_start` | 完成 |
| 采样步（1 次 `engine.step()`）| 完成 |
| 8 份 torch trace 导出 | **完成**（`raw/rank0..7.json`，各 ~2.8 MB）|
| `_gc_moe_profile_snapshot(first_epoch, 40)` | **失败** |
| `capture.json` 写入 | 未执行（因上一步抛错）|

### 3.3 失败的判据与实测

代码 `integration/deepep_compare.py:2324-2327`：

```python
if first_epoch is not None:
    if next_epoch - first_epoch != layers:
        raise RuntimeError(f'rank {buf.rank_idx}: expected {layers} calls, '
                           f'epoch advanced {next_epoch - first_epoch}')
```

- `layers = 40`（= `GC_MOE_LAYERS`）
- 实测 `next_epoch - first_epoch = 120 = 3 × 40`

### 3.4 关键证据：只跑了一个 iteration，但 epoch 推进 3 层组

`run.log` 里 `[trace] profiling` 之后只有一条 iteration：

```
[17:04:21] Iteration(319): 0 context requests, 0 context tokens, 256 generation requests
[trace] profiling -> /tmp/gcmoe-aligned-fSc3IBre/raw
[17:04:23] Iteration(320): 0 context requests, 0 context tokens, 256 generation requests
```

即 `measure_steps=1` 确实只调用了 1 次 `engine.step()`，
但该次调用使设备 epoch 前进了 **120**（3 个 40 层的完整 MoE 序列）。

### 3.5 我的判断（供设计端裁决，未据此改动任何东西）

`first_epoch` 由 `prof_start` **之前**的 `_gc_moe_profile_snapshot()` 读取
（`real_decode_100_100.py:880`）。在 `CUDAGRAPH_MODE=FULL_DECODE_ONLY` 下，
vLLM 的 decode graph 可能在一个 host `step()` 内 replay 多次；
也可能 `MEASURE_FLUSH=2` 之后仍有一次未计入的 flush forward 落在了两次快照之间。

无论哪种，**判据 `epoch_advance == layers` 在这套配置下不成立**，
这是任务书配置与校验代码之间的不一致，不是我能在「编译兼容性修复」范围内解决的。

## 4. 产物与校验值

采集目录：`/tmp/gcmoe-aligned-fSc3IBre`

内容：

```
start.sha            d92e7140f1b4ddadf1be4efa3d0e08d2a2a3e813
execution.sha        d92e7140f1b4ddadf1be4efa3d0e08d2a2a3e813   (相同：未改代码)
gpu.log              8 × H20 / 580.105.08 / 4 MiB used
runtime.log          2.13.0+cu130 13.0 0.31.0
cpu.log              ALIGNED_STEP_CPU PASS
compile-base.log     compiled roles [1, 2, 5, 3] in 9.4s
compile-aligned.log  compiled roles [1, 4, 2, 5, 3] in 12.1s
warm.log             advisory + 真实串行 dd 2m09s
run.log              完整采集日志（含 8 rank 的 RuntimeError）
raw/rank0..7.json    8 份 torch trace（均已导出，未参与交付）
raw/capture.json     缺失（校验失败，脚本未写）
```

```
86913521dfc047f783a7660eed0e07034d5304c4fc900105c61ac7b723b4fee4  start.sha
86913521dfc047f783a7660eed0e07034d5304c4fc900105c61ac7b723b4fee4  execution.sha
c6f718ff77732362aaa9213220ae5f374a1b21257302f708b88dc19198bef599  gpu.log
74064a5ac06f3b428b7e0480f7e8089af2a734d042f3597f79740e8b242160b4  runtime.log
eeea22f858445f4d27a731b40bab4e1ec317b9990e56c1f07c8b1fdfad3339c7  cpu.log
12203b74552fb52026c63de28b8379408e26fb3e0dbf6a18f3c2a02e6dba5a43  compile-base.log
bd040b466d68a521ebfde2edab4ce37059192d929fcde031f8799d636ada1487  compile-aligned.log
3d088163adbb06a89f64bb4031a61c0be125453bdcb31974e80b7b4820fa6f26  warm.log
dc191419ff1324dc7e8f621b63a617ea40e3955f2c73770506d47dc3f94d3402  run.log
eb754f5d072e9828c728fcb404417096acbac3aa14a388df5f1e19eba167dfe4  raw/rank0.json
d06a46d0bf73959009aa51b32c2843c22cd4de27b27d20564b8534966443ad6a  raw/rank1.json
0c81c3c33063b8c0db08aa1f17612f2a6edf212e0f905c53d6f11f1683e81ca4  raw/rank2.json
be4362d6a9f1638bab2ad328a26d550ffdabfecbe3414ad4fde9ee044f2186ab  raw/rank3.json
6fd943b283e934ac79188777d7209223271878c8b7a80ff1ffa889e84e4a84fa  raw/rank4.json
19a47f800a7f0ff519265677b363450cf5c5c99b97eee6f0ecd5194e4c58a09c  raw/rank5.json
9014e7f8673f8cd619a9fb2a08b63fba194d2c9c76b0156ffa2805d599c62ee3  raw/rank6.json
cf375933063bab25249b2e28e555f9ba682fe4763b8cfd7d5d58dcdc489871a0  raw/rank7.json
```

原始数据保留在上述目录（未删除），可供设计端直接复查。

## 5. 未执行

- 步骤 4（解析 layer 8/20）：未执行（缺 `capture.json`，且 epoch 校验未过）
- `results/`：空
- 未提交任何 perfetto / counts

## 6. 需要澄清

判据 `next_epoch - first_epoch == GC_MOE_LAYERS` 在
`ENFORCE_EAGER=0 + CUDAGRAPH_MODE=FULL_DECODE_ONLY + MEASURE_FLUSH=2`
下实测为 `3 × GC_MOE_LAYERS`。

请确认：
1. 采样步应当只推进 40（则 `prof_start` 的取点或 flush 计数需要调整）；还是
2. 判据应改为「推进量为 layers 的正整数倍」，由解析器按实际推进的层组数处理。
