# REPLY — 原始 trace 诊断（回执 ce91bb1 后续）

## 结论

**torch trace 本身完全正确：8 个 rank、每种角色恰好 40 个 kernel、跨度 60.8 ms
= 恰好 1 个 decode 步。**
上一轮 `expected 40 calls, epoch advanced 120` **不是 trace 的问题**，
而是**设备快照 `header[0]` 的读数比真实 epoch 落后 2 步**（120 = 3×40，
其中 1 步是采样步、另 2 步是 `MEASURE_FLUSH=2`）。

按 FOLLOW_UP 的判据，本次属于第一种情况：
> 若同一 capture ID 下每种角色有 40 个 kernel，说明设备快照范围比 torch trace
> 多记了调用，需要定位边界。

## 执行

未占卡、未加载模型、未改代码。仅运行诊断脚本：

```bash
cd /tmp/eco_publish/repo
python3 scripts/trace_layer/inspect_aligned_trace.py \
    /tmp/gcmoe-aligned-fSc3IBre/raw \
    | tee /tmp/gcmoe-aligned-fSc3IBre/trace-summary.json
```

工作 SHA：`f2ea062eb123e4e28498b2a6321876c5b309002a`

## 诊断结果

### 8 个 rank 完全一致

| 项 | 值（rank 0..7 全部相同）|
|---|---|
| capture ID | `85fba503198d4155b4241c34f8468862` |
| gc_moe kernel 总数 | **200** |
| `24x256`（dispatch）| **40** |
| `71x256`（shared）| **40** |
| `71x384`（routed）| **40** |
| `1x512`（plan）| **40** |
| `288x384`（combine）| **40** |
| `cudaGraphLaunch` | **1** |

- 8 个 rank 的 capture ID 是否相同：**True**（1 个不同值）
- 8 个 rank 的 kernel 几何是否相同：**True**

### 换算

```
200 kernels / 5 roles = 40 calls = 40 layer calls = 1.0 step
trace 跨度 60795.5 us = 60.80 ms
```

40 次调用对应 1 个 decode 步（40 层），耗时 60.8 ms —— 与 BS32 一步的量级一致。
**没有出现 120 个 kernel，所以不是 torch trace 多记了调用。**

## 边界定位：错在设备快照的 epoch 基准

### 代码路径

`scripts/real_decode_100_100.py:845-847, 880-884`：

```python
flush = int(os.environ.get('GC_MEASURE_FLUSH', '5'))    # 本次 = 2
for _ in range(flush):
    _collect(engine.step())                             # 2 步 flush
...
before = engine.collective_rpc('_gc_moe_profile_snapshot')   # 读 header[0]
first_epoch = first_epochs.pop()
engine.collective_rpc('_gc_moe_prof_start', ...)             # 之后才开始采样
```

`integration/deepep_compare.py:2320`：
```python
next_epoch = int(buf.header[0].item())
```

### 数量关系

```
next_epoch - first_epoch = 120 = 3 × 40
                              └─ 1 个采样步 + 2 个 flush 步
```

torch trace 只捕获了其中 1 步（40 调用，已验证）。
所以 120 里多出的 2 步，来自 **`first_epoch` 读到的值比实际早了 2 步**：
`header[0]` 是 host 读设备内存，在 `CUDAGRAPH_MODE=FULL_DECODE_ONLY` 下
（`ENFORCE_EAGER=0`），这个读在 flush 期间被 CUDA graph 重放"冻结"了 2 步。

这与仓库里其他地方已经记录过的同类问题一致（`irstep_begin()` 的注释明确写过：
header[0] 在 graph replay 下是 stale 的，因此改用 `eco_tl_step`）。

## 采样前后完整日志

### 最后两个 flush iteration（318、319）

```
[17:04:21] Engine 000: Iteration(318): 0 context requests, 0 context tokens, 256 generation requests
[17:04:21] Engine 000: Iteration(319): 0 context requests, 0 context tokens, 256 generation requests
```

### `[trace]` 起止

```
[17:04:21] [trace] profiling -> /tmp/gcmoe-aligned-fSc3IBre/raw
[17:04:23] Engine 000: Iteration(320): 0 context requests, 0 context tokens, 256 generation requests
[17:04:23] [trace] 8/8 workers exported
```

**注意：`profiling` 与 `8/8 workers exported` 之间只有 Iteration 320 一条**，
即 host 侧确实只跑了一个步 —— 与 trace 的 40 个调用一致。

### 异常（首个）

```
File "/tmp/eco_publish/repo/scripts/../integration/deepep_compare.py", line 2327, in gc_moe_profile_snapshot
    raise RuntimeError(f'rank {buf.rank_idx}: expected {layers} calls, '
RuntimeError: rank 7: expected 40 calls, epoch advanced 120
```

8 个 rank 全部抛出，数值均为 120。

## 产物与校验

诊断目录：`/tmp/gcmoe-aligned-fSc3IBre`

```
trace-summary.json      诊断脚本输出（本文件的数据来源）
raw/rank0..7.json       8 份原始 torch trace（保留，未删改）
run.log                 完整采集日志
capture.json            仍缺失（上轮校验失败未写）
```

```
bd55bc3b11a37aff295e8c267f62017842c9a9131aaf7f4dbf600f8dd1868d22  trace-summary.json
eb754f5d072e9828c728fcb404417096acbac3aa14a388df5f1e19eba167dfe4  raw/rank0.json
d06a46d0bf73959009aa51b32c2843c22cd4de27b27d20564b8534966443ad6a  raw/rank1.json
0c81c3c33063b8c0db08aa1f17612f2a6edf212e0f905c53d6f11f1683e81ca4  raw/rank2.json
be4362d6a9f1638bab2ad328a26d550ffdabfecbe3414ad4fde9ee044f2186ab  raw/rank3.json
6fd943b283e934ac79188777d7209223271878c8b7a80ff1ffa889e84e4a84fa  raw/rank4.json
19a47f800a7f0ff519265677b363450cf5c5c99b97eee6f0ecd5194e4c58a09c  raw/rank5.json
9014e7f8673f8cd619a9fb2a08b63fba194d2c9c76b0156ffa2805d599c62ee3  raw/rank6.json
cf375933063bab25249b2e28e555f9ba682fe4763b8cfd7d5d58dcdc489871a0  raw/rank7.json
```

## 未执行

- 步骤 4 解析：**未执行**（本轮指令是「只做原始 trace 诊断」）
- `results/`：空
- 未改任何代码（本 SHA 与 checkout 的 `origin/main` 一致）

## 建议的下一步（供设计端决定）

证据表明 torch trace 可用，问题只在 `first_epoch` 的取值。
可选修法（都属语义调整，需设计端授权）：

1. 把 `first_epoch` 的读取移到 `prof_start` **之后**、采样步 **之前**；
2. 或让 `_gc_moe_profile_snapshot` 改用 `eco_tl_step`（与 `irstep_begin` 的既有修法一致）；
3. 或接受 `advanced % 40 == 0` 且 `len(calls) == 40`，由 `capture.json` 记录实际基准。

按当前证据，**trace 本身已满足 40 层验收**，只差这个基准对齐。
