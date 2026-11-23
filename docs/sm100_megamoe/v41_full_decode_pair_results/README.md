# [SM100] V41 BS512 完整 decode 图 Mega/GC 双臂结果

任务书：`docs/sm100_megamoe/V41_BS512_FULL_DECODE_PAIR.md`  
执行 SHA：`8779b2948f43892509acb669c202de40afd1eecd`  
任务书 SHA256：`d5baee196081eab83db0036d9f760e0ac0487ee94f531c56388d9db83d070bf1`  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`

## 结论

- 正式锁定运行的 Mega 和 GC 两臂均以 exit code 0 完成 1000 步统计和随后 1000 步测量；两臂均启用 vLLM V2 的 BS512 完整 decode 图，采样和调度留在模型图外。
- 图覆盖通过：16/16 rank 均找到同一次匹配的 `cudaGraphLaunch`，关联号为 `751`，每个 rank 有 80 个阶段 marker、40 个 attention 节点、130 个 dense 节点、40 个 Mega MoE 节点或 120 个 GC MoE 节点，foreign kernel 为 0。
- 结构检查通过：每个 rank 每阶段记录 40,000 次完整图调用（32/32 arm-phase-rank）；两臂物理槽计数 16/16 phase-rank 完全相同；2000/2000 step-layer 的八卡 token 合计均为 3072。
- 强制回放路由通过：`replay_ids` 和 `replay_weights` 两臂均 16/16 相同。
- 严格双臂验收失败：`strict_pair_pass=false`，共 64 项。GC 的 `logical_ids`、`gate_weights`、`native_window` 与 Mega/原生结果不一致，GC 原生输出也未通过零差异检查。因此本回执中的性能差值只作描述，不能作为严格验收通过后的权威结论。

## 配置

- 模型：DeepSeek V4.1 Flash，TP8/EP8，融合 shared expert，40 个专家层，top-6，每 rank 48 个本地槽。
- 批量：BS512；固定请求顺序；逐阶段 KV checkpoint；保留原生布置。
- 图：vLLM 0.31.0 V2 `CUDAGraphMode.FULL_DECODE_ONLY`，capture size 512，每个 rank 每步一次模型图提交；TorchDynamo 编译关闭。
- Mega：本机 `deep_gemm` Mega MoE 后端。
- GC：`Natureal/gc-moe` 的 `eco` 分支，固定到上述 commit；实际 SM split 为 `dispatch=12`、`compute=136`。
- 两臂 prompt SHA256 均为 `b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。

## 验收明细

| 检查项 | 结果 |
|---|---|
| 严格双臂验收 | **失败**，`strict_pair_pass=false`，64 项 |
| 图覆盖 | 通过，16/16 rank |
| 每 rank 每阶段完整图调用 | 40,000/40,000，32/32 arm-phase-rank |
| 两臂物理槽计数 | 16/16 phase-rank 完全相同 |
| 逐步逐层八卡 token 合计 | 3072，2000/2000 step-layer |
| 强制回放 route id hash | 16/16 相同 |
| 强制回放 route weight hash | 16/16 相同 |
| 原生 logical id hash | 0/16 相同 |
| KV 边界 | 通过 |
| prefill 原生输出 | 通过，`prefill_difference.count=0` |
| GC 原生输出/路由 | 失败 |

失败不是图回退或调用次数不足：两臂确实在完整图中执行，且物理槽计数和强制回放路由一致。失败集中在原生路由和 gate 数值：Mega 的 `logical_ids`、`gate_weights` 与其原生/replay 结果一致，GC 的对应 hash 不同；GC 的 `native_window` 也与 Mega 不同。

诊断数据：

- GC `stats` 每 rank 原生 token 差异 1221；`measure` 每 rank 1037。
- GC 首处原生差异：`req=201, step=2`，原生 `36797`，参考 `25499`。
- GC 首处 gate 差异：`measure` 的 `step 0, L20, local_token 1, choice 0`；native/replay id 都是 353，权重分别为 `0.5604429244995117` 和 `0.5603961944580078`，差约 `4.67e-5`。
- 失败项完整列表见 `full_decode_pair.json` 和 `analysis_validation.txt`。

## L20 对齐

采样点为 `measure` 阶段 `step 2, L20`。八 rank 的 token 和 M 块在两臂相同：

| rank | tokens | M blocks |
|---:|---:|---:|
| 0 | 581 | 65 |
| 1 | 202 | 44 |
| 2 | 291 | 46 |
| 3 | 156 | 35 |
| 4 | 372 | 53 |
| 5 | 171 | 39 |
| 6 | 303 | 51 |
| 7 | 996 | 93 |
| 合计 | 3072 | 426 |

| 指标 | Mega | GC |
|---|---:|---:|
| 最早进入 MoE 的 rank | 2 | 3 |
| 最晚进入 MoE 的 rank | 3 | 6 |
| 八卡入口时差 | 1.0371 us | 3.3555 us |
| 跨 rank 完整层跨度（最早入口到最晚出口） | 240.6191 us | 272.6924 us |
| 最长 count exchange | rank 5，22.368 us | rank 2，16.576 us |
| 最大对齐半宽 | 0.800 us | 0.912 us |
| 最大绝对对齐漂移 | 1.018 us | 0.988 us |

GitHub 文件：

- `analysis/l20_summary.json`
- `analysis/l20_ranks.csv`
- `mega/perfetto/step002_layer20.perfetto.json`
- `gc/perfetto/step002_layer20.perfetto.json`

与上轮 L20 对照：

| 指标 | 上轮 Mega | 上轮 GC | 本轮 Mega | 本轮 GC |
|---|---:|---:|---:|---:|
| 入口时差 | 347.1162 us | 357.8896 us | 1.0371 us | 3.3555 us |
| 最长 count exchange | 366.0498 us | 368.6094 us | 22.368 us | 16.576 us |

上轮是 `step 1` 的 dense-submit 采样，本轮是 `step 2` 的完整 decode 图；工作负载采样步和提交路径不同，不能把两组绝对差值当作同一条件下的改进幅度。

## 图与逐层边界

- 每个 rank 仅匹配 1 次图 launch；两臂 correlation 都是 `751`。
- 每 rank marker 数 80，顺序为 40 对 `full_route_stage` 和 `full_collect_layer`。
- Mega MoE 节点 40 个；GC MoE 节点 120 个；两臂 attention 节点均为 40、dense 节点均为 130。
- `analysis/layer_boundaries_step002.csv` 保存两臂各 40 层、共 80 行逐层边界，包括最早/最晚 rank、入口时差、完整层跨度及本地 MoE 时长。

## 性能数字（非权威）

方法：排除采样步 2，对其余 999 步逐层取八 rank 本地完整 MoE 时长最大值，累加 40 层后对步求均值。

| 臂 | ms/step |
|---|---:|
| Mega | 7.7364914354 |
| GC | 8.1565011091 |

GC 相对 Mega 慢 `5.4289%`，等价于有符号降幅 `-5.4289%`。由于严格双臂验收失败，`analysis/performance_summary.json` 中 `authoritative=false`；该数字只能用于描述本轮运行，不能写成严格性能通过。

## GC 捕获修复与失败尝试

首次 GC 完整图捕获在 `gc_moe/runtime/gc_moe.py:264` 失败，异常为 `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`，退出码 1。原因是 GC 在 vLLM 的外部 stream capture 内调用了 `cuGraphInstantiate`。

最小修复只在正在捕获时不实例化子图，保留 capture 结束后的原路径：

```diff
-            self.instantiate()
+            if not torch.cuda.is_current_stream_capturing():
+                self.instantiate()
```

修复前日志、退出码和 instrumentation diff 在 `attempts/gc-capture-unsupported/`；正式 diff 在 `compat/gc_capture_fix.diff`。另有一次未加组锁的成功 GC 尝试归档在运行机 `run/attempt-gc-success-unlocked/`，不作为正式结果；正式 GC 证据来自 17:26:17 获取组锁、17:37:54 释放的八卡空闲校验运行。

## 环境说明

环境存在必须原样保留的标识差异：

- `nvidia-smi -L` 报告 8 张 **NVIDIA L20D**，`compute_cap=8.9`，每卡 275040 MiB。
- Torch 实际报告 compute capability **(10, 3)**，每卡约 267.69 GiB，8 卡可见。

两者均按实际观测记录，不把环境写成标准 SM100 产品标识。

版本：torch `2.13.0+cu130`、vLLM `0.31.0`、cutlass-dsl `4.8.1`、Triton `3.7.1`、NVIDIA 驱动 `580.105.08`、ptxas CUDA `13.5.37`。

实际八卡 UUID：

```text
GPU-3aad6312-0e20-d01b-4249-71cb090c2127
GPU-b0ac7f5d-738f-cdad-423f-456f4c26947e
GPU-c5a93c78-be0c-2648-564a-4b4666671960
GPU-77463de9-85e0-a6f3-2c7e-f1147d6f7f49
GPU-4a493131-84b9-4c88-18ee-50fc6f213a36
GPU-045e62f4-d04d-2f9e-e456-1fb82810029f
GPU-6b44066c-6712-30f2-02d0-fa0cb91733a3
GPU-31a509fd-e741-5ab8-b837-443ad986c404
```

## 证据与释放

交付目录包含两臂 `result.json`、各 8 个 `stats-rank*.npz` 和 8 个 `measure-rank*.npz`（共 32 个 NPZ）、step 2 的 L0/8/20/32/39 overview/Perfetto/counts、分析表、环境、修复 diff、日志和失败尝试。

大体积原始 KV、Kineto profile、原生路由表和工作负载证据保留在执行机：

- 归档根：`/data0/home/admin/workspace/megamoe/v41-full-decode-4F7lKSeP/run`
- `raw_evidence_manifest.json`：141 条，551,808,704,185 bytes（约 513.9 GiB）
- manifest SHA256：`4a8e772170692844bef9f1ebe45e080530b44974e73be895613366bb55a4689b`
- 完整路径清单：`raw_evidence_paths.txt`

正式运行结束后 `nvidia-smi --query-compute-apps=gpu_uuid,pid` 为空，`fuser` 显示无设备持有者，组锁 `/tmp/gcmoe-sm100-gpu-group.lock` 可立即获取。证据见 `gpu_release_after_run.txt` 和 `lock-release.txt`。
