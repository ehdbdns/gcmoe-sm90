# [SM100] BS512 原生 Mega/GC 完整 decode 图回执

执行者：SM100。未执行或混写 SM90 的任何任务。  
任务书：`docs/sm100_megamoe/V41_BS512_FULL_DECODE_PAIR.md`  
执行 SHA：`8779b2948f43892509acb669c202de40afd1eecd`  
任务书 SHA256：`d5baee196081eab83db0036d9f760e0ac0487ee94f531c56388d9db83d070bf1`  
结果目录：`docs/sm100_megamoe/v41_full_decode_pair_results/`

## 执行结果

两臂均按任务书完成 BS512 完整 decode 图的 1000 步统计和随后 1000 步测量，正式锁定运行 exit code 均为 0。模型为 DeepSeek V4.1 Flash，TP8/EP8，融合 shared，固定请求顺序和逐阶段 KV；vLLM V2 捕获完整 forward，采样和调度在模型图外，TorchDynamo 编译关闭。

1. 图覆盖通过：16/16 rank 均匹配同一次 `cudaGraphLaunch`，关联号 `751`；每 rank 80 个 marker、40 个 attention、130 个 dense、40 个 Mega MoE 或 120 个 GC MoE 节点，foreign kernel 为 0。
2. 调用与负载结构通过：每 rank 每阶段 40,000/40,000 次完整图调用；两臂物理槽计数 16/16 phase-rank 完全相同；2000/2000 step-layer 的八卡 token 合计均为 3072。
3. 强制回放路由通过：`replay_ids`、`replay_weights` 两臂均 16/16 相同。KV 边界和 prefill 原生输出也通过，`prefill_difference.count=0`。
4. **严格双臂验收失败**：`strict_pair_pass=false`，64 项。GC 的 `logical_ids`、`gate_weights`、`native_window` 与 Mega/原生结果不一致，且 GC 原生输出非零差异；不能据此宣称严格正确性通过。

失败诊断：GC `stats` 每 rank 原生 token 差异 1221，`measure` 每 rank 1037。首处原生差异为 `req=201, step=2`，原生 `36797`、参考 `25499`。首处 gate 差异在 `measure step 0, L20, local_token 1, choice 0`：id 均为 353，权重 `0.5604429244995117` 对 `0.5603961944580078`。完整失败列表见 `full_decode_pair.json` 和 `analysis_validation.txt`。这是路由/gate/原生输出一致性问题，不是图未捕获或调用次数不足。

## L20

采样点为 `measure step 2, L20`。八 rank token 为 `[581, 202, 291, 156, 372, 171, 303, 996]`，M 块为 `[65, 44, 46, 35, 53, 39, 51, 93]`，两臂相同，合计 3072 tokens / 426 M blocks。

| 指标 | Mega | GC |
|---|---:|---:|
| 最早/最晚进入 rank | 2 / 3 | 3 / 6 |
| 入口时差 | 1.0371 us | 3.3555 us |
| 跨 rank 完整层跨度 | 240.6191 us | 272.6924 us |
| 最长 count exchange | rank 5，22.368 us | rank 2，16.576 us |
| 最大对齐半宽 | 0.800 us | 0.912 us |
| 最大绝对对齐漂移 | 1.018 us | 0.988 us |

GitHub L20：

- https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_full_decode_pair_results/analysis/l20_summary.json
- https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_full_decode_pair_results/mega/perfetto/step002_layer20.perfetto.json
- https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_full_decode_pair_results/gc/perfetto/step002_layer20.perfetto.json

与上轮 L20 对照：上轮入口时差 Mega `347.1162 us`、GC `357.8896 us`，最长 count exchange Mega `366.0498 us`、GC `368.6094 us`。上轮是 `step 1` dense-submit，本轮是 `step 2` 完整 decode 图，采样步和提交路径不同，不能作为同条件改进幅度。

## 性能（非权威）

正式 MoE 指标排除采样步 2，对其余 999 步逐层取八 rank 本地完整 MoE 时长最大值，累加 40 层后对步求均值：

| 臂 | ms/step |
|---|---:|
| Mega | 7.7364914354 |
| GC | 8.1565011091 |

GC 相对 Mega 慢 `5.4289%`，有符号降幅为 `-5.4289%`。由于严格验收失败，`analysis/performance_summary.json` 标记 `authoritative=false`；该值仅描述本轮运行，不能写成严格性能通过。

## 修复、环境与证据

首次 GC 捕获在 `gc_moe/runtime/gc_moe.py:264` 报 `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`：GC 在 vLLM 外部 stream capture 内调用了 `cuGraphInstantiate`。最小修复为仅在非捕获时实例化子图，diff 见 `compat/gc_capture_fix.diff`，失败日志和退出码见 `attempts/gc-capture-unsupported/`。未加组锁的一次成功 GC 尝试仅归档为非正式证据；正式 GC 使用 17:26:17 获取组锁、17:37:54 释放的八卡空闲校验运行。

环境需原样保留：`nvidia-smi -L` 显示 NVIDIA L20D、`compute_cap=8.9`、每卡 275040 MiB；Torch 实际报告 capability `(10, 3)`、每卡约 267.69 GiB、8 卡。版本为 torch `2.13.0+cu130`、vLLM `0.31.0`、cutlass-dsl `4.8.1`、Triton `3.7.1`、驱动 `580.105.08`；GC SM split 为 dispatch 12 / compute 136。

大体积原始 KV、profile 和路由表留在执行机 `/data0/home/admin/workspace/megamoe/v41-full-decode-4F7lKSeP/run`。`raw_evidence_manifest.json` 共 141 条、551,808,704,185 bytes（约 513.9 GiB），SHA256 `4a8e772170692844bef9f1ebe45e080530b44974e73be895613366bb55a4689b`。运行后无 compute app、无设备持有者，组锁可获取，见 `gpu_release_after_run.txt` 和 `lock-release.txt`。
