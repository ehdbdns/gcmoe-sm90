# 【SM100】V4.1-Flash BS1024 图重放复测回执（中断，2/5）

执行者：**SM100**。本轮未执行任何 SM90 任务书。

## 状态

- 任务书：`docs/sm100_megamoe/V41_GRAPH_BS1024.md` @ `ea007382a58f836ff522e268313c1a1e28f0b047`
- 用户要求于 `2026-11-22T15:15:20+08:00` 停止当前实验并释放 GPU；`run_five.sh` 的 trap 在 `2026-11-22T15:15:24+08:00` 记录释放。
- 完成 **2/5** 臂：`mega`、`gc`，两臂 `exit-code=0`，每个 rank 均通过 40/40 层图的 eager/replay 输出、实际接收数和拍卖成交数校验。
- `mega_rr` 已启动但在出结果前被终止；`gc_rr`、`gc_eco` 未启动。
- 八卡已释放：`nvidia-smi --query-compute-apps` 为空，`fuser /dev/nvidia0..7` 为空，无本轮 launcher/worker 活进程。详见 `gpu-process-check.txt`。
- 本回执是**中断回执**，不是五臂最终验收；未生成最终 `delivery/`，工作目录和 JIT 缓存保留以便续跑。

## 两臂固定工作量结果

| 臂 | stats ms/step | measure ms/step | 旧 measure ms/step | 变化 | 完整 MoE ms（新/旧） | block M | 峰值 M | 活跃槽 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mega | 118.629883 | 117.518321 | 119.21163571998477 | -1.4204% | 19.478022208000002 / 21.299718111999997 | 32 | 68.167425 | 364.5883 |
| gc | 199.608525 | 123.410361 | 192.65551112685353 | -35.9425% | 20.910101248 / 81.288583808 | 32 | 68.167425 | 364.5883 |

- 新 `gc` 相对同轮 `mega` 为 `+5.0137%`；旧矩阵同口径为 `+61.6080%`。
- 两臂 `token_ir=1.498265`、`total_m_blocks=451.314925`、`m_block_ir=1.206684`；`gc` 非 ECO 臂，`accepted` 为空。
- 未舍入原始 CSV：`bs1024/metrics.csv`；同臂新旧并排：`comparison-old.csv`。

## 图重放校验

- `mega`、`gc` 的八个 rank 均为 `validated_layer_graphs=40`。
- 每层 eager 与 replay 的输出、实际接收计数、拍卖成交数逐项比较均无差异；否则 worker 会抛 `MoE eager/replay mismatch`。两臂均未出现该错误。
- 正式窗口五臂均使用图重放；模型外层仍 eager，保留原采集口径。

## 原生 gate 与回放的首个差异

- `mega`：stats 与 measure 两个窗口、八个 rank 均无差异（publisher 的 `raw` 与 `replay` 同源）。
- `gc`：两个窗口均在 stats/measure 全窗口摘要上不同于 `mega`；首个差异如下。所有记录 `native_id == replay_id`，差异只在 gate 权重浮点尾数。

| rank | stats 首个差异（step/layer/token） | measure 首个差异（step/layer/token） |
|---:|---|---|
| 0 | 0/10/59 | 0/2/16 |
| 1 | 2/19/114 | 0/2/79 |
| 2 | 0/10/59 | 0/2/36 |
| 3 | 7/22/69 | 0/2/69 |
| 4 | 0/10/59 | 0/2/88 |
| 5 | 0/34/124 | 0/2/23 |
| 6 | 0/10/59 | 0/2/24 |
| 7 | 2/35/124 | 0/2/10 |

- 完整权重数值见 `first-gate-differences.csv`。
- 严格配对：`mega` vs `gc` 的 `controlled_replay=true`，但 `native_gate_equal=false`、`native_token_equal=false`，因此 `strict=false`；本回执不把该计时差解释为已验证的五臂性能结论。

## 中止与资源

- 终止的仅是本轮 PID：launcher `312375`、`run_five.sh` `312394`、`five_matrix` `1905166`、vLLM workers `1911103 1911104 1911106 1911108 1911111 1911112 1911113 1911114`。
- 释放后检查：compute-apps 空、`fuser` 空、无 `run_five.sh`/`five_matrix.py`/`five_worker.py` 活进程；可能有本轮 `<defunct>` worker 僵尸，但不占 GPU。
- 工作目录：`/data0/home/admin/workspace/megamoe/v41-five-graph-bs1024-01`；保留 `overlay/`、`overlay.ready`、`dg-cache/`、`eco-cache/` 以便续跑，未做最终清理。
- 可续跑数据：`bs1024/{mega,gc}/` 的 result/log/start/end/exit-code 与 rank NPZ；`bs1024/plan.npz`；`bs1024/stats-routes-*.pt`、`measure-routes-*.pt`。

## 校验

- 结果目录 `docs/sm100_megamoe/v41_graph_bs1024_results/` 的 `SHA256SUMS` 已通过 `sha256sum -c`，共 29 个文件。
- `SHA256SUMS` 自身 sha256：`efdf7e52d483cebb2217f4c72b593cc7e9d0c50bf99efc5da11c1818a958166c`。
- 八卡 UUID、版本、ptxas 与预检：`gpus.txt`、`versions.txt`、`ptxas.txt`、`rr-preflight.log`。

## 续跑提示

若要完成五臂，保留同一工作目录，重新执行 `SM100_MATRIX_BS=1024 bash scripts/sm100/run_five.sh <workdir>`；脚本会跳过已成功的 `mega`、`gc`，从 `mega_rr` 继续。
