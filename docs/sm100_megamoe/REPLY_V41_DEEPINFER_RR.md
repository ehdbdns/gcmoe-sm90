# 【SM100】V4.1 两后端整模 DeepInfer RR 路由验证回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**两臂均通过，进程退出均为 `0`。** 使用与上轮相同的 checkpoint、原 vLLM venv、TP8/EP8、初始 EPLB 放置和固定 prompt `[100,200,300,400]`，分别启动 `deep_gemm_mega_moe` 与 `gc_moe`，两臂均追加 `--eplb-decode --deepinfer-rr`。每臂一次 cold start，同一请求连续贪心生成四步。

- 测量提交：`6fd3e35954328769e0b864e3099198c561e02af2`
- 任务书 blob：`9ce26ee4b06b248a60a79694a11c59268a0aa77c`
- 实际启动数：**2**（两臂各一次）
- 启动前等待了无关的 GPU 占用释放，未中断该占用；八卡空闲后才持有组锁运行
- 两臂启动模型前均执行 `python scripts/sm100/check_v41_rr.py 2>&1 | tee "$work/rr-preflight.log"`，均输出 `DeepInfer RR and physical load checks passed`，预检退出 `0`
- 两臂模型退出与 launcher 退出均为 `0`；均加载 checkpoint `48/48` 分片
- 两臂均无 `ERROR`、`Traceback` 或 `RuntimeError` 行；cgroup 的 `oom` / `oom_kill` / `oom_group_kill` 峰值始终为 `0 / 0 / 0`
- 本轮不采性能；逐层 DeepInfer 重排和 ECO 未包含在本轮

## 两臂结果

模型实际启动时间（锁内、预检后）：

| 项目 | MegaMoE | GCMoE |
|---|---|---|
| 后端参数 | `deep_gemm_mega_moe` | `gc_moe` |
| 模型启动 | `2026-11-22T00:08:25+08:00` | `2026-11-22T00:11:53+08:00` |
| 模型结束 / 释放 | `2026-11-22T00:11:05+08:00` | `2026-11-22T00:15:13+08:00` |
| 预检退出码 | `0` | `0` |
| 模型退出码 | `0` | `0` |
| launcher 退出码 | `0` | `0` |
| checkpoint 分片 | `48/48` | `48/48` |
| 输出 ID | `[75, 200, 18, 200]` | `[75, 200, 18, 200]` |
| 逐层布局 | `40/40` | `40/40` |

八卡生成期间的 `rr_calls`：

| rank | MegaMoE | GCMoE |
|---:|---:|---:|
| 0 | `160` | `160` |
| 1 | `160` | `160` |
| 2 | `160` | `160` |
| 3 | `160` | `160` |
| 4 | `160` | `160` |
| 5 | `160` | `160` |
| 6 | `160` | `160` |
| 7 | `160` | `160` |

两臂都是 `[160, 160, 160, 160, 160, 160, 160, 160]`，即每个 rank 在 4 步 decode、40 层 MoE 上的路由调用数。`v41_native_smoke.py` 的八 rank 计数断言通过。

## 布局与输出一致性

- 两臂各自返回 40 层 replica 布局，八 rank 的层列表一致。
- 当前两臂的 40/40 层布局哈希逐层完全一致；同时与上轮 `v41_eplb_decode_results` 的两臂 40/40 层哈希完全一致。
- 首个当前臂哈希差异：**无**；首个与上轮哈希差异：**无**。
- 共享的每层布局哈希仍为 `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa`。
- 两臂输出均为 `[75, 200, 18, 200]`，与上轮参考及其彼此之间均无首个 ID 差异。

## 工作进程

| rank | MegaMoE PID | GCMoE PID |
|---:|---:|---:|
| 0 | `2611416` | `2938922` |
| 1 | `2611417` | `2938923` |
| 2 | `2611421` | `2938924` |
| 3 | `2611422` | `2938925` |
| 4 | `2611424` | `2938926` |
| 5 | `2611426` | `2938927` |
| 6 | `2611427` | `2938928` |
| 7 | `2611428` | `2938929` |

GCMoE 的 8 个 worker PID 均输出 `V41 GC expert adapter installed`，adapter PID 集合恰等于上述 8 个 worker PID，未出现 ModelOpt 断言、`cache_key_str` 或 ptxas 版本错误。两臂均使用 CUDA 13.5 / `V13.5.37` 的 venv ptxas；`local-fix.diff` 为空，本轮未修改源码。

## 资源与产物

- cgroup 为 session cgroup；MegaMoE 峰值 `memory.current` 为 `855,843,864,576` bytes，GCMoE 为 `870,186,565,632` bytes
- MegaMoE 最低 `MemAvailable` 为 `884,299,628 kB`；GCMoE 为 `883,884,868 kB`
- 两臂结束后均记录八卡释放；最终复核 `2026-11-22T00:16:17+08:00` 无 compute-apps、`/dev/nvidia0..7` 无残留进程
- 两臂 system journal 均为 `PERMISSION_LIMITED`
- 工作目录：
  - MegaMoE：`/data0/home/admin/workspace/megamoe/native-v41-rrdeep-v6bW5f`
  - GCMoE：`/data0/home/admin/workspace/megamoe/native-v41-rrgc-pg3jCs`
- 结果目录：`docs/sm100_megamoe/v41_deepinfer_rr_results/`

结果目录保留两臂完整模型日志、RR 预检、版本、worker PID、GPU 等待与释放证据、cgroup/内存快照及 arm summary；顶层保存 `arm-summaries.json`、`summary.json`、`rr-calls.json`、`decode-ids.json`、`replica-comparison.json`、`replica-comparison.txt`、`layout-check.txt` 和 `SHA256SUMS`。未提交模型权重、venv 或编译缓存。
