# 【SM100】V4.1 逐层专家权重迁移验证回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**两臂均通过，预检、模型和 launcher 退出均为 `0`。** 使用同一个由 SM90 BS512 归档需求构建的 plan，分别在两后端上完成 40 层同步换槽、迁移后权重抽样核对、8 rank RR decode 和最终布局检查。两臂最终输出与迁移前参考 `[75, 200, 18, 200]` 一致，两臂之间也一致。

- 测量提交：`fe53bb7d5e4a7d86e61e03cfb240ecf6ac981135`
- 任务书 blob：`179c3945708551e41332369fbc778689b8f1c46e`
- 实际启动数：**2**（Mega、GC 各一次 cold start）
- plan 绝对路径（运行时）：`/tmp/sm100-placement-NrRm7rur/plan.npz`
- plan SHA-256：`4048ec19e4051ca7df85ccc3efe5c8d9b9e4eabe6d09e4b3d8305a2b6a6faf81`
- 两臂 `versions.txt` 与运行期 `placement.plan_sha256` 均记录同一 plan SHA
- plan 已复制到结果目录 `v41_deepinfer_placement_results/plan.npz`，可用 `plan.sha256` 复核
- 输入归档 fixture SHA-256：`024cec1f052f2e8fc29cd603a4cacb4d5a4d6c3955b87d22671fb7b97d3a9e50`
- 两臂启动前的 `check_v41_rr.py` 均输出 `DeepInfer RR and physical load checks passed`
- 两臂均加载 checkpoint `48/48` 分片；无 `ERROR`、`Traceback` 或 `RuntimeError`
- cgroup `oom` / `oom_kill` / `oom_group_kill` 峰值均为 `0 / 0 / 0`
- 本轮不做性能计时；正式五臂放置仍应由 SM100 自身统计窗口产生

## 两臂运行结果

| 项目 | MegaMoE | GCMoE |
|---|---|---|
| 后端 | `deep_gemm_mega_moe` | `gc_moe` |
| 模型启动 | `2026-11-22T02:24:30+08:00` | `2026-11-22T02:28:44+08:00` |
| 模型结束 / 释放记录 | `2026-11-22T02:27:50+08:00` | `2026-11-22T02:34:05+08:00` |
| 预检退出码 | `0` | `0` |
| 模型退出码 | `0` | `0` |
| launcher 退出码 | `0` | `0` |
| 总换槽数 | `17,873` | `17,873` |
| 抽样不符槽数（聚合） | `0` | `0` |
| RR calls（8 rank） | `[160] x 8` | `[160] x 8` |
| 最终布局层数 | `40` | `40` |
| 输出 ID | `[75, 200, 18, 200]` | `[75, 200, 18, 200]` |

### 每层实际换槽数

层号 `0..39` 依次为：

`[448, 448, 448, 447, 446, 448, 447, 448, 448, 446, 445, 447, 447, 448, 445, 447, 448, 446, 448, 447, 448, 446, 448, 446, 446, 447, 447, 445, 446, 447, 446, 446, 448, 446, 447, 447, 444, 447, 447, 447]`

两臂的换槽计数相同。每个槽的每个权重组件抽查 32 bytes；`sampled_slot_mismatches` 在两臂均为 **0**，这是抽样检查，不是全量逐字节审计。8 个 rank 返回的迁移结果相同。

## 迁移后布局与输出

- 两臂各自返回 40 层最终布局，八 rank 层列表一致；两臂 40/40 层哈希逐层一致。
- 本次 40 层最终布局哈希没有一层与迁移前的布局相同；每层都有实际换槽。
- 两臂首个布局差异：**无**。
- 两臂输出均为 `[75, 200, 18, 200]`；与迁移前参考的首个 ID 差异：**无**。
- 每层最终布局哈希完整保存在 `placement-comparison.txt` 和 `placement-comparison.json`；结构化的每层换槽数保存在 `changed-slots.json`。

## 八卡 worker 与 RR 计数

| rank | MegaMoE PID | GCMoE PID | Mega rr_calls | GC rr_calls |
|---:|---:|---:|---:|---:|
| 0 | `2742938` | `3139305` | `160` | `160` |
| 1 | `2742939` | `3139307` | `160` | `160` |
| 2 | `2742942` | `3139308` | `160` | `160` |
| 3 | `2742943` | `3139311` | `160` | `160` |
| 4 | `2742945` | `3139312` | `160` | `160` |
| 5 | `2742948` | `3139314` | `160` | `160` |
| 6 | `2742949` | `3139316` | `160` | `160` |
| 7 | `2742956` | `3139317` | `160` | `160` |

GCMoE 的 8 个 adapter PID 恰好等于上述 8 个 worker PID；两臂均无 ModelOpt 断言、`cache_key_str` 或 ptxas 版本错误。`local-fix.diff` 为空，本轮未修改源码。

## 释放与资源

- MegaMoE 结束时 `gpu-release.txt` 无 compute-apps，八卡为空。
- GCMoE 结束时组锁仍持有时，释放快照记录了本任务 worker 的拆除状态；另一外部 `rtp_llm` 作业随后开始占用 GPU。
- 最终复核 `2026-11-22T02:35:07+08:00`：本任务 16 个 worker PID 全部不存在；当前 compute-apps 均为 `ruixuan+` 的外部 `rtp_llm_rank-*` 进程，不是本任务进程。证据见 `gpu-release-recheck.txt`。
- cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`
- MegaMoE 峰值 `memory.current`：`878,223,355,904` bytes；最低 `MemAvailable`：`858,090,540` kB
- GCMoE 峰值 `memory.current`：`865,739,055,104` bytes；最低 `MemAvailable`：`712,515,092` kB
- 两臂 system journal 均为 `PERMISSION_LIMITED`
- 工作目录：
  - MegaMoE：`/data0/home/admin/workspace/megamoe/native-v41-placedeep-kE6gvG`
  - GCMoE：`/data0/home/admin/workspace/megamoe/native-v41-placegc-9vAwIG`

## 产物

结果目录：`docs/sm100_megamoe/v41_deepinfer_placement_results/`。保留两臂完整模型日志、RR 预检、版本、worker/adapter PID、cgroup 与释放证据；顶层保留 `plan.npz`、`summary.json`、`arm-summaries.json`、每层换槽、布局哈希、RR 计数、输出 ID、迁移计划构建信息和 `SHA256SUMS`。未提交模型权重、venv 或编译缓存。
