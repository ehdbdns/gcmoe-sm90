# 【SM100】MegaMoE 与 GCMoE 整模冗余专家 decode 验证回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**两臂均通过，进程退出均为 `0`。** 使用同一 checkpoint、原 vLLM venv、TP8/EP8、固定 prompt `[100,200,300,400]` 和贪心连续 4 步 decode，分别启动 `deep_gemm_mega_moe --eplb-decode` 与 `gc_moe --eplb-decode`。两臂的 4 个输出 ID 完全一致，40 层 EPLB replica 布局哈希也逐层完全一致；首个哈希差异和首个 ID 差异均为 **无**。

- 测量提交：`040eba834d6e51c338e895440ef6c45646199613`
- 任务书 blob：`d13dbd2aab9a7e85b3de9fa219d6a5b03371df50`
- 实际启动数：**2**（两臂各一次 cold start）
- MegaMoE 臂：`2026-11-21T23:33:18+08:00` 至 `23:36:58+08:00`，模型退出 `0`，launcher 退出 `0`
- GCMoE 臂：`2026-11-21T23:37:40+08:00` 至 `23:41:00+08:00`，模型退出 `0`，launcher 退出 `0`
- 两次均加载 checkpoint `48/48` 分片；`external_timeout=none`
- 两臂日志均无 `ERROR`、无 `Traceback`
- 本轮只验证 EPLB 初始放置和连续 decode，不报告性能；DeepInfer 放置、RR/ECO 分发与计时未包含在本轮

## 运行配置

- vLLM `0.31.0`；torch `2.13.0+cu130`；CUDA `13.0`
- `nvidia-cutlass-dsl 4.7.1`；Python `3.12.7`
- venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- checkpoint：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- 权重索引 SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 两臂启动前均执行任务书要求的环境设置：
  `GC_MOE_PTXAS="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"`
- 该汇编器实际版本：`CUDA 13.5, V13.5.37`；JIT 预检 target：`sm_103f`
- EPLB：`enable_eplb=True`，`num_redundant_experts=64`，`step_interval=3000`，`use_async=True`，`policy=default`

## 两臂结果

| 项目 | MegaMoE | GCMoE |
|---|---|---|
| 实际后端参数 | `deep_gemm_mega_moe` | `gc_moe` |
| 模型退出码 | `0` | `0` |
| launcher 退出码 | `0` | `0` |
| checkpoint 分片 | 48/48 | 48/48 |
| worker 数 | 8 | 8 |
| GC adapter 安装 | 不适用 | 8/8 个不同 PID |
| 4 个输出 ID | `[75,200,18,200]` | `[75,200,18,200]` |
| 40 层 replica 哈希 | 40/40 | 40/40 |
| 首个 ID 差异 | 无 | 无 |
| 首个哈希差异 | 无 | 无 |

工作目录：

- MegaMoE：`/data0/home/admin/workspace/megamoe/native-v41-epdeep-SjXbZu`
- GCMoE：`/data0/home/admin/workspace/megamoe/native-v41-epgc-gVMjP8`

八卡 worker PID：

| rank | MegaMoE PID | GCMoE PID |
|---:|---:|---:|
| 0 | `3529559` | `3934522` |
| 1 | `3529563` | `3934523` |
| 2 | `3529566` | `3934526` |
| 3 | `3529567` | `3934527` |
| 4 | `3529570` | `3934528` |
| 5 | `3529572` | `3934532` |
| 6 | `3529574` | `3934536` |
| 7 | `3529576` | `3934538` |

GCMoE 的 8 个不同 worker PID 均输出 `V41 GC expert adapter installed`；没有 ModelOpt 断言、`cache_key_str` 或 ptxas 版本错误。

## EPLB 布局与融合检查

`inspect_v41_replicas` 在每一层执行以下运行期断言，两臂均无异常：

- `len(logical_replica_count) == 384`
- `min(logical_replica_count) >= 1`
- `len(logical_to_physical_map) == 384`
- `sum(logical_replica_count) == 448`
- 选中的物理槽排序后恰为 `0..447`
- `experts.num_local_experts == 56`
- `experts.has_fused_shared_experts is true`
- `module.use_sequence_parallel is true`

随后 `v41_native_smoke.py` 断言 8 个 rank 都返回 40 层，且 8 个 rank 的 replica 列表完全相同。两臂均通过。完整断言上下文见 `layout-check.txt`。

## 40 层 replica 哈希逐层对比

40/40 层逐项相等；首个差异为 **无**。所有层的共享哈希均为：

```text
4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa
```

| layer | MegaMoE placement SHA-256 | GCMoE placement SHA-256 | equal |
|---:|---|---|:---:|
| 0 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 1 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 2 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 3 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 4 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 5 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 6 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 7 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 8 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 9 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 10 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 11 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 12 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 13 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 14 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 15 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 16 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 17 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 18 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 19 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 20 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 21 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 22 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 23 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 24 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 25 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 26 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 27 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 28 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 29 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 30 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 31 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 32 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 33 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 34 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 35 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 36 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 37 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 38 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |
| 39 | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa` | true |

## 4 步 decode 逐 token 对比

| step | MegaMoE | GCMoE | equal |
|---:|---:|---:|:---:|
| 1 | 75 | 75 | true |
| 2 | 200 | 200 | true |
| 3 | 18 | 18 | true |
| 4 | 200 | 200 | true |

首个 ID 差异：**无**。两臂都是同一请求上的连续自回归 decode，不是四次独立 prefill。

## 资源与释放

- cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`，`memory.max=max`
- MegaMoE：`memory.current` 前 `501,917,077,504` bytes；峰值 `869,132,124,160` bytes（`23:35:58`）；后 `523,174,215,680` bytes
- GCMoE：`memory.current` 前 `449,006,125,056` bytes；峰值 `869,047,676,928` bytes（`23:40:00`）；后 `490,897,764,352` bytes
- 两臂观测的 `oom` / `oom_kill` / `oom_group_kill` 始终为 `0 / 0 / 0`
- MegaMoE `MemAvailable` 最低 `885,153,048 kB`，结束 `1,099,644,384 kB`
- GCMoE `MemAvailable` 最低 `886,307,600 kB`，结束 `1,137,391,040 kB`
- `memory.current` 是整个 session cgroup 的计数，包含可回收页缓存，不是单模型 RSS
- 两臂结束后均记录八卡释放；`2026-11-21T23:41:30+08:00` 复核 compute-apps 为空，`/dev/nvidia0..7` 均无残留进程
- system journal：两臂均为 `PERMISSION_LIMITED`，无法据此完成 system-wide Xid/OOM 审计；cgroup、退出码与释放复核单独保留

## 产物

`docs/sm100_megamoe/v41_eplb_decode_results/` 按 `deep_gemm_mega_moe/` 与 `gc_moe/` 保存两次启动的完整模型日志、退出码、worker PID、版本、预检、内存/cgroup、GPU/journal 与锁证据；顶层保存 `replica-comparison.json`、`replica-comparison.txt`、`decode-ids.json`、`summary.json`、`layout-check.txt`、launcher 和 `SHA256SUMS`。未提交模型权重、venv 或编译缓存。
