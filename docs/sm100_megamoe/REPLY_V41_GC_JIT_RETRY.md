# 【SM100】CuTe DSL 4.7.1 缓存键修正后的 GCMoE 整模复测回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**验收通过，成功路径进程退出 `0`。** 上游 `82987eb` 的 `_environment_cache_key` 修复生效：上轮的 `EnvironmentVarManager.cache_key_str` 错误没有再现。首次使用任务原样默认路径冷启动时，失败点已后移到 PTX 汇编阶段；默认 `/usr/local/cuda/bin/ptxas` 是 CUDA 13.0，无法汇编为 `sm_103f` 生成的 PTX 9.1。按任务允许的“保持语义的编译修复”，增加一行可选的 `GC_MOE_PTXAS` 入口，并显式使用当前已验证 vLLM venv 内原本已安装的 CUDA 13.5 `ptxas`，随后冷启动、四前缀审计和 8×40 层运行期 introspection 全部完成。

- 测量提交：`82987eb0a1da70e0c3b3cbb6493ec6c501fe6a08`
- 任务书 blob：`d126cfac4d3c8429c3c414f230627209f4c2951a`
- Attempt 1：`2026-11-21T23:15:40+08:00` 至 `23:18:40+08:00`，退出 `1`
- Attempt 2：`2026-11-21T23:21:12+08:00` 至 `23:24:12+08:00`，退出 `0`
- 两次 launcher 退出码均与模型进程一致；`external_timeout=none`
- 两次均加载 checkpoint `48/48` 分片，均见 8/8 个不同 worker PID 安装 GC adapter
- 成功路径无 `ERROR`、无 `Traceback`；未进入任何 SM90 路径

## 编译修复

唯一源码修改是以下一行，已保存在 `v41_gc_jit_retry_results/local-fix.diff`：

```diff
文件：source/gc_moe/runtime/jit.py
-SYSTEM_PTXAS = '/usr/local/cuda/bin/ptxas'
+SYSTEM_PTXAS = os.environ.get('GC_MOE_PTXAS', '/usr/local/cuda/bin/ptxas')
```

默认值仍是 `/usr/local/cuda/bin/ptxas`，不改变默认行为；它只允许显式选择环境中已有的汇编器。模型、权重、采样参数、vLLM、GCMoE kernel、TP8/EP8、launcher harness 和四前缀输入均未改变。未安装、升级或替换 Python 包，未下载依赖。

## 工具链事实

预检在两次运行都通过，输出 `nvidia-cutlass-dsl 4.7.1`、target `sm_103f`。Attempt 1 的预检显示默认汇编器为 `CUDA 13.0, V13.0.48`；Attempt 2 的预检显示所选汇编器为 `CUDA 13.5, V13.5.37`。

环境中存在多个 `ptxas`，完整清单和 SHA-256 见 `ptxas-inventory.txt`：

| 路径 | 版本 | 用途 |
|---|---|---|
| `/usr/local/cuda/bin/ptxas` | 13.0 / V13.0.48 | 默认系统路径；Attempt 1 |
| venv `nvidia/cu13/bin/ptxas` | 13.5 / V13.5.37 | 原 venv 已安装；Attempt 2 |
| venv `triton/backends/nvidia/bin/ptxas` | 12.8 / V12.8.93 | 未使用 |
| venv `tokenspeed_triton/backends/nvidia/bin/ptxas` | 13.4 / V13.4.59 | 未使用 |

另以 PTX 9.1、`sm_103f` 的最小输入单独验证 13.5 汇编器，返回 `ptxas_rc=0`，见 `attempt2-venv-ptxas-13.5/ptxas-sm103f-probe.txt`。

## Attempt 1：默认 CUDA 13.0

任务原样默认路径已完成缓存键修复所针对的阶段，8/8 个 worker 均安装 adapter，48/48 分片加载完成。首个错误发生在 engine profile/dummy run 的 GCMoE JIT 汇编：

```text
profile_run
  -> _dummy_run
  -> model forward
  -> deep_gemm.fp8_fp4_mega_moe
  -> source/gc_moe/gc/__init__.py:240 fp8_fp4_gc_moe
  -> source/gc_moe/apis/gc_moe.py:169 launch_gc_moe
  -> source/gc_moe/jit_kernels/impls/sm100_fp8_fp4_gc_moe.py:743 prepare_role_binaries
  -> ...:431 _assemble_role_value_params
  -> source/gc_moe/runtime/jit.py:183 assemble_ptx
RuntimeError: CUDA 13.2 or newer ptxas is required at /usr/local/cuda/bin/ptxas for this PTX version
```

EngineCore 随后报告 `RuntimeError: Worker failed with error ...` 与 `RuntimeError: Engine core initialization failed`。该尝试未进入四前缀推理，无 top1/top10，未完成 8×40 层 introspection。完整首个 worker 栈见 `attempt1-system-ptxas-13.0/first-worker-traceback.txt`，PID 与 cgroup/释放证据同目录保留。

## Attempt 2：venv 内 CUDA 13.5

使用上述一行可选入口后，同一冷启动流程退出 `0`。8 个不同 worker PID 均输出 `V41 GC expert adapter installed`：

| rank | PID |
|---:|---:|
| 0 | `2379812` |
| 1 | `2379813` |
| 2 | `2379814` |
| 3 | `2379815` |
| 4 | `2379816` |
| 5 | `2379817` |
| 6 | `2379818` |
| 7 | `2379819` |

## 四前缀：GC vs Mega top1/top10

四组均为固定前缀重新 prefill、单请求贪心一步，不用上一轮生成 token 续写。GC 首选为 `75, 300, 200, 19`，与参考 Mega 首选完全相同；如下每个 rank 的 `id` 与 `logprob` 均逐项相等（`top10_exact_equal=true`），不是仅 top1 相同。机器可读完整结果在 `attempt2-venv-ptxas-13.5/prefix-comparison.json`。

| 固定前缀 | GC top1 | Mega top1 | top10 全项相等 |
|---|---:|---:|:---:|
| `[100,200,300,400]` | 75 | 75 | true |
| `[100,200,300,400,200]` | 300 | 300 | true |
| `[100,200,300,400,200,19]` | 200 | 200 | true |
| `[100,200,300,400,200,19,200]` | 19 | 19 | true |

#### `[100,200,300,400]`

| rank | GC id | GC logprob | Mega id | Mega logprob | equal |
|---:|---:|---:|---:|---:|:---:|
| 1 | 75 | -0.921821237 | 75 | -0.921821237 | true |
| 2 | 200 | -2.421821117 | 200 | -2.421821117 | true |
| 3 | 223 | -2.921821117 | 223 | -2.921821117 | true |
| 4 | 13 | -3.671821117 | 13 | -3.671821117 | true |
| 5 | 18 | -3.671821117 | 18 | -3.671821117 | true |
| 6 | 19 | -3.796821117 | 19 | -3.796821117 | true |
| 7 | 14639 | -3.796821117 | 14639 | -3.796821117 | true |
| 8 | 67082 | -3.984321117 | 67082 | -3.984321117 | true |
| 9 | 31 | -3.984321117 | 31 | -3.984321117 | true |
| 10 | 54155 | -4.109321117 | 54155 | -4.109321117 | true |

#### `[100,200,300,400,200]`

| rank | GC id | GC logprob | Mega id | Mega logprob | equal |
|---:|---:|---:|---:|---:|:---:|
| 1 | 300 | -0.084429950 | 300 | -0.084429950 | true |
| 2 | 18 | -2.959429979 | 18 | -2.959429979 | true |
| 3 | 20 | -4.959429741 | 20 | -4.959429741 | true |
| 4 | 22 | -5.834429741 | 22 | -5.834429741 | true |
| 5 | 21 | -5.834429741 | 21 | -5.834429741 | true |
| 6 | 23 | -5.959429741 | 23 | -5.959429741 | true |
| 7 | 19 | -6.209429741 | 19 | -6.209429741 | true |
| 8 | 31 | -6.459429741 | 31 | -6.459429741 | true |
| 9 | 200 | -6.834429741 | 200 | -6.834429741 | true |
| 10 | 162 | -7.459429741 | 162 | -7.459429741 | true |

#### `[100,200,300,400,200,19]`

| rank | GC id | GC logprob | Mega id | Mega logprob | equal |
|---:|---:|---:|---:|---:|:---:|
| 1 | 200 | -0.024628157 | 200 | -0.024628157 | true |
| 2 | 300 | -4.524628162 | 300 | -4.524628162 | true |
| 3 | 16 | -5.274628162 | 16 | -5.274628162 | true |
| 4 | 160 | -5.649628162 | 160 | -5.649628162 | true |
| 5 | 17299 | -7.774628162 | 17299 | -7.774628162 | true |
| 6 | 51757 | -8.149627686 | 51757 | -8.149627686 | true |
| 7 | 75 | -8.149627686 | 75 | -8.149627686 | true |
| 8 | 28 | -8.399627686 | 28 | -8.399627686 | true |
| 9 | 463 | -8.524627686 | 463 | -8.524627686 | true |
| 10 | 86 | -8.649627686 | 86 | -8.649627686 | true |

#### `[100,200,300,400,200,19,200]`

| rank | GC id | GC logprob | Mega id | Mega logprob | equal |
|---:|---:|---:|---:|---:|:---:|
| 1 | 19 | -0.382356703 | 19 | -0.382356703 | true |
| 2 | 300 | -2.132356644 | 300 | -2.132356644 | true |
| 3 | 18 | -2.632356644 | 18 | -2.632356644 | true |
| 4 | 20 | -3.382356644 | 20 | -3.382356644 | true |
| 5 | 160 | -4.257356644 | 160 | -4.257356644 | true |
| 6 | 26 | -4.632356644 | 26 | -4.632356644 | true |
| 7 | 9 | -4.882356644 | 9 | -4.882356644 | true |
| 8 | 25 | -5.132356644 | 25 | -5.132356644 | true |
| 9 | 23 | -5.132356644 | 23 | -5.132356644 | true |
| 10 | 22 | -5.132356644 | 22 | -5.132356644 | true |

## 8×40 层运行期核验

`inspect_v41_moe` 在成功路径返回 8 个 worker，每个 worker 恰好 40 层，共 320 层。每层 expert class 均为 `DeepseekV4MegaMoEExperts`，`sequence_parallel=true` 为 320/320，`fused_shared=true` 为 320/320。机器可读结果见 `attempt2-venv-ptxas-13.5/inspect-v41-moe.json` 与 `introspection-summary.json`。本轮没有发生 MegaMoE shared-expert 融合回退。

## 资源与释放

- cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`，`memory.max=max`
- Attempt 1：`memory.current` 前 `443,500,027,904` bytes；峰值 `869,367,078,912` bytes（`23:18:00`）；后 `567,338,127,360` bytes
- Attempt 2：`memory.current` 前 `567,479,795,712` bytes；峰值 `851,762,585,600` bytes（`23:23:12`）；后 `718,116,794,368` bytes
- 两次观测的 `oom` / `oom_kill` / `oom_group_kill` 始终为 `0 / 0 / 0`
- Attempt 2 `MemAvailable` 最低 `883,155,148 kB`；结束 `972,738,236 kB`
- `memory.current` 是整个 session cgroup 的计数，包含可回收页缓存，不是单模型 RSS
- 成功路径 `2026-11-21T23:24:12+08:00` 退出；`23:24:47` 复核 compute-apps 为空，`/dev/nvidia0..7` 均无残留进程
- system journal：`PERMISSION_LIMITED`，无法据此完成 system-wide Xid/OOM 审计；cgroup、进程退出码与释放复核单独保留

## 环境

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- `nvidia-cutlass-dsl` `4.7.1`
- cuda-python / cuda-bindings `13.5.0`（venv 原有）
- Python `3.12.7`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- GCMoE：本工作树 `source/gc_moe`，版本 `0.3.1`
- `VLLM_USE_V2_MODEL_RUNNER=1`，`HF_HUB_OFFLINE=1`
- `enable_flashinfer_autotune=False`，`gpu_memory_utilization=0.8`，`enforce_eager=True`

## 产物

`docs/sm100_megamoe/v41_gc_jit_retry_results/` 按 `attempt1-system-ptxas-13.0/` 与 `attempt2-venv-ptxas-13.5/` 分开保存完整日志、退出码、版本、预检、adapter PID、首个 traceback、四前缀 top10、8×40 introspection、cgroup/GPU/journal 证据、参考 Mega 结果、汇编器清单与本轮一行 diff；`SHA256SUMS` 覆盖全部产物。未提交模型权重、venv 或编译缓存。
