# 【SM100】修正 shared expert 权重布局后的 GCMoE 复测回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**验收失败，进程退出 `1`。** 上游 `c4a1410` 的 `preserve_loader_weight` 修复生效：8/8 个 worker 都记录了 GC adapter 安装，上轮的 ModelOpt `MXFP8_VALUE_DTYPE` 断言未再出现。但引擎在 profile/dummy run 中首次执行 GCMoE launch 时，JIT 编译缓存身份构建调用 `settings.cache_key_str()` 失败，EngineCore 未能完成初始化。没有进入四前缀推理，没有 top1/top10，也没有生成 token。

- 测量提交：`c4a1410ae78317ccba5911aeeb25a0298a333320`
- 任务书 blob：`2a87dae1765ebc0bebc77e84b467aea493652a9f`
- 工作目录：`/data0/home/admin/workspace/megamoe/native-v41-DJ0Nezlq`
- 运行窗口：`2026-11-21T22:58:24+08:00` 至 `23:01:24+08:00`
- 模型进程退出码：`1`；launcher 退出码：`1`
- checkpoint：`48/48` 分片已加载
- 实际推理：**未开始**
- 输出 ID：**无**
- 外部定时终止：`none`
- 八卡释放：`2026-11-21T23:01:24+08:00`；`23:01:55` 复核 compute-apps 与 `/dev/nvidia0..7` 均为空

## 首个错误

首个 worker 错误出现在 `2026-11-21T23:00:49+08:00`，调用链为：

```text
profile_run / _dummy_run
  -> model forward
  -> deep_gemm.fp8_fp4_mega_moe
  -> source/gc_moe/gc/__init__.py:240 fp8_fp4_gc_moe
  -> source/gc_moe/apis/gc_moe.py:169 launch_gc_moe
  -> source/gc_moe/jit_kernels/impls/sm100_fp8_fp4_gc_moe.py:575 prepare_role_binaries
  -> source/gc_moe/jit_kernels/impls/sm100_fp8_fp4_gc_moe.py:167 _compiler_identity
  -> source/gc_moe/runtime/jit.py:129 compiler_identity
```

原始错误：

```text
settings.cache_key_str(), normalized,
AttributeError: 'EnvironmentVarManager' object has no attribute 'cache_key_str'
```

EngineCore 随后报告：

```text
RuntimeError: Worker failed with error ''EnvironmentVarManager' object has no attribute 'cache_key_str''
RuntimeError: Engine core initialization failed. See root cause above.
```

当前已验证 vLLM venv 的 `nvidia-cutlass-dsl` 版本为 `4.7.1`。该环境用于本轮，未安装、覆盖或替换共享 DSL 依赖。此失败发生在 GCMoE JIT 编译缓存身份构建阶段；本轮没有修改 GCMoE、vLLM 或 harness 代码，也没有可以附上的额外编译修复 diff。

完整首个 worker traceback 见 `first-worker-traceback.txt`。

## shared expert 修复与状态

- `scripts/sm100/v41_gc_worker.py` 已包含任务要求的 `preserve_loader_weight`。
- 上游修复 diff 原样保存在 `applied-fix.diff`。
- 8 个不同 worker PID 均输出 `V41 GC expert adapter installed`：

| worker | PID |
|---|---:|
| TP0/EP0 | 222532 |
| TP1/EP1 | 222534 |
| TP2/EP2 | 222536 |
| TP3/EP3 | 222537 |
| TP4/EP4 | 222540 |
| TP5/EP5 | 222541 |
| TP6/EP6 | 222543 |
| TP7/EP7 | 222545 |

上轮 ModelOpt 原始断言已单独保留在 `modelopt-assert-preserved.txt`。本轮日志中没有再次出现该断言，也没有出现 `Disabling native MegaMoE shared-expert fusion`。但由于 profile/dummy run 在层 RPC 之前失败，`sequence_parallel`、`fused_shared` 和每 rank 40 层的运行期状态均为 **NOT_REACHED**，不能声称已完成 8×40 层融合核对。

## 四前缀与 MegaMoE 对照

四个固定前缀推理均未执行。对照对象仍是 `v41_fixed_prefix_results/prefix-audit.json` 中上一轮 `deep_gemm_mega_moe` 的首选：

```text
[100,200,300,400]                     -> 75
[100,200,300,400,200]                 -> 300
[100,200,300,400,200,19]              -> 200
[100,200,300,400,200,19,200]          -> 19
```

本轮没有 GCMoE top1/top10 可并列，故首选 ID、名次、top1/top2 logprob 差值以及融合回退情况均为 **N/A / 未测得**。

## 环境与资源

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- `nvidia-cutlass-dsl` `4.7.1`
- Python `3.12.7`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- GCMoE：本工作树 `source/gc_moe`，版本 `0.3.1`
- cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`，`memory.max=max`
- `memory.current`：前 `560,153,497,600` bytes；峰值 `858,715,082,752` bytes（`23:00:24`）；后 `624,622,370,816` bytes
- `oom` / `oom_kill` / `oom_group_kill`：前、观测峰值、后均为 `0 / 0 / 0`
- `MemAvailable` 最低：`889,693,952 kB`；结束：`1,013,078,492 kB`
- system journal：`PERMISSION_LIMITED`，无法据 system journal 完成 Xid/OOM 审计
- GPU：8 × `NVIDIA L20D`，capability `(10, 3)`；启动前 compute-apps 为空

## 产物

`docs/sm100_megamoe/v41_gc_shared_retry_results/` 包含完整日志、版本与依赖、adapter 安装证据、首个 worker traceback、上游修复 diff、上轮 ModelOpt 断言原文、cgroup/GPU/journal 证据、参考 prefix audit 与 `SHA256SUMS`。未提交模型权重、venv 或编译缓存。
