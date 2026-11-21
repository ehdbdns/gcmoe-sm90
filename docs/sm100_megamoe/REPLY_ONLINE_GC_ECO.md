# 【SM100】GCMoE 在线 ECO 冷启动八卡验证回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

验收失败。GCMoE 在线 ECO 冷启动运行实际输出 56 条 rank JSON，但进程退出码为 1。前六臂 48/48 rank 通过；新增的 `gc_eplb_eco_gpu` 臂 0/8 rank 通过：

- `gc_eplb_eco_gpu` 的 GPU 计划状态与参考规划一致：8/8 rank `plan_matches_reference=true`。
- 全局逻辑计数状态与 history 一致：8/8 rank `counts_match_history=true`。
- 本 rank 实际接收行数与计划一致：8/8 rank `slot_count_match=true`。
- 输出有限且非零：8/8 rank `valid_output=true`。
- 但与 MegaMoE shared identity 的逐元素输出不一致：8/8 rank `different_elements` 为 5045–30382，`max_abs_diff_from_mega_identity` 为 `1.0–5.0`。

首个失败臂为 `gc_eplb_eco_gpu`；最终异常为：

```text
RuntimeError: gc_eplb_eco_gpu: 8 ranks failed the correctness check
```

该失败由脚本既有的 `different_elements != 0` 验收条件触发，不是编译失败。未修改输入、验收条件、运行参数或代码，也未重跑或掩盖结果。完整 stdout/stderr 在 `smoke.log`。

## 执行提交与代码

- 测试仓库提交：`cb0226f292d7a1d37392f31196a36fed11d802a1`
- 任务书 blob：`5806694ee893da472e868b4de697e92f6f900c2c`
- DeepGEMM：`5081f8513af57e61696de4f4f04b0e707ffb503e`
- 测试脚本 blob：`mega_rr_smoke.py=6f6b34555509bf1abd32519dd783f44ef194e33c`
- Blackwell ECO 元数据 blob：`meta_sm100.py=2b353a352070173c978f6b19c8af2f40a7d6e7fe`
- GCMoE blob：`source/gc_moe/__init__.py=5fe1eacd3aba8df85a5e4d4fe32ce51d1bbeb4cb`，`source/gc_moe/gc/__init__.py=d37a9b58fe08e13e5daf943a620280e42bbccbbc`
- 未做代码或编译补丁。

实际 smoke 命令：

```bash
python scripts/sm100/mega_rr_smoke.py --live-eco --compare-gc --eco-reference --shared --checkpoint "$SM100_V41_MODEL"
```

## 依赖与执行环境

- 节点：`e01-cn-v8n4s5sfd01`，单节点 8 卡。
- 基础环境：`/home/admin/workspace/megamoe/env.sh`；Python 3.12.7；Torch `2.14.1+cu130`；CUDA 13.0；NCCL 2.30.7；compute capability `(10, 3)`。
- 任务指定 venv：`/data0/home/admin/workspace/megamoe/rr-smoke-ezB7Fty7/venv`。
- 任务依赖：`nvidia-cutlass-dsl==4.8.1`、`nvidia-cutlass-dsl-libs-base==4.8.1`、`nvidia-cutlass-dsl-libs-core==4.8.1`、`nvidia-cutlass-dsl-libs-cu12==4.8.1`、`cuda-python==13.2.0`、`cuda-bindings==13.2.0`、`cuda-pathfinder==1.9.1`、`protobuf==7.36.2`、`typing_extensions==4.16.0`。
- `GC_MOE_NVCC=/usr/local/cuda/bin/nvcc`；`GC_MOE_ECO_CACHE_DIR` 指向本次临时工作目录。
- import 校验成功：Torch、本工作树 GCMoE、venv 中 cutlass、DeepGEMM 路径正确。

透明说明：上一份 GC parity 回执提交后，其临时工作目录按清理流程删除，因此任务指定的 venv 路径已不存在。本轮在该路径按上次记录的同版本依赖重建 venv，并继承同一基础 site-packages；`pip-freeze.log` 和 `deps.log` 记录实际结果。这没有改变任务依赖版本或测试输入。

## 模型与用例

- 模型：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 架构：`DeepseekV41ForCausalLM`
- shape：`hidden=5120`、`moe_intermediate_size=2304`、`routed_experts=384`、`top-k=6`、`layers=40`
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- `model.safetensors.index.json` SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 每 rank 256 tokens、8 ranks、全局 448 个物理槽、每 rank 56 个；输入与 gate 为合成数据；routed/shared 权重来自第 0 层真实 checkpoint。
- 在线 ECO 使用按每卡 single-copy 前缀重排后的物理槽和对应权重。
- 在线配置：`online_config=[64, 260, 111, 1, 6, 0]`。

## 八卡分配与释放

开始时目标 8 卡被其他任务占用，运行器按任务书要求等待；占用方结束后才在 `2026-11-21T17:09:02+08:00` 开始本次测试，未与占用方重叠。

分配开始：`2026-11-21T17:09:02+08:00`  
失败发生：`2026-11-21T17:13:06.518+08:00`  
释放/退出：`2026-11-21T17:13:06+08:00`

进程以退出码 1 结束，因此任务脚本没有打印正常路径的 `released_at` 行；失败时间来自 `smoke.log` 的终止告警，释放时间与进程退出同时发生。外部复查于 `2026-11-21T17:13:28+08:00` 完成：compute app 为空、`/dev/nvidia0..7` 无 fd 占用。

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

## 全部 56 条 rank 结果

`different_elements` 是相对 MegaMoE shared identity 的逐元素 BF16 差异数；`max_abs_diff` 是 float 最大绝对差。`routed_control` 是首个执行臂，identity 建立参考，因此这两臂没有参考差值。`plan_matches_reference`、`counts_match_history` 只在在线 ECO 臂产生。`valid_output` 要求有限且非全零；`slot_count_match` 要求本 rank 实际接收行数与计划一致。

| arm | rank | different_elements | max_abs_diff | changed_from_routed_only | valid | slot_match | plan_match | counts_match |
|---|---:|---:|---:|---:|---|---|---|---|
| routed_control | 0 | 0 | — | — | true | true | — | — |
| routed_control | 1 | 0 | — | — | true | true | — | — |
| routed_control | 2 | 0 | — | — | true | true | — | — |
| routed_control | 3 | 0 | — | — | true | true | — | — |
| routed_control | 4 | 0 | — | — | true | true | — | — |
| routed_control | 5 | 0 | — | — | true | true | — | — |
| routed_control | 6 | 0 | — | — | true | true | — | — |
| routed_control | 7 | 0 | — | — | true | true | — | — |
| identity | 0 | 0 | — | 1309263 | true | true | — | — |
| identity | 1 | 0 | — | 1309272 | true | true | — | — |
| identity | 2 | 0 | — | 1309270 | true | true | — | — |
| identity | 3 | 0 | — | 1309263 | true | true | — | — |
| identity | 4 | 0 | — | 1309314 | true | true | — | — |
| identity | 5 | 0 | — | 1309224 | true | true | — | — |
| identity | 6 | 0 | — | 1309231 | true | true | — | — |
| identity | 7 | 0 | — | 1309299 | true | true | — | — |
| eplb_rr | 0 | 0 | 0.0 | 1309263 | true | true | — | — |
| eplb_rr | 1 | 0 | 0.0 | 1309272 | true | true | — | — |
| eplb_rr | 2 | 0 | 0.0 | 1309270 | true | true | — | — |
| eplb_rr | 3 | 0 | 0.0 | 1309263 | true | true | — | — |
| eplb_rr | 4 | 0 | 0.0 | 1309314 | true | true | — | — |
| eplb_rr | 5 | 0 | 0.0 | 1309224 | true | true | — | — |
| eplb_rr | 6 | 0 | 0.0 | 1309231 | true | true | — | — |
| eplb_rr | 7 | 0 | 0.0 | 1309299 | true | true | — | — |
| eplb_eco_reference | 0 | 0 | 0.0 | 1309263 | true | true | — | — |
| eplb_eco_reference | 1 | 0 | 0.0 | 1309272 | true | true | — | — |
| eplb_eco_reference | 2 | 0 | 0.0 | 1309270 | true | true | — | — |
| eplb_eco_reference | 3 | 0 | 0.0 | 1309263 | true | true | — | — |
| eplb_eco_reference | 4 | 0 | 0.0 | 1309314 | true | true | — | — |
| eplb_eco_reference | 5 | 0 | 0.0 | 1309224 | true | true | — | — |
| eplb_eco_reference | 6 | 0 | 0.0 | 1309231 | true | true | — | — |
| eplb_eco_reference | 7 | 0 | 0.0 | 1309299 | true | true | — | — |
| gc_eplb_rr | 0 | 0 | 0.0 | 1309263 | true | true | — | — |
| gc_eplb_rr | 1 | 0 | 0.0 | 1309272 | true | true | — | — |
| gc_eplb_rr | 2 | 0 | 0.0 | 1309270 | true | true | — | — |
| gc_eplb_rr | 3 | 0 | 0.0 | 1309263 | true | true | — | — |
| gc_eplb_rr | 4 | 0 | 0.0 | 1309314 | true | true | — | — |
| gc_eplb_rr | 5 | 0 | 0.0 | 1309224 | true | true | — | — |
| gc_eplb_rr | 6 | 0 | 0.0 | 1309231 | true | true | — | — |
| gc_eplb_rr | 7 | 0 | 0.0 | 1309299 | true | true | — | — |
| gc_eplb_eco_reference | 0 | 0 | 0.0 | 1309263 | true | true | — | — |
| gc_eplb_eco_reference | 1 | 0 | 0.0 | 1309272 | true | true | — | — |
| gc_eplb_eco_reference | 2 | 0 | 0.0 | 1309270 | true | true | — | — |
| gc_eplb_eco_reference | 3 | 0 | 0.0 | 1309263 | true | true | — | — |
| gc_eplb_eco_reference | 4 | 0 | 0.0 | 1309314 | true | true | — | — |
| gc_eplb_eco_reference | 5 | 0 | 0.0 | 1309224 | true | true | — | — |
| gc_eplb_eco_reference | 6 | 0 | 0.0 | 1309231 | true | true | — | — |
| gc_eplb_eco_reference | 7 | 0 | 0.0 | 1309299 | true | true | — | — |
| gc_eplb_eco_gpu | 0 | 10069 | 1.0 | 1309264 | true | true | true | true |
| gc_eplb_eco_gpu | 1 | 5045 | 1.375 | 1309276 | true | true | true | true |
| gc_eplb_eco_gpu | 2 | 15205 | 1.796875 | 1309267 | true | true | true | true |
| gc_eplb_eco_gpu | 3 | 10136 | 1.90234375 | 1309269 | true | true | true | true |
| gc_eplb_eco_gpu | 4 | 30382 | 5.0 | 1309310 | true | true | true | true |
| gc_eplb_eco_gpu | 5 | 15219 | 1.4375 | 1309227 | true | true | true | true |
| gc_eplb_eco_gpu | 6 | 20275 | 2.25 | 1309238 | true | true | true | true |
| gc_eplb_eco_gpu | 7 | 15174 | 3.625 | 1309287 | true | true | true | true |

`smoke.log` 包含 56 条 JSON rank 记录和最终失败栈。逐条记录同时保存为 `rank_results.jsonl`，并按 `ranks/<arm>/rank<rank>.json` 分开保存。

## 警告、失败与限制

- 每次启动均出现已知 `MLOPart is unavailable; localization has been disabled` 与 `1 MLOPart devices` 告警；这些进程继续运行。
- 在线 ECO 内核编译成功，缓存产物清单见 `eco_cache_manifest.txt`。
- 最终错误是 `gc_eplb_eco_gpu` 的输出差异验收失败，不是编译、路由、histogram 或接收行数失败。
- 本用例只覆盖第 0 层数值与分流，不覆盖完整模型前向、端到端精度或性能。

## 附件

- `online_gc_eco_results/config.log`：V4.1 shape 校验与 index SHA-256。
- `online_gc_eco_results/checkpoint.log`：第 0 层真实 routed 权重元数据。
- `online_gc_eco_results/deps.log`、`pip-freeze.log`：重建 venv 的实际依赖。
- `online_gc_eco_results/import.log`：Torch、GCMoE、cutlass、DeepGEMM 导入路径。
- `online_gc_eco_results/smoke.log`：完整 stdout/stderr、56 条 rank JSON、失败栈及时间。
- `online_gc_eco_results/failure_summary.txt`：失败摘要。
- `online_gc_eco_results/gpu_allocation.txt`：GPU UUID、占用/释放和前后置检查。
- `online_gc_eco_results/rank_results.jsonl`：56 条 rank 结果汇总。
- `online_gc_eco_results/ranks/`：56 个独立 rank JSON 文件。
- `online_gc_eco_results/summary.csv`：七臂通过数与差异范围汇总。
- `online_gc_eco_results/eco_cache_manifest.txt`：在线 ECO 编译缓存清单。
- `online_gc_eco_results/SHA256SUMS`：附件校验和。
