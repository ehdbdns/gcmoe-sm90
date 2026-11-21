# 【SM100】V4.1-Flash 的 GCMoE 与 MegaMoE 单层正确性对照回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

通过。使用同一次进程、同一组 BF16 输入与量化结果、同一份 DeepSeek-V4.1-Flash 第 0 层真实权重、同一路由及 shared expert，完成 MegaMoE 与 GCMoE 的 EPLB+RR、ECO reference 六臂对照。共 48 条 rank 结果全部通过：

- `routed_control`：8/8 rank 通过。
- shared `identity`：8/8 rank 通过；相对 `routed_control` 每 rank 有超过 130 万个元素改变。
- MegaMoE `eplb_rr`：8/8 rank 通过；对 MegaMoE identity 的 `different_elements=0`，`max_abs_diff=0.0`。
- MegaMoE `eplb_eco_reference`：8/8 rank 通过；对 MegaMoE identity 的 `different_elements=0`，`max_abs_diff=0.0`。
- GCMoE `gc_eplb_rr`：8/8 rank 通过；对 MegaMoE identity 的 `different_elements=0`，`max_abs_diff=0.0`。
- GCMoE `gc_eplb_eco_reference`：8/8 rank 通过；对 MegaMoE identity 的 `different_elements=0`，`max_abs_diff=0.0`。

本用例未遇到首个差异；两个 GCMoE 臂的最终 BF16 输出均与 MegaMoE shared identity 逐元素相同。所有 48 条结果均为 `valid_output=true`、`slot_count_match=true`。未做性能测量。

## 执行提交与代码

- 测试仓库提交：`4a426d2a4f225f3e8cb2ed74d1436951914ddb8c`
- 任务书 blob：`6b6f43aced86eb897d6e84363a930c7441f5da82`
- DeepGEMM：`5081f8513af57e61696de4f4f04b0e707ffb503e`
- 测试脚本 blob：`mega_rr_smoke.py=0d3ef10d7f0c8b91f758913bbed88cb205578ba0`
- GCMoE blob：`source/gc_moe/__init__.py=5fe1eacd3aba8df85a5e4d4fe32ce51d1bbeb4cb`，`source/gc_moe/gc/__init__.py=d37a9b58fe08e13e5daf943a620280e42bbccbbc`
- 未修改输入、路由、权重、验收条件或运行参数；未做代码/编译补丁。

实际 smoke 命令：

```bash
python scripts/sm100/mega_rr_smoke.py --compare-gc --eco-reference --shared --checkpoint "$SM100_V41_MODEL"
```

## 依赖与环境

- 节点：`e01-cn-v8n4s5sfd01`，单节点 8 卡。
- 基础环境：`/home/admin/workspace/megamoe/env.sh`；Python 3.12.7；Torch `2.14.1+cu130`；CUDA 13.0；NCCL 2.30.7；compute capability `(10, 3)`。
- 临时 venv 继承已验证的基础 site-packages，任务依赖只装在临时目录。
- 安装结果：`nvidia-cutlass-dsl==4.8.1`、`nvidia-cutlass-dsl-libs-base==4.8.1`、`nvidia-cutlass-dsl-libs-core==4.8.1`、`nvidia-cutlass-dsl-libs-cu12==4.8.1`、`cuda-python==13.2.0`、`cuda-bindings==13.2.0`、`cuda-pathfinder==1.9.1`、`protobuf==7.36.2`、`typing_extensions==4.16.0`。
- 依赖版本与任务要求一致。默认 PyPI 下载速度约 7 KB/s，首次尝试在 91.7 MB wheel 下载中停顿；随后改用清华镜像安装相同 wheel，镜像返回的 `nvidia-cutlass-dsl-libs-cu12` wheel SHA-256 为 `440b88739120c65d96c5f15bbe496a3bd7c26392874d5af6061d38497f84c70c`。两次日志均保留。
- import 校验：Torch `2.14.1+cu130`、本工作树 `gc_moe`、临时 venv 的 `cutlass`、`/home/admin/workspace/megamoe/DeepGEMM` 的 `deep_gemm` 均成功导入。

## 模型与用例

- 模型：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 架构：`DeepseekV41ForCausalLM`
- shape：`hidden=5120`、`moe_intermediate_size=2304`、`routed_experts=384`、`top-k=6`、`layers=40`
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- `model.safetensors.index.json` SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 用例：每 rank 256 tokens、8 ranks、全局 448 个物理槽、每 rank 56 个；routed 与 shared 权重来自第 0 层真实 checkpoint；输入与 gate 为合成数据。
- 同一份量化后的 FP8 激活、scale、top-k 索引与权重、同一 shared expert 同时用于 MegaMoE 和 GCMoE 对照。

ECO reference 参数仍为 `block_m=64`、`wb=260`、`wa=111`，仅验证接口与数值，不用于性能判断。两个 ECO 臂均继续执行配额非负、物理配额和逻辑需求一致、物理槽映射回逻辑专家、all-reduce 后实际直方图等于配额等检查。

## 八卡分配与释放

无作业调度器；使用共享锁 `/tmp/gcmoe-sm100-gpu-group.lock`。启动前确认 compute app 为空、`/dev/nvidia0..7` 无 fd 占用，UUID 与 `ENV.json` 登记一致，CUDA 只可见这 8 卡。

分配开始：`2026-11-21T16:32:55+08:00`  
释放完成：`2026-11-21T16:40:17+08:00`

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

用例结束后复查：compute app 为空，`/dev/nvidia0..7` 无 fd 占用。记录见 `gpu_allocation.txt`。

## 全部 48 条 rank 结果

`different_elements` 是相对 MegaMoE shared identity 的逐元素 BF16 差异数；`max_abs_diff` 是在 float 上计算的同参考最大绝对差。`routed_control` 是首个执行臂；identity 是首个非 control 臂并建立参考，因此两者没有参考列。`changed_from_routed_only` 是相对 `routed_control` 的元素变化数。`valid_output` 要求输出有限且非全零；`slot_count_match` 要求每 rank 实际接收行数与路由期望一致。

| arm | rank | different_elements | max_abs_diff | changed_from_routed_only | valid_output | slot_count_match |
|---|---:|---:|---:|---:|---|---|
| routed_control | 0 | 0 | — | — | true | true |
| routed_control | 1 | 0 | — | — | true | true |
| routed_control | 2 | 0 | — | — | true | true |
| routed_control | 3 | 0 | — | — | true | true |
| routed_control | 4 | 0 | — | — | true | true |
| routed_control | 5 | 0 | — | — | true | true |
| routed_control | 6 | 0 | — | — | true | true |
| routed_control | 7 | 0 | — | — | true | true |
| identity | 0 | 0 | — | 1309263 | true | true |
| identity | 1 | 0 | — | 1309272 | true | true |
| identity | 2 | 0 | — | 1309270 | true | true |
| identity | 3 | 0 | — | 1309263 | true | true |
| identity | 4 | 0 | — | 1309314 | true | true |
| identity | 5 | 0 | — | 1309224 | true | true |
| identity | 6 | 0 | — | 1309231 | true | true |
| identity | 7 | 0 | — | 1309299 | true | true |
| eplb_rr | 0 | 0 | 0.0 | 1309263 | true | true |
| eplb_rr | 1 | 0 | 0.0 | 1309272 | true | true |
| eplb_rr | 2 | 0 | 0.0 | 1309270 | true | true |
| eplb_rr | 3 | 0 | 0.0 | 1309263 | true | true |
| eplb_rr | 4 | 0 | 0.0 | 1309314 | true | true |
| eplb_rr | 5 | 0 | 0.0 | 1309224 | true | true |
| eplb_rr | 6 | 0 | 0.0 | 1309231 | true | true |
| eplb_rr | 7 | 0 | 0.0 | 1309299 | true | true |
| eplb_eco_reference | 0 | 0 | 0.0 | 1309263 | true | true |
| eplb_eco_reference | 1 | 0 | 0.0 | 1309272 | true | true |
| eplb_eco_reference | 2 | 0 | 0.0 | 1309270 | true | true |
| eplb_eco_reference | 3 | 0 | 0.0 | 1309263 | true | true |
| eplb_eco_reference | 4 | 0 | 0.0 | 1309314 | true | true |
| eplb_eco_reference | 5 | 0 | 0.0 | 1309224 | true | true |
| eplb_eco_reference | 6 | 0 | 0.0 | 1309231 | true | true |
| eplb_eco_reference | 7 | 0 | 0.0 | 1309299 | true | true |
| gc_eplb_rr | 0 | 0 | 0.0 | 1309263 | true | true |
| gc_eplb_rr | 1 | 0 | 0.0 | 1309272 | true | true |
| gc_eplb_rr | 2 | 0 | 0.0 | 1309270 | true | true |
| gc_eplb_rr | 3 | 0 | 0.0 | 1309263 | true | true |
| gc_eplb_rr | 4 | 0 | 0.0 | 1309314 | true | true |
| gc_eplb_rr | 5 | 0 | 0.0 | 1309224 | true | true |
| gc_eplb_rr | 6 | 0 | 0.0 | 1309231 | true | true |
| gc_eplb_rr | 7 | 0 | 0.0 | 1309299 | true | true |
| gc_eplb_eco_reference | 0 | 0 | 0.0 | 1309263 | true | true |
| gc_eplb_eco_reference | 1 | 0 | 0.0 | 1309272 | true | true |
| gc_eplb_eco_reference | 2 | 0 | 0.0 | 1309270 | true | true |
| gc_eplb_eco_reference | 3 | 0 | 0.0 | 1309263 | true | true |
| gc_eplb_eco_reference | 4 | 0 | 0.0 | 1309314 | true | true |
| gc_eplb_eco_reference | 5 | 0 | 0.0 | 1309224 | true | true |
| gc_eplb_eco_reference | 6 | 0 | 0.0 | 1309231 | true | true |
| gc_eplb_eco_reference | 7 | 0 | 0.0 | 1309299 | true | true |

`smoke.log` 共包含 48 条 JSON rank 记录，进程退出码 0。逐条记录同时保存为 `rank_results.jsonl`，并按 `ranks/<arm>/rank<rank>.json` 分开保存。

## 警告与限制

启动阶段每个 rank 仍出现 `MLOPart is unavailable; localization has been disabled` 与 `1 MLOPart devices` 断言；这是可选本地化能力不可用，并非本次对照失败，后续六臂均完成。完整 stdout/stderr 保留在 `smoke.log`。

本用例只验证第 0 层数值与分流一致性，不覆盖完整模型前向、端到端精度或性能。

## 附件

- `gc_parity_results/config.log`：V4.1 shape 校验与 index SHA-256。
- `gc_parity_results/checkpoint.log`：第 0 层真实 routed 权重元数据。
- `gc_parity_results/deps.log`：实际安装日志；`deps.default-index.log`：默认索引停滞记录。
- `gc_parity_results/pip-freeze.log`：临时 venv 的依赖版本。
- `gc_parity_results/import.log`：Torch、GCMoE、cutlass、DeepGEMM 导入路径。
- `gc_parity_results/smoke.log`：完整 stdout/stderr、48 条 rank JSON、分配与释放时间。
- `gc_parity_results/gpu_allocation.txt`：GPU UUID、占用/释放时间与前后置检查。
- `gc_parity_results/rank_results.jsonl`：48 条 rank 结果汇总。
- `gc_parity_results/ranks/`：48 个独立 rank JSON 文件。
- `gc_parity_results/summary.csv`：六臂通过数与差异范围汇总。
- `gc_parity_results/SHA256SUMS`：附件校验和。
