# 【SM100】修正副本表结束标记后的在线 ECO 复测回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

通过。按任务书修正主机侧 SM100 副本表空项编码后，使用相同输入、相同真实第 0 层权重、相同 logical experts、相同环境与验收条件复测，56/56 条 rank 结果全部通过：

- 前六臂 48/48 rank 通过。
- `gc_eplb_eco_gpu`：8/8 rank 通过；`plan_matches_reference=true`、`counts_match_history=true`、`slot_count_match=true`、`valid_output=true`。
- `gc_eplb_eco_gpu` 对 MegaMoE shared identity 的 `different_elements=0`，`max_abs_diff=0.0`。
- 在线配置仍为 `[64, 260, 111, 1, 6, 0]`。
- 进程退出码 0，未修改输入、路由、权重、验收条件或运行参数。

上一轮在线 ECO 的首个失败记录仍保留在 `online_gc_eco_results/` 与 `REPLY_ONLINE_GC_ECO.md`；本轮结果单独写入 `online_gc_eco_sentinel_results/`，没有覆盖原失败记录。

## 执行提交与相关改动

- 测试仓库提交：`cf4fcf484c5f644ad945402fdaccd35346f56560`
- 任务书 blob：`56b87770f86345717b0c69644ed9b2204d42e05c`
- DeepGEMM：`5081f8513af57e61696de4f4f04b0e707ffb503e`
- 修正文件 blob：`source/gc_moe/eco/meta_sm100.py=feb427342d487cf7f4c2affe8df5d901e2608ea3`，`eco_ref/eco_meta.py=6144791c327d63ddfdd10a6f0bde6a0c38abe7b0`
- 测试脚本 blob：`mega_rr_smoke.py=6f6b34555509bf1abd32519dd783f44ef194e33c`
- 使用修复后的 Blackwell 副本表结束标记：空项编码为 `0xffff`；`eco_ref` 也改为调用同一 SM100 构建函数。
- 未做其他代码或编译补丁。

实际 smoke 命令：

```bash
python scripts/sm100/mega_rr_smoke.py --live-eco --compare-gc --eco-reference --shared --checkpoint "$SM100_V41_MODEL"
```

## 依赖与执行环境

- 节点：`e01-cn-v8n4s5sfd01`，单节点 8 卡。
- 基础环境：`/home/admin/workspace/megamoe/env.sh`；Python 3.12.7；Torch `2.14.1+cu130`；CUDA 13.0；NCCL 2.30.7；compute capability `(10, 3)`。
- 复用 `/data0/home/admin/workspace/megamoe/rr-smoke-ezB7Fty7/venv`；本轮未安装新包。
- 依赖仍为：`nvidia-cutlass-dsl==4.8.1`、其 base/core/cu12 libs `4.8.1`、`cuda-python==13.2.0`、`cuda-bindings==13.2.0`、`cuda-pathfinder==1.9.1`、`protobuf==7.36.2`、`typing_extensions==4.16.0`。
- `GC_MOE_NVCC=/usr/local/cuda/bin/nvcc`；`GC_MOE_ECO_CACHE_DIR` 指向本轮临时目录。
- import 校验成功：Torch、本工作树 GCMoE、venv 中 cutlass、DeepGEMM 路径正确。

## 模型与用例

- 模型：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 架构：`DeepseekV41ForCausalLM`
- shape：`hidden=5120`、`moe_intermediate_size=2304`、`routed_experts=384`、`top-k=6`、`layers=40`
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- `model.safetensors.index.json` SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 每 rank 256 tokens、8 ranks、全局 448 个物理槽、每 rank 56 个；输入与 gate 为合成数据；routed/shared 权重来自第 0 层真实 checkpoint。
- 在线 ECO 使用按每卡 single-copy 前缀重排后的物理槽和对应权重。

## 八卡分配与释放

启动前目标 8 卡空闲；未与外部任务重叠。使用共享锁 `/tmp/gcmoe-sm100-gpu-group.lock`。

分配开始：`2026-11-21T17:20:24+08:00`  
释放完成：`2026-11-21T17:23:31+08:00`

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

## 全部 56 条 rank 结果

`different_elements` 是相对 MegaMoE shared identity 的逐元素 BF16 差异数；`max_abs_diff` 是 float 最大绝对差。`routed_control` 是首个执行臂，identity 建立参考，因此这两臂没有参考差值。`valid` 要求输出有限且非全零；`slot_match` 要求本 rank 实际接收行数与计划一致。

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
| gc_eplb_eco_gpu | 0 | 0 | 0.0 | 1309263 | true | true | true | true |
| gc_eplb_eco_gpu | 1 | 0 | 0.0 | 1309272 | true | true | true | true |
| gc_eplb_eco_gpu | 2 | 0 | 0.0 | 1309270 | true | true | true | true |
| gc_eplb_eco_gpu | 3 | 0 | 0.0 | 1309263 | true | true | true | true |
| gc_eplb_eco_gpu | 4 | 0 | 0.0 | 1309314 | true | true | true | true |
| gc_eplb_eco_gpu | 5 | 0 | 0.0 | 1309224 | true | true | true | true |
| gc_eplb_eco_gpu | 6 | 0 | 0.0 | 1309231 | true | true | true | true |
| gc_eplb_eco_gpu | 7 | 0 | 0.0 | 1309299 | true | true | true | true |

`smoke.log` 包含 56 条 JSON rank 记录，进程退出码 0。逐条记录同时保存为 `rank_results.jsonl`，并按 `ranks/<arm>/rank<rank>.json` 分开保存。

## 警告与限制

启动阶段每个 rank 仍出现 `MLOPart is unavailable; localization has been disabled` 与 `1 MLOPart devices` 断言；这是可选本地化能力不可用，并非本次测试失败，后续七臂全部完成。完整 stdout/stderr 保留在 `smoke.log`。

本用例只覆盖第 0 层数值与分流，不覆盖完整模型前向、端到端精度或性能。

## 附件

- `online_gc_eco_sentinel_results/config.log`：V4.1 shape 校验与 index SHA-256。
- `online_gc_eco_sentinel_results/checkpoint.log`：第 0 层真实 routed 权重元数据。
- `online_gc_eco_sentinel_results/deps.log`、`pip-freeze.log`：复用 venv 的依赖状态。
- `online_gc_eco_sentinel_results/import.log`：Torch、GCMoE、cutlass、DeepGEMM 导入路径。
- `online_gc_eco_sentinel_results/smoke.log`：完整 stdout/stderr、56 条 rank JSON、分配与释放时间。
- `online_gc_eco_sentinel_results/gpu_allocation.txt`：GPU UUID、占用/释放和前后置检查。
- `online_gc_eco_sentinel_results/rank_results.jsonl`：56 条 rank 结果汇总。
- `online_gc_eco_sentinel_results/ranks/`：56 个独立 rank JSON 文件。
- `online_gc_eco_sentinel_results/summary.csv`：七臂通过数与差异范围汇总。
- `online_gc_eco_sentinel_results/eco_cache_manifest.txt`：在线 ECO 编译缓存清单。
- `online_gc_eco_sentinel_results/SHA256SUMS`：附件校验和。
