# 【SM100】V4.1-Flash 第 0 层真实 shared + routed 权重正确性回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

通过。使用同一份 DeepSeek-V4.1-Flash 第 0 层真实 checkpoint、同一批合成输入与 gate，在 8 张 SM100 上顺序完成三臂：

- `routed_control`：8/8 rank 通过，输出有限非零，物理槽计数匹配。
- 带 shared expert 的 `identity`：8/8 rank 通过；相对 `routed_control` 每 rank 均有超过 130 万个元素改变；输出有限非零，物理槽计数匹配。
- 带 shared expert 的 `eplb_rr`：8/8 rank 通过；相对 identity 的元素差异为 0；相对 `routed_control` 每 rank 同样有超过 130 万个元素改变；输出有限非零，物理槽计数匹配。

后两臂逐元素相等，说明在本用例范围内共享 expert 路径产生实际贡献且换槽保持一致。该结果不提供独立模型输出基准。

执行 SHA：

- 测试仓库：`aafe4d427a07d027c418f2d1dd8534a62a306891`
- 任务书 blob：`05ca6fb903fe9120df3bef89e3785c8e4900a289`
- DeepGEMM：`5081f8513af57e61696de4f4f04b0e707ffb503e`
- 脚本 blob：`mega_rr_smoke.py=592b4ebb6dd3337d0d746f57fe902d8fba3e372f`，`real_routed_weights.py=7fe4cb169bf866b43baa4af81df09f957fd6963f`，`v41_shape.py=0b655f2549cdf210e554324837877d3cc1184050`，`checkpoint_experts.py=50342700435dca931c4156c0c4e16e78cc1f79c5`

## 模型与来源

- 实际路径：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 模型卡：`DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression`；README 链接 `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`。
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- `model.safetensors.index.json` SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 架构：`DeepseekV41ForCausalLM`；shape 校验从 `text_config` 读取，结果为 `hidden=5120`、`intermediate=2304`、`routed_experts=384`、`topk=6`、`layers=40`。

## 执行环境与规模

- 节点：`e01-cn-v8n4s5sfd01`，单节点 8 卡。
- 环境：`/home/admin/workspace/megamoe/env.sh`；Python 3.12.7；Torch `2.14.1+cu130`；CUDA 13.0；NCCL 2.30.7；实际 compute capability `(10, 3)`。
- 用例规模：384 routed experts、hidden 5120、intermediate 2304、top-k 6、每 rank 256 tokens、8 ranks；全局 448 个物理槽，每 rank 56 个。
- routed 与 shared 权重均读取自 V4.1 第 0 层真实 checkpoint；输入与 gate 为合成数据。
- 未启动模型、未安装框架、未测性能。

## 八卡分配与释放

无作业调度器；使用共享锁 `/tmp/gcmoe-sm100-gpu-group.lock`。启动前确认 compute app 为空、`/dev/nvidia0..7` 无 fd 占用、UUID 与 `ENV.json` 登记一致，CUDA 仅可见这 8 卡。

分配开始：`2026-11-21T15:45:42+08:00`  
释放完成：`2026-11-21T15:50:44+08:00`

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

用例结束后复查：无 compute app、无 nvidia 设备 fd、无锁持有者。

## 全部 24 条 rank 结果

`different_elements`：对 `eplb_rr` 是相对 identity 的逐元素差异；identity 作为首个非 control 臂建立参考。`changed_from_routed_only`：相对 `routed_control` 的元素变化数。`valid_output` 要求 BF16 输出有限且非全零；`slot_count_match` 要求每 rank 物理槽实际接收行数与路由期望一致。

| arm | rank | different_elements | changed_from_routed_only | valid_output | slot_count_match |
|---|---:|---:|---:|---|---|
| routed_control | 0 | 0 | — | true | true |
| routed_control | 1 | 0 | — | true | true |
| routed_control | 2 | 0 | — | true | true |
| routed_control | 3 | 0 | — | true | true |
| routed_control | 4 | 0 | — | true | true |
| routed_control | 5 | 0 | — | true | true |
| routed_control | 6 | 0 | — | true | true |
| routed_control | 7 | 0 | — | true | true |
| identity | 0 | 0 | 1309263 | true | true |
| identity | 1 | 0 | 1309272 | true | true |
| identity | 2 | 0 | 1309270 | true | true |
| identity | 3 | 0 | 1309263 | true | true |
| identity | 4 | 0 | 1309314 | true | true |
| identity | 5 | 0 | 1309224 | true | true |
| identity | 6 | 0 | 1309231 | true | true |
| identity | 7 | 0 | 1309299 | true | true |
| eplb_rr | 0 | 0 | 1309263 | true | true |
| eplb_rr | 1 | 0 | 1309272 | true | true |
| eplb_rr | 2 | 0 | 1309270 | true | true |
| eplb_rr | 3 | 0 | 1309263 | true | true |
| eplb_rr | 4 | 0 | 1309314 | true | true |
| eplb_rr | 5 | 0 | 1309224 | true | true |
| eplb_rr | 6 | 0 | 1309231 | true | true |
| eplb_rr | 7 | 0 | 1309299 | true | true |

`smoke.log` 共包含 24 条 JSON rank 记录，`smoke_exit=0`。

## 警告与限制

启动阶段每个 rank 打印 `MLOPart is unavailable; localization has been disabled` 与 `1 MLOPart devices` 断言告警；这是可选本地化能力不可用，进程未失败，所有内核检查均通过。完整输出保留在 `smoke.log`。

本用例只验证 shared 路径实际改变输出、identity 与 EPLB+RR 在共享加载器下逐元素一致，以及物理槽行数正确；不覆盖完整模型前向、外部参考精度或性能。

## 附件

- `shared_rr_results/config.log`：V4.1 shape 校验与 index SHA-256。
- `shared_rr_results/checkpoint.log`：第 0 层真实 routed 权重元数据。
- `shared_rr_results/smoke.log`：分配、启动告警、24 条 rank JSON、退出码与释放时间。
- `shared_rr_results/SHA256SUMS`：附件校验和。
