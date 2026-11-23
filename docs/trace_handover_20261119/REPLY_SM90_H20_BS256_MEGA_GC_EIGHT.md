# 【SM90】REPLY_SM90_H20_BS256_MEGA_GC_EIGHT — 第一阶段 CPU 核验完成；正式八臂**受阻于模型与启动器**

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 状态总览

| 项 | 结果 |
|---|---|
| 任务书 SHA | `a9f58ca` |
| **第一阶段 CPU 核验** | ✅ **已完成**（日志见 `cpu-preflight.log`）|
| **八臂正式启动** | ❌ **未启动**（两条阻塞，见 §3）|
| GPU | 八卡全空闲（各 4 MiB，无 compute app），本轮**未占用任何一卡** |

## 1. CPU 核验结果（任务书第一阶段）

### 1.1 DeepGEMM 官方源码

| 项 | 值 |
|---|---|
| 分支 | `nv_dev` |
| 检出 commit | **`7292e7bd0ef3b497ca948a7239d55f3f01539048`** ← 与任务书指定值一致 |
| 路径 | `/tmp/sm90-h20-eight-run1/DeepGEMM`（临时，重启后不保留）|
| 说明 | `nv_dev` 分支的 HEAD 本身就等于该 commit，验证无漂移 |

三个关键入口在源码中均存在：

```
deep_gemm/__init__.py:97:    get_symm_buffer_for_sm90_mega_moe,
deep_gemm/__init__.py:98:    transform_weights_for_mega_moe_sm90,
deep_gemm/__init__.py:99:    fp8_mega_moe,
deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:54:  sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(...)
deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:159: sm90_fp8_mega_moe_core(...)
deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:2323: sm90_fp8_mega_moe_l1_impl(...)
tests/test_mega_moe_sm90.py:548:    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
tests/test_mega_moe_sm90.py:558:    deep_gemm.fp8_mega_moe(
```

### 1.2 本机 Python 环境（与上次成功的 SM90 环境一致）

| 项 | 值 |
|---|---|
| Python | `/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python` **3.10.9** |
| torch | **2.13.0**（`env3/lib/python3.10/site-packages/torch`）|
| vllm | **0.31.0**（`env3/lib/python3.10/site-packages/vllm`）|
| nvidia-cutlass-dsl | **4.8.1** |
| `gc_moe` | `/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/gc_moe_src/gc_moe/__init__.py` |

**源码与运行环境不同目录**（已按要求分别记录）：
- 源码：`docs/.../sm90_h20_bs256_mega_gc_eight_results` 对应的仓库 worktree（`a9f58ca`）
- 运行环境：`/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3`

### 1.3 `deep_gemm` 与 `sm90_fp8_fp4_mega_moe` 的可用性

| 检查 | 结果 |
|---|---|
| `find_spec("deep_gemm")` | **None**（未安装为可导入包）|
| `importlib.metadata` 的 `deep-gemm` | **no installed distribution metadata** |
| `find_spec("sm90_fp8_fp4_mega_moe")` | **None** |
| `dir(gc_moe)` 中含 `mega` 的符号 | **空列表** |
| site-packages 下 `deep_gemm*` / `*ggemm*` | **不存在** |

**结论**：官方 `fp8_mega_moe` 只存在于刚 clone 的源码树里，**尚未构建/安装到运行环境**；
本机也**没有**任何已安装包暴露 `sm90_fp8_fp4_mega_moe` 入口。

> 按任务书："若已安装包另外暴露 sm90_fp8_fp4_mega_moe，记录签名和源码出处。"
> → 本机**未暴露**，故无签名可记。矩阵 1/5 采用官方 `fp8_mega_moe`（源码已在 §1.1 定位）。

## 2. 现场与容量

| 项 | 值 |
|---|---|
| **保留的 BS256 KV** | `/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123/retained_bs256/kv` = **320 GiB**（8 × 20 GiB `rank*.bin` + `prefill/` 同规模 + JSON 清单）|
| **保留的路由回放** | 同目录 `replay/` = **12 GiB**（`stats-rank*.pt`、`measure-rank*.pt`、`tokens.json`、`placement.json`、各臂 layout）|
| 可用私有磁盘 | `/mnt/fuse` **5.0 T 总量 / 5.0 T 可用**（ext4 本地盘）；NAS `/home/admin/workspace/aop_lab` 剩余 **90 T** |
| GPU | 8 × NVIDIA H20，driver **580.105.08** |
| 活跃 HCA（重启后变化）| `mlx5_4`、`mlx5_5`、`mlx5_9`、`mlx5_17`、`mlx5_21`、`mlx5_29`（**共 6 个**，与旧配置的 4 个不同）|

## 3. 两条阻塞（正式八臂未启动的原因）

### 3.1 模型权重不在本机

任务书要求的 `DeepSeek-V4.1-Flash` 在本机**不存在**：

| 检查 | 结果 |
|---|---|
| `/home/admin/local_models/DeepSeek-V4.1-Flash` | **不存在**（重启后 `/home/admin/local_models` 整个目录也不存在）|
| 全盘 `find` 名为 `DeepSeek-V4.1-Flash` 的目录 | **无** |
| 全盘 `*.safetensors`（`/home/admin`、`/mnt` 下 6 层深）| **无** |
| `/mnt/nas2/...` 数据集路径 | **不存在**（`/mnt` 下只有 `fuse`）|

因此**无法计算**任务书要求的"原模型配置与权重索引 SHA256"——文件本身不在。

### 3.2 八臂启动器尚未交付

任务书原文：

> 本仓当前 `run_main24.sh` 实现的是既有 GC 矩阵，**不能执行以上八臂**。
> 八臂启动器、Mega 的实际模型适配和核外 shared 计时接好后，
> **由设计端在本任务书补充固定 SHA 和正式命令**。

我核对了 `a9f58ca` 的仓库状态：**文档中没有补充 SHA，也没有正式命令**。
因此没有可执行的八臂入口——这属于设计端待办，执行者不得"自行移植后端或改精度、路由、验收口径"。

## 4. 本轮未做的事（合规说明）

| 项目 | 原因 |
|---|---|
| 未启动任何模型进程 | 任务书规定第一阶段只做 CPU 核验；且 §3 两条阻塞未解除 |
| 未占用任何 GPU | 八卡始终空闲；未做 GPU 预检（任务书明确"无需上卡验证安装"）|
| 未构建/安装 `deep_gemm` | 任务书说"无需上卡验证安装"；且"不自行移植后端" |
| 未改写 `run_main24.sh` | 任务书明令禁止 |
| 未修改 HCA 配置 | 重启后 HCA 已变化（6 个 ACTIVE），但本轮未跑 GPU，留待正式阶段由设计端确认 |

## 5. 交付物

`docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/`：

| 文件 | 内容 |
|---|---|
| `cpu-preflight.log` | 任务书要求的完整 CPU 核验日志（35 行：task SHA、DeepGEMM SHA、符号 grep、Python/包版本）|
| `gpu.txt` | 八卡型号/驱动/占用（全 4 MiB 空闲）|
| `active-hca.txt` | 重启后 6 个 ACTIVE HCA |
| `state-path.txt` | 临时工作目录（`/tmp`，重启后不保留）|

## 6. 需要设计端确认（按优先级）

1. **模型权重位置**：重启后 `DeepSeek-V4.1-Flash` 不在本机。请提供新路径（或恢复挂载），
   我才能计算权重索引 SHA256 并启动任何 GPU 阶段。
2. **数据集位置**：`/mnt/nas2/...dsv4-flash_online_input_ids_26-10-08_{1,2}.jsonl` 也随重启消失，
   请一并提供新路径。
3. **八臂启动器 SHA 与正式命令**：任务书说由设计端补充，但 `a9f58ca` 中尚未出现。
4. **`deep_gemm` 的构建方式**：是否需要我按 `nv_dev@7292e7b` 在 `env3` 中构建安装？
   （这会改动共享环境，需要授权。）
5. **HCA 变化**：重启后 ACTIVE HCA 变为 `mlx5_4/5/9/17/21/29`（6 个），
   旧配置默认是 `mlx5_2/8/20/24`。正式阶段需要新的 `NCCL_IB_HCA`。

## 7. 环境恢复情况（供参考）

重启导致 `/tmp` 全部清空，本轮已从持久备份恢复：

| 项 | 状态 |
|---|---|
| 仓库 | `app_source/sm90_persist_20261123/repo`，HEAD `a9f58ca`，工作区**干净**（2842 个文件已从 `.git` 完整还原）|
| 保留 KV（320 G）+ 回放（12 G）| ✅ 恢复可用，路径见 §2 |
| 历史回执与产物 | ✅ 完整（`docs/trace_handover_20261119`、`docs/trace_executor_20261119`）|

> 注：重启后仓库的 `remote` 仍带 token；`git fetch` 正常，已与 `origin/main` 同步到 `a9f58ca`。
