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

## 3. 阻塞项（正式八臂未启动的原因；3.1 已由设计端解决）

### 3.1 模型权重（**已解决**）

重启后原路径 `/home/admin/local_models/DeepSeek-V4.1-Flash` 消失，设计端提供了新 NAS 并完成挂载：

```bash
sudo mkdir -p /mnt/dsv41
sudo mount -t nfs -o vers=4,minorversion=0,noresvport \
  954af48778-koo23.cn-zhangjiakou.nas.aliyuncs.com:/ /mnt/dsv41
```

**挂载后绝对路径**：`/mnt/dsv41/models/DeepSeek-V4.1-Flash`

| 检查 | 结果 |
|---|---|
| `config.json` 存在 | ✅ |
| `*.safetensors` 个数 | ✅ **48**（与设计端说明一致）|
| 目录总量 | ✅ **476 G**（与设计端说明一致）|
| **`config.json` SHA256** | **`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`** |
| **`model.safetensors.index.json` SHA256** | **`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`** |

### 3.1b 数据集（**已解决**）

同一 NAS，无需另挂：

| 文件 | 行数 | SHA256 |
|---|---:|---|
| `dsv4-flash_online_input_ids_26-10-08_1.jsonl` | 1,500 | `3664daca1619617b76026e6a75e4b3738316320aceb2852c76f2897873e99073` |
| `dsv4-flash_online_input_ids_26-10-08_2.jsonl` | 1,500 | `957d06fed999f9fe517dab4581a2fe79f162debc1c21468502d3e988c83511f0` |

目录：`/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets/`

**两份 SHA256 与重启前备份时记录的值逐位相同**，说明 NAS 上的数据未被改动。

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

1. ~~**模型权重位置**~~ → **已解决**：`/mnt/dsv41/models/DeepSeek-V4.1-Flash`（476 G / 48 shards）
2. ~~**数据集位置**~~ → **已解决**：`/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets/`
3. **八臂启动器 SHA 与正式命令**：任务书说由设计端补充，但 `a9f58ca` 中尚未出现 ← **当前唯一硬阻塞**
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


---

# 追加：CPU 私有构建（`27473c1`）— ✅ **成功**

## A. 构建结果

`deep_gemm` 已在**私有持久目录**成功构建并安装，**共享 `env3` 未被改动**（`--target` 到私有 `site/`）。

| 项 | 值 |
|---|---|
| 构建目录 | `app_source/sm90_persist_20261123/dg-sm90-build-ohe59RVP` |
| 源码 revision | **`7292e7bd0ef3b497ca948a7239d55f3f01539048`**（= 任务书指定）|
| submodules | cutlass `f3fde58`（v4.2.1）、deep_jit `2efdab4` |
| **wheel** | `deep_gemm-2.8.2+7292e7b-cp310-cp310-linux_x86_64.whl` |
| **wheel SHA256** | **`7c848494dbd01daa77ea80765888073cb63b85d7f74054fbcead404598ddf61e`** |
| 安装目标 | `<area>/site/deep_gemm` + `deep_gemm-2.8.2+7292e7b.dist-info` |
| 编译架构 | `TORCH_CUDA_ARCH_LIST=9.0`（SM90/H20）、`DG_FORCE_BUILD=1`、`MAX_JOBS=8`、`CUDA_VISIBLE_DEVICES=''` |

## B. 三个官方入口已可用（CPU 导入验证，未上卡）

```
deep_gemm.__version__ = 2.8.2
fp8_mega_moe -> (y, l1_weights, l2_weights, sym_buffer,
                 cumulative_local_expert_recv_stats=None, recipe=(128,128,128),
                 activation='swiglu', activation_clamp=None, fast_math=True)
get_symm_buffer_for_sm90_mega_moe -> (group, num_experts, num_max_tokens_per_rank,
                 num_topk, hidden, intermediate_hidden, use_fp8_dispatch=True, activation='swiglu')
transform_weights_for_mega_moe_sm90 -> (l1_weights, l2_weights) -> ...
```

完整记录见 `deepgemm_wheel_and_import.txt`。

## C. 途中解决的两个环境阻塞（均为**不改语义**的系统/构建配置）

| # | 阻塞 | 解决方式 | 是否改动实验代码 |
|---|---|---|---|
| 1 | 头文件 `elfutils/libdwfl.h` 缺失 | `sudo yum install -y --disablerepo='epel*' elfutils-devel`（**禁用不可达的 epel 源**后用 Alinux 源成功；`libdw.so.1` 本就存在）| ❌ 否（仅系统开发包）|
| 2 | `g++` 10.2 不支持 C++20 `<format>` | 改用本机已有的 **conda-forge GCC 15.3.0**（`app_source/deep_ep25_ar_probe_20261025/toolchain`）作为 `CC`/`CXX` | ❌ 否（仅编译器选择）|
| 3 | 导入时 `GLIBCXX_3.4.31` 缺失 | 运行时把该 toolchain 的 `lib` 前置到 `LD_LIBRARY_PATH`（仓库 `scripts/env.sh` 本就这么做，第 18 行 `LD_LIBRARY_PATH=$NCCL_ROOT/lib:$TOOLCHAIN/lib:...`）| ❌ 否（与既有 env.sh 一致）|

**未做**：没有改写 `deep_jit` 源码、没有伪造头文件、没有改 `setup.py` 的宏或编译开关。
三次尝试的原始记录留在 `dgbuild-blocker.txt`（首错）与 `implicit-build-dependency.txt`（grep 摘录）。

## D. 正式运行时如何使用这个私有包

任务书要求"正式运行使用单进程私有包路径"。启动器需要：

```bash
export PYTHONPATH=<area>/site:$PWD/source:$PWD/patches:$PWD/integration
export LD_LIBRARY_PATH=<toolchain>/lib:$LD_LIBRARY_PATH   # scripts/env.sh 已含同类设置
```

（`<area>` = `app_source/sm90_persist_20261123/dg-sm90-build-ohe59RVP`）

## E. 仍缺的一项

八臂启动器与正式命令：任务书 §"用卡、交付和执行边界"说由设计端补充固定 SHA，
`27473c1` 只补了 CPU 构建这一段，**尚未给出八臂命令**。模型、数据集、`deep_gemm` 均已就绪，
只等这一项即可在八卡全空闲时开跑。

## F. 现场

| 内容 | 路径 |
|---|---|
| 构建区（源码/submodule/wheel/site/日志）| `app_source/sm90_persist_20261123/dg-sm90-build-ohe59RVP/` |
| wheel 与导入验证日志 | `.../sm90_h20_bs256_mega_gc_eight_results/deepgemm_wheel_and_import.txt` |
| 首错与三次尝试记录 | `.../dgbuild-blocker.txt`、`.../implicit-build-dependency.txt` |

GPU：八卡仍全空闲（各 4 MiB，0 个 compute app），**本轮未占用**。

---

# 正式八臂（`3682a33` + 我的一处修复）— **arm2 在 CUDA graph 捕获处失败，版本不兼容**

## A. 启动与前置条件（全部通过）

| 步 | 结果 |
|---|---|
| 输入校验 `sha256sum -c model_and_dataset.sha256` | ✅ 4/4 OK（模型 config、index、两份数据集）|
| 私有 wheel 摘要 | ✅ `7c848494…` OK |
| CPU 权重转换 `check_mega_weights.py` | ✅ 3 个投影（w1/w3/w2）全部成功，`max_abs_error` = 0.00446 / 0.00446 / 0.00893 |
| HCA ACTIVE 检查 | ✅ mlx5_4/5/9/17/21/29 全部 ACTIVE |
| 八卡空闲门禁 | ✅ 8 张全 H20、0 个 compute app |
| 磁盘 | ✅ `/mnt/fuse` 5.0 T 可用（> 405 GB 要求）|

启动命令即任务书原文（`scripts/sm90/run_eight.sh`），worktree 建于 `/mnt/fuse/sm90-h20-eight-wteT0PXZ`。

## B. 我修的第一个 bug（保持语义）

**现象**（第一次启动 `3682a33` 原样）：arm2 在 `process_weights_after_loading` 处崩：

```
File ".../integration/sm90_eight.py", line 34, in install_clocks
    buf.ffn_counter = torch.zeros(40, ..., device=buf.buffer.device)
AttributeError: 'NoneType' object has no attribute 'buffer'
```

**根因**：`integration/deepep_compare.py` 的 `_allocate_all_gc_workspaces()` 遍历
`model.modules()` 查找 `GCMoEExperts`，但该类**不是 nn.Module**（同文件第 41 行注释明说
"vLLM creates one per layer inside Mxfp4MoEMethod"），所以永远找不到实例、
`symmetric buffer` 保持 `None`。本文件其它地方（第 1497 行）用的是 `_EXPERT_REGISTRY`。

**修复**（已提交 `be92873`，diff 见 `executor_impl_fix.diff`）：

```python
-    for module in model.modules():
-        if not isinstance(module, GCMoEExperts):
-            continue
+    live = [m for m in _EXPERT_REGISTRY if isinstance(m, GCMoEExperts)]
+    for module in live:
         if module._buffer is None:
             module.allocate_workspace()
-    if any(isinstance(m, GCMoEExperts) for m in model.modules()):
-        _warm_gc_child_graph(model)
+    if live:
+        _warm_gc_child_graph(live[0])
```

并对 `_warm_gc_child_graph` 增加"接受活的 expert 对象或 model"两种入参。

**效果**：该错误消失，arm2 前进到 CUDA graph 捕获阶段（见 C）。**未改动**分流、精度、计时或验收口径。

## C. 当前硬阻塞：DeepGEMM wheel 与 vLLM 0.31.0 的 sparse-indexer 签名不兼容

修复后 arm2 在 **CUDA graph 捕获**时，8 个 rank 全部失败于同一处：

```
TypeError: fp8_fp4_paged_mqa_logits(): incompatible function arguments.
（先由 fp8_fp4_mqa_logits 触发，见下）
```

**精确原因**（已定位到具体调用与签名）：

| 侧 | 内容 |
|---|---|
| **vLLM 0.31.0 调用**（`env3/.../layers/sparse_attn_indexer.py:583`）| `fp8_fp4_mqa_logits(q_tuple, kv_tuple, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False)` —— **第 6 个参数是关键字 `clean_logits`** |
| **DeepGEMM 2.8.2 的签名**（我们按任务书构建的 wheel）| `fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, **max_seqlen_k**, logits_dtype=None, schedule_meta=None)` —— **没有 `clean_logits`，第 6 位是整数 `max_seqlen_k`** |

即：**任务书指定的 DeepGEMM `7292e7b`（v2.8.2）与这台机器的 vLLM 0.31.0 不匹配**。
vLLM 期望的是另一版 DeepGEMM（带 `clean_logits` / `indices` 的 sparse-indexer API）。

## D. 为什么这不是我该自行修的

| 可能的"修法" | 为什么不做 |
|---|---|
| 改 `env3` 里的 `sparse_attn_indexer.py`（改调用签名）| 改动 vLLM 共享包，影响所有其他任务；且需设计端确认语义 |
| 改我们构建的 DeepGEMM 源码加 `clean_logits` 参数 | 等于**自行移植后端**，任务书明令禁止（"不自行移植后端或改精度、路由、验收口径"）|
| 换一个 DeepGEMM 版本 | 任务书钉死 `7292e7b` / wheel 摘要 `7c848494…` |
| 关闭 sparse indexer | 改变模型语义 |

**按任务书"运行时数值、显存、框架兼容或硬件问题按下面的失败流程记录"** 处理：
保留首错、日志、退出码，不绕过。

## E. 八臂状态（本轮）

| 臂 | 状态 | 说明 |
|---|---|---|
| 2（发布者）| ❌ `exit=1` | 见 C；**已到达 CUDA graph 捕获**，说明 A/B 两项都通过 |
| 1,3,4,5,6,7,8 | ⏭ `skipped=publisher_failed` | 依赖 arm2 的 KV/token/路由参考，按启动器逻辑跳过 |

**没有任何臂产出性能数字**，因此**不填主表**（任务书要求"第 2 臂失败会阻断依赖它的参考回放"）。

## F. 退卡与清理

| 项 | 结果 |
|---|---|
| GPU | **八卡全空**（各 4 MiB，`--query-compute-apps` = 0）|
| worktree | `/mnt/fuse/sm90-h20-eight-wteT0PXZ/code` 已 `git worktree remove` + `prune` |
| 现场 | `/mnt/fuse/sm90-h20-eight-wteT0PXZ/results/`（2.4 MB）保留：launch.log、arm2/runner.log、gpus.csv、conversion.jsonl、各 arm 的 exit/skipped |
| 私有 wheel | 保留，未改动 |
| 历史 retained_bs256 | 未触碰 |

## G. 需要设计端决定（唯一阻塞）

1. **DeepGEMM 版本**：本机 vLLM 0.31.0 需要带 `clean_logits` 的 sparse-indexer API。
   请确认是否改用别的 DeepGEMM commit（并给出新 SHA/期望 wheel 摘要），
   还是由设计端提供匹配 vLLM 0.31.0 的 wheel。
2. **若坚持 `7292e7b`**：需要设计端授权我修改 `vllm` 的调用签名（或加兼容 shim），
   并明确这属于"保持语义的适配"——否则我无法继续。
