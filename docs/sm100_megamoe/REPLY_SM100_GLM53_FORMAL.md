# SM100 · GLM‑5.3 BS256 三臂执行回执（未启动）

**状态：三臂 0/3 完成，无任何测量结果。** 本回执只报告阻塞现场、已核验前提，以及一个需构建者裁定的硬件矛盾。执行者未修改模型配置、权重或仓库代码。

| 项目 | 结果 |
|---|---|
| 任务书 | `docs/sm100_megamoe/SM100_GLM53_FORMAL.md` |
| 任务书提交 | `4f26de7f6deac59fb5b72f05eecde0f28910c59f` |
| 已完成臂数 | 0 / 3（vanilla、rr、eco 均未启动） |
| `run_glm53_three.sh` | **未执行**，卡在私有 checkout 的文件取得阶段 |
| 机器 | `e01-cn-il24rbk1805` |
| 证据采集时间 | 2026-11-28T18:57:51+08:00 |
| 私有目录 | `/data0/home/admin/workspace/glm53-work/formal.jQxs5nOk`（空） |
| 本回执证据目录 | `docs/sm100_megamoe/glm53_formal_results/blocked-20261128T185751Z-e01' |

## 未启动的客观证据

按任务书，`run_glm53_three.sh` 在进入 GPU 阶段前必须先建 `/tmp/gcmoe-sm100-gpu-group.lock` 并 `flock` 等待八卡全空。现场显示这一步从未发生：

- `/tmp/gcmoe-sm100-gpu-group.lock` **不存在**（`No such file or directory`）
- `nvidia-smi --query-compute-apps` 输出为空，8 卡 `utilization.gpu` 全 `0 %`，`memory.used` 仅 5–12 MiB
- 私有目录 `formal.jQxs5nOk` 内无 `revision`、无 `results/run.log`、无 `dsl/`、无 `wheels/`，即脚本主体一行都没跑到
- 本任务全程未占卡、未创建 CUDA 上下文、未编译任何 kernel

## 阻塞原因：本机到 github.com 的 git 传输不稳定

`api.github.com` 全程稳定可用（HTTP 200），但 `github.com` 的 git 传输间歇性失败，实测到的错误：

```
fatal: unable to access 'https://github.com/ehdbdns/gcmoe-sm90.git/': Failed to connect to github.com port 443: Connection timed out
error: RPC failed; curl 52 Empty reply from server
fatal: expected flush after ref listing
fatal: unable to access '...': Empty reply from server
fatal: could not fetch <oid> from promisor remote
```

失败点随阶段不同：

| 阶段 | 现象 |
|---|---|
| `git clone` | 间歇性挂死或 `Connection timed out`，需重试 1–6 次才成功 |
| `git sparse-checkout` 取 blob | partial clone 需回源逐个取 blob，`Empty reply from server` |
| `git commit -- <path>` | 部分提交语义要回源取 HEAD 全部 tree 对象（`--pack_header=2,8771`），单次耗时约 5 分钟 |

执行者的应对与结论：第二阶段已验证「用 `api.github.com` 的 `contents` 端点逐文件取回、并用 `git hash-object` 核对 blob 哈希」这条路可靠（4 个脚本全部 `MATCH`），建议后续阶段把任务书里的 clone/sparse-checkout 替换为该方式，或改用 `git init` + `read-tree` 造提交以避开 promisor 回源。**这属于交付通道问题，不是任务口径问题，执行者未自行改动任务书流程。**

## 需构建者裁定：架构判定自相矛盾

同一批物理卡（UUID 在 `nvidia-smi -L` 与 driver API 输出中完全一致），三处来源给出的架构不一致：

| 来源 | 报告值 | 含义 |
|---|---|---|
| `nvidia-smi --query-gpu=compute_cap` | **8.9** | Ada（SM89） |
| `torch.cuda.get_device_capability()` | **(10, 3)** | Blackwell（SM103） |
| `cuDeviceGetAttribute(75/76)` driver API | **sm_103** | Blackwell（SM103） |
| `cuDeviceGetName` / `nvidia-smi` name | `NVIDIA L20D` | 商用 L20 为 sm_89 / 48 GB |

补充事实：

- 每卡 `148` SM、`267.7 GiB`（`nvidia-smi` 报 `275040 MiB`），与商用 L20 的规格差距极大
- driver `580.105.08`，CUDA driver version `13000`，vbios `97.10.64.00.0C`
- `torch 2.13.0+cu130` 编译 arch 列表为 `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`，**含 sm_100/sm_120，不含 sm_89**
- `hasattr(torch,'float8_e8m0fnu')` 为 `True`；`float8_e4m3fn` 与 `float8_e8m0fnu` 均可在设备上构造
- 实测 `cuda` 上 `1024×1024` matmul 可正常执行并同步返回

**影响**：`glm53_matrix.py` 的门禁是 `torch.cuda.get_device_capability(r)[0] != 10`，按 torch 口径**会通过**（实测 8 卡全为 `False`）。但任务书标题与验收口径明确要求 **Blackwell**。若 `nvidia-smi` 的 8.9 才是真实架构，则 MXFP4 + E8M0 的 SM100 kernel 无法运行，三臂数据整体无效；若 driver/torch 的 sm_103 为真，则 `nvidia-smi` 的 `compute_cap` 字段不可信。执行者不裁定架构，按要求交回构建者。

## 已核验前提（均满足）

| 检查项 | 结果 |
|---|---|
| 八卡是否全空闲 | 是，util 0%、无 compute 进程 |
| `config.json` sha256 | `82bd28e0…86cb2b`，与回执 `68d019e` 一致 |
| `model.safetensors.index.json` sha256 | `fbf6cbce…1aac0c`，与回执 `68d019e` 一致 |
| 工作卷可用空间 | `/data0` 剩 `4079 GiB`，> 脚本要求的 800 GiB |
| `nvcc` / `ptxas` | `/usr/local/cuda/bin/`，CUDA `13.0 V13.0.48` |
| `fuser` / `flock` / `sha256sum` / `realpath` | 均在（`psmisc-23.1-5.1.al8`） |
| 上游 `gc-moe@af5a7a75` | 可达（HTTP 200），已缓存并 detach 到该提交 |
| DSL 4.8.1 wheel | 5 个包在镜像上齐备，`dsl-wheelhouse.sha256` 清单可用于校验 |
| 任务书所需 94 个文件 | 共 0.98 MB，上游 tree 中全部存在 |

## 环境备注

- pip 直连 `pypi.org` 实测握手 `12.2 s`、下载约 `16 KB/s`，装 526 MB 的 torch 需 9 小时以上，不可行；清华镜像实测 `345 MB/s`，第二阶段 8.4 GB venv 安装 + 权重校验共 `2 分 11 秒`
- 本机无任何 `pip.conf`（`/etc/pip.conf`、`~/.pip/pip.conf`、`~/.config/pip/pip.conf`、`/opt/aiext/pip.conf` 均不存在），故任务书「沿用机器已有 pip 镜像配置」实际无配置可沿用，镜像地址由执行时以 `PIP_INDEX_URL` 注入，未写入任何全局配置、未进入回执
- 第二阶段私有 venv 保留在 `/data0/home/admin/workspace/glm53-work/glm53-runtime-qgmxfwva/venv`（8.4 G），即 `run_glm53_three.sh` 的默认 `GLM_RUNTIME_PYTHON` 路径，可直接复用

## 已缓存资产（供恢复时省时间）

| 资产 | 路径 | 大小 |
|---|---|---|
| 第二阶段 venv（torch 2.13.0 + vllm 0.31.0） | `/data0/home/admin/workspace/glm53-work/glm53-runtime-qgmxfwva/venv` | 8.4 G |
| `gc-moe@af5a7a75` 工作副本 | `/data0/home/admin/workspace/glm53-cache/gc-moe` | 31 M |
| `gc-moe` 裸仓（可作 `insteadOf` 本地源，避开 github 抖动） | `/data0/home/admin/workspace/glm53-cache/gc-moe.git` | 124 K |
| 第一/二阶段交付说明与结果副本 | `/data0/home/admin/workspace/sm100-glm53-handoff` | — |

## 执行者请求

1. 裁定上述架构矛盾：本机是否即为任务书所指的「获准使用的同组八张 Blackwell GPU」
2. 确认交付通道：是否允许把 clone/sparse-checkout 替换为 `api.github.com` 取文件（第二阶段已验证），或提供一条稳定的 git 镜像
3. 上述两项确认后即可执行三臂；执行者将严格遵循「八卡全空闲才启动每一臂、每臂结束释放、卡忙每 60 秒复查、不预占空闲卡、不终止他人进程」

原始证据见 `docs/sm100_megamoe/glm53_formal_results/blocked-20261128T185751Z-e01/`，含 `gpu.txt`、`arch.txt`、`env.txt`。
