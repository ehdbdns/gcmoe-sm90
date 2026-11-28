# SM100 · GLM‑5.3 原生 GC、逐层 EPLB 和 ECO

> **给 SM100 执行者：八张卡全部空闲后才能开始 GPU 实验。** 当前 CPU 元数据采集可以立即执行。收到正式启动命令后，每一臂加载模型前，先核对调度分配，再用 `nvidia-smi` 查看分配给本任务的八张卡。只要其中一张仍有其他作业占用或已被他人预约，就让整组等待，60 秒后复查。等待期间不预占空闲卡，不终止他人进程。每臂结束即释放八张卡，下一臂启动前重新检查。

执行者为 **SM100**。模型取自本机 HF3FS 的 GLM5.3 FP4moe-FP8ue8m0。三个整模臂均用 Bruce GC fused shared，底座为 `af5a7a750800954f52b9ff28cc55d945ead48da6`，ECO 使用本仓库的 `scripts/sm100/bruce_sm100_eco.patch`。三臂均从进程启动设置 `GC_MOE_SM100_TRACE=2`。本轮先按上一份 SM100 回执实际要求的 BS256（32 个 decode token/rank）准备三臂。

## 新执行者从这里开始

在能访问 `/mnt/hf3fs/3fs/models` 的 **SM100 机器**打开 Bash。你负责执行命令和交回原始结果；代码、参数选择和数据分析由构建者负责。**本次任务完成的标志是 CPU 回执已经出现在 GitHub 的 main 分支。** 这一步随时可做，机器上的 GPU 正在运行其他人的作业也不影响。它读取 checkpoint 配置、索引、分片头和本机 vLLM 安装信息，通常只需 Git、Bash 和 Python 3.9+，不用安装 CUDA 或重新下载模型。

如果当前 shell 已激活该 GLM 实际使用的 vLLM 环境，直接复制下面整段。若你已知专用解释器或模型实际路径，在执行前分别设置 `export GLM_PYTHON='/实际路径/bin/python'` 和 `export GLM_MODEL='/实际 checkpoint 目录'`；路径含空格时保留引号。默认检查模型名与量化名连成一个目录、量化名作为子目录、量化名仅为说明这三种情况。解释器优先取当前环境中安装了 vLLM 的 Python，其次检查同机两处实验历史环境；回执区分自动发现与显式指定。即使只有系统 Python，也先交回模型头信息。

### 一次复制，采集并交回

~~~bash
(
set -euo pipefail
umask 077
job=$(mktemp -d "${TMPDIR:-/tmp}/sm100-glm53.XXXXXXXX")
echo "本次私有工作目录为 $job"
repo="$job/gcmoe-sm90"
git clone --depth=1 --filter=blob:none --sparse --branch main \
  https://github.com/ehdbdns/gcmoe-sm90.git "$repo"
cd "$repo"
git sparse-checkout set --no-cone \
  '/scripts/sm100/glm53_checkpoint.py' '/scripts/sm100/run_glm53_probe.sh' \
  '/docs/sm100_megamoe/SM100_GLM53_EP8_TRACE2.md' \
  '/docs/sm100_megamoe/REPLY_SM100_GLM53_EP8_TRACE2.md' \
  '/docs/sm100_megamoe/glm53_ep8_trace2_results/'
bash scripts/sm100/run_glm53_probe.sh
git add -- docs/sm100_megamoe/REPLY_SM100_GLM53_EP8_TRACE2.md \
  docs/sm100_megamoe/glm53_ep8_trace2_results/
git -c user.name="SM100 executor" -c user.email="sm100-executor@users.noreply.github.com" \
  commit -m "[SM100] Report GLM-5.3 checkpoint and loader inventory"
git push origin HEAD:refs/heads/main
git fetch --depth=8 origin main
git merge-base --is-ancestor HEAD FETCH_HEAD
echo "回执已在 main，对应提交 $(git rev-parse HEAD)"
)
~~~

脚本自动生成 Markdown 回执，以及本次独立目录里的 `revision.txt`、`probe-exit-code.txt`、`probe.log` 和 `checkpoint.json`。若解释器启动即失败，以日志和退出码为准。回执显示的“采集器退出码”反映采集结果；即使采集失败，脚本也会生成回执、继续提交。只有看到最后一行的提交号才算已交付。GPU 实验在此阶段的正确计数为 **0 臂**。

### 出现问题时照此处理

| 看到的情况 | 具体操作 |
|---|---|
| 找到零个或多个 checkpoint | 在该机运行 `ls -ld /mnt/hf3fs/3fs/models/GLM* /mnt/hf3fs/3fs/models/GLM5.3/*`。若可确认是指定量化模型，设置 `GLM_MODEL` 到同时含 `config.json` 和 `model.safetensors.index.json` 的目录，再执行整段命令。若挂载缺失，保留失败回执。 |
| 已采到权重、vLLM 未检出，或机器实际用另一个 Python | 已知实际环境时指定 `GLM_PYTHON` 再执行整段命令。自动识别只证明找到一种已安装的 vLLM。若该机另有启用 GLM 专用 loader 的本地模块，将模块名、类名、版本和相关启动参数补入回执；忽略凭证、模型权重和提示词正文。 |
| 模型分片缺失、无法读取或 shape 与预期不同 | 交回该次失败 JSON 和日志，由构建者处置。执行者无需改权重、跳过专家或修 probe。 |
| Git 提示 non-fast-forward | 进入第一行打印的私有目录下的 `gcmoe-sm90`，执行 `git pull --rebase origin main`、`git push origin HEAD:refs/heads/main`，然后重新执行上面的 fetch 和祖先检查。若回执文件本身冲突，保留现场并报告本地提交号及目录，避免覆盖另一份回执。 |
| clone/push 提示无读写权限、网络错误或分支保护 | 使用本机已获授权的仓库凭据重试同一条 Git 命令。若仍失败，交回错误信息与私有目录；已经 commit 时一并给出本地提交号。 |
| shell 在生成回执前停止 | 根据打印的目录查看现场，记录失败命令与退出信息。若机器缺 Python 3.9+，改用本机已有环境；共享环境无需升级。 |

可在 GitHub 核对回执。[SM100 回执位置](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/REPLY_SM100_GLM53_EP8_TRACE2.md)

## 三臂的运行口径

| 臂 | 放置与分流 |
|---|---|
| 原生 GC | checkpoint 逻辑专家的原生连续放置，发布同 BS 的输入、原生反馈、逻辑 top-k ID 和 gate 权重 |
| GC + EPLB | 原生臂前 1000 个完整 decode 步的逐层专家直方图决定 DeepInfer 放置；用 RR 分流 |
| GC + EPLB + ECO | 与 RR 共用逐层放置、物理权重排列和冗余数；设备 ECO 消费相同逻辑路由，plan 用通信分区的 1 个 SM |

按现有 SM100 比较口径两条 EPLB 臂每 rank 额外 8 个槽。按真实 checkpoint 读取层号、稠密前缀、top-k、shared 数、激活与量化配方。采用 Natureal 的 BM 对应价格，在回执记录实际 BM、价格和 SM 分配。使用适合 GLM tokenizer 的同一批请求。每臂 1000 步统计，随即在相同 KV 边界上测量后 1000 步。保存每步每层逻辑路由/权重、反馈、KV 边界与覆盖前原生差异的校验。记录每 rank 每层实收槽计数、M 块、活跃槽、graph 和 routed 时间，给出 MoE µs/层、两组相对变化、各 IR、峰值、总 M 块，以及 ECO routed MAE、成交和接受率。trace 2 的耗时单独标明。

第 2 个测量 replay 对实际 decoder 层 8、20、32 导出每 rank 的 torch.profiler 原件、rank 计数、概览 Perfetto 与 absolute Perfetto。如实际层为 dense，以配置中邻近且互异的 routed 层补足并记录实际编号。概览同轨展示 count exchange 和 pull token，另列 shared L1/L2、wait token、首个真实 MMA 到最后 L2 push 的 routed GEMM、combine 及 ECO plan。记录采样溢出和时钟锚点误差。

## 本提交可执行范围与机器占用

本版可执行的是 checkpoint 与环境采集。仓内原有 `run_ep8_four.sh` 挂在 V4 的 `DeepseekV4MegaMoEExperts`，该入口对 GLM 不适用。负责写代码的一方将根据这份实物清单接入 GLM loader、路由与归并并给出整模启动命令。CPU 回执结束后即可释放工作环境，无须占卡等待。正式运行只在调度授权且同组八张 Blackwell 卡均空闲后整组取得，Busy 时整组等待。每臂退出后释放；只清理自己的 worker、私有环境及临时缓存，保留失败日志和原始数据。执行者只可处理编译兼容性问题；发生量化、路由、输出或图捕获差异时，保留首差并回传。
