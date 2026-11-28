# SM100 · GLM‑5.3 原生 GC、逐层 EPLB 和 ECO

执行者为 **SM100**。模型取自本机 HF3FS 的 GLM5.3 FP4moe-FP8ue8m0。三个整模臂均用 Bruce GC fused shared，底座为 `af5a7a750800954f52b9ff28cc55d945ead48da6`，ECO 使用本仓库的 `scripts/sm100/bruce_sm100_eco.patch`。三臂均从进程启动设置 `GC_MOE_SM100_TRACE=2`。本轮先按上一份 SM100 回执实际要求的 BS256（32 个 decode token/rank）准备三臂。

## 立即执行，先回传私有权重的接口信息

在运行 GLM 的 vLLM Python 环境中，从项目根目录执行。全程只用 CPU；八卡被占用也可以立即完成。可使用旧 SM100 环境启动，但需写明本机真正用于加载这份 GLM 的 Python 和 vLLM 版本，若是不同环境要用后者再采一份。

~~~bash
set -euo pipefail
out=docs/sm100_megamoe/glm53_ep8_trace2_results
mkdir -p "$out"
git rev-parse HEAD > "$out/revision.txt"
export GC_MOE_SM100_TRACE=2
rc=0
CUDA_VISIBLE_DEVICES='' python scripts/sm100/glm53_checkpoint.py --out "$out/checkpoint.json" || rc=$?
echo "$rc" > "$out/probe-exit-code.txt"
~~~

程序检查目录名带空格、量化名为子目录、量化名仅为说明这三种路径。若有歧义，使用 `--model "/实际的/GLM 量化目录"` 指定用户的这份权重。若某个环境已有 GLM 的启动命令或本地 loader 补丁，在回执列出实际入口、版本、影响专家权重加载的类/方法及补丁；只提交与接口相关的源码片段。模型拥有独立量化配置时以配置和 tensor header 为准。仅向 GitHub 提交元数据，不上传原始权重或用户提示。立即将 JSON、退出码和 `docs/sm100_megamoe/REPLY_SM100_GLM53_EP8_TRACE2.md` 提交到 main。失败同样回传 JSON。

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
