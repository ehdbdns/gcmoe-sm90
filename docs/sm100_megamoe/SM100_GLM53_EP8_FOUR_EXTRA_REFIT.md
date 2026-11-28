# SM100 · GLM-5.3 EP8 · 每卡 32 个独立流、4 个冗余槽、shared 优先

在原来的八张设备上执行三臂整模 decode。比较原生 GC、逐层 EPLB + RR、同一放置的 EPLB + ECO。模型为 GLM-5.3 FP4 MoE / FP8 UE8M0。站点设备名为 L20D，上次实测 CUDA capability 10.3，并通过了 sm_103f 原生指令探针。本轮代码版本为 `2fd9fba8f37604bfba24ff6e6e486f5078af31f4`。

## 1. 拟合依据和本轮价格

旧回执 `formal.QLfEZmKt` 中的三个臂，均有第 2 个测量步、解码层 8、20、32 的八 rank 数据。共 9 次层调用、72 条 rank 观测。每条记录可用物理槽计数复算本地接收的 routed 行数 T、M 块数 M、非空物理槽数 A，实际时长取首个 routed CTA 的首次 MMA 至最后一个 routed CTA 完成 L2 push。各槽分别按 M16 向上取整再求和。

固定 T 系数为 1，最小化每次层调用中八个预测耗时占比与八个实测占比之差的平方和，每个 rank 等权。连续解为 **wb=66.1573、wa=74.4479**。设备整数价格固定为 **T + 66M + 74A**，满 M 块进入空槽的 slack 为 156，暖轮 1、冷轮 6、双跳 0。旧 M16 配置为 T + 249M + 140A，旧 slack 为 405。

| 占比 MAE，单位为百分点 | 上游旧价格 | 本次取整价格 |
|---|---:|---:|
| 原生，24 行 | 0.30465 | 0.25652 |
| EPLB + RR，24 行 | 0.19440 | 0.20866 |
| EPLB + ECO，24 行 | 0.15600 | 0.16119 |
| 三臂，72 行 | 0.21835 | 0.20879 |
| 按层留一验证，3 折等权 | 0.21835 | 0.21772 |

末行每折只用其余两层训练连续系数，再评估留出的整层。每折一起留出三个臂及全部 rank。完整旧窗口的原始 NPZ 存在执行节点，公开回执提供文件摘要。本任务首先在 CPU 核验之前三臂的 24 份正式 NPZ，然后按完整的 3 × 1000 步 × 75 层评估两套固定价格。额外记录全量重拟合以及在前 800 步训练、后 200 步检验的结果。写入 `historical-price/scores.json`。本次 GPU 实验固定使用上文已经汇报的 66、74。公开样本上的改善集中在原生臂，能否降低新放置下的完整 MoE 耗时由实测决定。[旧实验主表、三臂 Perfetto 和 rank 表](glm53_formal_results/formal.QLfEZmKt/) · [拟合和复核脚本](../../scripts/sm100/glm53_price.py)

公开样本中 2,412 个非空槽观测有 2,260 个只接收一个 M 块。将 wa 固定为 0 或 1、在各折重新拟合 wb，两组留层 MAE 都为 0.31633 个百分点。允许 wa 自由拟合后为 0.21772。三折独立拟合的 wa 分别约为 134、55、62。完整历史窗口在相同的前 800 步拟合、后 200 步检验上也分别报告 wa=0、wa=1 和自由拟合。

## 2. 三臂配置和公平性

| 顺序 | 臂 | 每卡物理槽 | 分发 | CUDA 子图节点提交顺序 |
|---|---|---:|---|---|
| 1 | vanilla | 32（0 个冗余） | 原生无副本 | compute → dispatch → combine |
| 2 | rr | 36（4 个冗余） | 逐层 DeepInfer EPLB + RR | compute → dispatch → combine |
| 3 | eco | 36（4 个冗余） | 同一 EPLB 放置 + 线性 ECO 拍卖 | compute → dispatch → plan → combine |

- TP8/EP8，全局 256 个不同请求。每个源 rank 每步 32 行，在 gate 前取本 rank 的序列分片，或按 rank 从 TP 全局张量取 32 行；top-8 后整个层应接收 2048 行。每个请求每步输出 1 个 token。记录两窗口八个源 rank 的请求 ID，验证切片互不相交、覆盖 256 个请求、成员与顺序保持一致。
- 每臂 1000 步统计、1000 步测量，每步覆盖解码层 3–77 共 75 个 MoE 层。rr、eco 的 288 个物理槽根据本轮原生统计计算。两臂使用同一份 `placement.npy`，逐步逐层核对按逻辑专家归并后与原生实收相等，并逐槽验证 ECO quota。
- 三臂都启用 `GC_MOE_SM100_SHARED_FIRST=1`。SM100 的 shared 在 compute 角色中执行。提交 compute 节点优先，dispatch、plan 可并行入场。ECO 从 12 个通信 SM 中给 plan 1 个，dispatch 使用另外 11 个，compute 保留 136 个。记录八个 rank 实际子图角色、首次 shared MMA 和三阶段重合时间。
- 保留 trace 2、FULL_DECODE_ONLY CUDA Graph、48 GiB KV/rank，使用同一模型、prompt、请求顺序、统计和测量窗口的 KV 与路由参考。检查每个 rank 每个窗口 1000 次 full replay，以及替换参考值前的原始 gate ID、gate weight、模型原生 greedy top-1 和预填充输出差异。运行中启用按 EP 筛选权重、本地分片 mmap 和权重 readahead。

## 3. 从空闲八卡开始

在原节点 `e01-cn-il24rbk1805` 使用站点授权账号、原八卡和原 rank 顺序。先核对调度器预约。上一个执行者仍在运行时等它退卡；八卡中任一张在用就等待，不占用部分空闲卡。保留站点分配的设备掩码。执行器在每臂前取得协作锁并连续两次检查 GPU 进程、句柄、利用率和显存，结束该臂即释放 GPU 锁。权重、旧实验和其他用户作业均保留。

上一轮启动在历史 CPU 拟合时遇到缺少 SciPy，完成 0 个 GPU 臂，首错归档于 `formal.7MYZCaAQ`。使用下面命令新建 job；启动器先查找可用的 CPU 拟合解释器，需要时在作业私有目录安装固定版本的 NumPy 和 SciPy。

默认上次完整原件目录为 `/data0/home/admin/workspace/glm53-work/formal.QLfEZmKt/results`。若站点已经迁移原件，通过 `GLM_PRICE_REFERENCE` 指向原件。脚本核对已发布的文件哈希。使用足以容纳本次模型参考、KV 快照和三个臂结果的私有工作盘。八卡 CUDA 进程之外的准备、历史复核和归档均在 CPU 上运行。执行下面整段命令，记录打印的 job 路径。

~~~bash
(
set -euo pipefail
set +x
umask 077
test "$(hostname -s)" = e01-cn-il24rbk1805
old=${GLM_PRICE_REFERENCE:-/data0/home/admin/workspace/glm53-work/formal.QLfEZmKt/results}
for arm in vanilla rr eco; do test -s "$old/$arm/result.json"; done
export GLM_WORK_ROOT=${GLM_WORK_ROOT:-/data0/home/$(id -un)/workspace/glm53-work}
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
token=${GH_TOKEN:-${GITHUB_TOKEN:-}}
if [[ -z $token ]] && command -v gh >/dev/null; then token=$(gh auth token --hostname github.com 2>/dev/null) || token=; fi
if [[ -z $token ]]; then
  token=$(printf 'protocol=https\nhost=github.com\n\n' | GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null | sed -n 's/^password=//p') || token=
fi
test -n "$token" || { echo '需要本站授权的 GitHub 下载与回执身份'; exit 2; }
export GH_TOKEN=$token
mkdir -p "$GLM_WORK_ROOT"
job=$(mktemp -d "$GLM_WORK_ROOT/formal.XXXXXXXX")
echo "本次任务 $job"
mkdir "$job/repo"
cd "$job"
blob=9d0147c47f2457af4a3790ed9de14ee3bccd7403
printf 'Authorization: Bearer %s\n' "$GH_TOKEN" |
  curl -fsS --connect-timeout 10 --max-time 90 --retry 3 -H @- \
    -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/ehdbdns/gcmoe-sm90/git/blobs/$blob" -o bundle.json
python3 - "$blob" <<'PY'
import base64, hashlib, json, pathlib, sys
obj = json.loads(pathlib.Path("bundle.json").read_text())
if obj["encoding"] != "base64" or obj["sha"] != sys.argv[1]:
    raise SystemExit("GitHub blob mismatch")
payload = base64.b64decode(obj["content"])
if hashlib.sha256(payload).hexdigest() != "125ab3d8e708954f0ffc0391421c0b90e28a338439a2a15594d8714ea0aba8fd":
    raise SystemExit("executor archive SHA256 mismatch")
pathlib.Path("executor.tar.gz").write_bytes(payload)
PY
tar xzf executor.tar.gz -C repo
echo 2fd9fba8f37604bfba24ff6e6e486f5078af31f4 > repo/SOURCE_REVISION
bash repo/scripts/sm100/glm53_execute.sh "$job" --refit4 "$old"
)
~~~

## 4. 回执

1. `main.csv` 有三行，`bs=32`，`global_streams=256`，额外槽数依次为 0、4、4，`full_native_match=True`。列出 MoE µs/层、相对原生和同轮 RR 的有符号变化、token IR、总 M 块、峰值 rank M 块、活跃槽、峰值 rank 活跃槽、M 块 IR，以及 ECO routed MAE、拍卖报价／成交和零成交调用数。
2. 提交 `historical-price/` 和 `new-price/`。各自有三份逐步逐层逐 rank 的 T、M、A、routed_ns 压缩数组及 `scores.json`。其中列出首次 shared MMA 相对 dispatch 的有符号偏移、首个角色至首次 shared MMA 的时间，ECO 另有 shared L1、dispatch、plan 的交集。历史与新一轮同时改变了副本数、启动优先级及 ECO 价格，完整 MoE 变化按组合效果解释。相同已执行分配上的两套价格 MAE 可以直接对比。
3. 保留每臂测量第 2 步 L8、L20、L32 的阶段 Perfetto、absolute Perfetto 和逐 rank 实收表，以及放置、运行版本、GPU 空闲记录和原始数据清单。执行器自动发布到 `docs/sm100_megamoe/glm53_formal_results/formal.<本次编号>/`。只有发布失败时，在本次 job 下补传 `python3 repo/scripts/sm100/glm53_publish.py "receipt/$(basename "$PWD")"`。
4. 执行者只处理保持语义的编译兼容问题；如需修复，保存首错、diff 和版本，确保同组结果来自同一实现。遇到资源故障或一致性失败，保留首个错误与已完成结果，释放本任务的占用并回执。发布后确认本任务工作进程已退出，清理本次私有临时构建文件，保留可核查的原始实验和归档。
