# SM100 · GLM-5.3 EP8 · 每卡 32 流 · RR 1 槽与 ECO 8、12 槽

本任务用于原八卡节点的正式整模 decode。GLM-5.3 FP4 MoE / FP8 UE8M0，源端每 rank 32 个不同请求，八个 rank 共 256 个不同流，top-8，每步每层实收 2048 行。每臂 1000 个满批步统计，再测 1000 个满批步，解码层 3–77 共 75 个 MoE 层。四个臂各加载一次模型。执行版本为 `797eb29cb28aa7c4af078542c57e64b10ace2697`。

| 顺序 | 结果名 | 放置与分发 | 每卡额外槽 | 每卡物理槽 | 全局物理槽 |
|---:|---|---|---:|---:|---:|
| 1 | vanilla | 原生 GC，生成同组参考 | 0 | 32 | 256 |
| 2 | rr | 逐层 DeepInfer EPLB + RR | 1 | 33 | 264 |
| 3 | eco8 | 逐层 DeepInfer EPLB + ECO | 8 | 40 | 320 |
| 4 | eco12 | 逐层 DeepInfer EPLB + ECO | 12 | 44 | 352 |

每档的放置由本组原生臂的同一份逐步逐层统计独立生成，归档为 `placement-extra1.npy`、`placement-extra8.npy`、`placement-extra12.npy`。权重跟随各自放置搬迁，ECO 允许 rank 内做 single-copy-first 排序。升高预算后，每个逻辑专家获得的副本数由放置算法决定。

两条 ECO 均沿用前一份四槽任务书登记的线性价格 **T + 66M + 74A**，M16，slack 156，冷启动最多 6 轮、暖启动 1 轮、双跳 0。使用 TP8/EP8、48 GiB KV/rank、FULL_DECODE_ONLY CUDA Graph、trace 2、按 EP 筛选权重、本地 mmap 与权重 readahead。四臂都启用 shared 优先提交，shared 在 compute 角色中执行；dispatch 与 plan 可重叠。compute 保留 136 SM，通信分区有 12 SM，ECO 分出其中 1 个给 plan。原生和 RR 节点顺序为 compute → dispatch → combine，ECO 为 compute → dispatch → plan → combine。

## 从原执行节点启动

使用站点授权的 `e01-cn-il24rbk1805` 和原来八卡的 UUID 顺序。确认调度器分配及旧任务状态，等待在途作业释放，八卡全部空闲后再开始。执行器逐臂加协作锁并连续两次核验 GPU 进程、利用率、句柄与显存。新建独立 job，保留旧任务和失败现场。原历史数据默认位于 `/data0/home/admin/workspace/glm53-work/formal.QLfEZmKt/results`。历史原件迁移后可用 `GLM_PRICE_REFERENCE` 指定。将以下整段复制到已授权账号的 bash 运行。

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
blob=7d505b4fe598d25f8e9c8f451b52653b79e64594
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
if hashlib.sha256(payload).hexdigest() != "fcdb8534a5f47523358554808e1141cadc46a8f911a4195eca6c106e6d8eee1d":
    raise SystemExit("executor archive SHA256 mismatch")
pathlib.Path("executor.tar.gz").write_bytes(payload)
PY
tar xzf executor.tar.gz -C repo
echo 797eb29cb28aa7c4af078542c57e64b10ace2697 > repo/SOURCE_REVISION
bash repo/scripts/sm100/glm53_execute.sh "$job" --slot-budgets "$old"
)
~~~

## 交付

1. `main.csv` 共四行，`bs=32`、`global_streams=256`，额外槽依次为 0、1、8、12。交付四臂的 MoE µs/层、相对原生、相对 RR1、有符号变化、token IR、总 M 块、峰值 rank M 块、活跃槽和 M 块 IR。ECO 两行各自报告 routed MAE、报价、成交、零成交。另列同轮 ECO12 对 ECO8 的 MoE 变化，按 100 × (ECO12/ECO8 − 1) 计算。RR1 与 ECO8/12 的差异同时包含放置容量和分发策略。
2. 每个 arm 发布 `setup.json`、`result.json` 和测量第 2 步 L8、L20、L32 的阶段 Perfetto、absolute Perfetto、逐 rank 槽计数。保留 `historical-price/`、含四个本轮逐 rank NPZ 的 `new-price/`、三份放置、GPU 空闲记录、原始文件清单。两窗口逐步核对每源 32 个请求的归属，完整图 replay、路由、反馈、原生 top-1、KV 边界、逻辑专家行数，ECO 逐槽校验 quota。四个 `full_native_match` 全为 True 后发布比较结论。
3. 执行者仅修复可保持语义的编译问题，附首错日志、diff 和 SHA，同组数据来自同一实现。空闲卡 OOM 时保留已完成臂，报告失败的 BS、臂、阶段和显存。各臂沿用指定 BS、槽数、窗口和 KV。程序自动归档至 `docs/sm100_megamoe/glm53_formal_results/formal.<本次编号>/`。如果发布阶段失败，从原 job 重试 `python3 repo/scripts/sm100/glm53_publish.py "receipt/$(basename "$PWD")"`。结束时释放 GPU 和本作业私有锁，确认进程退出，保留可核查的原始数据，清理本作业私有的可再生临时文件。
