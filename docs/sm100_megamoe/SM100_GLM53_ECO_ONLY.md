# SM100 · GLM-5.3 BS256 · 只执行第三臂 EPLB + ECO

本轮只启动一次 ECO 模型进程组。前两臂已完成。原生结果来自 `formal.20261128200535`，8 槽 EPLB + RR 来自 `formal.VHnr9l8N`。使用同一份 75 层 EPLB 放置，每 rank 8 个冗余槽，TP8/EP8，BS256，每 rank 32 个输入，1000 步统计和 1000 步测量，48 GiB KV/rank，FULL_DECODE_ONLY CUDA Graph，trace 2。模型、路由、输入、反馈、KV、原生输出检查和权重加载优化沿用前两臂。

## 续跑内容

上轮第三臂返回 143，公开尾日志记录了 CUDA launch failure。检查调度器发现初始放置中 rank 0、7 的单副本槽均为零，旧计算调度器可能抢先领取 L2，然后等待尚未发出的 L1。本版本在单副本块数不足以启动第一波 L2 时，先读取该 rank 的完整配额，再决定首批 L1 工作量。已在构建机离线编译本次形状的 sm_103f 计算内核，是否消除现场故障由这次实测验证。

## 执行

1. 在原节点 `e01-cn-il24rbk1805` 使用能读取两次原件的实验账号。保留已有调度器掩码，在站点获准的同一组八张卡全部空闲时运行。各卡 UUID 和 rank 顺序必须与原生臂一致。有其他用户或预约则等待。启动器对进程、显存、设备句柄和利用率连续检查两次。若站点分配会到期，先取得覆盖本次 ECO 的预约。
2. 在持久 Bash 会话执行下面完整命令。保留原节点上的 Python 环境、本地已验证模型和原件。启动器在新目录校验前两臂原始文件和 KV 清单，复用其输入，离线生成两行 `reference-main.csv` 后只启动 ECO。`GLM_RR_RESULTS` 仅用于指定已搬迁的同一份 RR results。请勿运行旧版 RR → ECO 任务书。

~~~bash
(
set -euo pipefail
set +x
umask 077
[[ $(hostname -s) == e01-cn-il24rbk1805 ]]
seed=${GLM_RR_RESULTS:-/data0/home/admin/workspace/glm53-work/formal.VHnr9l8N/results}
test -s "$seed/rr/result.json"
export GLM_WORK_ROOT=${GLM_WORK_ROOT:-/data0/home/$(id -un)/workspace/glm53-work}
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
token=${GH_TOKEN:-${GITHUB_TOKEN:-}}
if [[ -z $token ]] && command -v gh >/dev/null; then token=$(gh auth token --hostname github.com 2>/dev/null) || token=; fi
if [[ -z $token ]]; then
  token=$(printf 'protocol=https\nhost=github.com\n\n' | GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null | sed -n 's/^password=//p') || token=
fi
[[ -n $token ]] || { echo '需要本站已获准的 GitHub 身份以下载和回传'; exit 2; }
export GH_TOKEN=$token
mkdir -p "$GLM_WORK_ROOT"
job=$(mktemp -d "$GLM_WORK_ROOT/formal.XXXXXXXX")
echo "本次续跑目录 $job"
mkdir "$job/repo"
cd "$job"
blob=82f7256724fff315613df6a87de6ad73ee417119
printf 'Authorization: Bearer %s\n' "$GH_TOKEN" |
  curl -fsS --connect-timeout 10 --max-time 90 --retry 3 -H @- \
    -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/ehdbdns/gcmoe-sm90/git/blobs/$blob" -o bundle.json
python3 - "$blob" <<'PY'
import base64, hashlib, json, pathlib, sys
obj = json.loads(pathlib.Path("bundle.json").read_text())
if obj.get("encoding") != "base64" or obj.get("sha") != sys.argv[1]:
    raise SystemExit("wrong GitHub blob")
payload = base64.b64decode(obj["content"])
if hashlib.sha256(payload).hexdigest() != "b3f00ea9f321caa144189b9febed596e1b6eba71adce374f14548f74b5334aeb":
    raise SystemExit("wrong bundle SHA256")
pathlib.Path("executor.tar.gz").write_bytes(payload)
PY
tar xzf executor.tar.gz -C repo
printf '%s\n' 4ac5e9d6720f42caef669001c6fb9ca8aa33b27f > repo/SOURCE_REVISION
bash repo/scripts/sm100/glm53_execute.sh "$job" --eco-only "$seed"
)
~~~

## 验收与交付

- `resume-attempts.txt` 本轮应只有一条 `arm=eco extra=8`。遇到 OOM、kernel 超时或其他失败即退卡，保持 8 槽放置和现场，只执行这一次。执行者只处理保持语义的编译兼容问题并记录首错和 diff。
- 成功后用前两臂原始 NPZ 和本轮 ECO 生成三行 `main.csv`。检查两段的路由、KV、请求顺序、原生 top-1、每步每层逻辑专家行数和 ECO 实收配额。报告实际测量 SHA、完整 MoE、token 和 M 块 IR、峰值与总 M 块、活跃槽、routed MAE、拍卖成交，以及公平性状态。保存第 2 步 L8、L20、L32 的概要及 absolute Perfetto 和逐 rank 数据。
- 八卡退出后立即释放本站预约。启动器随后在 CPU 上整理并推送 `docs/sm100_megamoe/glm53_formal_results/formal.<本次编号>/`。回执带上一轮 ECO 的最早错误行；如果本轮失败，也带上本轮错误行和退出尾日志。若错误行仍不能定位，请从本地独立的 ECO 日志摘录最早故障前后的 rank、阶段、调用栈，随回执补充，避免只传析构异常。
- 发布失败可从同一目录执行 `python3 "$job/repo/scripts/sm100/glm53_publish.py" "$job/receipt/$(basename "$job")"` 补传。确认远端归档后只清理本轮可重建的下载件和缓存。保留三个臂的原始结果、KV 和固定路由。
