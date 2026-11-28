# SM100 · GLM-5.3 EP8 · shared 优先与三角色重叠

本轮只运行一次 BS256 的 EPLB+ECO。对照是已完成的 `formal.QLfEZmKt`，其全窗口 MoE 为 142.0892 µs。原生和 EPLB+RR 的分段分析直接读取旧数据。待验证的改动是 CUDA 子图先放入负责 shared 的 compute 节点，再放 dispatch、plan。三者都可独立启动，plan 从 dispatch 分区取一个 SM。CUDA 最终采用的设备顺序由时钟验证。

## 现场与配置

1. 在原节点 `e01-cn-il24rbk1805` 使用有权读取前几次实验的账号和持久 Bash 会话。沿用原来的八张卡及 rank 顺序，只在本站允许使用且八张全部空闲时开始。已有作业或预约时等待。保留本站分配的设备掩码。启动器会复核 UUID，并连续两次检查进程、设备句柄、利用率和显存。
2. 复用原生统计、同一份 8 槽 EPLB 放置和两段输入、KV、路由、反馈。TP8/EP8，每 rank 32 输入，统计 1000 步，再测 1000 步 × 75 层，FULL_DECODE_ONLY CUDA Graph、trace 2、原权重加载优化及 48 GiB KV/rank。运行器自己启用 `GC_MOE_SM100_SHARED_FIRST=1` 并检查每个 rank 的实际图节点顺序。原件目录如曾迁移，可指定 `GLM_RR_RESULTS` 和 `GLM_LAUNCH_REFERENCE` 指向对应的完整 results；脚本会核对已发布的原件清单。
3. 执行以下整段命令。已完成的两个对照臂仍由同一报告脚本汇总。CPU 分析、打包和发布在释放 GPU 后进行。

~~~bash
(
set -euo pipefail
set +x
umask 077
[[ $(hostname -s) == e01-cn-il24rbk1805 ]]
seed=${GLM_RR_RESULTS:-/data0/home/admin/workspace/glm53-work/formal.VHnr9l8N/results}
test -s "$seed/rr/result.json"
export GLM_LAUNCH_REFERENCE=${GLM_LAUNCH_REFERENCE:-/data0/home/admin/workspace/glm53-work/formal.QLfEZmKt/results}
test -s "$GLM_LAUNCH_REFERENCE/eco/result.json"
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
blob=3e8797f2496dcb2cea4249f53bea67041ee7e0c0
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
if hashlib.sha256(payload).hexdigest() != "88683210c3ad9d1688cc803a86c5a97bae2123d2e350297db1eeb01d5969a91b":
    raise SystemExit("wrong bundle SHA256")
pathlib.Path("executor.tar.gz").write_bytes(payload)
PY
tar xzf executor.tar.gz -C repo
printf '%s\n' 91b437e66ec4f6b39d69631446971f3594899247 > repo/SOURCE_REVISION
bash repo/scripts/sm100/glm53_execute.sh "$job" --eco-only "$seed"
)
~~~

## 验收

- `resume-attempts.txt` 应只有一条 `arm=eco extra=8`。保留两段输入/路由、KV、top-1、每步每层物理槽实收、拍卖和 price 校验结果。归档 `main.csv`、`launch-comparison.json` 及第 2 步 L8、20、32 的概要、absolute Perfetto 和逐 rank 表。执行者只处理保持实验语义的编译兼容，记录首错与 diff。遇 OOM 或运行故障收集日志并退卡，保持原配置。
- `launch-comparison.json` 给出按每步每层最慢 rank 选取的 graph 到第一条 routed MMA、routed（含 push）、至 combine 结束三段，全窗均值之和等于主表 MoE。用同一文件的 native 与 rr 行定位 RR 的额外耗时。
- 对比旧、新 ECO 的首个角色至 shared 首条 MMA、compute 入口至 shared 首条 MMA、shared 首条 MMA 相对 dispatch 的有符号偏移，同时核对 shared L1、dispatch、plan 三个阶段跨度的交集以及完整 MoE。三者交集是时间线重合长度。若偏移改善仅因 dispatch 推迟，或完整 MoE 变慢，照实保留。`unmatched_ranks` 为空才可作固定工作量对照。
- 第 2 步三个采样层在 `replay2_by_layer` 列出已按 Kineto 双锚点对齐的逐 rank 时钟和锚点半宽。先看上层 combine 收尾时 rank 是否已经错开，再看从上层 combine 结束到本层首个角色之间，哪些 rank 额外花了时间。这个区间包括层间算子、路由和记录操作。概要中的 `compute kernel entry` 是首个 compute CTA 到达入口的时刻，`shared expert L1` 始于首条实际 MMA，两者各保留一个标记。
- 释放站点预约，保留原件、KV、固定路由和本轮私有结果。启动器将公开回执自动归档到 `docs/sm100_megamoe/glm53_formal_results/formal.<本次编号>/`。若只有发布失败，从本次 job 目录运行 `python3 repo/scripts/sm100/glm53_publish.py "receipt/$(basename "$PWD")"` 补传，无需再次使用 GPU。
