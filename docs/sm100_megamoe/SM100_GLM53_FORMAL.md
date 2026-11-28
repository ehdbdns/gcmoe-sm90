# SM100 新执行者任务书 · GLM-5.3 EP8 三臂

你接手的是节点 `e01-cn-il24rbk1805` 上的正式实验。上次回执 `516235d` 为 **0/3**。旧执行者在下载代码时退出，三臂均未启动。模型和权重已在这台机器上校验过。你只需要这份任务书、该节点的 Bash 登录权限、指定八卡的使用授权以及可向本仓库提交回执的 GitHub 身份。可以从空目录和新的 Linux 账号开始。

## 本次交付

| 项目 | 固定设定 |
|---|---|
| 节点与模型 | NVIDIA L20D 节点，实际模型目录为 `/mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0` |
| 三臂顺序 | `vanilla` 原生 GC → `rr` 逐层 EPLB + RR → `eco` 同一份 EPLB 放置 + Natureal ECO |
| 工作量 | TP8/EP8，BS256，每 rank 32 个 decode token，75 个 routed 层；每臂前 1000 步统计、后 1000 步测量 |
| 共同配置 | GC 上游底座 `af5a7a75`，模型原始 FP4 routed / FP8 shared 权重，CUDA Graph，`GC_MOE_SM100_TRACE=2`；原生 32 槽/rank，两条 EPLB 臂 40 槽/rank，ECO plan 从 dispatch 分区取 1 SM |
| 采样 | 后半段第 2 个 replay，实际 decoder 第 8、20、32 层，每层八个 rank 的 token、M 块、概要及 absolute Perfetto；保留对应 torch.profiler 原件 |
| 返回位置 | `main` 分支的 `docs/sm100_megamoe/glm53_formal_results/formal.<本次编号>/`；完成后回复 GitHub 返回的提交地址、三臂状态和首个错误（如有） |

## 登录后先看现场

在获准使用的这台节点上运行以下只读命令。确认自己的调度分配或预约覆盖同一组八卡。即使卡的利用率为零，也要尊重他人预约。调度器已设置的 `CUDA_VISIBLE_DEVICES` 原样保留。无调度器且节点仅有这八张卡时，在确认可使用整组后执行。若能看到更多卡，先把获准的八个完整 GPU UUID 以逗号分隔写入 `SM100_GPU_UUIDS`。

~~~bash
hostname -s
id -un
nvidia-smi -L
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
pgrep -af '^([^ ]*/)?bash[[:space:]]+([^ ]*/)?(glm53_execute|run_glm53_three)[.]sh([[:space:]]|$)' || true
ls -d /data0/home/admin/workspace/glm53-work/formal.*/results/{main.csv,*/result.json} 2>/dev/null || true
~~~

回执中的旧目录 `formal.jQxs5nOk` 是空的。若上面出现新进程，先看该进程所属任务的 `results/run.log`，让已启动的任务继续。如果发现已有正式臂的 `result.json`，先按末节交回已有记录。仅有空目录时直接继续。不要终止其他人的进程。

## 从空目录启动

1. 在有持久终端的 Bash 中执行下面两段，可以用节点已有的 tmux 会话，或先运行 `tmux new -s glm53-ep8`。断开连接后，用 `tmux attach -t glm53-ep8` 回到原任务。代码从现场已验证可用的 GitHub Contents API 下载。执行包约 0.4 MiB，含 GC 上游源、补丁、运行及回传脚本，下载一次即齐备。
2. 默认私有盘为 `/data0/home/当前用户名/workspace/glm53-work`。新账号的目录位置若不同，先把 `GLM_WORK_ROOT` 设为自己可写且空间充足的私有卷。两份 KV 快照逻辑大小合计最多约 768 GiB，实际磁盘占用取决于稀疏文件支持和非零数据量。先用 `df -h` 查看该卷。
3. 默认复用上轮的 vLLM 0.31.0、torch 2.13.0 / CUDA 13.0 环境。该环境对新账号不可访问时，脚本先找自己的同版环境，再使用 `python3.12 -m venv` 安装到本次目录。已有同版私有环境可通过 `GLM_RUNTIME_PYTHON` 指向其 `bin/python`。现场确认的 CUDA 工具在 `/usr/local/cuda/bin`。Python 镜像优先沿用本站配置，未设置时命令使用上轮验证过的清华镜像。

第一段定义本次 GitHub 登录方式。可使用现场已有的 `GH_TOKEN`、`GITHUB_TOKEN`、GitHub CLI 登录或 Git credential helper。所在账号需要本仓库 Contents 读写权限。新账号若使用 GitHub CLI，可先运行 `gh auth login --hostname github.com --web`。凭据只在本机使用。

~~~bash
glm53_auth() {
  set +x
  local token=${GH_TOKEN:-${GITHUB_TOKEN:-}}
  if [[ -z $token ]] && command -v gh >/dev/null; then
    token=$(gh auth token --hostname github.com 2>/dev/null) || token=
  fi
  if [[ -z $token ]]; then
    token=$(printf 'protocol=https\nhost=github.com\npath=ehdbdns/gcmoe-sm90.git\n\n' |
      GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null | sed -n 's/^password=//p') || token=
  fi
  [[ -n $token ]] || { echo '请在此账号登录获准的 GitHub 身份，或注入 GH_TOKEN 后重试。' >&2; return 1; }
  export GH_TOKEN=$token
}
~~~

第二段完成下载、校验、执行和回传。保留终端打印的私有目录，后续查看日志和补传都用它。

~~~bash
(
set -euo pipefail
set +x
umask 077
glm53_auth
[[ $(hostname -s) == e01-cn-il24rbk1805 ]] || { echo '请登录任务指定的节点'; exit 2; }
export GLM_WORK_ROOT="${GLM_WORK_ROOT:-/data0/home/$(id -un)/workspace/glm53-work}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
mkdir -p "$GLM_WORK_ROOT"
df -h "$GLM_WORK_ROOT"
job=$(mktemp -d "$GLM_WORK_ROOT/formal.XXXXXXXX")
echo "本次私有目录 $job"
mkdir "$job/repo"
cd "$job"
stage=download
trap 'rc=$?; if ((rc)); then echo "stage=$stage exit=$rc job=$job" | tee "$job/bootstrap-status.txt"; fi' EXIT
api=https://api.github.com/repos/ehdbdns/gcmoe-sm90/contents/docs/sm100_megamoe/glm53_executor.tar.gz
printf 'Authorization: Bearer %s\n' "$GH_TOKEN" |
  curl -fsS --connect-timeout 10 --max-time 90 --retry 3 -H @- \
    -H 'Accept: application/vnd.github+json' \
    "$api?ref=444969ae13dd438c12ef571fc0cb9708f0d2cb38" -o bundle.json
stage=verify
python3 - <<'PY'
import base64, hashlib, json, pathlib
obj = json.loads(pathlib.Path("bundle.json").read_text())
if obj.get("encoding") != "base64" or obj.get("sha") != "8e08c43d45044fef2034c0f75d2aa3a4e198fb2b":
    raise SystemExit("incorrect Contents API response")
payload = base64.b64decode(obj["content"])
if hashlib.sha256(payload).hexdigest() != "43d9ff691e59ba28900d61b3cbe9702946312d41eaaa8786a252311cf0eba33b":
    raise SystemExit("incorrect execution bundle SHA256")
pathlib.Path("executor.tar.gz").write_bytes(payload)
PY
tar xzf executor.tar.gz -C repo
printf '%s\n' 0daa8328d6d53211869b0de5a683054ca2be2c4a > repo/SOURCE_REVISION
stage=execution-or-return
bash repo/scripts/sm100/glm53_execute.sh "$job"
)
~~~

## 启动后如何判断进度

- 在另一个终端用 `tail -F <本次私有目录>/results/run.log`。准备阶段检查 checkpoint 摘要、准备 DSL 4.8.1 并预读分片。正式脚本在每臂开始前取共享锁，检查八卡进程、利用率和设备占用。任一卡忙则释放锁，每 60 秒复查。每臂退出即释放八卡，EPLB 放置在 CPU 上计算。
- 节点的 `nvidia-smi` 报 8.9，CUDA Driver 和 torch 对相同 UUID 均报 10.3。**裁定由设备执行结果决定。** 首臂取得八张空闲卡后，脚本在每张卡执行只含 `sm_103f` 原生机器码的探针，要求八卡逐一返回预期值。通过后立即加载模型，证据写在 `arch-proof.json`。若有设备失败，交回具体 UUID、CUDA 错误和已通过的设备；无需另起 GPU 作业判卡。
- 权重加载优化已启用。每臂对 282 个 HF 分片申请页缓存预读，vLLM 采用 safetensors 延迟 mmap 和 EP 本地专家过滤；GC 复用分片句柄。重排时从已加载的 packed 权重做设备间搬运。每臂 `result.json` 记录各 rank 读取的专家 ID、vLLM 与 GC 加载时间。
- 你负责执行以及留下原始证据。只允许处理可确认不改变语义的编译兼容问题，附原始错误、最小补丁及其使用的臂；涉及公共代码的补丁需整组保持一致。遇到 OOM、路由或数值差异、运行时错误，保留第一个失败，交回构建者。保持 BS、统计及测量步数、放置、量化、trace 和采样层配置。

## 验收和补传

成功回执的 `receipt.json` 有三个 `complete_arms`、退出码 0、源码版本和原始文件清单。`main.csv` 有三行，涵盖 MoE 微秒数、token IR、总/峰值 M 块、总/峰值活跃槽、M 块 IR、相对原生和 RR 的变化；ECO 另有 routed MAE、报价数、成交数和接受率。逐步逐层验证逻辑 dispatch 总数，各臂核对 token、请求顺序、KV 边界、gate 和原生 top1。仅在严格匹配时填写配对百分比。

每臂的 `perfetto/` 中，`step0002-layer{08,20,32}.perfetto.json` 看阶段重叠，`.absolute.perfetto.json` 看 CTA，`.ranks.json` 看逐 rank token 和 M 块。完整 Kineto、原始 clocks/counts、KV 与反馈留在回执注明的私有目录。失败回执保留已完成臂和 `failure-tail.txt`。等待 API 回传并核实终端打印的 GitHub 地址后，可清理自己本轮的 wheels 和编译缓存；保留可复用环境及复核材料。

如果只在回传时断网，使用第一段登录函数和下面的命令补传同一份回执。将第一行改成启动时打印的目录；补传会校验 GitHub 上的整棵结果目录，可以重复执行。若发现旧启动器已完成部分臂但尚未生成回执，在新执行包目录用 `python3 repo/scripts/sm100/glm53_receipt.py 旧目录/results 私有目录/receipt/formal.原编号 1` 打包（已有完整三臂及主表时用 0），再通过同一发布脚本上传。若在执行包下载前即失败，在交接消息附节点、API HTTP 状态与 `bootstrap-status.txt`；仅重试下载。所有回执都不含权重或登录凭据。

~~~bash
job='/替换为本次私有目录/formal.XXXXXXXX'
( glm53_auth && python3 "$job/repo/scripts/sm100/glm53_publish.py" "$job/receipt/$(basename "$job")" )
~~~
