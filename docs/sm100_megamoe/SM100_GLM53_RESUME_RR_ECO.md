# SM100 · GLM-5.3 BS256 续跑 RR 和 ECO

原生 GC 的 1000 步统计和 1000 步测量已完成。原件在节点 `e01-cn-il24rbk1805` 的 `/data0/home/admin/workspace/glm53-work/formal.20261128200535/results`。本轮复用该臂的输入、两套 KV、固定路由和测量结果，启动两条剩余臂。[上轮回执](glm53_formal_results/formal.20261128200535/receipt.json)

RR 在模型加载期间转换 routed 权重时申请 480 MiB 失败。GPU 7 只剩 40.38 MiB，另一个 PID 占用 197.88 GiB。GPU 4 的另一个 PID 占用 198.31 GiB。回执没有标明这些 PID 的归属。续跑先使用之前已经生成的每 rank 8 个冗余槽放置，并用相同的上游转换函数每批打包 4 个槽。[原始异常](glm53_formal_results/formal.20261128200535/failure-tail.txt)

## 执行条件

1. 由调度器或节点管理员确认同一组八卡的整组预约，覆盖两条剩余臂。在共享节点只使用获准的八张卡。卡忙或预约冲突时在 CPU 侧等待。其他 PID 的进程与文件由其所有者处理。
2. 保留已有的 `CUDA_VISIBLE_DEVICES`。若在有更多 GPU 的宿主上直接登录，用 `SM100_GPU_UUIDS` 指定已获准的八张卡。启动器按第一臂记录的 UUID 恢复 rank 顺序，并对每臂在协作锁内间隔 5 秒检查两次进程、设备句柄、利用率和显存。每张卡须没有计算进程或占用句柄、利用率为零、已用显存至多 1024 MiB。若上轮 PID 在另一个命名空间，显存检查仍会阻止启动。
3. 使用已跑通原生臂的账号或有读取原件权限的账号。默认复用该节点已验证的模型、vLLM 环境和 DSL。模型为 `/mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0`。原件搬迁时，设置 `GLM_REFERENCE_RESULTS` 为含 `vanilla/result.json` 的实际 results 目录。启动器在新建目录工作。私有 KV 以链接复用，无需再写出原生臂的快照。

## 一次启动

在持久 Bash 会话执行整段。使用现有 GitHub 登录凭据。命令从 GitHub API 取固定版本的执行包，核对 SHA256 后以 `--resume` 启动。执行过程中保持本站八卡预约有效。

~~~bash
(
set -euo pipefail
set +x
umask 077
[[ $(hostname -s) == e01-cn-il24rbk1805 ]]
seed=${GLM_REFERENCE_RESULTS:-/data0/home/admin/workspace/glm53-work/formal.20261128200535/results}
test -s "$seed/vanilla/result.json"
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
blob=27bfc0843aec89423f055850b9df7bd3aaac2203
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
if hashlib.sha256(payload).hexdigest() != "cf6a0f4104063c144400a5c4161b888809603617978b1f575487ae595f0afed7":
    raise SystemExit("wrong bundle SHA256")
pathlib.Path("executor.tar.gz").write_bytes(payload)
PY
tar xzf executor.tar.gz -C repo
printf '%s\n' 9e242246bde3200c3956f0fcf59d4a6879832308 > repo/SOURCE_REVISION
bash repo/scripts/sm100/glm53_execute.sh "$job" --resume "$seed"
)
~~~

## 运行与回执

- 正常顺序为 RR → ECO → 三行主表。两臂均为 BS256、TP8/EP8、每 rank 32 个 decode token、1000+1000 步、75 个 routed 层、每卡 48 GiB KV、trace 2。恢复到原 CUDA Graph 地址后重新读回 KV 与原生 SHA256 核对。执行包核验输入、八卡 rank 顺序、原生 rank 文件、实际反馈、放置和逻辑 expert token 总量。原生数据使用旧测量 SHA，RR 和 ECO 使用本次源码 SHA，均列在 `resume.json` 中。
- 进度见 `$job/results/run.log`。入场记录在 `gpu-admission.csv`，分臂日志名含冗余度和尝试编号。碰到其他 PID 占用导致 OOM 时，释放本任务的卡锁，待整组空闲后以相同设置重试当前臂一次。若再出现竞争，回执注明，由站点重新安排独占窗口。
- 只有在通过空闲检查、报错未显示其他大显存 PID、退出后八卡再次空闲且出现 CUDA OOM 时，脚本才切换到每卡 4 个冗余槽。它用原生臂留下的同一份 1000 步统计重新计算逐层 EPLB 放置，按新放置完成 RR 和 ECO 两臂。此前的 8 槽尝试保留在本次私有目录。4 槽仍 OOM 时交回首错和日志。
- 最终应有三行 `main.csv`、实际 `extra_slots_per_rank`、控制项与输出一致性结果、replay 2 的第 8、20、32 层 Perfetto。原始数据继续归档在原目录。脚本在八卡释放后把新回执发布到 `docs/sm100_megamoe/glm53_formal_results/formal.<本次编号>/`。若仅发布失败，用同一 GitHub 身份执行 `python3 "$job/repo/scripts/sm100/glm53_publish.py" "$job/receipt/$(basename "$job")"` 即可补传。
- 执行者只处理可保持语义的编译兼容问题并记录 diff。保留原件、失败证据与测量数据。结束或失败即释放本任务的 GPU 和预约。归档核验成功后只清理本次私有的临时缓存和下载包。
