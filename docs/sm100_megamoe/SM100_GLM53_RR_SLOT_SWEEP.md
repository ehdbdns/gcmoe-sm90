# SM100 · GLM-5.3 · EPLB 冗余度与 token IR

在保存 `formal.QLfEZmKt/results` 的执行节点做一次 CPU 离线扫描。GLM-5.3 EP8，每个源 rank 有 32 个独立流，全局 256 个请求，top-8，每次层调用有 2048 行路由，M=16。可以与另一个正式任务并行，限制为四个 CPU 线程，保持 GPU 隐藏。无需起模型、读权重或 KV、分配 GPU、重新生成 token。

## 数据和评价

- 每个 rank 的 EPLB 额外物理槽从 0 到 32 逐一取整。每档按原生臂统计窗口的 1000 步，针对 75 层分别运行原版 DeepInfer 复制分配和 rank 放置。0 槽同样重新放置专家，原生连续放置单独报告。
- 评分窗口使用后 1000 步。八个独立源 rank 各自读取原始 gate ID，按原实现 `(23333 * local_row + source_rank) % copies` 分派到该专家的物理槽。逐次计算最重 rank 的路由行数除以 256，最后平均 75,000 次，另报总 M 块和单 rank 峰值 M 块。每个物理槽独立按 16 向上取整。
- 启动时核对 24 个计数文件的历史 SHA-256。逐层核对原生逻辑专家计数、8 槽的已安装放置以及完整实收物理槽计数，共 2400 万个 8 槽计数。汇总必须复现正式原生和 RR 主表。
- `min_ir_extra` 是实扫 33 档的最低 IR 档。`within_one_row_extra` 是与最低值相差至多 1 条峰值路由行的最小档。另报 `sustained_one_row_extra`，要求从该档到 32 槽的所有结果之间，平均峰值相差至多 1 行，至少覆盖三档。1 行对应 IR 绝对差 1/256。曲线仍在变化时输出 null。

## 执行和回执

在节点上直接执行整段，使用原实验可读的私有目录与已有 GitHub 回执身份。命令只下载固定版本的扫描器、放置函数及公开参照数据。成功后发布一个结果文件。若原件已迁移，可用 `GLM_PRICE_REFERENCE` 指向同一历史回执的 results 目录；哈希和逐槽检查均保留。发生错误时回传 job 路径和首错，勿重启任何 GPU 任务。

~~~bash
(
set -euo pipefail
set +x
umask 077
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
base=${GLM_WORK_ROOT:-/data0/home/admin/workspace/glm53-work}
old=${GLM_PRICE_REFERENCE:-$base/formal.QLfEZmKt/results}
test -s "$old/measure-reference-7.pt"
job=$(mktemp -d "$base/rr-slots.XXXXXXXX")
cd "$job"
echo "CPU 扫描目录 $job"
py=''
for candidate in "${GLM_RUNTIME_PYTHON:-$base/glm53-runtime-qgmxfwva/venv/bin/python}" "$base"/formal.*/results/base/bin/python "$base"/formal.*/results/dsl/bin/python python3; do
  if "$candidate" -c 'import torch,numpy,hashlib; assert torch.__version__.split("+")[0]=="2.13.0" and hasattr(hashlib,"file_digest")' >/dev/null 2>&1; then py=$candidate; break; fi
done
test -n "$py" || { echo '通过 GLM_RUNTIME_PYTHON 指定正式实验使用的 Torch 2.13.0 Python'; exit 1; }
token=${GH_TOKEN:-${GITHUB_TOKEN:-}}
if [[ -z $token ]] && command -v gh >/dev/null; then token=$(gh auth token --hostname github.com 2>/dev/null) || token=; fi
if [[ -z $token ]]; then token=$(printf 'protocol=https\nhost=github.com\n\n' | GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null | sed -n 's/^password=//p') || token=; fi
test -n "$token"
export GH_TOKEN=$token
export SWEEP_REV=e66b91605d09008c9871259606ad0884d25578d1
python3 - <<'PY'
import base64, hashlib, json, os
from pathlib import Path
from urllib.request import Request, urlopen
revision = os.environ["SWEEP_REV"]
base = "https://api.github.com/repos/ehdbdns/gcmoe-sm90/contents/"
archive = "docs/sm100_megamoe/glm53_formal_results/formal.QLfEZmKt/"
paths = ["scripts/sm100/glm53_rr_sweep.py", "source/gc_moe/eplb/deepinfer_periodic/policy.py",
         "source/gc_moe/eco/meta_sm90.py", *(archive + name for name in ("receipt.json", "main.csv", "placement.npy"))]
for path in paths:
    req = Request(base + path + "?ref=" + revision, headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"]})
    with urlopen(req, timeout=90) as stream:
        obj = json.load(stream)
    data = base64.b64decode(obj["content"])
    assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == obj["sha"]
    dest = Path(path)
    dest.parent.mkdir(exist_ok=True, parents=True)
    dest.write_bytes(data)
PY
env -u GH_TOKEN -u GITHUB_TOKEN nice -n 10 "$py" scripts/sm100/glm53_rr_sweep.py "$old" summary.json 2>&1 | tee scan.log
python3 - <<'PY'
import base64, json, os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
path = "docs/sm100_megamoe/glm53_rr_slot_sweep/summary.json"
url = "https://api.github.com/repos/ehdbdns/gcmoe-sm90/contents/" + path
data = Path("summary.json").read_bytes()
headers = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Content-Type": "application/json"}
def get_result():
    with urlopen(Request(url + "?ref=main", headers=headers), timeout=60) as f:
        return base64.b64decode(json.load(f)["content"])
try:
    prior = get_result()
except HTTPError as exc:
    if exc.code != 404:
        raise
    body = dict(message="[SM100] GLM EP8 historical RR token IR by slot budget", branch="main",
                content=base64.b64encode(data).decode())
    with urlopen(Request(url, method="PUT", data=json.dumps(body).encode(), headers=headers), timeout=90) as f:
        json.load(f)
    prior = get_result()
if prior != data:
    raise SystemExit("main 上已有不同回执，请保留两份原件并汇报")
print("https://github.com/ehdbdns/gcmoe-sm90/blob/main/" + path)
PY
)
~~~
