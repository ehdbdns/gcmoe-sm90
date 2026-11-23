# SM100 · BS512 原生 GC 与 Mega 的输入公平性和数值定位

收到即执行。同组八张空闲卡，沿用 fb8b65f 的 GC 上游默认配置、V4.1 Flash、TP8/EP8、融合 shared、整步图和上一轮的 512 个请求。准备代码、wheel、空间检查和离线判读均在 CPU 上做。启动器核验八卡 UUID，持共享锁复查占用；有预约冲突或卡上有人则在组外等候。每次释放后再用卡都要重新确认。仅允许保持语义的编译修复，附 diff。准备阶段沿用本机已校验的 wheelhouse。

## 执行

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_FAIR_BOUNDARY.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-fair-boundary-XXXXXXXX)
echo "本轮结果目录 $area/run"
git worktree add --detach "$area/code" "$rev"
cd "$area/code"
fixture=docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla
if [[ $(git config --bool core.sparseCheckout || true) == true ]]; then
  git sparse-checkout add "$fixture" scripts/sm100 source/gc_moe integration docs/sm100_megamoe
fi
test -r "$fixture/rank0.npz"
base=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv
run="$area/run"
mkdir -p "$run/wheels"
git clone --branch eco https://github.com/Natureal/gc-moe.git "$run/upstream"
git -C "$run/upstream" checkout --detach fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736
"$base/bin/python" -m venv "$run/dsl-env"
py="$run/dsl-env/bin/python"
packages=(nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13)
wheel_args=()
if [[ -n ${SM100_DSL_WHEELS:-} ]]; then wheel_args=(--no-index --find-links "$SM100_DSL_WHEELS"); fi
"$py" -m pip download "${wheel_args[@]}" --no-deps -d "$run/wheels" "${packages[@]/%/==4.8.1}"
manifest="$PWD/docs/sm100_megamoe/v41_native_phase_results/dsl-wheelhouse.sha256"
(cd "$run/wheels" && sha256sum -c "$manifest")
"$py" -m pip install --no-index --find-links "$run/wheels" --no-deps "${packages[@]/%/==4.8.1}"
lib=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
base_lib=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf "import site; site.addsitedir('%s')\n" "$base_lib" > "$lib/zz_v41_common.pth"
cp docs/sm100_megamoe/v41_full_decode_pair_results/compat/sitecustomize.py "$lib/sitecustomize.py"
export TORCHINDUCTOR_CACHE_DIR="$run/inductor" TRITON_CACHE_DIR="$run/triton"
export VLLM_CACHE_ROOT="$run/vllm-cache"
# 共享文件系统上检查空间，八卡仍未预约。
"$py" - "$area" <<'PYSPACE'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < (290 << 30):
    raise SystemExit("短程 checkpoint 与快照需要至少 290 GiB，换本机足量存储后启动")
PYSPACE
bash scripts/sm100/run_five.sh "$run" fair-boundary-bs512 2>&1 | tee "$run/launcher.log"
# 此时两臂进程已退出，共享锁已释放。CPU 离线判读；退出码 3 表示拿到了分叉证据。
set +e
OMP_NUM_THREADS=2 "$py" scripts/sm100/fair_boundary_report.py "$run/bs512"
verdict=$?
set -e
if [[ $verdict == 0 ]]; then
  formal="$area/formal"
  mkdir -p "$formal"
  ln -s "$run/dsl-env" "$formal/dsl-env"
  git clone --local "$run/upstream" "$formal/upstream"
  git -C "$formal/upstream" checkout --detach fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736
  "$py" - "$formal" <<'PYSPACE'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < (550 << 30):
    raise SystemExit("正式复测需要 550 GiB 可用空间；移存已校验的短程 checkpoint 后继续")
PYSPACE
  bash scripts/sm100/run_five.sh "$formal" strict-native-bs512 2>&1 | tee "$formal/launcher.log"
elif [[ $verdict != 3 ]]; then
  exit "$verdict"
fi
```

## 回执与判读

1. 短程只加载两个模型进程。各自 prefill 后，在相同请求历史和物理块表下，把 Mega 的 KV 写回两个后端的实际地址并校验读回。保留 GC 恢复前摘要；整块摘要包含闲置区域，单凭它不同无法判定有效 token 的 KV 不同。各跑前三个原生 greedy decode，无路由或输出覆盖。每步 40 层、全八卡，保留 attention 输入/输出摘要、专家输入 hidden、top-k IDs、gate 和 BF16 输出。核对两臂步骤计数及第 3 层旧锚点。若分叉消失，表述为在共同 KV 干预下消失。若仍分叉，报告最早 step/layer、全部相关 rank 和阶段。
2. 同一进程在自然轨迹采完后，两个后端逐层重新计算同一份 Mega 原始 BF16 hidden、相同路由及 gate，核验量化后输入、routed/shared scale。Mega 自回放作为检查。输出与记录的 Mega 自然图逐元素比较，报告不相符数量和最大绝对差；检查返回槽之和能复原实际输出后，列有差异的具体 top-k 槽和 shared 槽。该重放在整图外执行；分别报告自然图、同输入重放和自回放的结果。限这三个样本。
3. fair_boundary.json 中 short_strict_pass 通过时，脚本自动进入正式 BS512 两臂 1000+1000。共用相同 GPU 配置、请求、固定 token、KV 和路由，检查原生 gate/ID、输出、整步图覆盖和槽计数。以 full_decode_pair.json 的 strict_pair_pass 为正式判据。失败时写出首差，保留原始数据。本轮短程插桩不计入性能；仅在正式验收通过时报告 999 个非采样步完整 MoE 的耗时及有符号差值。
4. 回执写 docs/sm100_megamoe/REPLY_V41_BS512_FAIR_BOUNDARY.md。提交短程和已启动正式阶段的报告、退出码、两臂版本和实际 SM/M；原始张量及 KV 留在私有存储，附路径和校验清单。完成 GPU 工作立即退卡，清理本轮私有进程、缓存、worktree 与依赖，CPU 整理后推送 main。遇短程失败也直接提交已有分叉和同输入证据。
