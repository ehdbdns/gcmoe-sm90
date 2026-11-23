# SM100 · BS512 原生 greedy 的 Mega 与 GC 两臂

收到即执行。同一预约的八张空闲卡，依次启动 Mega 和 GC 各一次。使用上轮相同的模型、提示词、TP8/EP8、融合 shared、greedy、BS512 和整步 decode graph。每臂独立生成并沿用自身 KV。两个后端均直接消费模型产生的 top‑6 和 gate 权重，采样记录只读。GC 用锁定上游的 Green Context 默认值和自动 M 块大小。每臂 1000 步统计，随后 1000 步测量。

## 执行

在原仓库运行完整代码块。启动器核验 GPU UUID、等待八卡全部空闲并持组锁。沿用已验证的私有 DSL 兼容和 GC 外层 graph capture 修复。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_UPSTREAM_GREEDY_PAIR.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-native-greedy-XXXXXXXX)
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
bash scripts/sm100/run_five.sh "$run" upstream-greedy-bs512 2>&1 | tee "$run/launcher.log"
```

## 交付

1. 两臂提交实际 backend、版本与 SHA、独立原生输出、所有 rank 实际 block M 和 GC 的 dispatch/compute SM 数。核实 GC_MOE_DISPATCH_SMS 和 GC_MOE_BALANCED_WORKQUEUES 均未设置，驱动实际分区照实填写。
2. 交付 native_greedy_pair.json、两臂 result.json、prefill.json、两阶段 tokens.npy 和每 rank 计数/时钟 NPZ。列出原生 token、路由 id、gate 权重、物理槽计数的差异数量和首次分歧（输出索引为阶段步/请求，路由为阶段步/层/本地行/top-k 序号）。按独立生成轨迹解释结果。大型路由 NPY 与原始 profile 在执行机归档并提交路径、大小、SHA256。
3. 列出 16 份整步图覆盖证据。比较排除测量 step 2 后 999 步完整 MoE 均值（每层取八 rank 本地时长最大值，每步累加 40 层），以同轮 Mega 为基准列有符号耗时变化。交付 step 2 的 L0、8、20、32、39 的 overview、详细 Perfetto 和每 rank token/M 块，并附 L20 两臂 GitHub 链接。
4. 执行者仅可修不改变语义的编译问题，附 diff，配对使用同一最终版本。遇捕获、正确性、测量或运行失败，保留首错、退出码和已有结果并回执。回执放 docs/sm100_megamoe/REPLY_V41_BS512_UPSTREAM_GREEDY_PAIR.md，产物放同级 v41_upstream_greedy_pair_results/。推送 main 并核对远端提交后，释放卡与锁并清理本轮私有 worktree、依赖与缓存，保留归档证据。
