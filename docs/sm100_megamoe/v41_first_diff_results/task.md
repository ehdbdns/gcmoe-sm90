# SM100 · BS512 首次 decode 输出分叉定位

收到即执行。使用上一轮同一模型、提示词、greedy、原生 GC/Mega 配置和整步 CUDA Graph。两臂各自完成 prefill 并执行一次 stats decode，采集 L0–L3 全八卡。仅在原八卡均空闲时持锁执行。复用上一轮已校验的本地 wheelhouse（SM100_DSL_WHEELS）。性能计时不作结论。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_FIRST_DIFF.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-first-diff-XXXXXXXX)
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
bash scripts/sm100/run_five.sh "$run" first-diff-bs512 2>&1 | tee "$run/launcher.log"
```

先读 run/bs512/first_diff.json，按层、先输入后输出排列，汇总全八卡。x 为 FP8 字节，sf 为打包的 scale，y 为专家层最终输出。检查 stats step 0、rank 7、L3、row 33 的六个 gate，与 37bc326 的两臂值分别比对。每 rank 必须实际记录 40 个有序层。比对两臂的 prefill token、请求顺序以及实际后端配置。若旧分叉未复现，报告这次实测首差与旧锚点的值。

回执写在 docs/sm100_megamoe/REPLY_V41_BS512_FIRST_DIFF.md，附 first_diff.json、两臂日志和结果元数据，原始快照登记路径与 SHA256。说明最早差异层、rank、行、通道、两侧数值，并说明它出现在已采集的 MoE 输入还是输出。如果输出先变，逐项确认同层八卡输入；若首层输入已经不同，标明 prefill KV/attention 仍待区分。只对实测张量归因。本轮只允许修复不改变语义的编译问题，附 diff；遇运行错误，提交首错与已有产物。推送 main 并核对远端提交，清理本轮私有环境和缓存、释放 GPU 和锁，保留快照。
