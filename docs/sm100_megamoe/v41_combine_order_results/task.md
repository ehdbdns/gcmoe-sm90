# SM100 · BS512 对齐 vLLM Mega combine 顺序后的正式双臂

收到即执行。使用空闲的同组八卡；准备和核对 wheel、模型、提示词、磁盘空间在预约前完成。组内任一卡有人使用或已被预约，整组等待。退出、失败和中止时释放本任务的资源。复用本机已校验的 SM100_DSL_WHEELS。

74b8b6b 已归档共同 KV 与同输入诊断。CPU 重算 15 个首差元素都显示相同七个 BF16 partial 按不同顺序用 FP32 累加即可精确重现两侧输出。本轮直接执行一个正式 Mega 和一个正式 GC，各 1000+1000。GC 私有 checkout 只在原有兼容修复和时钟之外应用 vllm_combine_order.patch，保持上游 SM 默认值。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_COMBINE_ORDER.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-combine-order-XXXXXXXX)
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
"$py" - "$run" <<'PYSPACE'
import shutil,sys
if shutil.disk_usage(sys.argv[1]).free < (550 << 30):
    raise SystemExit("请在足量的私有存储上准备正式双臂目录")
PYSPACE
export SM100_VLLM_COMBINE_ORDER=1
bash scripts/sm100/run_five.sh "$run" strict-native-bs512 2>&1 | tee "$run/launcher.log"
```

保存 launcher.log、两臂 result.json、实际 GPU UUID 和 block M/SM、两阶段逐 rank 数据、full_decode_pair.json 以及两臂采样层 Perfetto。对 GC 记录上游 SHA 与最终 diff，对 Mega 记录实际 overlay 中 combine 段与头文件 SHA256。逐项核对 prefill、KV、覆盖前的原生 top-k/gate 和原生 token、回放输入、槽计数、整步图以及 999 个非采样步的完整 MoE。只有 strict_pair_pass 为真才能报告该配对的性能胜负。失败则保留首差的步、层、rank、原值和其他已生成证据。执行者只可修不改变语义的编译问题并附 diff。退出后清理私有缓存和依赖，保留结果、释放 GPU，回执写 docs/sm100_megamoe/REPLY_V41_BS512_COMBINE_ORDER.md，推送 main 并核对。
