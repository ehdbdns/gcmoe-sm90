# SM100 · BS512 L20 稀疏注意力 top-k 对照

收到即执行。使用空闲的同组八卡；准备和核对 wheel、模型、提示词、磁盘空间在预约前完成。组内任一卡有人使用或已被预约，整组等待。退出、失败和中止时释放本任务的资源。复用本机已校验的 SM100_DSL_WHEELS。

85a4eed 两臂首次原生 gate 差异在 stats 502/L20，模型 L20 的压缩率为 1、index_topk 为 512，同时承担候选块选择。本轮检验稀疏注意力索引器的影响。两臂都使用 vLLM 自带的 torch decode top-k 参考后端；专家 MoE、SM 默认值、combine 顺序和严格验收规则沿用上一轮。此变更属于两臂共同的 attention 控制。归档实际 kernel_config。按下列命令执行一个配对，不追加独立重复。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_SPARSE_CONTROL.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-sparse-control-XXXXXXXX)
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
export SM100_VLLM_COMBINE_ORDER=1 SM100_SPARSE_CONTROL=1
bash scripts/sm100/run_five.sh "$run" strict-native-bs512 2>&1 | tee "$run/launcher.log"
```

保存 launcher.log、两臂 result.json、实际 GPU UUID 和 block M/SM、两阶段逐 rank 数据、full_decode_pair.json 以及两臂采样层 Perfetto。对 GC 记录上游 SHA 与最终 diff，对 Mega 记录实际 overlay 中 combine 段与头文件 SHA256。逐项核对 prefill、KV、覆盖前的原生 top-k/gate 和原生 token、回放输入、槽计数、整步图以及 999 个非采样步的完整 MoE。只有 strict_pair_pass 为真才能报告该配对的性能胜负。失败则保留首差的步、层、rank、原值和其他已生成证据。执行者只可修不改变语义的编译问题并附 diff。退出后清理私有缓存和依赖，保留结果、释放 GPU，回执写 docs/sm100_megamoe/REPLY_V41_BS512_SPARSE_CONTROL.md，推送 main 并核对。

另附两臂 sparse-*.npz 清单。在 CPU 上对同一步、同 rank、同采样行列出 positions、L20 attention 输入/输出位型以及 index 序列是否相等；比较排序后的有效位置集合，区分选集与顺序。记录相同输入下仍有差异的首处。若 early-failure.json 已生成，提交该文件与两臂已有快照；立即释放八卡，结束本轮。缺失文件如实标注。参考后端通过只说明此共同配置下的公平性，单独一轮不能证明默认索引器内某条指令有错。
