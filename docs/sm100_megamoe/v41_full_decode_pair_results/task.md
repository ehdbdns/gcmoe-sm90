# SM100 执行任务 · BS512 原生 Mega 与 GC 的完整 decode 图

收到即执行。本轮占用同组八张空闲卡，Mega 和 GC 串行各一次，分别统计 1000 步并测量随后 1000 步。两臂均用 DeepSeek V4.1 Flash、TP8/EP8、融合 shared、固定请求顺序和逐阶段 KV。保留原生布置。vLLM V2 捕获 BS512 完整 forward，每个 rank 每步提交一次模型图。两臂关闭 TorchDynamo 编译，让同一套模型代码在捕获阶段展开。采样和调度在模型图外。

## 执行

在原仓库执行完整代码块。启动器核对八卡 UUID、进程占用和组锁，有其他使用者则等待。预留两个阶段各 8 × 32 GiB KV checkpoint、原始 profile、路由、日志所需的本轮私有磁盘。共用模型、依赖和他人进程均须保留。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_FULL_DECODE_PAIR.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-full-decode-XXXXXXXX)
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
export TORCHINDUCTOR_CACHE_DIR="$run/inductor" TRITON_CACHE_DIR="$run/triton"
export VLLM_CACHE_ROOT="$run/vllm-cache"
bash scripts/sm100/run_five.sh "$run" full-native-bs512 2>&1 | tee "$run/launcher.log"
```

## 验收及回执

1. 保留启动日志、退出码、环境版本、模型与代码 SHA、acquired/released 记录。若捕获失败或回退，记录首个异常和 rank。执行者仅可修复编译错误，并提交最小 diff、修复前后日志。实验参数、计时范围、采样点和验收条件由任务书固定。
2. 每臂交付 bs512 下的 result.json、8 个 stats 和 8 个 measure NPZ。full_decode_pair.json 校验每个 rank 每阶段按层顺序记录了 40,000 次，逐步逐层八卡合计 3072 个路由条目，且两臂物理槽计数完全相同。分别列原生 greedy、原生 route id/weight、实际回放路由、KV 边界校验及首处分歧。某项失败时保留诊断数据，并说明其对性能比较的限制。
3. 提交两臂测量 step 2 的 L0、8、20、32、39 的 overview、详细 Perfetto JSON 和 counts JSON。回执给出 L20 的 GitHub 地址、八 rank 的 token/M 块、最早和最晚进入 MoE 的 rank 与到达时差、最长 count exchange、跨 rank 完整层跨度，以及校准误差。与上轮 L20 同口径对照，注明采样步和工作负载差异。
4. 对两臂各八 rank 打印同一次 cudaGraphLaunch 的关联号、attention/dense/MoE 节点数及 40 对逐层边界。正式 MoE 指标取其余 999 步，每步按层取八 rank 本地完整 MoE 时长最大值，再累加 40 层并对步求均值。该指标与采样步跨卡最早入口到最晚出口的跨度分别列出。通过严格验收后给出 GC 相对 Mega 的有符号 MoE 降幅。
5. 将回执保存为 docs/sm100_megamoe/REPLY_V41_BS512_FULL_DECODE_PAIR.md，将上述交付物存入同级 v41_full_decode_pair_results/ 并推送 main。过大的原始 profile、校准和工作负载证据在执行机归档，提交路径、字节数与 SHA256。失败也回执，附已完成产物。核对推送后清理本轮私有 worktree、venv、wheel、上游副本及 JIT 缓存，释放卡和锁并记录证据。
