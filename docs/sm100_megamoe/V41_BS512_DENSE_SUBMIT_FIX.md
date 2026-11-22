# SM100 · BS512 原生 GC / Mega 的层间提交错峰复测

执行机为 SM100。仅在预约的同一组八张卡均空闲时运行，沿用 run_five.sh 的 UUID、占用检查和组锁。Mega 和 GC 串行，各启动一次。保留 V4.1 Flash、原生 greedy、32 GiB KV、TP8/EP8、原 MoE 内核、shared 角色和逐层 MoE 图。

旧归档中 GC 第 20 层八 rank 的最长交换在未开 profiler 的 step 0、1 也有 369.8、356.3 µs。此次同时检验两个现象：相同 MXFP8 量化和 dense GEMM 由一张图提交，能缩短多少连续 kernel 的间隔；这是否也能减少各 rank 的 MoE 到达时差和跨八卡完整 MoE 跨度。开启回放包含输入复制和独立输出复制，两者成本计入实际运行。

## 执行

在原仓库运行下列命令。任务书提交锁定代码和脚本；在共享环境之外创建临时目录。沿用经过上轮五个 SHA 校验的 DSL wheels，可将 SM100_DSL_WHEELS 指向已有 wheel 镜像。未提供时 pip 优先使用机器缓存。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_BS512_DENSE_SUBMIT_FIX.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-dense-submit-XXXXXXXX)
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
bash scripts/sm100/run_five.sh "$run" quiet-native-bs512 2>&1 | tee "$run/launcher.log"
```

## 窗口与判读

- 两臂各 1000 个统计步。最后两个真实 decode 步分别建立 MXFP8 投影图、用新输入再与 eager 逐元素比较。两步均返回原 eager 输出以保留预热轨迹。第一个测量步前继续执行原 MoE eager/graph 输出与计数核验。只接受八个 rank 都记录了经过校验的 dense 图且按步回放开关生效的结果。
- 7 个测量步中 step 0 预热，step 1 开启 dense 回放并只采 CUDA 活动，给出原始 kernel 和五个采样层。step 2 到 6 的 dense 回放依次开、关、开、关、开。这五个 forward 在 profiler 关闭时执行；只在整段前后各测一次 Kineto 时钟锚点，各层使用内核内部 19 个设备时间戳。每次 MoE graph replay 的主机调用前读一次单调时钟，全程没有新增层内跨 rank 屏障。本次错峰对照取这五个步；整轮 wall_ms 含准备、RPC 和导出，五个步的耗时也不可替代长窗吞吐。
- 查 bs512/step001_dense_gaps.csv，同一 CUDA stream 上量化到 dense 的间隔按 rank 逐条给出。与旧 launch_gap_evidence.csv 核对 kernel 名、形状及调用位置；缺少同一次调用匹配时单独说明。表内没有配对时写清缺失覆盖。
- 用 within_run_skew.csv 打印两臂 L20 五步逐项结果，列早晚 rank、主机 replay 调用时差、GPU 入口时差、最长交换耗时、跨八卡完整 MoE 跨度；另外打印五个采样层按开关分组的中位数。相邻步为自然解码，结合每 rank token、M 块和物理槽计数判断负载差异。重点看关闭→开启的两次相邻转换（3→4、5→6）。若 dense 局部间隔缩短而到达时差或完整 MoE 跨度没有一起改善，结论逐项报告。
- 用 step002_skew.csv 比较同臂旧 step 2 与新 step 2。检查 changed_slot_counts 及对齐误差；存在路由量差异的层注明。旧 host spread 取 CUDA API 入口，新值取 Python graph.replay 调用前，均为跨 rank 的差值。不要将二者解释为同一 API 的精确时长。旧 trace 的两次最大空档可能来自不同 dense 调用。
- 图和 counts 位于各臂 overview/ 与 perfetto/。step 1 有真实 Kineto kernel；step 2–6 的详细图只画同一步实测阶段，禁止拼接相邻步的 kernel。回执列出两臂第 20 层 step 1/2 的 overview 及详细图 GitHub 链接。

## 交付和清理

提交 docs/sm100_megamoe/REPLY_V41_BS512_DENSE_SUBMIT_FIX.md 和同级 v41_dense_submit_results/。交付三个 CSV、每臂结果 JSON、八个 measure NPZ、Perfetto/overview/counts、环境 SHA、日志、退出码、GPU 及锁释放证据。原始 CUDA profile 和两校准会话及 NPY 保存在执行机并提交 SHA256 与路径。若遇首错，附报错、阶段、已完成的产物和释放证据，执行者仅可做保持语义的编译修复并附 diff。将回执和产物推送 main，确认远端提交后清理本轮 worktree、私有依赖、overlay 和缓存，保留原始证据和共享环境。
