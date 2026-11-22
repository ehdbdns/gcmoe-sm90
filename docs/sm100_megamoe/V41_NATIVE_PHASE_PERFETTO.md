# SM100 · BS512 原生 GCMoE / MegaMoE 阶段 Perfetto

只跑 mega、gc，各一次模型启动。沿用 V4.1 Flash、原五臂提示、TP8/EP8、原生放置、top-6、融合 shared expert、32 GiB KV 和正式窗口逐层 CUDA Graph。每臂先走原有 1000 个统计步，再运行 3 个测量步，只围住零基 step 2 的一次 engine.step()。输出零基层 0、8、20、32、39 各一张八 rank 图。各臂自然 greedy 解码，路由可不同。此轮用于定位阶段，速度结论沿用直跑表。

## 执行

原 SM100 节点保留同组八卡预约，空闲时串行执行；只使用本轮新的私有代码、环境和缓存。启动器检查实际 UUID 与占用，其他用户在用时等待。

```bash
set -euo pipefail
git switch main
git pull --ff-only origin main
cd "$(git rev-parse --show-toplevel)"
fixture=docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla
if [[ $(git config --bool core.sparseCheckout || true) == true ]]; then
  git sparse-checkout add "$fixture" scripts/sm100 source/gc_moe integration docs/sm100_megamoe
fi
test -r "$fixture/rank0.npz"
work=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-phase-bs512-XXXXXXXX)
base=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv
git clone --branch eco https://github.com/Natureal/gc-moe.git "$work/upstream"
git -C "$work/upstream" checkout --detach fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736
"$base/bin/python" -m venv "$work/dsl-env"
py="$work/dsl-env/bin/python"
"$py" -m pip install --no-deps nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 nvidia-cutlass-dsl-libs-cu13==4.8.1
lib=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
base_lib=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf "import site; site.addsitedir('%s')\n" "$base_lib" > "$lib/zz_v41_common.pth"
bash scripts/sm100/run_five.sh "$work" trace-native-bs512 2>&1 | tee "$work/launcher.log"
```

## 读取与交付

每臂的 bs512/{mega,gc}/perfetto/ 有五个 step002_layerXX.perfetto.json 和对应 counts.json。每张图的八个进程对应八个 rank。Kineto 捕获真实 CUDA kernel 事件，内部阶段来自同一次调用的设备 globaltimer。各 rank 从同一步前后两个专用 kernel 取时钟锚点，以 Kineto 的公共时间基准对齐。counts.json 保存时钟漂移、锚点时长对应的对齐误差和 19 个原始时间戳。

每条阶段轨道表示该 rank 上最早进入到最晚退出的跨度。count exchange 包含路由索引写入、计数交换及屏障；pull 从进入拉取循环准备到完成。shared/routed 的 L1、L2 从首个有效 tile 的首段 MMA 前到最后一批 epilogue 边界。routed L2 包含输出 epilogue/push，push 轨从处理 routed L2 输出到写回后的 epilogue 边界。combine 含就绪等待和归并。entry to first routed MMA 同时可能包含 shared 计算及通信。区间可重叠。用同一个 rank 对照首个 routed MMA 的开始时刻、pull 尾部、shared 跨度、最后一次 routed 写回及 combine 完成；附该 rank 的 token/M 块。

提交 docs/sm100_megamoe/v41_native_phase_results/{mega,gc}/ 和 docs/sm100_megamoe/REPLY_V41_NATIVE_PHASE_PERFETTO.md 到 main。回执给出 10 张图的 GitHub blob 链接，列五层各自关键 rank 及各阶段实测时长。附 counts、两臂测量 NPZ、result.json、版本/SHA、校验清单、日志和 GPU 释放证据。每 rank 的原始 torch profile 与 anchor 文件保留在执行机，记录路径和哈希。采集或校验失败时保留首栈和已完成产物。执行者仅可修复保持语义的编译错误并附 diff。完成或失败后释放八卡和组锁，核对归档后清理本轮的临时 checkout、DSL 环境、overlay 和编译缓存，保留共享环境及历史实验。
