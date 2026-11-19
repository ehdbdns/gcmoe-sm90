# 八卡任务书 · 按就绪 M 块发布 routed 工作

验证 SM90 流式发票。保持已验收的 BS256、逐层 EPLB、MB-CLAIM、CUDA Graph、TMA、SM-reuse=0 和固定 step 2 的 layer 8、20。每次领取仍是一个 `(slot, M block, N tile, phase)`。

## 回执证据

旧的 layer 8 / rank 6 图来自会自动选择 epoch 的绘图脚本。它有 684 个 tile，对应 rank 6 的 layer 20 的 18 个 M 块。真正的 layer 8 / rank 6 有 36 个 M 块和 1368 个 tile。其热点 local slot 48 在 319 µs 拉完，434 µs 开始 L1，618 µs 完成最后一个 L1 tile，714 µs 开始首个 L2，800 µs 完成最后一个 L2 tile。原始 CTA 活跃数据在这些区间仍有其它槽的计算。先前回执对跨 rank L1 依赖的推测需要按当前源码复核。L2 输入由本 rank 的本槽位生成。

旧 dispatch 只在整槽拉完后发票。旧 L2 领取还可能选中 L1 未完成的槽，随后等该槽的计数。新版每拉完一个满块或尾部残块，就在该块路由元数据、激活和 scale 可见后发布这一批 L1 票。同一 M 块的所有 L1 N-tile 均完成后，再发布它的 L2 票。计算 CTA 只领取已发布的任务，可以在 L1 没票时领取已就绪的 L2。

## 回执 9a02879 后续

上轮基线两组编译停在 dispatch 的专家循环，尚无 A/B 采样。设计端本地复现后确认，CuTe 会把某些带 `self` 的下标赋值目标误判成动态控制流携带的值，即使该赋值所在的 `const_expr` 分支没有启用。本次将 dispatch 的两处写入、compute 的 L2 batch 发布和 combine 的 L2 队列清零改为先绑定局部变量，再按原地址写入。流式 compute 的 L2 batch 读取显式转为 `Uint32`，与 L1 分支的类型一致。

设计端用 CuTe DSL 4.8.1 和 CUDA 13.2 `ptxas` 离线生成了四组全部角色的 `sm_90a` cubin。设备启动和 CUDA Graph 校验在八卡机执行。

## 八卡编译预检

在八卡机仓库根目录确认工作区干净、GPU 空闲，再同步 `main`。先用正式运行的工具链检查四种调度和 profiling 组合。每组独立运行，分别保留完整日志。

```bash
set -eo pipefail
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main

(
  source config/experiment.env
  source scripts/env.sh
  export GC_MOE_SRC="$PWD/source"
  export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration"
  export GC_MOE_MB_CLAIM=1 GC_MOE_SM90_PULL_TMA=1 GC_MOE_BLOCK_M_FORCE=64
  export GC_MOE_SM90_SM_REUSE=0 GC_MOE_SM90_DISPATCH_SMS=6
  export GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 GC_MOE_IR_STEP=0
  export GC_MOE_FINE_FLOOR=1 GC_MOE_FINE_EPOCHS=1
  export GC_MOE_FINE_RECS=4194304 GC_MOE_FINE_MIN_EPOCH=0
  export SMOKE_ECO=1 SMOKE_NULL=0 SMOKE_NOAUC=0
  logs=$(mktemp -d /tmp/gcmoe-stream-compile-XXXXXXXX)
  echo "Compile logs: $logs"
  git rev-parse HEAD | tee "$logs/execution.sha"
  nvidia-smi --query-gpu=name,driver_version --format=csv | tee "$logs/gpu.log"
  /usr/local/cuda/bin/ptxas --version | tee "$logs/ptxas.log"
  "$GC_ENV/bin/python" -c 'import torch, cutlass; print(torch.__version__, torch.version.cuda); print(getattr(cutlass, "__version__", "unknown"), cutlass.__file__)' \
    | tee "$logs/python.log"
  failed=0
  for mode in 0 1; do
    for profile in 0 1; do
      roles='[1, 2, 5, 3]'
      wb=64
      if [ "$profile" = 1 ]; then
        roles='[1, 4, 2, 5, 3]'
        wb=26
      fi
      log="$logs/stream-$mode-profile-$profile.log"
      if GC_MOE_SM90_STREAM_M_BLOCKS="$mode" GC_MOE_STEP_PROFILE="$profile" \
         GC_MOE_FINE_TRACE="$profile" SMOKE_ECO_WB="$wb" \
         "$GC_ENV/bin/python" scripts/smoke_compile_eco.py 2>&1 | tee "$log" \
         && grep -Fq "compiled roles $roles" "$log"; then
        printf '%s PASS\n' "$log"
      else
        printf '%s FAIL\n' "$log"
        failed=1
      fi
    done
  done
  test "$failed" -eq 0
)
```

如果仍有编译失败，执行者只可按首条诊断修改对应的 CuTe DSL 写法，例如局部指针绑定、同位宽显式类型转换或等价的编译期分支写法。每次只改一处，记录失败配置、原始报错、改动和复测结果。一处修复后先复跑该组合，通过后重跑四组。兼容性修复单独提交，使用修复后的同一 SHA 验证 A/B。发票时机、领取粒度、队列与等待条件、内存顺序、CUDA Graph、计数与计时口径、实验配置和验收门槛均按原实现执行。若编译问题无法在上述范围内解决，或出现运行、正确性、采样失败，保存命令与完整日志，停止并回执。

## 执行

四组均通过后，用同一模型、数据集和工具链按 A → B 顺序运行。入口自动校验并记录执行 SHA、失败步骤、命令和日志，分别保留两个输出目录。

```bash
set -e
test -z "$(git status --porcelain)"
revision=$(git rev-parse HEAD)
GC_MOE_SM90_STREAM_M_BLOCKS=0 bash scripts/trace_layer/run_aligned_ep8.sh "$revision" --fine-trace
GC_MOE_SM90_STREAM_M_BLOCKS=1 bash scripts/trace_layer/run_aligned_ep8.sh "$revision" --fine-trace
```

每轮先跑 CPU 回归和角色编译冒烟。新方案还要过无需加载模型的八卡 CUDA Graph 正确性门禁。该门禁给一个单副本专家分配 135 行，覆盖两个满 M 块和一个 7 行尾块。旧的按 N-tile 领取路径作为数值参考，CUDA Graph 重放后八卡 BF16 输出的 SHA-256 必须逐 rank 一致。正式 A/B 的两轮都使用 MB-CLAIM。

## 验收与回执

1. 每轮都须保留八卡 source trace、`capture.json` 和 `fine.json`，并通过同一 capture ID 与 epoch 校验。保留两层七段 Perfetto、逐 rank token／M 块计数、细粒度 Perfetto 和 detail JSON。原始数据留在执行机，回执提供位置及校验值。
2. 新方案每个非空 M 块都必须有唯一的 L1、L2 发布记录。L1 只能在本块发布后开始。L2 发布必须晚于或等于本块最后一个 L1 tile 完成。同一槽位的任务数、有效行数和 L2 push 必须与原始 owned 一致。
3. 逐层列出最后退出 routed 的 rank 和最重槽。画出该槽每个 M 块的 L1 发布、首个 L1 tile、最后一个 L1 tile、L2 发布和最后一个 L2 tile。报告多少个满块的 L1 在整槽拉完前开始。对照旧方案 rank 6 的 319/434/618/714/800 µs 五个边界，测量首票提前量、L1→L2 间隔和 routed 退出时刻。
4. 比较前先检查两轮的实际逐槽分布。如果热点 rank 或 owned 分布变了，先报告采样内的因果顺序，跨轮耗时只做有注明的对照。细 trace 的时间不代表未插桩稳态性能。

在本目录更新 `REPLY_STREAM_MBLOCK.md`，列出实际执行 SHA、编译矩阵和四份日志路径。有兼容性修复时附提交 SHA。成功时把 A/B 的生成产物分别放进 `stream_results/baseline/` 和 `stream_results/streamed/`，附 CUDA Graph 校验、两轮 capture ID 和 epoch、热点块发布表及剩余等待位置。运行或采样失败时，回执写明首个失败步骤、命令、报错与已有数据位置。将回执及本轮验收产物提交并推送到 `main`。
