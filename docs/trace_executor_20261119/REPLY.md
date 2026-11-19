# REPLY — 八卡执行回执（失败：停在步骤 2）

## 结论

**停在步骤 2（两套编译冒烟），未进入步骤 3 采集。**
原因：插桩路径（`GC_MOE_STEP_PROFILE=1`）编译出的 role 集合是 `[1, 2, 5, 3]`，
**缺少任务书要求的 role 4**（要求 `1, 4, 2, 5, 3`）。
这是 `scripts/smoke_compile_eco.py` 的硬编码，属于实验配置而非编译兼容性问题，
不在我的修改授权内（任务书：代码修改仅限编译器诊断定位的最小兼容性修复）。

## 版本

| 项 | 值 |
|---|---|
| 起始 SHA | `3c537d20787853b038cc5e77b206c6654b1acdaa` |
| 执行 SHA | `3c537d20787853b038cc5e77b206c6654b1acdaa`（**未做任何代码修改**）|
| `cffe483` 是否为祖先 | 是 |
| 工作区 | 干净（步骤 1 检查通过）|

## 机器与配置

```
8 × NVIDIA H20, driver 580.105.08, 每卡 97871 MiB, 起始全部 4 MiB 空闲
torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / 8 卡 capability 9.x
MODEL=/home/admin/local_models/DeepSeek-V4.1-Flash
DATASET=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
```

环境变量按任务书步骤 1 原文设置（EPLB=1, ECO=1, ECO_WB=26, WA=0, MAX_COPIES=8,
EXTRA_PER_RANK=8, 逐层 EPLB=1, MB_CLAIM=1, PULL_TMA=1, SHARED_IN_GC=1,
SM_REUSE=0, DISPATCH_SMS=6, CAPACITY/BOUND=2048, BLOCK_M=64,
PULL_TRACE=1, ECO_TRACE=1, FINE_TRACE=0, IR_STEP=0, LAYERS=40,
MEASURE_FLUSH=2, ENFORCE_EAGER=0, CUDAGRAPH_MODE=FULL_DECODE_ONLY）。

## 步骤 1 结果：通过

```
workspace clean
cffe483 is ancestor
HEAD = 3c537d20787853b038cc5e77b206c6654b1acdaa
2.13.0+cu130 13.0 0.31.0        # torch / cuda / vllm，8 卡 SM90 断言通过
```

## 步骤 2 结果：CPU 通过，常规编译通过，**插桩编译不满足判据**

### 2.1 CPU 检查
```
ALIGNED_STEP_CPU PASS
```

### 2.2 常规路径（`GC_MOE_STEP_PROFILE=0`）
```
roles [(1, 43008, 256, 24), (2, 199680, 384, 71), (5, 51200, 512, 1), (3, 0, 384, 288)]
compiled roles [1, 2, 5, 3] in 9.6s
```
符合任务书「常规路径应包含角色 1、2、5、3」。✅

### 2.3 插桩路径（`GC_MOE_STEP_PROFILE=1 SMOKE_ECO_WB=26`）
```
roles [(1, 43008, 256, 24), (2, 199680, 384, 71), (5, 53248, 512, 1), (3, 0, 384, 288)]
compiled roles [1, 2, 5, 3] in 10.1s
```
**缺少 role 4。** 任务书要求「插桩路径应包含角色 1、4、2、5、3」。

## 根因（已定位到行）

`source/gc_moe/jit_kernels/impls/sm90_fp8_fp4_gc_moe.py:92-95`：

```python
shared = get_shared_role_config(schedule) if config.shared_experts else None
eco = bool(getattr(config, 'eco_logical_experts', 0))
base_roles = (1, 4, 2, 3) if config.shared_experts else (1, 2, 3)
```

role 4 是否存在**只取决于 `config.shared_experts`**。

而 `scripts/smoke_compile_eco.py:32` 把它硬编码为 0：

```python
clamp=10.0, fast_math=True, shared_experts=0,
```

同一文件的第 26-27 行确实按开关设了 layout 的 shared 槽位：

```python
step_profile = os.environ.get('GC_MOE_STEP_PROFILE') == '1'
shared = int(step_profile)
```

但 `layout` 的 shared 与 `config.shared_experts` 是两个不同的入参，
前者只影响缓冲区布局，后者才决定 role 4 是否被编译。

## 为什么我没有修

任务书「执行权限」写明：

> 代码修改仅限步骤 2 中由**编译器诊断定位**的最小兼容性修复，例如 CuTe DSL 不支持的表达式写法。

这里没有编译错误——两套构建都成功退出（`compiled roles ... in 9.6s / 10.1s`）。
缺 role 4 不是编译器诊断，而是**实验配置与验收判据不一致**。
把 `shared_experts` 改成 1 会改变被编译/被运行的内核集合，属于
「冻结实验配置、内核行为、调度」明确禁止的改动。

因此按任务书步骤 5「失败」流程回传，等待明确指令。

## 需要设计端澄清的一个问题

任务书步骤 2 要求插桩路径编译出 role 4，但步骤 1 的环境变量里
`GC_MOE_SHARED_IN_GC=1` 只影响真实运行时的 `deepep_compare` 接入，
**不会传到 `smoke_compile_eco.py` 的 `config.shared_experts`**。

两种可能的正确做法，请择一指示：

1. `smoke_compile_eco.py` 应改为 `shared_experts=int(os.environ.get('GC_MOE_SHARED_IN_GC', '0'))`
   （我认为这是本意；改动 1 行，属脚本 bug 修复）；
2. 或者步骤 2 的期望值应改为 `[1, 2, 5, 3]`（即 role 4 只在真实运行时出现，
   smoke 编译不覆盖它）。

## 产物位置与校验值

采集工作目录（**未产生任何采集数据，仅含步骤 1-2 的日志**）：

```
/tmp/gcmoe-aligned-W7nZLoqX
```

内容：

```
start.sha              3c537d20787853b038cc5e77b206c6654b1acdaa
gpu.log                8 × H20 / 580.105.08 / 4 MiB used
runtime.log            2.13.0+cu130 13.0 0.31.0
cpu.log                ALIGNED_STEP_CPU PASS
compile-base.log       compiled roles [1, 2, 5, 3] in 9.6s
compile-aligned.log    compiled roles [1, 2, 5, 3] in 10.1s   <-- 缺 role 4
```

校验值：

```
68625da1fe41de44dcf521cddee98efd7fa9bb4087599d35bed5ff70aa9a6407  start.sha
52fec3535694e27e323df103991686727b00d566fd9c3fbc4f5eff1b2a78c772  gpu.log
74064a5ac06f3b428b7e0480f7e8089af2a734d042f3597f79740e8b242160b4  runtime.log
eeea22f858445f4d27a731b40bab4e1ec317b9990e56c1f07c8b1fdfad3339c7  cpu.log
bda1d6cc8db79f318dfeb8e964d98727bb636554259590505f14299114512bf0  compile-base.log
bbfd8939e06bb309a864eec2c0d745ce2433b9570b1b88d55a73b3bd74b8efac  compile-aligned.log
```

## 未执行的部分

- 步骤 3（BS256 采集）：**未执行**（避免在已知不合规的构建上浪费 10 分钟机时）
- 步骤 4（解析 layer 8/20）：**未执行**
- `results/`：**空**（无产物）
