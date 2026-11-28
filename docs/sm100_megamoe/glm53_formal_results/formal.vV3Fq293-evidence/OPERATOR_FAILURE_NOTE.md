# 执行者补充说明 · SM100 GLM EP8 32-stream four-extra-slot refit

任务书提交 `6b278a37684c`，执行包 blob `24b79e0ec1d9`，SOURCE_REVISION `4516d6e81878`。

## 结果

三次尝试全部在 vanilla 臂启动校验处退出，`full_native_match` 未产生，`main.csv` 未生成。GPU 已释放，无外部资源占用。

| 目录 | 失败点 | 退出 |
|---|---|---|
| `formal.7MYZCaAQ` | 历史复核阶段，DSL venv 缺 `scipy` | 1 |
| `formal.Nxv2rahn` | `vanilla rank0: incorrect shared-first graph, device, price or slot count` | 1 |
| `formal.vV3Fq293` | 同上（干净重试） | 1 |

## 首个错误（`formal.vV3Fq293/results/run.log`）

```
File ".../repo/scripts/sm100/glm53_matrix.py", line 203, in arm_run
ValueError: vanilla rank0: incorrect shared-first graph, device, price or slot count
```

校验在 `glm53_matrix.py:190-203`，比较 `glm_setup` 返回的 `rank/gpu/slots/block_m/trace_level/roles/eco_unit`
与硬编码期望 `[[2,1,136],[1,0,12],[3,1,64]]`（vanilla）。引擎初始化本身成功
（`init engine (profile, create kv cache, warmup model) took 282.24 s`，CUDA Graph 捕获完成，KV 48 GiB 分配完成，
显存 155.6 GB，无 OOM、无 Xid、无 device_trap）。

## 已排除

- GPU 空闲：三次启动前八卡均 <=12 MiB、0 个计算进程，执行器连续两次检查通过。
- 设备与 rank 顺序：UUID 序列与原生臂一致（`729ae225,…,6ad27e1f`），`gpu-admission.csv` 无 busy。
- 提示词队列：`prompt_sha256` 与 `GLM_PRICE_REFERENCE` 原生臂一致（校验在该比较之后才失败）。
- 环境：`GC_MOE_SM100_SHARED_FIRST=1`、`GC_MOE_ECO_UNIT_CFG=66,74,1,6,0` 已由 `run_glm53_three.sh:139-141` 导出。
- 语义代码：未修改。仅给 `glm53_matrix.py` 加了只打印的诊断行（未提交，也未参与上述两次失败）。

## 需要构建者确认

`dispatch_sms` 在 `sm100_fp8_fp4_gc_moe.py:593` 为 `context.dispatch_sms - int(plan_on_dispatch)`；
`plan_on_dispatch = eco and GC_MOE_ECO_PLAN_DISPATCH==1`。vanilla 的期望是 12（136+12=148 SM），
若该路径在 `GLM_REFIT4=1` 下把 12 变成 11，则会命中此校验。请在构建机确认本次 32 槽 shared-first
vanilla 的实际 `ep8_roles` 与 `dispatch_sms`，并给出正确的期望值或修复。

历史价格复核本身已跑通（`historical-price/scores.json` 已归档，含前 800 步拟合、后 200 步检验）。
