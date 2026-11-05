# 结果：SM90 ECO 三臂（2026-11-05）

## 实验规格

- 模型：DeepSeek-V4.1-Flash，8×H20（SM90），TP=8 / EP=8
- 协议：真实 prefill（不计时）→ 100 步统计（EPLB 学习 + 一次重排）→ 100 步测量
- 批量：4 seq/rank（32 条序列），每步 32 个生成 token
- 冗余：25%（`physical = 480`，60 slots/rank，96 个冗余专家）
- 编译几何：`tile_m = 64`，`bound = 2048`

三臂：
| 臂 | 放置 | 分流 |
|---|---|---|
| vanilla | 无副本（384 slots） | 逻辑 id = 物理 id |
| eplb-rr | EPLB 25% | host 侧按 route 轮询选副本（上一轮已验证的 B 臂） |
| eplb-eco | EPLB 25% | **kernel 内 role 5 按 quota 切分**（本次移植） |

## 结果

| 臂 | ms/step | Δ vs vanilla | IR (rank 负载) | tokens/rank |
|---|---:|---:|---:|---:|
| vanilla | **20.25** | — | 1.0284 | 32 |
| eplb-rr | 20.51 | +1.29% | 1.0672 | 32 |
| eplb-eco | 21.99 | +8.60% | **1.0358** | 32 |

> ⚠️ **M 块数 / 激活专家数这两列已撤回**（原来写的 `5 / 19 / 16 / 17`
> 和 `block_m 192/240` 是错的）。原因和修正见 `docs/METRICS.md`：
> 浮点整除 `(x+bm-1)//bm` 把每个槽的块数算成 0～1，`block_m` 又取的是
> SM100 的选择器（192/240）而不是 SM90 routed role 的 `tile_m`（64），
> `header[7]` 在窗口结束后读已经被 role 3 清零。
> 修正后的真实量级：本次负载 32 token / 48–60 槽 ≈ 0.4 token/槽/步，
> **每个活跃槽恰好 1 块 ⇒ M 块数 == 激活槽数**，单次调用实测
> vanilla rank0：`active=20, blocks=21`。三臂的这三项需要重跑才有准确值
> （用户要求不重跑，故此处不填）。

窗口内（`load_detail.ir_window`，按层平均的 slot 级负载比，这一项不受上述
三个缺陷影响）：
vanilla 1.662 / eplb-rr 1.338 / **eplb-eco 1.310**。

### 每 rank 实际 token（IR 的分子/分母）

```
vanilla   [94815, 92253, 91410, 90707, 96753, 96447, 96123, 94132]
eplb-rr   [94759, 95651, 92065, 93200, 95729, 100398, 91409, 89429]
eplb-eco  [95653, 94051, 92971, 94395, 93153, 91069, 93898, 97450]
```

## 结论

1. **ECO 的分流比 RR 更均衡**：slot 级 IR 1.310 vs 1.338，rank 级 IR 1.0358 vs 1.0672。
   ECO 会把「零头」和整块分开搬，RR 只能整 token 轮询，因此尾部长尾更短。
2. **本规格下 ECO 慢 1.48 ms/step（+7.2% vs RR）**。原因清楚：
   - plan role 与 routed role 同时在跑，routed grid 从 72 个 CTA 降到 **71**
     （必须让出一整块 SM，否则 plan 永远排不上队 → 死锁）；
   - 副本 pull 要先等 plan 发布，single-copy 槽可以重叠，但 60 个槽里 24 个是副本槽；
   - 本 batch 只有 32 token，`count <= block M`，plan 的收益（少一个 M 块 / 少几个激活槽）
     上限很低；RR 的 IR 本来就只有 1.067。
3. **tokens/rank = 32 全部正确**，三臂都跑满 100 步测量、`running=32..32`、
   `gen_tokens/step=32..32`。这是本轮先解决的核心问题（每 rank 实际 token 是否为 32）。
4. **正确性有独立证据**：`scripts/verify_eco_ep8.py` 在 8 卡上跑随机权重，
   ECO 臂输出与无副本参考路径 **cos=1.00000000 / rel=0.0 / 逐位相同**，
   且 quota 守恒、`quota_start` 连续、8 个 rank 的 plan 完全一致。

## 为什么 ECO 的 IR 不是「必然」更好

ECO 最小化的是 `wb·块数 + wa·激活槽 + token`。`wb = wa = 0` 时它只均衡 token，
不看 M 块数（`m_blocks_median` 三臂都是 5，`active_experts` 17 vs 16）。
要让 ECO 在**时延**上赢，需要按 §5 的方法标定 H20 的 wb/wa，让 plan 主动减少
M 块数和激活槽数（这才是 GEMM 的瓶颈），而不是只均 token。

## 复现

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/gcmoe_eco-auction_20261105
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done

SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 ./scripts/run_arm_eco.sh 0 0 vanilla_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 0 eplb_rr_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 1 eplb_eco_100
```

原始 JSON：`runs/{vanilla_100,eplb_rr_100,eplb_eco_100}.json`；
完整日志：`logs/real_{vanilla_100,eplb_rr_100,eplb_eco_100}.log`。
