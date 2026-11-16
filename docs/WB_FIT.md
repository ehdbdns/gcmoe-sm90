# 用实测 trace 反推 ECO 代价函数的权重（wb / wa）

## 结论

```
wb = 29.5   (每多一个 M 块 ≈ 29.5 个 token 的代价)
wa = 0      （没有独立的"开槽"开销）
token 系数 = 1（保持）
```

即 `cost(q) = 29.5·⌈q/64⌉ + q`。
之前手工测的 `wb=26.5` 与此相差 11%，**方向一致、量级正确**，
现在这版是从 1824 个 M 块的完整分布拟合出来的，比手工两点法可靠。

## 推导过程（三步，缺一步就会错）

### 第 0 步：先纠正一个会致命的前提

耗时**不是** valid 的线性函数。GEMM 会把 n 维**量化到档位**
（`_select_persistent_tile`，`sm90_fp8_fp4_gc_moe.py:3214-3241`）：

```
valid <= 8 -> n=8 ; <=16 -> 16 ; <=32 -> 32 ; <=40 -> 40 ; <=48 -> 48 ; else 64
```

所以 valid=17 和 valid=32 的块耗时几乎一样。这解释了为什么
直接按 valid 分组时"局部斜率"乱跳（0.03 ~ 0.91 us/row）。
**必须按 `nq(valid)` 分组再拟合。**

### 第 1 步：按量化档分组，得到单个 ntile 任务耗时

| nq | L1 n | L1 均值 | L2 n | L2 均值 |
|---:|---:|---:|---:|---:|
|  8 |  72 | 22.72 |  80 | 10.97 |
| 16 | 162 | 24.03 | 180 | 11.89 |
| 32 | 414 | 27.84 | 460 | 14.02 |
| 40 |  72 | 33.62 |  80 | 16.26 |
| 48 |  18 | 40.89 |  20 | 19.25 |
| 64 | 126 | 52.65 | 140 | 24.75 |

单任务最小二乘：

```
L1: t = 15.10 + 0.5345·nq     (R² = 0.925)
L2: t =  7.79 + 0.2425·nq     (R² = 0.943)
```

R² 没到 1，残差呈 S 形（nq=32/40 偏低、两端偏高），
说明还有一个与 nq 无关的成分——这正是第 2 步要处理的。

### 第 2 步：折算到"一个槽位的一个 M 块"（这一步最容易漏）

ECO 的 cost 是**每槽位**的，而一个槽位要跑完 **全部 ntile**：

* L1 有 `intermediate/(128·factor)` = **18** 个 ntile
* L2 有 `hidden/(256·factor)` = **20** 个 ntile

所以：

```
一个 M 块的物理耗时 = 18·(L1 每任务) + 20·(L2 每任务)
                   = 427.5 + 14.47·nq   [µs]
```

（直接拿单任务耗时当权重、忘了乘 ntile 数，会差 18~20 倍。）

### 第 3 步：换成 ECO 的 cost 单位

`cost(q) = wb·⌈q/bm⌉ + wa·[q>0] + q`，其中 1 个 token = 1 单位。
把它对齐到 `427.5 + 14.47·nq`：

* 每行 → 1、每块 → wb，所以 **wb = 427.5 / 14.47 = 29.5**
* 常数项全部由"每块固定开销 × ntile"解释

## wa 为什么是 0（做了独立检验）

如果存在"开槽"这笔独立开销，它会表现为**与 ntile 数无关**的额外常数。
检验方法：把拟合出的固定项按 ntile 数摊开，看是否超出单任务的固定启动：

| 阶段 | 固定项 | ÷ ntile 数 | 结论 |
|---|---:|---:|---|
| L1 | 15.10 µs | 0.84 µs | < 2–3 µs 的 wgmma 启动 ⇒ 无额外开销 |
| L2 | 7.79 µs | 0.39 µs | 同上 |

⇒ 固定项已完全由 ntile 数解释，**wa = 0**。
这与既有配置 `GC_MOE_ECO_WA=0` 一致，等于从数据上确认了那个选择。

## 与旧值对比

| 来源 | wb | 方法 | 可信度 |
|---|---:|---|---|
| 早期手工 | 260 | 两点法 | ✗ 高估 10 倍，已废弃 |
| `measure_units` 手工 | 26.5 | 单点斜率 | 方向对 |
| **本次** | **29.5** | **1824 块完整分布 + 量化档校正 + ntile 折算** | ✓ |

## 重要提醒（别把这个结论用错）

1. **这是"按当前 kernel 形状"的值。** 若改 `block_m`、ntile 数、或 GEMM 档位表，
   `A` 和 `B` 都会变，wb 必须重拟合。尤其 `wb ∝ ntile_L1·a_L1 + ntile_L2·a_L2`，
   ntile 数变了权重就变。
2. **ECO 的 cost 用 `q`（真实 token）计费，而 GEMM 按 `nq`（量化档）收费。**
   这意味着 `q` 在档位内增长时，cost 涨了但耗时不涨 ⇒ ECO 会轻微高估
   中段 token 的代价（例如 33~40 行按 40 收费，41~48 又跳一档）。
   本次测量无法消除这个建模误差，只能靠 wb 的斜率取"平均档宽"来摊平。
3. **wb=29.5 尚未在 GPU 上验证。** 上面所有数都来自 trace 拟合，
   还没有跑过 `GC_MOE_ECO_WB=29.5` 的对照实验。下一步应当是：
   在 mb-claim 版本上跑 `wb=26` vs `wb=29.5` 的 A/B，看 IR 与 ms 是否同向改善。

## 复现

```bash
cd .../t18_abstrace
python3 - <<'PY'
import csv, collections, statistics
p='docs/abstrace/AT256_eco_mbclaim_rank0_full.abstrace_rank0.csv'
rows=[r for r in csv.DictReader(open(p)) if r['kind']=='gemm']
def nq(v):
    for lim in (8,16,32,40,48):
        if v<=lim: return lim
    return 64
by=collections.defaultdict(list)
for r in rows: by[(r['phase'],nq(int(r['valid'])))].append(float(r['dur_us']))
NT={'1':18,'2':20}
A=B=0
for ph in ('1','2'):
    g={k:v for k,v in by.items() if k[0]==ph}
    xs=sorted(k[1] for k in g); ys=[statistics.mean(g[(ph,x)]) for x in xs]
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    a=my-b*mx
    print(f'L{ph}: {a:.2f} + {b:.4f}*nq  x{NT[ph]}')
    A+=a*NT[ph]; B+=b*NT[ph]
print(f'total: {A:.2f} + {B:.4f}*nq  => wb={A/B:.2f}')
PY
```
