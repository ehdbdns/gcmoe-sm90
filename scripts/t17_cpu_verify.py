"""T17 CPU 验证：逐层放置 vs 合并放置。

输入：GC_EPLB_STATS_DUMP 导出的真实 stats[ranks, layers, per_rank]。
两边都用【完全相同的】DeepInfer policy 源码（gc_moe.eplb.deepinfer_periodic.policy），
只改变喂给它的 layer 维度：
   合并 : hist = sum(layers)[1, 1, E]
   逐层 : hist = [1, L, E]
然后按层评估 IR（口径 A）：
   IR_layer = max_r(该层 rank token) / mean_r(...)
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'source'))
sys.path.insert(0, os.path.join(HERE, 'integration'))
from eplb_logic import layer_demand_from_stats
from gc_moe.eplb.deepinfer_periodic.policy import rebalance_experts_global


def load(npz_path):
    z = np.load(npz_path)
    stats = torch.as_tensor(z['stats'], dtype=torch.float32)   # [ranks, layers, per_rank]
    layout = np.asarray(z['layout'], np.int64)
    return stats, layout, int(z['per_rank']), int(z['ranks']), int(z['experts'])


def demand_per_layer(stats, layout, per_rank, experts):
    """[ranks, layers, per_rank] -> [layers, experts]，每层独立、跨 rank 求和。

    与 integration/eplb_logic.py::demand_from_stats 相同的变换，但保留 layer 维。
    """
    return sum(layer_demand_from_stats(row, layout, r, per_rank, experts)
               for r, row in enumerate(stats))


def eval_placement(plan_flat, demand_row, layout, per_rank, experts, ranks):
    """给定一份 [physical] placement，算该层的 IR = max_r/mean_r。"""
    lay = np.asarray(layout, np.int64)
    phy2log = np.zeros(len(plan_flat), np.int64)
    phy2log[:] = np.asarray(plan_flat, np.int64)     # plan 本身就是 slot->expert
    rank_of = np.arange(len(plan_flat)) // per_rank
    loads = np.zeros(ranks)
    d = np.asarray(demand_row, np.float64)
    np.add.at(loads, rank_of, d[phy2log])
    return loads.max() / loads.mean(), loads


def main():
    npz = sys.argv[1]
    stats, layout, per_rank, ranks, experts = load(npz)
    L = stats.shape[1]
    print(f'stats {tuple(stats.shape)}  layout {layout.shape}  per_rank={per_rank} '
          f'ranks={ranks} experts={experts} layers={L}')
    P = layout.shape[-1]

    dem = demand_per_layer(stats, layout, per_rank, experts)      # [L, E]
    torch.set_printoptions(precision=1, linewidth=200)
    print(f'每层 demand 总和: mean={dem.sum(1).mean():.0f} '
          f'min={dem.sum(1).min():.0f} max={dem.sum(1).max():.0f}')

    # ---------- 我们的实现：合并 40 层 ----------
    merged = dem.sum(0, keepdim=True).view(1, 1, experts)         # [1,1,E]
    plan_m, _, cnt_m = rebalance_experts_global(merged, P, per_rank)
    plan_merged = plan_m[0].tolist()

    # ---------- DeepInfer：逐层 ----------
    hist = dem.view(1, L, experts)                                # [1,L,E]
    plan_l, _, cnt_l = rebalance_experts_global(hist, P, per_rank)

    # ---------- 评估（口径 A: 每层独立算 IR 再平均） ----------
    irs_m, irs_l = [], []
    for l in range(L):
        a, _ = eval_placement(plan_merged, dem[l], layout, per_rank, experts, ranks)
        b, _ = eval_placement(plan_l[l].tolist(), dem[l], layout, per_rank, experts, ranks)
        irs_m.append(a); irs_l.append(b)

    import statistics as st
    print()
    print(f'{"口径":18s} {"IR mean":>9s} {"IR p50":>8s} {"IR max":>8s}')
    print('-'*48)
    print(f'{"合并(现状)":18s} {st.mean(irs_m):9.4f} {st.median(irs_m):8.4f} {max(irs_m):8.4f}')
    print(f'{"逐层(DeepInfer)":18s} {st.mean(irs_l):9.4f} {st.median(irs_l):8.4f} {max(irs_l):8.4f}')
    print()
    gain = 100*(st.mean(irs_m)-st.mean(irs_l))/st.mean(irs_m)
    print(f'★ 逐层放置相对合并的改善: {gain:+.2f}%')

    # 逐层放置下每层的副本数是否真的不同
    print()
    print(f'合并: 一份 plan，副本数 max={int(cnt_m.max())} 的专家数={int((cnt_m==cnt_m.max()).sum())}')
    per_layer_maxcopy = [int(cnt_l[l].max()) for l in range(L)]
    print(f'逐层: 每层最大副本数 mean={st.mean(per_layer_maxcopy):.2f} '
          f'min={min(per_layer_maxcopy)} max={max(per_layer_maxcopy)}')
    same = all(plan_l[0].tolist() == plan_l[l].tolist() for l in range(L))
    print(f'逐层: 40 层 placement {"全部相同(异常)" if same else "各不相同(符合 DeepInfer)"}')


if __name__ == '__main__':
    main()
