"""CPU sweep: how do EPLB's gains scale with batch size?

Run:  PYTHONPATH=$PWD/source:$PWD/integration python scripts/eplb_batch_sweep_cpu.py
No GPU, no model, ~8 s.

Three placements of the SAME 384 logical experts, on EP8:

  vanilla : P=384, no copies, slot s serves expert s (rank = s//48)
  init    : P=480, the untuned `initial_layout` (96 extra copies appended)
  eplb    : P=480, DeepInfer `rebalance_experts_global` on measured demand

Routing model calibrated to the measured decode run: 32 tokens/rank/layer,
topk 6, and a popularity law whose distinct-expert count matches the observed
~150 per layer (Zipf exponent 0.35).

Distribution uses DeepInfer's gate line for the replicated arms:
    duplicate_idx = (ep_rank + token_idx * 23333) % copies[e]

The point of the sweep is the RELATIVE gain vs batch size, and where it lands
relative to the Poisson floor (the part that is physically optimisable).
"""
import torch
from gc_moe.eplb.deepinfer_periodic.config import initial_layout
from gc_moe.eplb.deepinfer_periodic.policy import rebalance_experts_global

E, R, TOPK, LAYERS, PRIME = 384, 8, 6, 40, 23333
P_VAN, P_EPLB = E, 416   # 4 redundant slots per rank (GC_MOE_EPLB_EXTRA_PER_RANK=4)


def popularity(seed=0, zipf=0.15):
    g = torch.Generator().manual_seed(seed)
    w = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** zipf
    w = w[torch.randperm(E, generator=g)]
    return (w / w.sum()).float()


def sample(ids_pop, tokens, gen):
    r = torch.rand(tokens, E, generator=gen) ** (1.0 / ids_pop)
    return r.topk(TOPK, dim=-1).indices


class Placement:
    def __init__(self, plan):
        self.plan = torch.as_tensor(plan, dtype=torch.long)
        self.P = self.plan.numel()
        self.per = self.P // R
        self.copies = torch.bincount(self.plan, minlength=E)
        width = int(self.copies.max()) + 1
        self.table = torch.full((E, width), -1, dtype=torch.long)
        seen = torch.zeros(E, dtype=torch.long)
        for slot, e in enumerate(self.plan.tolist()):
            self.table[e, seen[e]] = slot
            seen[e] += 1
        self.rank_of = torch.arange(self.P) // self.per

    def slots(self, ids, ep_rank):
        n, tk = ids.shape
        tok = torch.arange(n).unsqueeze(1).expand(n, tk)
        dup = torch.remainder(ep_rank + tok * PRIME,
                             self.copies[ids].clamp_min(1))
        return torch.gather(self.table[ids], -1, dup.unsqueeze(-1)).squeeze(-1)


def run(batch, pop, place, rounds=60, seed=7):
    g = torch.Generator().manual_seed(seed)
    tot = torch.zeros(R, dtype=torch.float64)
    for _ in range(rounds):
        for r in range(R):
            ids = sample(pop, batch, g)
            sl = place.slots(ids, r)
            tot += torch.bincount(place.rank_of[sl.reshape(-1)],
                                  minlength=R).double()
    return tot


def main():
    pop = popularity()
    # demand for the placement decision, measured over a window of layers
    g = torch.Generator().manual_seed(11)
    demand = torch.zeros(E)
    for _ in range(LAYERS):
        demand += torch.bincount(sample(pop, 64, g).reshape(-1),
                                 minlength=E).float()
    bal, _, cnt = rebalance_experts_global((demand / LAYERS).view(1, 1, E),
                                          P_EPLB, P_EPLB // R)
    arms = {
        'van': Placement(list(range(E))),
        'init': Placement(initial_layout(E, P_EPLB - E)),
        'eplb': Placement(bal[0]),
    }
    label = {'van': 'vanilla(P=384)', 'init': 'init(P=480)', 'eplb': 'eplb(P=480)'}
    print(f'E={E} EP={R} TOPK={TOPK} layers={LAYERS}  '
          f'eplb: max_copies={int(cnt.max())} replicated={int((cnt>1).sum())}')
    print('routes = tokens x topk per rank per layer; '
          'IR = max(rank)/mean(rank) over the whole window\n')
    hdr = (f'{"tok/rank":>9} {"routes":>7} | {"vanilla":>8} {"init":>8} '
           f'{"eplb":>8} | {"eplb vs init":>13} {"eplb vs van":>12} | {"poisson":>8}')
    print(hdr); print('-' * len(hdr))
    for batch in (4, 8, 16, 32, 64, 128, 256, 512):
        res = {k: run(batch, pop, v) for k, v in arms.items()}
        ir = {k: float(v.max() / v.mean()) for k, v in res.items()}
        mean = float(res['eplb'].mean())
        floor = 1.0 + 1.4 * (mean ** 0.5) / mean
        g1 = (ir['init'] - ir['eplb']) / ir['init'] * 100
        g2 = (ir['van'] - ir['eplb']) / ir['van'] * 100
        print(f'{batch:>9} {batch*TOPK:>7} | {ir["van"]:8.4f} '
              f'{ir["init"]:8.4f} {ir["eplb"]:8.4f} | '
              f'{g1:+12.2f}% {g2:+11.2f}% | {floor:8.4f}')


if __name__ == '__main__':
    main()
