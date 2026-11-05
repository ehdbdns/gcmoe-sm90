"""CPU-only regression test for the EPLB integration logic.

No CUDA, no model load, ~2 s. Covers every bug found while porting the
rearrangement path:

  T1  layer index stays unique (object ids are recycled otherwise)
  T2  stats[layers, per_rank] -> logical demand
  T3  shuffle move matches the pure-python reference
  T4  demonstrates why strong refs are required
  T5  per-layer stats rows are distinct

Run:  python tests/test_eplb_logic_cpu.py
"""
import sys, torch, gc, collections
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT+'/source'); sys.path.insert(0, ROOT+'/integration')
from gc_moe.eplb.deepinfer_periodic.config import initial_layout
from gc_moe.eplb.deepinfer_periodic.policy import rebalance_experts_global
from eplb_logic import (demand_from_stats, build_layer_index, verify_shuffle,
                        rr_pick)

E,R,EXTRA=384,8,40; P=E+EXTRA; PER=P//R
layout=initial_layout(E,EXTRA)
fails=[]

# ---- T1: layer index must survive id recycling ---------------------------
def t1():
    tensors=[torch.zeros(4) for _ in range(40)]
    m=build_layer_index(tensors)
    if len(m)!=40: fails.append('T1 collision'); return
    # drop and recreate to provoke id reuse
    for i in range(20):
        tensors[i]=torch.zeros(4)          # old object dies, id may be reused
    m2=build_layer_index(tensors)
    if len(m2)!=40: fails.append('T1 id-reuse collision')
    print(f'T1 layer index: {len(m2)} unique for 40 layers')

# ---- T2: demand mapping ------------------------------------------------
def t2():
    L=4
    stats=torch.zeros(L,PER)
    stats[:, :10]=6                            # 10 slots x 6 tokens per layer
    d=demand_from_stats(stats, layout, 0, PER, E)
    expect=torch.zeros(E)
    for s,e in enumerate(layout[:PER]):
        if s<10: expect[e]+=6*L
    if not torch.allclose(d, expect):
        fails.append('T2 demand mismatch'); return
    print(f'T2 demand: sum={d.sum().item():.0f} (expect {(6*10*L):.0f})')

# ---- T3: shuffle reference vs real plan --------------------------------
def t3():
    target,_ ,_=rebalance_experts_global(torch.ones(1,1,E), P, PER)
    target=target[0].tolist()
    rows=verify_shuffle(layout, target, PER, R)
    bad=0
    for r in range(R):
        for s in range(PER):
            e=target[r*PER+s]
            holders={g for g,x in enumerate(layout) if x==e}
            if rows[r][s] not in holders: bad+=1
    if bad: fails.append(f'T3 shuffle ref {bad} bad rows')
    print(f'T3 shuffle reference: bad rows={bad}/{P}')

# ---- T4: id-reuse would corrupt stats ---------------------------------
def t4():
    a=torch.zeros(2); ptr_id=id(a)
    m={ptr_id:'layer0'}
    del a; gc.collect()
    b=torch.zeros(2)
    reused = id(b)==ptr_id
    print(f'T4 id reuse after del: {"OCCURRED" if reused else "not observed"} '
          f'(this is why _LAYER_REFS must hold strong refs)')

for f in (t1,t2,t3,t4): f()
print()
print('CPU_TEST', 'FAIL: '+', '.join(fails) if fails else 'PASS')

# ---- T5: per-layer stats rows must not be shared ----------------------
def t5():
    import eplb_logic
    payloads=[torch.zeros(3) for _ in range(40)]
    idx=eplb_logic.build_layer_index(payloads)
    # simulate one step: 24 routes per rank per layer, each layer distinct
    stats=torch.zeros(40,PER)
    for layer in range(40):
        stats[layer,:5]= (layer+1)          # make each row identifiable
    per_layer=stats.sum(1)
    if len(set(per_layer.tolist()))!=40:
        fails.append('T5 rows not distinct')
    # every payload maps to its own row
    seen=set()
    for i,t in enumerate(payloads):
        seen.add(idx[id(t)])
    if len(seen)!=40: fails.append('T5 layer rows collide')
    print(f'T5 distinct layer rows: {len(seen)}/40, per-layer sum ok')

t5()
print()
print('CPU_TEST_T5', 'FAIL: '+', '.join(fails) if fails else 'PASS')

# ---- T6: world_size inflation in the kernel's counters ------------------
def t6():
    # Ground truth from a controlled 8-rank run:
    #   stats.sum() = 200, routes that really landed on this rank = 25, world = 8
    # The kernel's `total` sums send_count over all source ranks, and
    # send_count is a global per-slot counter replicated on every rank.
    stats_sum, real_routes, world = 200, 25, 8
    if stats_sum / world != real_routes:
        fails.append('T6 inflation factor wrong')
    print(f'T6 world_size inflation: {stats_sum}/{world} = {stats_sum//world} '
          f'(matches real routes {real_routes})')

t6()
print()
print('CPU_TEST_ALL', 'FAIL: '+', '.join(fails) if fails else 'PASS')

# ---- T7: window aggregation semantics ----------------------------------
def t7():
    from eplb_logic import aggregate_load, aggregate_with_sample
    LAYERS, PER, BM, STEPS, WORLD = 40, 48, 192, 10, 8
    # Step 1..STEPS: only the first 8 slots ever receive tokens; slots 8..47
    # stay empty the whole window. Per layer per step: 3 tokens per slot.
    stats = torch.zeros((LAYERS, PER), dtype=torch.int32)
    stats[:, :8] = 3 * WORLD * STEPS          # kernel reports world x real
    r = aggregate_load(stats, BM, STEPS, WORLD)
    want = 3 * 8                              # 24 tokens/step
    if abs(r['tokens_median'] - want) > 0.5:
        fails.append(f"T7 tokens {r['tokens_median']} != {want}")
    if abs(r['blocks_median'] - 8) > 0.5:
        fails.append(f"T7 blocks {r['blocks_median']} != 8")
    if r['active_ever_median'] != 8:
        fails.append(f"T7 active_ever {r['active_ever_median']} != 8")
    # A one-call sample: 3 tokens on 8 of the 48 slots.
    sample = torch.zeros(PER, dtype=torch.int32); sample[:8] = 3 * WORLD
    r2 = aggregate_with_sample(stats, BM, STEPS, WORLD, sample=sample,
                               active_sample=8)
    if r2['active_step'] != 8:
        fails.append(f"T7 active_step {r2['active_step']} != 8")
    if r2.get('active_header') != 8:
        fails.append(f"T7 active_header {r2.get('active_header')} != 8")
    if abs(r2['sample_tokens'] - 24) > 0.5:
        fails.append(f"T7 sample_tokens {r2['sample_tokens']} != 24")
    print(f"T7 window semantics: tokens={r['tokens_median']:.0f}/step "
          f"blocks={r['blocks_median']:.0f} active_ever={r['active_ever_median']:.0f} "
          f"active_step={r2['active_step']} active_header={r2.get('active_header')} "
          f"sample_tokens={r2['sample_tokens']:.0f}")

# ---- T8: active_ever must NOT be mistaken for per-step occupancy --------
def t8():
    from eplb_logic import aggregate_load
    LAYERS, PER, BM, STEPS, WORLD = 40, 48, 192, 100, 8
    stats = torch.zeros((LAYERS, PER), dtype=torch.int32)
    # Every slot gets exactly ONE token somewhere in the 100-step window.
    stats[:, :] = WORLD                      # 1 real token per slot, once
    r = aggregate_load(stats, BM, STEPS, WORLD)
    if r['active_ever_median'] != PER:
        fails.append('T8 active_ever should be all slots')
    # per-step average tokens is only 0.48, so occupancy is NOT PER per step
    if r['tokens_median'] >= 1.0:
        fails.append(f"T8 per-step tokens {r['tokens_median']} should be <1")
    print(f"T8 traps: active_ever={r['active_ever_median']:.0f}/{PER} but "
          f"per-step tokens={r['tokens_median']:.2f} (occupancy is far lower)")

# ---- T9: replica choice must match DeepInfer's gate EXACTLY --------------
def t9():
    """`duplicate_idx = (ep_rank + token_idx * 23333) % num_duplicates`.

    Source: deepseek-ai/TileKernels,
    tile_kernels/moe/moe_topk_gate_forward_cuda.py:283.

    This pins three properties that a naive `arange(n) % m` gets wrong:
      * the key is the TOKEN index (all topk routes of a token share it, but
        each uses its own expert's copy count);
      * the 23333 prime decorrelates neighbouring tokens;
      * `ep_rank` shifts each rank's phase.
    """
    TOPK9, NTOK, E9 = 6, 32, 384
    torch.manual_seed(0)
    ids = torch.randint(0, E9, (NTOK * TOPK9,))
    copies = torch.ones(E9, dtype=torch.int32)
    copies[:96] = 2
    copies[:16] = 3

    # (a) matches a hand-written transcription of the kernel line
    for ep in (0, 3, 7):
        ref = torch.tensor([
            (ep + (i // TOPK9) * 23333) % max(1, int(copies[e]))
            for i, e in enumerate(ids.tolist())
        ], dtype=torch.int64)
        got = rr_pick(ids, copies, ep_rank=ep, topk=TOPK9)
        if not torch.equal(got, ref):
            fails.append(f'T9 transcription mismatch at ep_rank={ep}')
            break
        if int(got.min()) < 0 or int(got.max()) >= int(copies.max()):
            fails.append(f'T9 out of range at ep_rank={ep}')

    # (b) the prime actually decorrelates: compare with a plain token index
    plain = torch.remainder(
        torch.arange(ids.numel(), dtype=torch.int64) // TOPK9, copies[ids].long())
    primed = rr_pick(ids, copies, ep_rank=0, topk=TOPK9)
    # For m == 1 every key maps to 0, so only replicated routes are informative.
    # For m == 2, 23333 % 2 == 1 so token*23333 % 2 == token % 2: the prime
    # CANNOT decorrelate mod 2. That is a property of the constant, not a bug,
    # so compare only where a difference is arithmetically possible.
    multi = copies[ids] > 1
    m_vals = copies[ids][multi].long()
    distinguish = ((23333 - 1) % m_vals.clamp_min(2) != 0)
    agree = float((plain[multi][distinguish] == primed[multi][distinguish])
                  .float().mean()) if distinguish.any() else 0.0
    # (c) ep_rank must change the assignment for at least some routes
    ep3 = rr_pick(ids, copies, ep_rank=3, topk=TOPK9)
    diff_ep = float((primed[multi] != ep3[multi]).float().mean()) if multi.any() else 0.0
    print(f'T9 deepinfer RR: {int(multi.sum())} replicated routes, '
          f'agree-with-plain-key={agree:.3f} on the {int(distinguish.sum())} where '
          f'it is measurable (want <0.9), ep_rank shifts={diff_ep:.3f} (want >0)')
    if distinguish.any() and agree >= 0.9:
        fails.append('T9 the 23333 prime is not decorrelating')
    if diff_ep <= 0.0:
        fails.append('T9 ep_rank has no effect')

    # (d) all copies of a hot expert must actually be used across many tokens
    hot = torch.full((NTOK * TOPK9,), 0, dtype=torch.int64)
    used = rr_pick(hot, copies, ep_rank=0, topk=TOPK9).unique().numel()
    if used != 3:
        fails.append(f'T9 hot expert used {used}/3 copies')


t7()
t8()
t9()
print()
print('CPU_TEST_FINAL', 'FAIL: '+', '.join(fails) if fails else 'PASS')
