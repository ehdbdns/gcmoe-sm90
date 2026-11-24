"""Real vLLM inference: real prefill, then a *separate* decode window.

Exactly the requested protocol:
  1. load DeepSeek-V4.1-Flash (8x H20, GC MoE backend);
  2. submit ``seqs_per_rank * 8`` NAS prompts and run a REAL prefill; vLLM keeps
     the KV cache. This window is **not** timed;
  3. decode 100 steps on those same sequences -- KV cache reused, no
     re-prefill. EPLB learns the expert distribution here, then the plan is
     rearranged once and frozen;
  4. decode 100 more steps and time **only** those.

Why ``LLMEngine.step()`` instead of ``llm.generate()``: ``generate()``
re-prefills on every call (``enable_prefix_caching=False``), so a second call
would time prefill again. ``step()`` advances the same running sequences, and
the prefill/decode split is explicit: every sequence's *first* generated token
belongs to the prefill window, everything after it is decode.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'patches'))
sys.path.insert(0, os.path.join(_HERE, '..', 'integration'))
import vllm_warmup_compat  # noqa: E402

vllm_warmup_compat.install()
import deepep_compare  # noqa: E402,F401  (registers the gc_moe backend)


DATASET = os.environ.get(
    'DATASET',
    '/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/'
    'dsv4-flash_online_input_ids_26-10-08_1.jsonl')


def _read_prompts(path, lo, hi):
    """All prompts of `path` whose length is in [lo, hi], in file order."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            ids = json.loads(line).get('input_ids') or []
            if lo <= len(ids) <= hi:
                out.append(ids)
    return out


def _stride_sample(pool, n):
    """Take `n` entries spread EVENLY over `pool` (which must be length-sorted).

    Why not the first `n`: file order correlates with arrival time, and the
    online log has bursts, so a prefix is not a fair length sample. A stride of
    `len(pool)/n` walks the whole length distribution and yields small, medium
    and large prompts in the same proportion as the corpus.
    """
    if n >= len(pool):
        return list(pool)
    step = len(pool) / n
    return [pool[min(len(pool) - 1, int(i * step))] for i in range(n)]


def load_prompts(n, lo, hi):
    """Pick `n` prompts with an even length spread, from one or more files.

    [DATA-MIX] Two requirements now that warm+measure is 1000+1000 steps:

    1. the length distribution must be REPRESENTATIVE (small AND large
       prompts), not "the first N that fit `[lo, hi]`; and
    2. the whole batch must fit the KV budget. Every stream keeps growing by
       one token per decode step and vLLM reserves `max_model_len` slots for
       it, so the requirement is

           sum_i (len_i + warm + measure)  <=  KV_BUDGET

       With 1000+1000 that is what stops BS=256 from fitting 8192-token
       prompts -- not the sequence length itself.

    Implementation: per file, sort that file's usable prompts by length, walk
    the files round-robin taking one prompt at a time at an even stride, and
    stop early if the next pick would break the budget. The result keeps the
    corpus's length mix while guaranteeing the run fits.
    """
    files = [p.strip() for p in DATASET.split(',') if p.strip()]
    pools = []
    for path in files:
        pool = _read_prompts(path, lo, hi)
        pool.sort(key=len)
        pools.append(pool)
    if not all(pools):
        raise SystemExit(f'no prompts in [{lo},{hi}] for {files}')

    # even stride per file, round-robin across files
    take = max(1, n // len(pools))
    queues = []
    for pool in pools:
        queues.append(list(_stride_sample(pool, min(take, len(pool)))))
    # leftover slots come from the files that still have entries
    got = sum(len(q) for q in queues)
    if got < n:
        for pool in pools:
            extra = min(n - got, len(pool))
            if extra > 0:
                queues[pools.index(pool)].extend(_stride_sample(pool, extra))
                got += extra
            if got >= n:
                break

    budget = int(os.environ.get('GC_KV_BUDGET', '0') or 0)
    # [KV] Per-stream KV overhead, in tokens. MUST be the same quantity the
    # KV reservation uses (`max_model_len = max_len + decode_budget + margin
    # + 64`), otherwise this selector admits prompts the engine cannot fit.
    # `decode_budget` depends on the SELECTED prompts, so it is iterated: see
    # `_kv_ok`.  Passed in via GC_KV_OVERHEAD by the caller once known.
    per_step = int(os.environ.get('GC_KV_OVERHEAD', '0') or 0)
    picked = []
    cursor = [0] * len(queues)
    while len(picked) < n:
        progressed = False
        for qi, q in enumerate(queues):
            if len(picked) >= n:
                break
            if cursor[qi] >= len(q):
                continue
            cand = q[cursor[qi]]
            if budget > 0:
                need = sum(len(p) + per_step for p in picked) + len(cand) + per_step
                if need > budget:
                    cursor[qi] += 1        # try a shorter one from this file
                    progressed = True
                    continue
            picked.append(cand)
            cursor[qi] += 1
            progressed = True
        if not progressed:
            break

    if len(picked) < n:
        raise SystemExit(
            f'only {len(picked)} prompts fit (need {n}, len range [{lo},{hi}], '
            f'KV budget {budget or "unlimited"}, steps {per_step})')

    lens = sorted(len(p) for p in picked)
    print(f'[data] picked {len(picked)} prompts, len '
          f'{lens[0]}/{lens[len(lens)//2]}/{lens[-1]} (min/med/max), '
          f'KV need {sum(l + per_step for l in lens)}'
          + (f' / budget {budget}' if budget else ''), flush=True)
    return picked


def select_decode_prompts(n, lo, hi):
    reference = os.environ.get('GC_REPEAT_BS512_PROMPT_SHA256')
    if not reference:
        return load_prompts(n, lo, hi)
    if n not in (1024, 2048):
        raise ValueError('BS512 repetition requires 1024 or 2048 requests')
    base = load_prompts(512, lo, hi)
    digest = hashlib.sha256(json.dumps(base, separators=(',', ':')).encode()).hexdigest()
    if digest != reference:
        raise ValueError(f'BS512 prompt reference mismatch: {digest}')
    prompts = base * (n // 512)
    overhead = int(os.environ.get('GC_KV_OVERHEAD', 0))
    budget = int(os.environ.get('GC_KV_BUDGET', 0))
    if budget and sum(len(p) + overhead for p in prompts) > budget:
        raise ValueError('Repeated BS512 workload exceeds the configured KV token budget')
    return prompts


def _compilation_config():
    from vllm.config.compilation import CUDAGraphMode
    mode = os.environ.get('CUDAGRAPH_MODE', 'FULL_DECODE_ONLY')
    if mode == 'default':
        return None
    config = {'cudagraph_mode': getattr(CUDAGraphMode, mode)}
    batch = int(os.environ.get('GC_MOE_MATRIX_BS', '0'))
    if os.environ.get('GC_MOE_MATRIX_ARM') and (batch > 256 or os.getenv('SM90_PAIRED_SMALL') == '1'):
        # Pure decode has exactly this shape. Avoid retaining dozens of
        # capture pools or falling back to eager above the default graph cap.
        config.update(cudagraph_capture_sizes=[batch], max_cudagraph_capture_size=batch)
    return config


def _irstep_reduce(dumps):
    """# [IR-STEP] avg over (step, layer) of max_rank/mean_rank.

    `dumps` is one dict per RANK, each holding its own `[layers][steps]` token
    table produced by the dispatch role. For every (layer, step) pair we take

        IR = max_r tok[r][layer][step] / mean_r tok[r][layer][step]

    and average over all pairs that every rank actually filled.

    A cell is only counted when ALL ranks have a non-zero value for it: a zero
    means the dispatch role never wrote there (outside the window, or a step
    where that rank sent nothing), and treating it as a real zero would
    manufacture imbalance.

    A POISSON NULL BASELINE is returned alongside. One layer of one step
    carries only `tokens/8` routes per rank, so even a perfectly balanced
    placement gives IR > 1 from sampling noise alone:
        sigma = sqrt(n),  E[max of 8 draws] ~ mean + 1.42*sigma
    Without it, an IR of 1.03 looks like imbalance when it may be noise.
    """
    import math
    import statistics

    ok = [d for d in (dumps or []) if isinstance(d, dict) and d.get('ok')]
    if len(ok) < 2:
        return {'ok': False, 'reason': f'need >=2 ranks, got {len(ok)}'}
    n_rank = len(ok)
    layers = min(int(d['layers']) for d in ok)
    steps = min(int(d['steps']) for d in ok)
    # [IR-STEP] The kernel writes `stats`-scale numbers, and `stats` is
    # inflated by `world`: `send_count` is replicated on every rank and the
    # pull sums ALL peers' counters, so a cell holds world x this rank's real
    # share. `ir_end` divides it out ("world_size x inflation ... so it is
    # divided out before the gather"). The max/mean RATIO is scale-invariant,
    # so IR itself is unaffected -- but the Poisson baseline uses the ABSOLUTE
    # count, and with the inflation left in it is computed from 192 instead of
    # 24, understating the noise floor (1.10 instead of 1.29) and therefore
    # OVERSTATING `excess`. Divide here.
    world = n_rank
    has_mb = all(isinstance(d.get('rows_mb'), list) for d in ok)
    has_ae = all(isinstance(d.get('rows_ae'), list) for d in ok)

    irs, per_layer, nulls, per_step_tot = [], {}, [], []
    mb_irs, mb_total, ae_total = [], [], []
    # [LAYER-DETAIL] One (layer, step) slice with EVERY per-rank number side by
    # side, so "why is this step unbalanced" can be answered directly:
    #   tokens[r], m_blocks[r], active_slots[r]  for the same epoch.
    # We keep the FIRST fully-filled cell (deterministic, and the earliest one
    # is closest to the start of the measured window).
    layer_detail = None
    # [MB-PEAK] Peak M blocks over the 8 RANKS for each (step, layer) pair.
    # `rows_mb` already holds this rank's real block count, so the peak is the
    # max across ranks -- the number that decides how long that (step, layer)
    # takes on the slowest rank. Reported as mean / p90 / max over all pairs.
    mb_peak = []
    for k in range(steps):
        step_total = 0.0
        any_cell = False
        for lay in range(layers):
            vals = [d['rows'][lay][k] / world for d in ok]
            # A zero is a REAL count, not a missing value: one layer handles
            # the whole batch (every token picks `topk` experts in every
            # layer), so at BS=32 a rank averages 32*6/8 = 24 tokens per layer
            # per step, and a particular rank/expert pair really can be zero.
            # Only cells where EVERY rank reads zero (never written) are
            # dropped -- those are steps outside the window.
            if max(vals) <= 0:
                continue
            m = sum(vals) / n_rank
            ir = max(vals) / m if m > 0 else 0.0
            irs.append(ir)
            per_layer.setdefault(lay, []).append(ir)
            nulls.append(1.0 + 1.42 * math.sqrt(max(m, 1e-9)) / m)
            step_total += m
            any_cell = True
            # [IR-STEP] M blocks and active slots for this same (step, layer).
            # `mb_tbl` / `ae_tbl` are already per-rank counts (atoms of
            # ceil(owned/block_m) and of 1-per-nonempty-slot), so only the
            # TOKEN table needs the world division above.
            if has_mb:
                mvals = [d['rows_mb'][lay][k] for d in ok]
                mm = sum(mvals) / n_rank
                mb_irs.append(max(mvals) / mm if mm > 0 else 0.0)
                mb_total.append(sum(mvals))       # ★ cross-rank SUM
                mb_peak.append(max(mvals))        # ★ cross-rank PEAK
            if has_ae:
                avals = [d['rows_ae'][lay][k] for d in ok]
                ae_total.append(sum(avals))       # ★ cross-rank SUM
            if layer_detail is None:
                layer_detail = {
                    'layer': int(lay), 'step': int(k),
                    'tokens': [round(v, 2) for v in vals],
                    'm_blocks': ([int(d['rows_mb'][lay][k]) for d in ok]
                                 if has_mb else None),
                    'active_slots': ([int(d['rows_ae'][lay][k]) for d in ok]
                                     if has_ae else None),
                    'token_ir': round(max(vals) / (sum(vals) / n_rank), 4)
                                if sum(vals) > 0 else None,
                }
        if any_cell:
            per_step_tot.append(step_total)

    if not irs:
        return {'ok': False, 'reason': 'no fully-filled (layer, step) cell'}

    def _q(xs, p):
        xs = sorted(xs)
        return xs[max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))]

    # [IR-STEP] Two COARSER aggregations, reported next to the fine-grained
    # one because they answer different questions. At BS=32 a rank sees
    # 32*6/8 = 24 tokens per layer per step, so the per-(layer, step) ratio has
    # ~20% Poisson noise; summing first shrinks it:
    #
    #   ir_step_alllayers : sum the 40 LAYERS first, then max/mean over ranks
    #                       -> one value per step (960 tokens/rank -> ~3%).
    #   ir_layer_window   : sum all 1000 STEPS first, then max/mean per layer
    #                       -> one value per layer (24000 tokens/rank -> ~0.6%).
    #
    # These are NOT the headline number; they exist to show that the
    # fine-grained result survives aggregation (if placement is balanced, all
    # three collapse toward 1.0 + null).
    n_steps = min(len(d['rows'][0]) for d in ok)
    step_all, layer_win, null_step = [], [], []
    for k in range(n_steps):
        vals = [sum(d['rows'][lay][k] for lay in range(layers)) / world
                for d in ok]
        if max(vals) <= 0:
            continue
        m = sum(vals) / n_rank
        step_all.append(max(vals) / m if m > 0 else 0.0)
        null_step.append(1.0 + 1.42 * math.sqrt(max(m, 1e-9)) / m)
    for lay in range(layers):
        vals = [sum(d['rows'][lay]) / world for d in ok]
        if max(vals) <= 0:
            continue
        m = sum(vals) / n_rank
        layer_win.append(max(vals) / m if m > 0 else 0.0)

    def _mean(xs):
        return statistics.mean(xs) if xs else -1.0


    return {
        'ok': True,
        'method': 'in-kernel (layer, step) table; avg of max_rank/mean_rank',
        'layer_detail': layer_detail,
        # [LAYER-TABLE] Keep the per-(layer, step) tables of ALL ranks, so any
        # single layer can be re-derived later and cross-checked against the
        # torch trace. Without this the reduction below throws the raw arrays
        # away and only an averaged IR survives -- that is why an earlier
        # report paired a trace layer with a DIFFERENT layer's token counts.
        'tables': {
            'epoch0': int(ok[0].get('epoch0', 0)),
            'layers': layers, 'steps': steps,
            'tokens': [[ok[r]['rows'][lay][k] // world
                        for r in range(n_rank)] for lay in range(layers)
                       for k in range(steps)],
            'm_blocks': ([[ok[r]['rows_mb'][lay][k] for r in range(n_rank)]
                          for lay in range(layers) for k in range(steps)]
                         if has_mb else None),
            'active': ([[ok[r]['rows_ae'][lay][k] for r in range(n_rank)]
                        for lay in range(layers) for k in range(steps)]
                       if has_ae else None),
            # [IR-STEP-EPOCH] kernel-written epoch for every (layer, step) cell.
            # This is the AUTHORITATIVE key for joining with pull_tl: the host
            # `epoch0` above can be stale under CUDA-graph replay.
            'epochs': ([[ok[r]['rows_epoch'][lay][k] for r in range(n_rank)]
                        for lay in range(layers) for k in range(steps)]
                       if all(d.get('rows_epoch') for d in ok) else None),
        },
        'ir_step_layer_mean': statistics.mean(irs),
        'ir_step_layer_p50': _q(irs, 0.50),
        'ir_step_layer_p90': _q(irs, 0.90),
        'ir_step_layer_p99': _q(irs, 0.99),
        'ir_step_layer_max': max(irs),
        'ir_poisson_null_mean': statistics.mean(nulls),
        'ir_excess_over_null': statistics.mean(irs) - statistics.mean(nulls),
        'n_pairs': len(irs),
        'n_ranks': n_rank,
        'n_layers': layers,
        'n_steps_sampled': len(per_step_tot),
        # per_step_tot is ALREADY per-rank (see `m` above), summed over layers.
        # Dividing by layers gives the per-(layer, step) count a single rank
        # sees; at BS=32 that must be 32*6/8 = 24.
        'routes_per_rank_per_layer': (statistics.mean(per_step_tot) / layers
                                      if per_step_tot else -1.0),
        'per_layer_ir_mean': {str(k): statistics.mean(v)
                              for k, v in sorted(per_layer.items())},
        # coarser, statistically usable aggregates
        'ir_step_alllayers_mean': _mean(step_all),
        'ir_step_alllayers_null': _mean(null_step),
        'ir_layer_window_mean': _mean(layer_win),
        'ir_layer_window_p90': _q(layer_win, 0.90),
        # [IR-STEP] the two extra headline numbers, both averaged over ALL
        # (step, layer) pairs:
        #   mb_ir_mean      = mean of max_r mb[r] / mean_r mb[r]
        #   mb_total_mean   = mean of SUM_r mb[r]      (cross-rank, whole work)
        #   ae_total_mean   = mean of SUM_r ae[r]      (cross-rank)
        'mb_ir_mean': _mean(mb_irs),
        'mb_ir_p90': _q(mb_irs, 0.90),
        'mb_total_mean': _mean(mb_total),
        'mb_peak_mean': _mean(mb_peak),
        'mb_peak_p90': _q(mb_peak, 0.90),
        'mb_peak_max': max(mb_peak) if mb_peak else -1.0,
        'ae_total_mean': _mean(ae_total),
        'n_pairs_mb': len(mb_total),
        'n_steps_alllayers': len(step_all),
        'n_layers_window': len(layer_win),
        # Average tokens a rank receives in one layer of one step
        # (= routes_per_rank_per_layer; kept under the descriptive name too).
        'tokens_per_rank_per_layer_per_step': (
            statistics.mean(per_step_tot) / layers
            if per_step_tot else -1.0),
    }




def time_ir(eco_trace, expect_ms=None):
    """[TIME-IR] Per-(step,layer) RANK-TIME imbalance, then averaged.

    WHAT COUNTS AS "A RANK'S TIME". The combine kernel starts concurrently
    with compute, so it cannot be folded into the same span. This measures:

        rank MoE cost  = dispatch kernel entry  -> compute kernel exit
        combine cost   = combine kernel entry   -> combine kernel exit

    and reports an IR for each.

    WHY NOT the older `[9]-[4]` (pull start -> combine done): both of those
    points are CROSS-RANK BARRIERS -- every rank waits for every other -- so
    their difference came out identical to within 2-3 us on all 8 ranks and
    reported a meaningless IR of 1.0017. The stamps used here are
    per-rank: dispatch entry is the first thing the role-1 kernel does, and
    compute exit is the last CTA of THAT rank leaving, with no peer wait.

    Slots (see the layout's PULL-TL reserve):
        [10] dispatch entry   [11] compute exit (max over CTAs)
        [12] combine entry    [13] combine exit  (max over CTAs)
        [0]  epoch tag        (written with slot 10 so rows can be matched)

    A call is used only when all four stamps are non-zero and the span is
    plausible (0.1..20 ms); everything else is a stale ring row.
    """
    import statistics
    ranks = (eco_trace or {}).get('ranks') or []
    if len(ranks) < 2:
        return {'ok': False, 'why': f'need >=2 ranks, got {len(ranks)}'}

    def collect(idx_a, idx_b):
        per, tagged = [], False
        for r in ranks:
            d = {}
            for row in (r.get('pull_rows') or []):
                if len(row) < 14:
                    continue
                ep, t0, t1 = row[0], row[idx_a], row[idx_b]
                if t0 == 0 or t1 == 0:
                    continue
                span = t1 - t0
                if not (100 <= span <= 20000):
                    continue
                if ep:
                    tagged = True
                d[ep] = span
            per.append(d)
        if not tagged:
            return None, 'no epoch tag (pre-RANK-TS trace)'
        common = set(per[0])
        for d in per[1:]:
            common &= set(d)
        common.discard(0)
        if not common:
            return None, 'no shared epoch across ranks'
        eps = sorted(common)
        means = [statistics.mean(per[i][e] for i in range(len(per))) for e in eps]
        irs = [max(per[i][e] for i in range(len(per))) / m
               for e, m in zip(eps, means)]
        rank_mean = [statistics.mean(per[i][e] for e in eps)
                     for i in range(len(per))]
        return dict(n=len(eps), ir=statistics.mean(irs),
                    p50=statistics.median(irs),
                    p90=sorted(irs)[int(0.9 * (len(irs) - 1))],
                    mean_us=statistics.mean(means),
                    rank_max_us=max(rank_mean)), None

    moe, err = collect(10, 11)
    if moe is None:
        return {'ok': False, 'why': err}
    comb, _ = collect(12, 13) or (None, None)

    if expect_ms:
        want = expect_ms * 1000.0 / max(1, int(os.environ.get('GC_MOE_LAYERS', '40')))
        # a MoE layer (dispatch->compute) is a large part of one step's layer;
        # allow a wide band, this only catches gross pairing errors
        if not (0.15 * want <= moe['mean_us'] <= 3.0 * want):
            return {'ok': False,
                    'why': f"mean MoE layer {moe['mean_us']:.0f}us implausible "
                           f"vs {want:.0f}us/step/layer"}

    out = {'ok': True, 'time_layers': moe['n'],
           'time_ir': moe['ir'], 'time_ir_p50': moe['p50'],
           'time_ir_p90': moe['p90'],
           'time_rank_mean_us': moe['mean_us'],
           'time_rank_max_us': moe['rank_max_us']}

    # ---- [PLAN-PRED] predicted vs measured per-rank RATIO ----------------
    # The only honest judge of a cost function: does the 8-rank time ratio it
    # predicts match the ratio that actually happened? A plan can be better on
    # a noisy end-to-end ms while modelling the rank cost worse.
    #
    # Predicted: T[r] that the plan's own `_eco_rank_time` produced, exported
    #            into eco_tl[epoch][14+r].
    # Measured : dispatch-entry -> compute-exit span per rank (the same
    #            quantity `moe` above averaged).
    # Both are divided by their own mean, so only the SHAPE is compared.
    pred_err = None
    try:
        # Each rank's row carries ITS OWN T[r] at column 14+r, so the vector is
        # assembled by reading rank r's row at 14+r. Rows are matched by the
        # epoch tag in column 0 (a stale ring row from an older call must not
        # be mixed in).
        # T[r] lives in rank r's own pull_tl row at column 14+r (see
        # _pull_tl_write_raw), so the vector is gathered across ranks.
        per = []
        # [PHASES] T[r] used to live at column 14+r, which collided with the
        # phase stamps at 15/16/17 ("first compute task" read back as a plan
        # unit count). It now lives at 24+r; derive the base from the row
        # width so an older trace still parses.
        rows0 = [row for row in (ranks[0].get('pull_rows') or []) if row and row[0]]
        tbase = 24 if (rows0 and len(rows0[0]) >= 24 + len(ranks)) else 14
        for r in ranks:
            d = {}
            for row in (r.get('pull_rows') or []):
                if len(row) < tbase + len(ranks) or not row[0]:
                    continue
                d[row[0]] = row
            per.append(d)
        common = set(per[0])
        for d in per[1:]:
            common &= set(d)
        if common:
            ep = sorted(common)[-1]
            T = [per[k][ep][tbase + k] for k in range(len(ranks))]
            # [PLAN-PRED-FIX] Take THIS epoch's measured span, not the mean of
            # every call in the ring buffer. Comparing one call's predicted T
            # against an average over 256 unrelated layers mixes two different
            # quantities and inflates the apparent error.
            meas = []
            # `ranks` holds dicts; `per` is positional, so iterate INDICES.
            for ri in range(len(ranks)):
                row = per[ri].get(ep)
                span = (row[11] - row[10]) if (row and row[10] and row[11]) else 0
                meas.append(span if 100 <= span <= 20000 else 0)
            if all(meas) and all(T):
                tm, mm = statistics.mean(T), statistics.mean(meas)
                pr = [t / tm for t in T]
                mr = [m / mm for m in meas]
                _nlay = int(os.environ.get('GC_MOE_LAYERS', '40'))
                pred_err = {
                    'epoch': ep, 'layer': ep % _nlay, 'pred_T': T,
                    'meas_us': [round(x) for x in meas],
                    'pred_ratio': [round(x, 4) for x in pr],
                    'meas_ratio': [round(x, 4) for x in mr],
                    'ratio_l1': round(sum(abs(a - b) for a, b in zip(pr, mr)), 4),
                    'ratio_maxerr': round(max(abs(a - b) for a, b in zip(pr, mr)), 4),
                }
            else:
                pred_err = {'error': f'T={T} meas={meas}'}
        else:
            pred_err = {'error': 'no common epoch in tl_rows'}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        pred_err = {'error': f'{type(exc).__name__}: {exc}'}
    if pred_err:
        out['plan_pred'] = pred_err
    if comb:
        out.update(comb_layers=comb['n'], comb_ir=comb['ir'],
                   comb_mean_us=comb['mean_us'])
    return out


def moe_real(eco_trace, max_layers=4):
    """[MOE-REAL] Per-rank TRUE MoE layer cost, straight from the fine trace.

    The end-to-end ms/step is dominated by attention/comm and carries ~0.5%
    run-to-run noise, which is larger than the differences we are trying to
    resolve between plan variants. The MoE layer itself is what a plan
    controls, so measure THAT: for each rank take one clean decode epoch and
    report the wall clock from its first to its last GEMM span.

    Returns per-layer averages over the sampled epochs:
      * `wall_max`   : max over ranks  (= the critical rank, what ms/step feels)
      * `wall_mean`  : mean over ranks
      * `ir_real`    : wall_max / wall_mean  (true rank imbalance)
      * `layers`     : how many epochs were averaged
    """
    ranks = (eco_trace or {}).get('ranks') or []
    if not ranks:
        return {'ok': False, 'why': 'no eco_trace ranks'}
    import collections, statistics
    per_rank = []
    for r in ranks:
        byep = collections.defaultdict(list)
        for x in (r.get('fine_rows') or []):
            if x[1] == 6:                       # type 6 = GEMM task begin
                byep[x[8]].append(x[0])         # x[8] = epoch
        per_rank.append(byep)
    common = set(per_rank[0])
    for p in per_rank[1:]:
        common &= set(p)
    # a real decode layer carries 1200..2600 GEMM spans; prefill layers do not
    common = {e for e in common if 1200 <= len(per_rank[0][e]) <= 2600}
    if not common:
        return {'ok': False, 'why': 'no common clean epoch'}
    eps = sorted(common)[:max_layers]
    walls, means, irs = [], [], []
    for ep in eps:
        w = []
        for p in per_rank:
            ts = sorted(p[ep])
            w.append(ts[-1] - ts[0])
        walls.append(max(w))
        means.append(statistics.mean(w))
        irs.append(max(w) / statistics.mean(w))
    return {'ok': True, 'layers': len(eps), 'epochs': eps,
            'wall_max': statistics.mean(walls),
            'wall_mean': statistics.mean(means),
            'ir_real': statistics.mean(irs),
            'per_epoch_wall_max': walls}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--seqs-per-rank', type=int, default=32)
    ap.add_argument('--prefill-lo', type=int, default=256)
    ap.add_argument('--prefill-hi', type=int, default=4096)
    ap.add_argument('--warm-steps', type=int, default=100)
    ap.add_argument('--measure-steps', type=int, default=100)
    # [TRACE] Optional torch profile over the measured window, so a 1000+1000
    # run can also yield a chrome trace of the post-rearrange layers.
    ap.add_argument('--profile-out', default=None,
                    help='if set, profile the measured window into this dir')
    ap.add_argument('--profile-steps', type=int, default=0,
                    help='profile only the first N measured steps (0 = all)')
    ap.add_argument('--profile-aligned-step', action='store_true',
                    help='capture exactly one post-rearrangement decode step and device rows')
    ap.add_argument('--fine-layers', type=int, nargs='+', default=[],
                    help='export fine rows for these zero-based layers of the aligned step')
    ap.add_argument('--timing-tail-steps', type=int, default=0,
                    help='time additional decode steps in the same process after the aligned trace')
    ap.add_argument('--long-cost-study', action='store_true',
                    help='one 1000-step EPLB statistics window and one 1000-step paired cost window')
    a = ap.parse_args()
    long_cost = a.long_cost_study
    matrix = os.environ.get('GC_MOE_MATRIX_ARM')
    if os.environ.get('GC_MOE_EPLB') == '1':
        os.environ['GC_MOE_EPLB_WINDOW'] = str(max(1, a.warm_steps))
    if long_cost:
        required = ('GC_MOE_LONG_COST', 'GC_MOE_EPLB', 'GC_MOE_ECO', 'GC_MOE_COST_STUDY',
                    'GC_MOE_ORDER_EXTRA', 'GC_MOE_STEP_PROFILE', 'GC_MOE_SM90_STREAM_M_BLOCKS')
        if matrix:
            required = tuple(k for k in required if k not in ('GC_MOE_EPLB', 'GC_MOE_ECO', 'GC_MOE_COST_STUDY'))
            arms = ('vanilla', 'random', 'rr', 'eco', 'eco_smr')
            if os.getenv('GC_MOE_MATRIX_SIZE') == '24':
                arms += ('eco_linear',)
                expected_mode = '0' if matrix == 'eco_linear' else '2'
                if os.getenv('GC_MOE_ECO_COSTFN') != expected_mode:
                    ap.error(f'{matrix} requires GC_MOE_ECO_COSTFN={expected_mode}')
            if matrix not in arms or int(os.environ['GC_MOE_MATRIX_BS']) != 8*a.seqs_per_rank:
                ap.error('invalid main table arm or batch')
        if (any(os.getenv(k) != '1' for k in required)
                or not os.getenv('GC_MOE_FIXED_DIR') or not os.getenv('GC_MOE_STRICT_STATE_DIR')
                or a.profile_out or a.profile_aligned_step or a.timing_tail_steps or a.fine_layers
                or os.getenv('GC_MOE_FINE_TRACE') != ('1' if os.getenv('GC_BS256_TRACE_DIR') else '0') or os.getenv('GC_MEASURE_FLUSH') != '0'
                or (a.seqs_per_rank not in ((4,8,16,32,64,128,256) if matrix else (32,)) or (a.warm_steps, a.measure_steps) != (1000,1000))):
            ap.error('long cost study requires the fixed, unprofiled 1000+1000 EP8 configuration')
    if a.timing_tail_steps < 0 or (a.timing_tail_steps and
            (not a.fine_layers or os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') != '0')):
        ap.error('the timing tail requires a fine aligned capture with an in-process EngineCore')

    ranks = int(os.environ.get('WORLD', '8'))
    aligned_layers = int(os.environ.get('GC_MOE_LAYERS', '40'))
    if a.fine_layers:
        if not a.profile_aligned_step or any(not 0 <= i < aligned_layers for i in a.fine_layers):
            ap.error('--fine-layers requires aligned profiling and layer indices inside the step')
        os.environ.update(GC_MOE_FINE_TRACE='1', GC_MOE_FINE_FLOOR='1',
                          GC_MOE_FINE_EPOCHS='1', GC_MOE_FINE_MIN_EPOCH='0')
    if a.profile_aligned_step:
        required = ('GC_MOE_EPLB', 'GC_MOE_ECO', 'GC_MOE_ECOPLB_PER_LAYER',
                    'GC_MOE_PULL_TRACE', 'GC_MOE_ECO_TRACE', 'GC_MOE_SHARED_IN_GC')
        if any(os.environ.get(name) != '1' for name in required):
            ap.error('aligned profile requires ' + ', '.join(f'{name}=1' for name in required))
        if (os.environ.get('GC_MOE_EPLB_NO_REARRANGE') == '1'
                or os.environ.get('GC_MOE_SLOT_PLACEMENT') == 'random'
                or int(os.environ.get('GC_MOE_SM90_SM_REUSE', '0') or 0) != 0):
            ap.error('aligned profile requires EPLB rearrangement and exactly five GC MoE roles')
        if (not a.profile_out or a.measure_steps != 1 or a.warm_steps < 1
                or ranks != 8 or not 1 <= aligned_layers <= 256):
            ap.error('aligned profile requires --profile-out, --measure-steps 1, warmup, '
                     '8 ranks and 1-256 MoE layers')
        os.environ['GC_MOE_STEP_PROFILE'] = '1'
    nseq = a.seqs_per_rank * ranks
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    chunk = int(os.environ.get('MAX_BATCHED', '8192'))

    # [KV-FIT] Two-pass selection. The per-stream KV cost is
    #     len(prompt) + decode_budget + margin + 64
    # and `decode_budget` itself depends on the prompts (slack = prefill
    # steps). Selecting with a fixed guess admits batches the engine then
    # cannot fit (measured: BS=256 picked 256 x 8k prompts and needed 2.1M KV
    # against a 990k pool). So: pick with a placeholder, compute the real
    # budget, then re-pick with that budget as the per-stream overhead.
    kv_pool = int(os.environ.get('GC_KV_BUDGET', '1150000'))
    if matrix and nseq > 256 and int(os.environ.get('GC_KV_BUDGET', '0')) <= 0:
        raise ValueError('set GC_KV_BUDGET to the calibrated token budget for this matrix batch')
    prompts = select_decode_prompts(nseq, a.prefill_lo, a.prefill_hi)
    flush = int(os.environ.get('GC_MEASURE_FLUSH', '5'))
    tail_warmup = 2 if a.timing_tail_steps else 0
    extra_steps = flush + tail_warmup + a.timing_tail_steps if a.timing_tail_steps else 0
    for _ in range(3):
        tot = sum(len(p) for p in prompts)
        slack = 2 * (-(-tot // chunk)) + 64
        budget = 1 + a.warm_steps + a.measure_steps + slack + extra_steps
        overhead = budget + int(os.environ.get('GC_PREFILL_MARGIN', '512')) + 64 if matrix and nseq > 256 else budget + 512 + 64
        os.environ['GC_KV_OVERHEAD'] = str(overhead)
        if sum(len(p) + overhead for p in prompts) <= kv_pool:
            break
        prompts = select_decode_prompts(nseq, a.prefill_lo, a.prefill_hi)
    max_len = max(len(p) for p in prompts)
    print(f'[data] {nseq} prompts ({a.seqs_per_rank}/rank x {ranks}), '
          f'len {min(map(len,prompts))}..{max_len}', flush=True)
    # Chunked prefill finishes the 256 streams at *different* engine steps: the
    # first one after 1 step, the last after ~N/chunk steps. If every request
    # only asks for `1 + warm + measure` tokens, the ones that started early
    # retire before the measurement window ends (observed: Running fell
    # 256 -> 188 inside the timed window, so the numbers described a shrinking
    # batch, not a steady 32 streams/rank). The margin keeps every stream alive
    # past the last measured step; it only costs KV space, never GPU time.
    margin = int(os.environ.get('GC_PREFILL_MARGIN', '512'))
    # [TOKEN-BUDGET] One number drives BOTH the KV reservation
    # (`max_model_len`) and the per-request decode budget (`max_tokens`);
    # they must agree or vLLM truncates one of them.
    #
    # Why a slack term: requests are submitted all at once, so chunked prefill
    # finishes them at DIFFERENT engine steps and the early ones start decoding
    # immediately. Measured on BS=32 (`sum(len)=150k`, chunk=2048):
    #
    #   iter=1 -> 2 streams already generating
    #   iter=29 -> 19, iter=49 -> 26, iter=71 -> all 32
    #
    # A stream that began at iter 1 has spent ~70 of its `max_tokens` before
    # the measured window even opens. Without slack it retires mid-window and
    # the average silently describes a shrinking batch.
    #
    # `slack` is therefore DERIVED from this run's prompts, not guessed:
    # ceil(sum(len)/chunk) is the prefill step count, and +64 covers the
    # first token each stream emits while others are still prefilling.
    total_tokens = sum(len(p) for p in prompts)
    # 2x the estimated prefill step count: the estimate `ceil(sum/chunk)`
    # matched the measurement closely (74 est vs 71 measured on BS=32, 150k
    # tokens, chunk=2048), but the scheduler can take more steps when long and
    # short prompts interleave. A stream that runs dry retires mid-window and
    # silently turns the average into a shrinking-batch number, so paying
    # double here is cheap insurance -- it costs KV, never GPU time.
    slack = 2 * (-(-total_tokens // chunk)) + 64
    if os.environ.get('GC_PREFILL_SLACK'):
        slack = int(os.environ['GC_PREFILL_SLACK'])   # manual override
    decode_budget = 1 + a.warm_steps + a.measure_steps + slack + extra_steps
    print(f'[budget] decode_budget={decode_budget} '
          f'(1+{a.warm_steps}warm+{a.measure_steps}measure+{slack}slack) '
          f'prefill_tokens={total_tokens} chunk={chunk} kv_margin={margin}',
          flush=True)
    strict_state = os.environ.get('GC_MOE_STRICT_STATE_DIR')
    fixed_dir = os.environ.get('GC_MOE_FIXED_DIR')
    fixed = {}
    if fixed_dir:
        # Capture controls before the model runner creates per-process IDs.
        runtime_keys = ('GC_STEP_BOUNDARY', 'MODEL', 'DATASET', 'WORLD', 'GPU_UTIL', 'MAX_BATCHED', 'MOE_BACKEND',
                        'ALL2ALL_BACKEND', 'CUDAGRAPH_MODE', 'PYTHONHASHSEED', 'ENFORCE_EAGER',
                        'CUDA_VISIBLE_DEVICES', 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_ENABLE_V1_MULTIPROCESSING',
                        'VLLM_DISABLE_REQUEST_ID_RANDOMIZATION', 'VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD',
                        'NCCL_IB_HCA', 'NVSHMEM_HCA_LIST', 'NCCL_NET')
        fixed_controls = {k: v for k, v in os.environ.items()
                          if (k.startswith(('GC_MOE_', 'GC_MEASURE_')) or k in runtime_keys)
                          and k != 'GC_MOE_SM90_STREAM_M_BLOCKS'}
        from pathlib import Path
        from fixed_workload import publishes_checkpoint, requires_layer_aligned_warmup
        recording_tokens = long_cost and publishes_checkpoint() and os.getenv("GC_MOE_PREFIX_DIAG") != "1"
        if not recording_tokens:
            fixed = json.loads((Path(fixed_dir) / 'tokens.json').read_text())
        prompt_hash = hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest()
        if ((not recording_tokens and (prompt_hash != fixed['prompt_hash']
                                       or set(fixed['tokens']) != {f'req{i}' for i in range(nseq)}))
                or (not long_cost and (nseq, a.warm_steps, flush, a.measure_steps, tail_warmup, a.timing_tail_steps)
                    != (256, 200, 2, 1, 2, 100))):
            raise ValueError('fixed workload does not match the prompts or decode protocol')
    llm = LLM(
        model=os.environ['MODEL'], tensor_parallel_size=8,
        enable_expert_parallel=True, enable_prefix_caching=False,
        max_model_len=max_len + decode_budget + margin + 64,
        gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.80')),
        **({'kv_cache_memory_bytes': int(os.environ['GC_MOE_KV_BYTES'])} if strict_state else {}),
        max_num_seqs=nseq,
        moe_backend=os.environ.get('MOE_BACKEND', 'gc_moe'),
        all2all_backend=os.environ.get('ALL2ALL_BACKEND',
                                              'allgather_reducescatter'),
        max_num_batched_tokens=chunk,
        trust_remote_code=True, dtype='bfloat16', disable_log_stats=False,
        enable_logging_iteration_details=True,
        compilation_config=_compilation_config(),
        enforce_eager=os.environ.get('ENFORCE_EAGER', '0') == '1',
        **({'async_scheduling': False, 'pipeline_parallel_size': 1, 'seed': 0}
           if a.profile_aligned_step or long_cost else {}))

    engine = llm.llm_engine
    if a.timing_tail_steps or long_cost:
        from vllm.v1.engine.core_client import InprocClient

        if not isinstance(engine.engine_core, InprocClient):
            raise RuntimeError('aligned capture requires driver-stepped EngineCore')
    # [TOKEN-BUDGET] `max_tokens` must cover EVERY decode step any stream will
    # ever be asked to produce, and chunked prefill makes that more than
    # warm+measure:
    #
    #   * the prefill loop below returns once every sequence has emitted its
    #     first token, but the sequences that finished prefill early were
    #     ALREADY decoding during that loop (they consumed up to
    #     `prefill_steps` tokens of their budget);
    #   * after the timed window there is a separate IR sampling window of
    #     `GC_IRSTEP_STEPS` steps.
    #
    # With 100 warm + 100 measure the old `1+warm+measure+margin` (2513) was
    # far above anything consumed, so a stream never ran dry. At 1000+1000 plus
    # a 1000-step IR window it does: measured "Running: 32 -> 1 reqs" dropping
    # inside the window, which silently turns the measurement into a
    # single-stream latency. Add both terms explicitly.
    total = decode_budget

    # vLLM 0.31 enables async scheduling by default: the scheduler keeps one
    # batch in flight, so ``LLMEngine.step()`` returns as soon as the request
    # is *submitted* to the worker. Timing individual step() calls therefore
    # measures queue latency, not GPU time -- the old logs contain a 443 ms
    # "step" followed by four 3.3 ms ones, which is that artefact. Time whole
    # windows instead, bracketed by a real device synchronize in every worker.
    def _sync():
        engine.collective_rpc('_gc_moe_sync')

    # ---- token capture (correctness check across arms) -------------------
    # Every step's RequestOutput carries DELTA token ids; keep them so two arms
    # can be compared bit-for-bit. This is outside the timed window's logic --
    # it only appends to a dict.
    cap_tokens = os.environ.get('GC_CAPTURE_TOKENS', '1') == '1'
    got = {}          # request_id -> list of generated token ids (filled lazily)

    def _collect(outs):
        if not cap_tokens:
            return
        for o in outs:
            if o.outputs and o.outputs[0].token_ids:
                got.setdefault(o.request_id, []).extend(o.outputs[0].token_ids)

    def time_window(n, blocks=None, host=None, capture=False, check_first=False):
        _sync()                       # drain batches queued before the window
        t0 = time.perf_counter()
        last = t0
        for i in range(1, n + 1):
            if capture and i == 3:
                from trace_layer.bs256_capture import capture_step
                _collect(capture_step(engine, matrix))
            elif host is None:
                _collect(engine.step())
            else:
                t = time.perf_counter_ns()
                outs = engine.step()
                returned = time.perf_counter_ns()
                _collect(outs)
                host.append([t, returned, time.perf_counter_ns()])
            if check_first and i == 1:
                if engine.collective_rpc('_gc_moe_long_first_step') != [True] * ranks:
                    raise ValueError('missing worker confirmation for the first statistics step')
            if blocks is not None and i < n and i % 10 == 0:
                now = time.perf_counter()
                blocks.append((now - last) * 100)
                last = now
        _sync()
        end = time.perf_counter()
        if blocks is not None:
            blocks.append((end - last) * 100)
        return (end - t0) / n * 1000.0

    # Per-step scheduler truth (running requests / scheduled tokens). Read from
    # the EngineCore output before LLMEngine consumes it.
    rec = {'on': False, 'running': [], 'gen_tokens': [], 'ctx_tokens': []}
    _orig_get_output = engine.engine_core.get_output

    def _get_output():
        out = _orig_get_output()
        if rec['on']:
            st = getattr(out, 'scheduler_stats', None)
            it = getattr(st, 'iteration_details', None)
            rec['running'].append(getattr(st, 'num_running_reqs', -1)
                                  if st is not None else -1)
            rec['gen_tokens'].append(getattr(it, 'num_generation_tokens', -1)
                                     if it is not None else -1)
            rec['ctx_tokens'].append(getattr(it, 'num_ctx_tokens', -1)
                                     if it is not None else -1)
        return out

    engine.engine_core.get_output = _get_output
    params = SamplingParams(max_tokens=total, temperature=0.0, ignore_eos=True,
                            output_kind=RequestOutputKind.DELTA)

    # [SME-MEAS] Enable the counter table BEFORE any forward: CUDA graphs bake
    # in the kernel variant they capture, so a table added later is ignored.
    if os.environ.get('GC_MEASURE_LOAD', '1') == '1':
        engine.collective_rpc('_gc_moe_meas_enable')

    if fixed_dir:
        installed = (engine.collective_rpc('_gc_moe_fixed_tokens', args=(decode_budget,)) if long_cost
                     else engine.collective_rpc('_gc_moe_fixed_tokens'))
        if installed != [True] * ranks:
            raise RuntimeError('reference sampler was not installed on every worker')

    # ---- submit -----------------------------------------------------------
    ids = []
    for i, p in enumerate(prompts):
        rid = f'req{i}'
        ids.append(rid)
        engine.add_request(rid, {'prompt_token_ids': p}, params)

    # ---- 1. REAL prefill (not timed) -------------------------------------
    # A sequence's first generated token marks the end of its prefill.
    t0 = time.perf_counter()
    stepped = set()
    prefill_steps = 0
    while len(stepped) < nseq:
        outs = engine.step()
        if a.timing_tail_steps or long_cost:
            _collect(outs)
        prefill_steps += 1
        for o in outs:
            if o.outputs and o.outputs[0].token_ids:
                stepped.add(o.request_id)
        if prefill_steps > 100000:
            raise SystemExit('prefill did not converge')
    prefill_s = time.perf_counter() - t0
    print(f'[prefill] {nseq} seqs, {prefill_steps} engine steps, '
          f'{prefill_s:.1f}s (not timed)', flush=True)

    if os.getenv("SM90_COLLECT_SE_ROUTES") == "1":
        if not (long_cost and matrix == "vanilla" and recording_tokens and cap_tokens):
            raise ValueError("route collection requires the vanilla publisher")
        for phase, steps in (("stats", 1000), ("measure", 3)):
            starts = engine.collective_rpc('_gc_se_capture_begin', args=(phase,))
            if len(starts) != ranks or len(set(starts)) != 1:
                raise ValueError(f'{phase}: rank epoch mismatch')
            for key in ('running', 'gen_tokens', 'ctx_tokens'):
                rec[key].clear()
            before = [len(got[rid]) for rid in ids]
            rec['on'] = True
            time_window(steps)
            rec['on'] = False
            if (rec['gen_tokens'] != [nseq]*steps or rec['ctx_tokens'] != [0]*steps or
                    any(len(got[rid])-previous != steps for rid,previous in zip(ids,before))):
                raise ValueError(f'{phase}: expected {steps} full pure decode steps')
            result = engine.collective_rpc('_gc_se_capture_finish', args=(phase,))
            if len(result) != ranks:
                raise ValueError(f'{phase}: incomplete worker collection')
            if phase == 'stats' and len(set(result)) != 1:
                raise ValueError('statistics demand hash differs across ranks')
            if phase == 'measure' and sorted(result) != list(range(ranks)):
                raise ValueError('missing source rank in collected routes')
        with (Path(fixed_dir)/'capture.json').open('x') as f:
            json.dump(dict(batch=nseq, stats_steps=1000, capture_steps=3, layers=[8,20,32],
                           prompt_sha256=prompt_hash, prefill_steps=prefill_steps, controls=fixed_controls,
                           output_sha256=hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest()), f)
        return

    if os.getenv("GC_MOE_PREFIX_DIAG") == "1":
        if not long_cost or matrix not in ("vanilla", "random") or recording_tokens:
            raise ValueError("prefill diagnostic requires a matrix reference consumer")
        reports = engine.collective_rpc('_gc_moe_prefix_diagnostic')
        extra = {}
        if os.getenv("GC_MOE_PREFIX_BLOCK_AA") == "1":
            extra["block_by_rank"] = [report.pop("block_records") for report in reports]
        if os.getenv("GC_MOE_PREFIX_LAYER_AA") == "1":
            extra["moe_by_rank"] = [report.pop("moe_boundaries") for report in reports]
        if os.getenv("GC_MOE_PREFIX_HIDDEN_AA") == "1":
            extra["hidden_by_rank"] = [report.pop("hidden_samples") for report in reports]
        if len(reports) != ranks or any(report != reports[0] for report in reports):
            raise ValueError("prefill diagnostic differs across workers")
        if any(got[rid] != fixed['tokens'][rid][:n] for rid, n in zip(ids, reports[0]['positions'])):
            raise ValueError("prefill feedback differs from reference")
        with Path(a.output).open('x') as f:
            json.dump(dict(arm=matrix, prompt_hash=prompt_hash, prefill_steps=prefill_steps,
                           **reports[0], **extra), f)
        return

    # ---- 2. warm decode: EPLB learning window ----------------------------
    def long_window(phase):
        start = engine.collective_rpc('_gc_moe_long_begin', args=(phase,))
        for key in ('running', 'gen_tokens', 'ctx_tokens'):
            rec[key].clear()
        before = [len(got[rid]) for rid in ids]
        blocks = []
        rec['on'] = True
        diagnose = phase == "measure" and os.getenv("GC_STEP_BOUNDARY") == "1"
        if diagnose:
            from contextlib import ExitStack
            from unittest.mock import patch
            host, calls = [], {}
            def instrument(method, name):
                spans = calls[name] = []
                def timed(*args, **kwargs):
                    t = time.perf_counter_ns()
                    value = method(*args, **kwargs)
                    spans.append([len(host), t, time.perf_counter_ns()])
                    return value
                return timed
            client = engine.engine_core
            scheduler = client.engine_core.scheduler
            with ExitStack() as stack:
                for obj, name in ((client, 'get_output'), (scheduler, 'schedule'),
                                  (scheduler, 'update_from_output')):
                    stack.enter_context(patch.object(obj, name, instrument(getattr(obj, name), name)))
                if os.getenv("GC_COST_HOST_PROBE") == "1":
                    from cost_host_probe import install
                    calls["exclusive_host"] = install(stack, engine, host)
                elapsed = time_window(1000, blocks, host)
        else:
            elapsed = time_window(1000, blocks, capture=phase == "measure" and os.getenv("GC_BS256_TRACE_DIR"),
                                  check_first=phase == "stats" and requires_layer_aligned_warmup())
        rec['on'] = False
        if (rec['gen_tokens'] != [nseq]*1000 or rec['ctx_tokens'] != [0]*1000
                or any(len(got[rid]) != n+1000 for rid,n in zip(ids,before))):
            raise ValueError(f'{phase}: expected 1000 full pure-decode steps for all requests')
        audit = engine.collective_rpc('_gc_moe_long_finish', args=(phase, str(Path(a.output).parent)))
        return dict(ms_per_step=elapsed, ten_step_ms=blocks, initial=start, audit=audit,
                    **({"host_boundary_ns": dict(steps=host, **calls)} if diagnose else {}))

    if long_cost:
        training = long_window('stats')
        if (os.getenv("GC_MOE_VANILLA_STATS_REPLAY") == "1"
                or os.getenv("GC_MOE_DECODE_STATS_AA") == "1"):
            if recording_tokens:
                with (Path(fixed_dir) / "tokens.json").open("x") as f:
                    json.dump(dict(prompt_hash=prompt_hash, tokens=got), f, separators=(",", ":"))
            elif os.getenv("GC_MOE_DECODE_STATS_AA") == "1" and got != fixed["tokens"]:
                raise ValueError("statistics-window feedback differs from its publisher")
            with Path(a.output).open("x") as f:
                json.dump(dict(arm=matrix, phase="stats", prompt_hash=prompt_hash,
                               prefill_steps=prefill_steps, stats=training), f)
            return
        warn_s = training['ms_per_step']
    else:
        warn_s = time_window(a.warm_steps)
    print(f'[warm] {a.warm_steps} decode steps: {warn_s:.2f} ms/step '
          f'(device-synced window)', flush=True)

    if os.environ.get('GC_MOE_SLOT_STATS') == '1':
        deepep_compare.dump_slot_stats(a.label)

    # The registry lives in the *worker* processes, so the replan must run
    # there: collective_rpc broadcasts the call to every worker.
    rearranged = False
    if os.environ.get('GC_MOE_EPLB') == '1' and matrix != 'random':
        try:
            outs = engine.collective_rpc('_gc_moe_rearrange')
            rearranged = any(bool(x) for x in outs)
            if (a.profile_aligned_step or long_cost) and (len(outs) != ranks or not all(outs)):
                raise RuntimeError(f'rearrangement did not complete on all ranks: {outs}')
        except Exception as exc:  # noqa: BLE001
            print(f'[eplb] rearrangement FAILED: {exc}', flush=True)
            raise
    print(f'[eplb] rearrangement {"done" if rearranged else "not applicable"}',
          flush=True)

    if long_cost:
        measured = long_window('measure')
        if recording_tokens:
            with (Path(fixed_dir) / 'tokens.json').open('x') as f:
                json.dump(dict(prompt_hash=prompt_hash, tokens=got), f, separators=(',', ':'))
        elif got != fixed['tokens']:
            raise ValueError('feedback differs from the native linear-arm trajectory')
        result = dict(stats=training, measured=measured, prompt_hash=prompt_hash,
                      output_hash=hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest(),
                      prefill_steps=prefill_steps, steps=1000, layers=aligned_layers, batch=nseq,
                      stream=int(os.environ['GC_MOE_SM90_STREAM_M_BLOCKS']), controls=fixed_controls,
                      cost=int(os.environ['GC_MOE_ECO_COSTFN']), rearranged=rearranged)
        with Path(a.output).open('x') as f:
            json.dump(result, f)
        return

    aa_control = os.getenv('GC_MOE_AA_CONTROL') == '1'
    frozen_state = []
    if fixed_dir:
        frozen_state = engine.collective_rpc('_gc_moe_fixed_routes')
        if strict_state:
            if (sorted(row['rank'] for row in frozen_state) != list(range(ranks))
                    or any(row['checkpoint']['source_sha256'] != row['checkpoint']['readback_sha256']
                           for row in frozen_state)):
                raise ValueError('eight verified checkpoint readbacks required before stream activation')
            flag = int(os.environ['GC_MOE_SM90_STREAM_M_BLOCKS'])
            if not aa_control and engine.collective_rpc('_gc_moe_activate_stream') != [flag] * ranks:
                raise ValueError('workers did not activate the requested scheduling mode')
    # ---- 3. measured decode ---------------------------------------------
    # A few throw-away steps: the first forwards after the EPLB weight rewrite
    # (or after any host-side stall) are not representative, and one 400 ms
    # outlier in a 100-step average is worth 4 ms/step -- larger than the
    # effect we are trying to measure.
    for flush_step in range(flush):
        _collect(engine.step())
        if strict_state and flush_step == 0:
            probes = {row['rank']: row for row in engine.collective_rpc('_gc_moe_first_replay_call', args=(flush_step,))}
            for row in frozen_state:
                row['first_call'] = probes[row['rank']]
            if aa_control:
                with (Path(strict_state)/f'control-step-mode{flag}.json').open('x') as f:
                    json.dump(frozen_state, f, separators=(',', ':'))
                checkpoints = {r['rank']: r['checkpoint'] for r in engine.collective_rpc('_gc_moe_treatment_boundary')}
                if set(checkpoints) != set(range(ranks)) or any(
                        c['source_sha256'] != c['readback_sha256'] for c in checkpoints.values()):
                    raise ValueError('missing treatment checkpoint readback')
                for row in frozen_state:
                    row['treatment_checkpoint'] = checkpoints[row['rank']]
                if engine.collective_rpc('_gc_moe_activate_stream') != [flag] * ranks:
                    raise ValueError('workers did not switch modes after the control step')
    _sync()
    # [SME-IR] Record per-slot loads during the measured window so we can
    # report IR = max_rank(tokens)/mean_rank(tokens) for the same steps.
    collect_ir = os.environ.get('GC_MEASURE_IR', '1') == '1'
    if collect_ir:
        engine.collective_rpc('_gc_moe_ir_begin')
    # [SME-MEAS] Per-step per-layer/per-rank load, M blocks, active experts.
    collect_meas = os.environ.get('GC_MEASURE_LOAD', '1') == '1'
    if collect_meas:
        engine.collective_rpc('_gc_moe_meas_begin', args=(a.measure_steps,))
    # [IR-STEP] Anchor the in-kernel (layer, step) table at the window start.
    # The dispatch role records one cell per (layer, step) itself, so the
    # measured loop below needs NO extra RPC: this call only publishes
    # `epoch0` (a plain device store after draining the queue), and the dump
    # happens once at the end. That is what keeps the window clean AND keeps
    # `max_tokens` free of a second sampling window.
    collect_irstep = os.environ.get('GC_MEASURE_IR_STEP', '1') == '1'
    if collect_irstep:
        _sync()
        try:
            engine.collective_rpc('_gc_moe_irstep_begin')
        except Exception as exc:  # noqa: BLE001
            collect_irstep = False
            print(f'[ir-step] begin failed: {exc}', flush=True)
    capture_id = None
    if a.profile_out:
        if a.profile_aligned_step:
            from uuid import uuid4

            # Refuse stale artifacts if a worker fails to export this run.
            os.makedirs(a.profile_out, exist_ok=False)
            capture_id = uuid4().hex
            before = engine.collective_rpc('_gc_moe_profile_snapshot')
            first_epochs = {item['next_epoch'] for item in before}
            if len(first_epochs) != 1 or {item['rank'] for item in before} != set(range(ranks)):
                raise RuntimeError('ranks disagree on the pre-step workspace epoch')
            first_epoch = first_epochs.pop()
            engine.collective_rpc('_gc_moe_prof_start', args=(a.profile_out, capture_id, bool(a.fine_layers)))
        else:
            os.makedirs(a.profile_out, exist_ok=True)
            engine.collective_rpc('_gc_moe_prof_start', args=(a.profile_out,))
        print(f'[trace] profiling -> {a.profile_out}', flush=True)
    rec['on'] = True
    t0 = time.perf_counter()
    for _ in range(a.measure_steps):
        outputs = engine.step()
        _collect(outputs)
    t_submit_s = time.perf_counter() - t0
    rec['on'] = False
    _sync()
    window_s = time.perf_counter() - t0
    if a.profile_out:
        info = engine.collective_rpc('_gc_moe_prof_stop')
        ok = [x for x in info if isinstance(x, dict) and x.get('ok') and x.get('path')]
        print(f'[trace] {len(ok)}/{len(info)} workers exported', flush=True)
        if capture_id is not None:
            if len(ok) != ranks:
                raise RuntimeError(f'missing worker trace exports: {info}')
            workers = engine.collective_rpc('_gc_moe_profile_snapshot', args=(first_epoch,))
            decode_counts = {o.request_id: len(o.outputs[0].token_ids)
                             for o in outputs if o.outputs and o.outputs[0].token_ids}
            manifest = dict(id=capture_id, gemm_start='first_valid_cta_claim',
                            label=a.label, step=flush, layers=aligned_layers,
                            first_epoch=first_epoch, decode_tokens=nseq, ranks=workers)
            from pathlib import Path

            from trace_layer.aligned_step import resolve_epochs

            directory = Path(a.profile_out)
            try:
                if len(decode_counts) != nseq or set(decode_counts.values()) != {1}:
                    raise ValueError('the captured step did not generate one token per sequence')
                dispatch_grid = 4 * int(os.environ.get('GC_MOE_SM90_DISPATCH_PARTITION_SMS',
                                                       os.environ.get('GC_MOE_SM90_DISPATCH_SMS', '4')))
                if os.getenv("GC_MOE_ECO") == "1" and os.getenv("GC_MOE_SM90_PLAN_DISPATCH", "1") == "1":
                    dispatch_grid -= 4
                capture = resolve_epochs(directory, manifest, dispatch_grid)
            except (KeyError, OSError, ValueError) as exc:
                (directory / 'capture.invalid.json').write_text(json.dumps(dict(manifest, error=str(exc))))
                raise RuntimeError(f'{exc}; device snapshots saved in capture.invalid.json') from exc
            (directory / 'capture.json').write_text(json.dumps(capture))
            print(f'[trace] matched {aligned_layers} calls at epoch {capture["first_epoch"]} '
                  f'within snapshot {capture["device_snapshot_epochs"]}', flush=True)
            if a.fine_layers:
                epochs = [capture['first_epoch'] + layer for layer in sorted(set(a.fine_layers))]
                # Bound the worker RPC and host allocation to one layer in a wide capture.
                split = len(epochs) > 2
                for group in ([[epoch] for epoch in epochs] if split else [epochs]):
                    fine_ranks = engine.collective_rpc('_gc_moe_profile_fine_dump', args=(group,))
                    name = f'fine-epoch-{group[0]}.json' if split else 'fine.json'
                    (directory / name).write_text(json.dumps(dict(id=capture_id, epochs=group, ranks=fine_ranks)))
                    if len(fine_ranks) != ranks or {w['rank'] for w in fine_ranks} != set(range(ranks)):
                        raise RuntimeError(f'missing or duplicate fine-trace ranks; see {name}')
                    for worker in fine_ranks:
                        if worker.get('error') or any(not worker['epochs'].get(str(e)) for e in group):
                            raise RuntimeError(f'rank {worker["rank"]}: invalid fine rows; see {name}')
                if split:
                    (directory / 'fine.json').write_text(json.dumps(dict(id=capture_id, epochs=epochs,
                                                                          per_epoch_files=True)))
    ir = {}
    if collect_ir:
        irs = engine.collective_rpc('_gc_moe_ir_end')
        ir = next((x for x in irs if isinstance(x, dict) and x.get('ir', -1) > 0), {})
    load = {}
    if collect_meas:
        outs = engine.collective_rpc('_gc_moe_meas_end')
        load = next((x for x in outs if isinstance(x, dict) and x.get('ok')), {})
    irstep = {}
    if collect_irstep:
        try:
            outs = engine.collective_rpc('_gc_moe_irstep_end')
            irstep = _irstep_reduce(outs)
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a run
            irstep = {'error': str(exc)}
    # [ECO-TRACE] One dump at the end of the window (never per step): the raw
    # 30-slot timeline plus the per-step x per-rank rows. Averaging happens
    # here, in Python, so the kernel never reduces away "which rank was slow".
    eco_trace = {}
    # [ECO-TRACE] Dump whenever a timeline was requested. The dispatch-side
    # rows (plan_wake / push / pull) now exist for EVERY arm, so the RR and
    # random arms produce the same table as ECO.
    if os.environ.get('GC_MOE_ECO_TRACE') == '1' \
            or os.environ.get('GC_MOE_PULL_TRACE') == '1':
        try:
            trs = engine.collective_rpc('_gc_moe_eco_trace_dump', args=(not a.fine_layers,))
            eco_trace = {'ranks': [x for x in trs if isinstance(x, dict) and x.get('ok')]}
        except Exception as exc:  # noqa: BLE001 - never fail a run over telemetry
            eco_trace = {'error': str(exc)}

    def _stat(xs):
        return (min(xs), max(xs), statistics.median(xs)) if xs else (-1, -1, -1)

    running = rec['running']
    run_lo, run_hi, _ = _stat(running)
    gen_lo, gen_hi, gen_med = _stat(rec['gen_tokens'])
    ctx_lo, ctx_hi, _ = _stat(rec['ctx_tokens'])
    timing_tail = {}
    if a.timing_tail_steps:
        # The capture and its counters have already been exported. Park fine
        # writes before the timing window, keeping the same captured kernels.
        counters = engine.collective_rpc('_gc_moe_profile_park_fine')
        if len(counters) != ranks or not all(n > 0 for n in counters):
            raise RuntimeError(f'invalid per-worker fine counters: {counters}')
        for _ in range(tail_warmup):
            _collect(engine.step())
        for key in ('running', 'gen_tokens', 'ctx_tokens'):
            rec[key].clear()
        rec['on'] = True
        blocks = [] if strict_state else None
        tail_ms = time_window(a.timing_tail_steps, blocks)
        rec['on'] = False
        if (rec['gen_tokens'] != [nseq] * a.timing_tail_steps
                or rec['ctx_tokens'] != [0] * a.timing_tail_steps):
            raise RuntimeError('timing tail did not sustain the requested pure-decode batch')
        if engine.collective_rpc('_gc_moe_profile_park_fine') != counters:
            raise RuntimeError('fine records advanced during the timing tail')
        timing_tail = dict(steps=a.timing_tail_steps, ms_per_step=tail_ms,
                           engine_core='inproc', profiler_active=False, fine_writes=False)
        if blocks is not None:
            timing_tail['ten_step_host_ms_per_step'] = blocks
    ms = window_s / a.measure_steps * 1000.0
    print(f'[measure] {ms:.2f} ms/step (device-synced window, '
          f'{a.measure_steps} steps, submit={t_submit_s / a.measure_steps * 1000:.2f} ms/step) '
          f'running={run_lo}..{run_hi} (want {nseq}) '
          f'gen_tokens/step={gen_lo}..{gen_hi} (median {gen_med}) '
          f'ctx_tokens/step={ctx_lo}..{ctx_hi}', flush=True)

    print(f'[ir] IR={ir.get("ir", -1):.4f} '
          f'max={ir.get("max_rank_tokens", -1):.0f} '
          f'mean={ir.get("mean_rank_tokens", -1):.1f} '
          f'min={ir.get("min_rank_tokens", -1):.0f} '
          f'per_rank={[int(x) for x in ir.get("tokens_per_rank", [])]}', flush=True)

    if irstep:
        print(f'[ir-step] IR(step x layer avg)={irstep.get("ir_step_layer_mean", -1):.4f} '
              f'p50={irstep.get("ir_step_layer_p50", -1):.4f} '
              f'p90={irstep.get("ir_step_layer_p90", -1):.4f} '
              f'null={irstep.get("ir_poisson_null_mean", -1):.4f} '
              f'excess={irstep.get("ir_excess_over_null", -1):.4f} '
              f'pairs={irstep.get("n_pairs", 0)} steps={irstep.get("n_steps", 0)}',
              flush=True)
    result = dict(label=a.label, seqs=nseq, seqs_per_rank=a.seqs_per_rank,
                  warm_steps=a.warm_steps, measure_steps=a.measure_steps,
                  prompt_sha256=hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest(),
                  capture_id=capture_id, prefill_s=prefill_s, warm_ms_per_step=warn_s,
                  timing='device_synced_window', timing_tail=timing_tail,
                  submit_ms_per_step=t_submit_s / a.measure_steps * 1000,
                  ms_per_step_median=ms, ms_per_step_min=ms,
                  ms_per_step_max=ms, window_s=window_s,
                  running_min=run_lo, running_max=run_hi,
                  gen_tokens_per_step=[gen_lo, gen_hi, gen_med],
                  ctx_tokens_per_step=[ctx_lo, ctx_hi],
                  step_ms=[ms], eplb=bool(rearranged),
                  # [ECO-SM90] which split policy this arm used, and its
                  # rank-time weights (empty for the vanilla/rr arms)
                  eco=os.environ.get('GC_MOE_ECO') == '1',
                  eco_costfn=int(os.environ.get('GC_MOE_ECO_COSTFN', '0')),
                  stream_m_blocks=int(os.environ.get('GC_MOE_SM90_STREAM_M_BLOCKS', '0')),
                  eco_weights=dict(wb=int(os.environ.get('GC_MOE_ECO_WB', '0')),
                                   wa=int(os.environ.get('GC_MOE_ECO_WA', '0')),
                                   rounds=int(os.environ.get('GC_MOE_ECO_ROUNDS', '1')),
                                   cold_rounds=int(os.environ.get(
                                       'GC_MOE_ECO_COLD_ROUNDS', '6')))
                  if os.environ.get('GC_MOE_ECO') == '1' else {},
                  # correctness / IR evidence
                  tokens_per_rank=int(gen_med) if gen_med > 0 else -1,
                  ir=ir.get('ir', -1.0),
                  ir_detail=ir, ir_step=irstep,                   load_detail=load,
                  eco_trace=eco_trace,
                  moe_real=moe_real(eco_trace),
                  time_ir=time_ir(eco_trace, expect_ms=ms),
                  gen_tokens_total=sum(len(v) for v in got.values()),
                  tokens={k: v for k, v in sorted(got.items())})

    if fixed_dir:
        if got != fixed['tokens']:
            raise ValueError('observed per-request token histories differ from the input reference')
        result['fixed_controls'] = fixed_controls
        if aa_control:
            probes = {row['rank']: row for row in engine.collective_rpc('_gc_moe_first_replay_call', args=(1,))}
            for row in frozen_state:
                row['treatment_call'] = probes[row['rank']]
        result['fixed_workload'] = dict(initial=frozen_state,
                                        ranks=engine.collective_rpc('_gc_moe_fixed_routes', args=(True,)))
    _pp = (result.get('time_ir') or {}).get('plan_pred')
    if _pp and 'pred_ratio' in _pp:
        print(f"[plan-pred] pred_ratio={_pp['pred_ratio']}", flush=True)
        print(f"[plan-pred] meas_ratio={_pp['meas_ratio']}  "
              f"L1={_pp['ratio_l1']:.4f} maxerr={_pp['ratio_maxerr']:.4f}", flush=True)
    _ti = result.get('time_ir') or {}
    if _ti.get('ok'):
        print(f"[time-ir] layers={_ti['time_layers']} "
              f"time_IR={_ti['time_ir']:.4f} p90={_ti['time_ir_p90']:.4f} "
              f"rank_mean={_ti['time_rank_mean_us']:.0f}us "
              f"rank_max={_ti['time_rank_max_us']:.0f}us", flush=True)
    _mr = result.get('moe_real') or {}
    if _mr.get('ok'):
        print(f"[moe-real] layers={_mr['layers']} wall_max={_mr['wall_max']:.0f}us "
              f"wall_mean={_mr['wall_mean']:.0f}us ir_real={_mr['ir_real']:.4f}", flush=True)
    print('RESULT', json.dumps(result), flush=True)
    with open(a.output, 'w') as fh:
        json.dump(result, fh, indent=2)


if __name__ == '__main__':
    main()
