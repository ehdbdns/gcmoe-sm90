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


def _compilation_config():
    from vllm.config.compilation import CUDAGraphMode
    mode = os.environ.get('CUDAGRAPH_MODE', 'FULL_DECODE_ONLY')
    if mode == 'default':
        return None
    return {'cudagraph_mode': getattr(CUDAGraphMode, mode)}


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
            if has_ae:
                avals = [d['rows_ae'][lay][k] for d in ok]
                ae_total.append(sum(avals))       # ★ cross-rank SUM
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--seqs-per-rank', type=int, default=32)
    ap.add_argument('--prefill-lo', type=int, default=256)
    ap.add_argument('--prefill-hi', type=int, default=4096)
    ap.add_argument('--warm-steps', type=int, default=100)
    ap.add_argument('--measure-steps', type=int, default=100)
    a = ap.parse_args()

    ranks = int(os.environ.get('WORLD', '8'))
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
    prompts = load_prompts(nseq, a.prefill_lo, a.prefill_hi)
    for _ in range(3):
        tot = sum(len(p) for p in prompts)
        slack = 2 * (-(-tot // chunk)) + 64
        budget = 1 + a.warm_steps + a.measure_steps + slack
        overhead = budget + 512 + 64
        os.environ['GC_KV_OVERHEAD'] = str(overhead)
        if sum(len(p) + overhead for p in prompts) <= kv_pool:
            break
        prompts = load_prompts(nseq, a.prefill_lo, a.prefill_hi)
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
    decode_budget = 1 + a.warm_steps + a.measure_steps + slack
    print(f'[budget] decode_budget={decode_budget} '
          f'(1+{a.warm_steps}warm+{a.measure_steps}measure+{slack}slack) '
          f'prefill_tokens={total_tokens} chunk={chunk} kv_margin={margin}',
          flush=True)
    llm = LLM(
        model=os.environ['MODEL'], tensor_parallel_size=8,
        enable_expert_parallel=True, enable_prefix_caching=False,
        max_model_len=max_len + decode_budget + margin + 64,
        gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.80')),
        max_num_seqs=nseq,
        moe_backend=os.environ.get('MOE_BACKEND', 'gc_moe'),
        all2all_backend=os.environ.get('ALL2ALL_BACKEND',
                                              'allgather_reducescatter'),
        max_num_batched_tokens=chunk,
        trust_remote_code=True, dtype='bfloat16', disable_log_stats=False,
        enable_logging_iteration_details=True,
        compilation_config=_compilation_config(),
        enforce_eager=os.environ.get('ENFORCE_EAGER', '0') == '1')

    engine = llm.llm_engine
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

    def time_window(n):
        _sync()                       # drain batches queued before the window
        t0 = time.perf_counter()
        for _ in range(n):
            _collect(engine.step())
        _sync()
        return (time.perf_counter() - t0) / n * 1000.0

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
        prefill_steps += 1
        for o in outs:
            if o.outputs and o.outputs[0].token_ids:
                stepped.add(o.request_id)
        if prefill_steps > 100000:
            raise SystemExit('prefill did not converge')
    prefill_s = time.perf_counter() - t0
    print(f'[prefill] {nseq} seqs, {prefill_steps} engine steps, '
          f'{prefill_s:.1f}s (not timed)', flush=True)

    # ---- 2. warm decode: EPLB learning window ----------------------------
    warn_s = time_window(a.warm_steps)
    print(f'[warm] {a.warm_steps} decode steps: {warn_s:.2f} ms/step '
          f'(device-synced window)', flush=True)

    if os.environ.get('GC_MOE_SLOT_STATS') == '1':
        deepep_compare.dump_slot_stats(a.label)

    # The registry lives in the *worker* processes, so the replan must run
    # there: collective_rpc broadcasts the call to every worker.
    rearranged = False
    if os.environ.get('GC_MOE_EPLB') == '1':
        try:
            outs = engine.collective_rpc('_gc_moe_rearrange')
            rearranged = any(bool(x) for x in outs)
        except Exception as exc:  # noqa: BLE001
            print(f'[eplb] rearrangement FAILED: {exc}', flush=True)
            raise
    print(f'[eplb] rearrangement {"done" if rearranged else "not applicable"}',
          flush=True)

    # ---- 3. measured decode ---------------------------------------------
    # A few throw-away steps: the first forwards after the EPLB weight rewrite
    # (or after any host-side stall) are not representative, and one 400 ms
    # outlier in a 100-step average is worth 4 ms/step -- larger than the
    # effect we are trying to measure.
    flush = int(os.environ.get('GC_MEASURE_FLUSH', '5'))
    for _ in range(flush):
        _collect(engine.step())
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
    rec['on'] = True
    t0 = time.perf_counter()
    for _ in range(a.measure_steps):
        _collect(engine.step())
    t_submit_s = time.perf_counter() - t0
    rec['on'] = False
    _sync()                       # last real work still queued on the GPUs
    window_s = time.perf_counter() - t0
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
            trs = engine.collective_rpc('_gc_moe_eco_trace_dump')
            eco_trace = {'ranks': [x for x in trs if isinstance(x, dict) and x.get('ok')]}
        except Exception as exc:  # noqa: BLE001 - never fail a run over telemetry
            eco_trace = {'error': str(exc)}

    def _stat(xs):
        return (min(xs), max(xs), statistics.median(xs)) if xs else (-1, -1, -1)

    running = rec['running']
    run_lo, run_hi, _ = _stat(running)
    gen_lo, gen_hi, gen_med = _stat(rec['gen_tokens'])
    ctx_lo, ctx_hi, _ = _stat(rec['ctx_tokens'])
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
                  prefill_s=prefill_s, warm_ms_per_step=warn_s,
                  timing='device_synced_window',
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
                  gen_tokens_total=sum(len(v) for v in got.values()),
                  tokens={k: v for k, v in sorted(got.items())})

    print('RESULT', json.dumps(result), flush=True)
    with open(a.output, 'w') as fh:
        json.dump(result, fh, indent=2)


if __name__ == '__main__':
    main()
