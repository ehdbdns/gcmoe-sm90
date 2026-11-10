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


def load_prompts(n, lo, hi):
    """Collect `n` prompts of length [lo, hi] from one or more jsonl files.

    DATASET may be a comma-separated list: the short-prompt slice of the
    NAS corpus is spread over both ``_1`` and ``_2``, and a single file does
    not always hold enough sequences for 256 streams.
    """
    out = []
    for path in DATASET.split(','):
        path = path.strip()
        if not path:
            continue
        with open(path) as fh:
            for line in fh:
                if len(out) == n:
                    break
                if not line.strip():
                    continue
                ids = json.loads(line).get('input_ids') or []
                if lo <= len(ids) <= hi:
                    out.append(ids)
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f'only {len(out)} prompts in [{lo},{hi}], need {n}')
    return out


def _compilation_config():
    from vllm.config.compilation import CUDAGraphMode
    mode = os.environ.get('CUDAGRAPH_MODE', 'FULL_DECODE_ONLY')
    if mode == 'default':
        return None
    return {'cudagraph_mode': getattr(CUDAGraphMode, mode)}


def _irstep_reduce(rows):
    """# [IR-STEP] avg over (step, layer) of max_rank/mean_rank.

    `rows[t][r]` is rank r's per-layer token vector for step t. For every
    (step, layer) pair we take

        IR = max_r tok[r, layer] / mean_r tok[r, layer]

    and report the mean over all pairs (the "A" definition: layer is NOT
    unfolded into extra weight -- every pair counts once).

    A Poisson NULL BASELINE is returned alongside. One layer of one step
    carries only `tokens_total / 8` routes per rank, so even a perfectly
    balanced placement gives IR > 1 purely from sampling noise:
        sigma = sqrt(n),  E[max of 8 draws] ~ mean + sigma * 1.42
    Without this number an IR of 1.03 looks like imbalance when it may be
    nothing but noise.
    """
    import math
    import statistics

    # rows[0] is the baseline snapshot (warm), rows[1:] carry the deltas.
    steps = [r for r in rows[1:] if r]
    if not steps:
        return {'ok': False, 'reason': 'no samples'}

    irs, per_layer_irs, totals, nulls = [], {}, [], []
    n_rank = max(len(r) for r in steps)
    for row in steps:
        if len(row) != n_rank:
            continue
        vecs = []
        for r in row:
            pl = r.get('per_layer_tokens')
            if not pl:
                vecs = []
                break
            vecs.append(pl)
        if not vecs:
            continue
        n_layer = min(len(v) for v in vecs)
        step_total = 0.0
        for layer in range(n_layer):
            vals = [v[layer] for v in vecs]
            m = sum(vals) / len(vals)
            if m <= 0:
                continue
            irs.append(max(vals) / m)
            per_layer_irs.setdefault(layer, []).append(max(vals) / m)
            step_total += m
            # Poisson null for this (step, layer): same total, even split, but
            # with sqrt(n) sampling noise per rank.
            n_route = m
            nulls.append(1.0 + 1.42 * math.sqrt(max(n_route, 1e-9)) / n_route)
        totals.append(step_total)

    if not irs:
        return {'ok': False, 'reason': 'no positive (step, layer) pair'}

    def _q(xs, p):
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
        return xs[k]

    return {
        'ok': True,
        'ir_step_layer_mean': statistics.mean(irs),
        'ir_step_layer_p50': _q(irs, 0.50),
        'ir_step_layer_p90': _q(irs, 0.90),
        'ir_step_layer_p99': _q(irs, 0.99),
        'ir_step_layer_max': max(irs),
        'ir_poisson_null_mean': statistics.mean(nulls),
        # excess over what noise alone explains
        'ir_excess_over_null': statistics.mean(irs) - statistics.mean(nulls),
        'n_pairs': len(irs),
        'n_steps': len(totals),
        'n_ranks': n_rank,
        'routes_per_rank_per_layer': statistics.mean(
            [t / max(1, len(steps)) for t in totals]) if totals else -1.0,
        'per_layer_ir_mean': {str(k): statistics.mean(v)
                              for k, v in sorted(per_layer_irs.items())},
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
    prompts = load_prompts(nseq, a.prefill_lo, a.prefill_hi)
    max_len = max(len(p) for p in prompts)
    print(f'[data] {nseq} prompts ({a.seqs_per_rank}/rank x {ranks}), '
          f'len {min(map(len,prompts))}..{max_len}', flush=True)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    chunk = int(os.environ.get('MAX_BATCHED', '8192'))
    # Chunked prefill finishes the 256 streams at *different* engine steps: the
    # first one after 1 step, the last after ~N/chunk steps. If every request
    # only asks for `1 + warm + measure` tokens, the ones that started early
    # retire before the measurement window ends (observed: Running fell
    # 256 -> 188 inside the timed window, so the numbers described a shrinking
    # batch, not a steady 32 streams/rank). The margin keeps every stream alive
    # past the last measured step; it only costs KV space, never GPU time.
    margin = int(os.environ.get('GC_PREFILL_MARGIN', '512'))
    llm = LLM(
        model=os.environ['MODEL'], tensor_parallel_size=8,
        enable_expert_parallel=True, enable_prefix_caching=False,
        max_model_len=max_len + a.warm_steps + a.measure_steps + margin + 64,
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
    total = 1 + a.warm_steps + a.measure_steps + margin

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

    # [IR-STEP] Separate sampling window: one max/mean over the 8 ranks PER
    # LAYER PER STEP, averaged over (step, layer) afterwards.
    #
    # It runs AFTER the timed window and its RPCs are excluded from ms/step on
    # purpose: `collective_rpc` costs ~60 ms and would otherwise inflate the
    # number we report.  Decode replays a CUDA graph, so the engine cannot
    # sample itself -- the harness has to drive one snapshot per step.
    irstep = {}
    if os.environ.get('GC_MEASURE_IR_STEP', '1') == '1':
        try:
            engine.collective_rpc('_gc_moe_irstep_begin')
            rows = []                      # rows[step][rank] = [layers]
            n_irstep = int(os.environ.get('GC_IRSTEP_STEPS',
                                          str(a.measure_steps)))
            for _ in range(n_irstep + 1):   # +1 establishes the baseline
                _collect(engine.step())
                outs = engine.collective_rpc('_gc_moe_irstep_sample')
                rows.append([x for x in outs if isinstance(x, dict) and x.get('ok')])
            irstep = _irstep_reduce(rows)
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a run
            irstep = {'error': str(exc)}

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
