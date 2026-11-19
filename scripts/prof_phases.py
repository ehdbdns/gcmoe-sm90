"""Torch-profile the GC-MoE phases per rank, on the NATIVE kernel (eager).

WHY EAGER. `gc_moe` normally packs its roles into ONE child CUDA graph and
replays it, so a torch profile of decode shows a single graph launch and none
of the internal structure. With ENFORCE_EAGER=1 no graph is captured and the
roles launch as separate kernels, which is exactly what we need:

    role 1  dispatch  routing + count exchange, then token pull
    role 4  shared    shared expert L1 + L2   (1 shared expert, as in V4.1)
    role 2  compute   routed GEMM L1 + L2
    role 3  combine   write tokens back to their owner ranks

Each worker profiles its OWN stream (via the _gc_moe_prof_start/stop RPCs) and
writes its own chrome trace, so the analyser can plot one decode step with the
8 ranks on the y axis at absolute time -- which answers "does dispatch overlap
the shared expert's compute?".
"""
import argparse
import json
import os
import sys

# Register the gc_moe backend and the vLLM warmup shim EXACTLY as the
# production driver does; without these, vLLM rejects moe_backend='gc_moe'
# for MXFP4 ("not supported for MXFP4 MoE").
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'patches'))
sys.path.insert(0, os.path.join(_HERE, '..', 'integration'))
import vllm_warmup_compat  # noqa: E402

vllm_warmup_compat.install()
import deepep_compare  # noqa: E402,F401  (registers the gc_moe backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, help='dir for the traces')
    ap.add_argument('--label', default='PROF')
    ap.add_argument('--seqs-per-rank', type=int, default=32)
    ap.add_argument('--prefill-lo', type=int, default=256)
    ap.add_argument('--prefill-hi', type=int, default=2048)
    ap.add_argument('--prof-steps', type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    ranks = int(os.environ.get('WORLD', '8'))
    nseq = a.seqs_per_rank * ranks
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.config.compilation import CUDAGraphMode

    chunk = int(os.environ.get('MAX_BATCHED', '2048'))
    kv_pool = int(os.environ.get('GC_KV_BUDGET', '1150000'))
    from real_decode_100_100 import load_prompts

    prompts = load_prompts(nseq, a.prefill_lo, a.prefill_hi)
    tot = sum(len(p) for p in prompts)
    slack = 2 * (-(-tot // chunk)) + 64
    budget = 1 + a.prof_steps + 64 + slack
    os.environ['GC_KV_OVERHEAD'] = str(budget + 512 + 64)
    max_len = max(len(p) for p in prompts)

    llm = LLM(
        model=os.environ['MODEL'], tensor_parallel_size=ranks,
        enable_expert_parallel=True, enable_prefix_caching=False,
        max_model_len=max_len + budget + 256,
        gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.80')),
        max_num_seqs=nseq,
        moe_backend=os.environ.get('MOE_BACKEND', 'gc_moe'),
        all2all_backend=os.environ.get('ALL2ALL_BACKEND',
                                       'allgather_reducescatter'),
        max_num_batched_tokens=chunk,
        trust_remote_code=True, dtype='bfloat16', disable_log_stats=False,
        enable_logging_iteration_details=True,
        # Same dict form the production script uses; only the graph mode is
        # switched off so gc_moe launches its roles as separate kernels.
        compilation_config={'cudagraph_mode': CUDAGraphMode.NONE},
        enforce_eager=True)          # ★ no child graph -> separate launches
    engine = llm.llm_engine

    sp = SamplingParams(max_tokens=1 + a.prof_steps + 8, temperature=0.0,
                        top_p=1.0, output_kind=RequestOutputKind.FINAL_ONLY)

    # ---- 1. prefill (NOT profiled) -------------------------------------
    llm.generate(prompts, sp, use_tqdm=False)
    print('[prof] prefill done', flush=True)

    # [EPLB] Rearrange the placement NOW, exactly as the production driver does
    # between its warm and measured windows. Profiling the identity placement
    # shows an artificially balanced picture: the whole point of EPLB is that
    # the hot experts get replica slots spread over the ranks.
    try:
        outs = engine.collective_rpc('_gc_moe_rearrange')
        done = any(bool(x) for x in outs)
        print(f'[prof] EPLB rearrangement {"done" if done else "not applicable"}',
              flush=True)
    except Exception as exc:                       # noqa: BLE001
        print(f'[prof] EPLB rearrangement FAILED: {exc}', flush=True)
    # a few eager steps so the new placement is in effect before recording
    llm.generate(prompts, sp, use_tqdm=False)
    print('[prof] post-rearrange warm done', flush=True)

    # ---- 2. profile a few decode steps on every worker ------------------
    trace_dir = os.path.join(a.output, 'traces')
    os.makedirs(trace_dir, exist_ok=True)
    engine.collective_rpc('_gc_moe_prof_start', args=(trace_dir,))
    print('[prof] profiling...', flush=True)
    llm.generate(prompts, sp, use_tqdm=False)
    info = engine.collective_rpc('_gc_moe_prof_stop')
    # [PHASES] Also dump the in-kernel phase stamps (pull_tl slots 10/11/15/
    # 16/17): torch profiler only sees KERNEL boundaries, so "when did the
    # first compute CTA get a task" and the shared expert's own span have to
    # come from the kernel's own timestamps.
    try:
        dumps = engine.collective_rpc('_gc_moe_eco_trace_dump')
        import json as _json
        with open(os.path.join(a.output, 'pull_rows.json'), 'w') as fh:
            _json.dump([d for d in dumps if isinstance(d, dict)], fh)
        print(f'[prof] pull_rows dumped ({len(dumps)} ranks)', flush=True)
    except Exception as exc:                       # noqa: BLE001
        print(f'[prof] pull_rows dump failed: {exc}', flush=True)
    ok = [x for x in info if isinstance(x, dict) and x.get('ok')]
    print(f'[prof] {len(ok)}/{len(info)} workers exported', flush=True)
    for x in info[:2]:
        print('   ', json.dumps(x)[:160], flush=True)
    print(f'[prof] traces in {trace_dir}', flush=True)


if __name__ == '__main__':
    main()
