"""SM90 eight-arm model adapters. Enabled only by the eight-arm launcher."""
import os
from functools import wraps

import torch
import triton
import triton.language as tl


@triton.jit
def _mark(counter, stamps, layer: tl.constexpr, end: tl.constexpr):
    step = tl.load(counter + layer)
    now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                    args=[], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(stamps + ((step % 1000) * 40 + layer) * 2 + end, now)
    if end:
        tl.store(counter + layer, step + 1)


def _wrap_forward(original, counter, stamps, layer):
    @wraps(original)
    def measured(*args, **kwargs):
        _mark[(1,)](counter, stamps, layer, False)
        result = original(*args, **kwargs)
        _mark[(1,)](counter, stamps, layer, True)
        return result
    return measured


def install_clocks(model, buf):
    layers = [m for m in model.modules() if hasattr(m, "ffn") and hasattr(m, "ffn_norm")]
    if len(layers) != 40:
        raise ValueError(f"expected 40 V4.1 decoder FFNs, found {len(layers)}")
    buf.ffn_counter = torch.zeros(40, dtype=torch.int32, device=buf.buffer.device)
    buf.ffn_stamps = torch.zeros((1000, 40, 2), dtype=torch.int64, device=buf.buffer.device)
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
    buf.shared_paths = []
    for i, layer in enumerate(layers):
        # A deferred FFN delegates the shared add to the next decoder boundary.
        # All eight arms need the locally finalized FFN contract.
        if getattr(layer, "fuse_mhc_all_reduce", False) or layer.ffn.defers_finalize(256):
            raise ValueError("FFN finalization extends past this layer")
        shared = [m for m in layer.ffn.modules() if isinstance(m, SharedExperts)]
        if len(shared) != 1:
            raise ValueError(f"layer {i} has {len(shared)} shared MLP runners")
        shared_type = type(shared[0]._layer).__name__
        if (shared_type == "_ZeroShared") != bool(buf.num_shared_experts):
            raise ValueError(f"layer {i}: inconsistent shared expert owner")
        buf.shared_paths.append(shared_type)
        layer.ffn.forward = _wrap_forward(layer.ffn.forward, buf.ffn_counter, buf.ffn_stamps, i)


def mega_weights(w13, s13, w2, s2):
    import deep_gemm
    from gc_moe.sm90_gc import _scale_codes
    from sm90_mega_weights import convert_projection

    def grouped(data, scales):
        e, n, half = data.shape
        w = torch.empty((e, n, 2 * half), device=data.device, dtype=torch.float8_e4m3fn)
        sf = torch.empty((e, n // 128, half // 64), device=data.device, dtype=torch.float32)
        for i in range(e):
            codes = _scale_codes(scales[i]).to(torch.uint8).view(torch.float8_e8m0fnu)
            wi, si, _ = convert_projection(data[i], codes)
            w[i].copy_(wi)
            sf[i].copy_(si)
        return w, sf

    return deep_gemm.transform_weights_for_mega_moe_sm90(grouped(w13, s13), grouped(w2, s2))


def init_mega(buf, hidden, intermediate):
    import deep_gemm
    from pathlib import Path
    if Path(os.environ["SM90_DG_SITE"]).resolve() not in Path(deep_gemm.__file__).resolve().parents:
        raise RuntimeError(f"wrong official DeepGEMM import: {deep_gemm.__file__}")
    buf.mega = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        buf.group, buf.physical, buf.num_max_tokens_per_rank, buf.num_topk, hidden, intermediate)
    buf.mega.topk_idx.fill_(-1)
    buf.mega.topk_weights.zero_()
    buf.mega.x.zero_()
    buf.mega.x_sf.fill_(1)
    buf.mega_received = torch.zeros(buf.physical // buf.group.size(), device=buf.buffer.device, dtype=torch.int32)


@triton.jit
def _record_received(received, total, epoch, last, ring, history,
                     width: tl.constexpr, rows: tl.constexpr, has_total: tl.constexpr,
                     block: tl.constexpr):
    i = tl.arange(0, block)
    n = tl.load(received + i, i < width, 0)
    seq = tl.load(epoch)
    tl.store(ring + (seq % 256) * width + i, n, i < width)
    if rows:
        start = (seq % rows) * (width + 1)
        tl.store(history + start + i, n, i < width)
        tl.store(history + start + width, seq)
    if has_total:
        before = tl.load(total + i, i < width, 0)
        tl.store(total + i, before + n, i < width)
    tl.store(received + i, 0, i < width)
    tl.store(last, seq)
    tl.store(epoch, seq + 1)


def launch_mega(buf, pair, output, tokens, stats):
    import deep_gemm
    target = buf.mega
    deep_gemm.fp8_mega_moe(output[:tokens], *pair, target, cumulative_local_expert_recv_stats=buf.mega_received,
                          recipe=(128, 128, 128), activation="swiglu", activation_clamp=10., fast_math=True)
    counts = buf.mega_received
    history = getattr(buf, "placement_history", None)
    _record_received[(1,)](counts, counts if stats is None else stats, buf.header,
                           buf.eco_tl_step, buf.profile_slot_counts, counts if history is None else history,
                           counts.numel(), 0 if history is None else len(history), stats is not None,
                           triton.next_power_of_2(counts.numel()), num_warps=1)


def count_graph_replays(buf):
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    original = CudaGraphManager.run_fullgraph
    buf.full_replays = 0

    @wraps(original)
    def counted(manager, desc):
        result = original(manager, desc)
        if desc.num_tokens == 256:
            buf.full_replays += 1
        return result

    CudaGraphManager.run_fullgraph = counted
    return lambda: setattr(CudaGraphManager, "run_fullgraph", original)
