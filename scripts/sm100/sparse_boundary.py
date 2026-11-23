"""Read-only L20 window samples for the BS512 sparse-indexer control."""
from functools import wraps

import torch


def install():
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4DecoderLayer
    original = DeepseekV4DecoderLayer.__init__

    @wraps(original)
    def init(layer, *args, **kwargs):
        original(layer, *args, **kwargs)
        if layer.ffn.experts.layer_id != 20:
            return
        layer.attn._bs512_sample = {}

        def record(module, inputs, out):
            positions, x = inputs[:2]
            if x.shape[0] != 512:
                return
            samples = module._bs512_sample
            # One row from each contiguous 64-row shard. Store local compressed
            # positions, before translating them into physical cache addresses.
            for name, tensor in (("positions", positions), ("indices", module.topk_indices_buffer[:512]),
                                 ("input", x), ("output", out)):
                view = tensor[::64]
                if name not in samples:
                    samples[name] = torch.empty_like(view)
                samples[name].copy_(view)
        layer.attn.register_forward_hook(record)
    DeepseekV4DecoderLayer.__init__ = init


def check(worker, phase, step):
    import numpy as np
    from five_worker import first_gate_difference
    from full_decode_worker import STATE

    m = worker._matrix
    module, = [mod for mod in worker.get_model().modules() if hasattr(mod, "_bs512_sample")]
    data = module._bs512_sample
    if not data:
        raise RuntimeError("L20 attention marker did not run")
    np.savez(m.folder / f"sparse-{phase}-{step}-rank{m.rank}.npz", **{
        name: (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).cpu().numpy()
        for name,t in data.items()})
    if m.publisher:
        return None
    n = int(STATE.cursor.item())
    if n != (step+1)*40:
        raise RuntimeError(f"incomplete route window: {n} calls at {step}")
    gate = first_gate_difference(m.raw_ids[:n], m.bank_ids[:n], m.raw_weights[:n], m.bank_weights[:n])
    offsets = m.phase_start[:,None] + torch.arange(step+1, device=m.phase_start.device)
    rows = torch.arange(len(offsets),device=offsets.device)[:,None]
    diff = (m.tokens.native[rows,offsets] != m.tokens.table[rows,offsets]).T
    token = diff.nonzero()[0].tolist() if diff.any().item() else None
    return dict(rank=m.rank, gate=gate, first_token_step_request=token) if gate or token else None
