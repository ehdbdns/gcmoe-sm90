"""Hash the effective sparse prefill inputs without reading unused KV workspace."""
from contextvars import ContextVar
from functools import wraps

import torch

from prefix_module_probe import tensors


def sparse_inputs(kwargs, heads):
    ids = kwargs["indices"][:, 0, :]
    lens = kwargs["topk_length"].reshape(-1)
    if len(lens) != len(ids) or (lens < 0).any() or (lens > ids.size(1)).any():
        raise ValueError("invalid sparse prefill lengths")
    active = torch.arange(ids.size(1), device=ids.device)[None, :] < lens[:, None]
    masked = torch.where(active, ids, -1)
    slots = masked[masked >= 0].unique(sorted=True).long()
    kv = kwargs["kv"]
    if slots.numel() and slots[-1] >= len(kv):
        raise ValueError("sparse prefill index outside gathered KV")
    # Keep duplicates and request boundaries in the multiset comparison.
    return dict(query=tensors(kwargs["q"][:, :heads]), lengths=tensors(lens),
                ordered=tensors(masked), sorted_per_row=tensors(masked.sort(dim=-1).values),
                slots=tensors(slots), selected_kv=tensors(kv.index_select(0, slots)),
                sink=tensors(kwargs["attn_sink"][:heads]) if kwargs["attn_sink"] is not None else None,
                scale=float(kwargs["sm_scale"]))


def install(ref, capturing):
    from vllm.models.deepseek_v41.nvidia import flashmla
    cls = flashmla.DeepseekV4FlashMLAAttention
    prefill = cls._forward_prefill
    sparse = flashmla.flash_mla_sparse_fwd
    active = ContextVar("gc_prefix_sparse_heads", default=None)

    @wraps(prefill)
    def observe_prefill(self, *args, **kwargs):
        if self.layer_id != 20 or len(ref.hidden_samples) not in (12, 13) or capturing():
            return prefill(self, *args, **kwargs)
        token = active.set(self.n_local_heads)
        try:
            return prefill(self, *args, **kwargs)
        finally:
            active.reset(token)

    @wraps(sparse)
    def observe_sparse(*args, **kwargs):
        heads = active.get()
        if heads is None:
            return sparse(*args, **kwargs)
        record = dict(step=len(ref.hidden_samples), module="L20.flash_mla_sparse_fwd")
        ref.block_records.append(dict(record, edge="in", tensors=sparse_inputs(kwargs, heads)))
        result = sparse(*args, **kwargs)
        ref.block_records.append(dict(record, edge="out", tensors=tensors(kwargs["out"][:, :heads])))
        return result

    cls._forward_prefill = observe_prefill
    flashmla.flash_mla_sparse_fwd = observe_sparse
