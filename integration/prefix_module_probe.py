"""Diagnostic-only eager boundaries around the first divergent decoder block."""
import re
import torch
from stream_checkpoint import tensor_hash


def tensors(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), digest=tensor_hash(value))
    if isinstance(value, (tuple, list)):
        return [tensors(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensors(v) for k, v in value.items() if isinstance(v, (torch.Tensor, tuple, list, dict))}
    return None


def install(runner, ref, capturing):
    ref.block_records = []

    def observer(name, edge):
        def record(module, args, kwargs, output=None):
            step = len(ref.hidden_samples)
            if step in (12, 13) and not capturing():
                values = (args, kwargs) if edge == "in" else output
                ref.block_records.append(dict(step=step, module=name, edge=edge, tensors=tensors(values)))
        return record

    for name, module in runner.model.named_modules():
        if re.search(r"(?:^|\.)layers\.(?:19|20)(?:\.[^.]+){0,2}$", name):
            module.register_forward_pre_hook(observer(name, "in"), with_kwargs=True)
            module.register_forward_hook(observer(name, "out"), with_kwargs=True)
