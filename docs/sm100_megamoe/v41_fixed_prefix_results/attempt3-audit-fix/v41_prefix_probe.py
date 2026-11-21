def moe_runtime(model):
    """Read post-load module state, including fused weights on each worker."""
    result = []
    for name, module in model.named_modules():
        if name.endswith(".ffn") and hasattr(module, "experts"):
            experts = module.experts
            result.append(dict(layer=name, expert_class=type(experts).__name__,
                               sequence_parallel=getattr(module, "use_sequence_parallel", None),
                               fused_shared=getattr(experts, "has_fused_shared_experts", None)))
    return result
