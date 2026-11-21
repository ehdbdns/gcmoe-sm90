"""vLLM worker extensions for the Blackwell V4.1 comparison."""
from functools import wraps


_rr_calls = 0


class MoeAuditWorker:
    def install_v41_placement(self, filename):
        from v41_placement import install
        return install(self, filename)

    def inspect_rr_calls(self):
        return _rr_calls

    def inspect_v41_replicas(self):
        import hashlib
        rows = []
        for name, module in self.get_model().named_modules():
            if not name.endswith(".ffn") or not hasattr(module, "experts"):
                continue
            experts = module.experts
            state = experts.eplb_state
            counts = state.logical_replica_count.tolist()
            mapping = state.logical_to_physical_map.tolist()
            if len(counts) != 384 or min(counts) < 1 or len(mapping) != len(counts):
                raise RuntimeError(f"{name}: incomplete logical replica map")
            slots = [slot for row, n in zip(mapping, counts) for slot in row[:n]]
            if (sum(counts) != 448 or sorted(slots) != list(range(448))
                    or experts.num_local_experts != 56 or not experts.has_fused_shared_experts
                    or not module.use_sequence_parallel):
                raise RuntimeError(f"{name}: unexpected physical layout or shared fusion state")
            digest = hashlib.sha256(str((counts, slots)).encode()).hexdigest()
            rows.append(dict(layer=name, placement_sha256=digest))
        return rows

    def inspect_v41_moe(self):
        rows = []
        for name, module in self.get_model().named_modules():
            if name.endswith(".ffn") and hasattr(module, "experts"):
                experts = module.experts
                rows.append(dict(layer=name, expert_class=type(experts).__name__,
                                 sequence_parallel=getattr(module, "use_sequence_parallel", None),
                                 fused_shared=getattr(experts, "has_fused_shared_experts", None)))
        return rows


def preserve_loader_weight(finalize_shared):
    @wraps(finalize_shared)
    def finalize(self, deep_gemm, shared_experts):
        # vLLM assigns the fused payload to the loader Parameter. GC packs it
        # in 3D, while ModelOpt still needs the original 2D checkpoint matrix.
        weight = shared_experts.gate_up_proj.weight
        checkpoint = weight.data
        try:
            return finalize_shared(self, deep_gemm, shared_experts)
        finally:
            # vLLM retained a separate tensor in _transformed_shared_l1_weights.
            weight.data = checkpoint

    return finalize


def install_gc_experts():
    import os
    if os.environ.get("GC_MOE_V41_ENGINE") != "gc":
        return

    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
    from vllm.utils.deep_gemm import _import_deep_gemm

    finalize = DeepseekV4MegaMoEExperts.finalize_weights

    @wraps(finalize)
    def finalize_gc(self, *args, **kwargs):
        # Worker device and distributed environment are ready at weight finalization.
        from gc_moe import (fp8_fp4_gc_moe, get_block_m_for_gc_moe,
                            get_symm_buffer_for_gc_moe, transform_weights_for_gc_moe)
        dg = _import_deep_gemm()
        if dg.fp8_fp4_mega_moe is not fp8_fp4_gc_moe:
            dg.fp8_fp4_mega_moe = fp8_fp4_gc_moe
            dg.get_block_m_for_mega_moe = get_block_m_for_gc_moe
            dg.get_symm_buffer_for_mega_moe = get_symm_buffer_for_gc_moe
            dg.transform_weights_for_mega_moe = transform_weights_for_gc_moe
            print(f"V41 GC expert adapter installed pid={os.getpid()}", flush=True)
        return finalize(self, *args, **kwargs)

    DeepseekV4MegaMoEExperts._finalize_shared_expert_weights = preserve_loader_weight(
        DeepseekV4MegaMoEExperts._finalize_shared_expert_weights
    )
    DeepseekV4MegaMoEExperts.finalize_weights = finalize_gc


def install_rr():
    import os
    if os.environ.get("GC_MOE_V41_DEEPINFER_RR") != "1":
        return
    from vllm.models.deepseek_v4.nvidia import model
    from v41_rr import map_and_record

    def route(**kwargs):
        global _rr_calls
        _rr_calls += 1
        return map_and_record(model.get_ep_group().rank_in_group, **kwargs)

    model.eplb_map_to_physical_and_record = route


install_rr()
install_gc_experts()
