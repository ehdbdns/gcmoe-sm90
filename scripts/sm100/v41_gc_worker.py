"""vLLM worker extensions for the Blackwell V4.1 comparison."""
from functools import wraps


class MoeAuditWorker:
    def inspect_v41_moe(self):
        rows = []
        for name, module in self.get_model().named_modules():
            if name.endswith(".ffn") and hasattr(module, "experts"):
                experts = module.experts
                rows.append(dict(layer=name, expert_class=type(experts).__name__,
                                 sequence_parallel=getattr(module, "use_sequence_parallel", None),
                                 fused_shared=getattr(experts, "has_fused_shared_experts", None)))
        return rows


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

    DeepseekV4MegaMoEExperts.finalize_weights = finalize_gc


install_gc_experts()
