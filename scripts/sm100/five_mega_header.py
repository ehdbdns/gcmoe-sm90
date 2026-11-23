"""Prepare a run-private copy of the installed MegaMoE package with CTA clocks."""

import argparse
import os
from pathlib import Path
import shutil


def instrument(source):
    def once(text, anchor, replacement):
        if text.count(anchor) != 1:
            raise ValueError(f"unexpected MegaMoE header at {anchor!r}")
        return text.replace(anchor, replacement)

    anchor = "    const uint32_t thread_idx = threadIdx.x;"
    source = once(
        source,
        anchor,
        anchor
        + """
    auto matrix_stamp = [&](int field) {
        if (cumulative_local_expert_recv_stats != nullptr) {
            auto *clock = reinterpret_cast<unsigned long long *>(
                cumulative_local_expert_recv_stats + kNumExpertsPerRank) + field;
            unsigned long long ns;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
            if (field < 2) atomicMin(clock, ns); else atomicMax(clock, ns);
        }
    };
    if (thread_idx == 0) matrix_stamp(0);""",
    )
    anchor = "                    const uint32_t a_desc_base_lo = a_desc_lo"
    source = once(
        source,
        anchor,
        """                    if (k_block_idx == 0 && !task_info.is_shared() && cute::elect_one_sync())
                        matrix_stamp(1);
"""
        + anchor,
    )
    anchor = "                // Ensure the next epilogue safe to use shared memory\n                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);"
    source = once(
        source,
        anchor,
        anchor
        + """
                if (!task_info.is_shared() && epilogue_warp_idx == 0 && cute::elect_one_sync())
                    matrix_stamp(2);""",
    )
    anchor = "\n    }\n#else\n    if (blockIdx.x == 0 and threadIdx.x == 0)"
    return once(
        source, anchor, "\n        if (cumulative_local_expert_recv_stats && cute::elect_one_sync()) { ptx::tma_store_wait<0>(); matrix_stamp(3); }" + anchor
    )


if __name__ == "__main__":
    from vllm.utils.deep_gemm import _import_deep_gemm

    p = argparse.ArgumentParser()
    p.add_argument("overlay", type=Path)
    a = p.parse_args()
    src = Path(_import_deep_gemm().__file__).resolve().parent
    dest = a.overlay / "deep_gemm"
    shutil.copytree(
        src, dest, symlinks=False, ignore=shutil.ignore_patterns("__pycache__")
    )
    # A stats view in this private build must own an aligned four-clock suffix.
    api = dest / "mega" / "__init__.py"
    with api.open("a") as f:
        f.write("""
from functools import wraps as _matrix_wraps
_matrix_original = fp8_fp4_mega_moe
@_matrix_wraps(_matrix_original)
def fp8_fp4_mega_moe(*args, **kwargs):
    counters = kwargs.get("cumulative_local_expert_recv_stats", args[6] if len(args)>6 else None)
    if counters is not None:
        end = (counters.storage_offset()+counters.numel())*counters.element_size()
        if (not getattr(counters,"_matrix_clock_scratch",False)
                or not counters.is_contiguous() or end % 8
                or counters.untyped_storage().nbytes()-end < 32):
            raise ValueError("MegaMoE matrix counters require four adjacent globaltimer slots")
    return _matrix_original(*args, **kwargs)
""")
    header = dest / "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
    if os.getenv("SM100_VLLM_COMBINE_ORDER") == "1":
        if "const uint32_t slot_idx = __ffs(mask) - 1;" not in header.read_text():
            raise RuntimeError("Installed Mega combine slot order differs from the pinned vLLM fork")
    source = instrument(header.read_text())
    if os.getenv("SM100_PHASE_TRACE"):
        from native_phase_instrument import mega
        source = mega(source)
        api.write_text(api.read_text().replace("nbytes()-end < 32", "nbytes()-end < 152"))
    header.write_text(source)
    print(f"private instrumented package {src} -> {dest}")
