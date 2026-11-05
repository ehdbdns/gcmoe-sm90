"""vLLM MoE backend backed by GC MoE (github.com/Bruce-Lee-LY/gc-moe).

Importing this module registers the ``gc_moe`` MoE backend, so it must be
imported before the LLM engine builds its kernels. The benchmark scripts do
that through PYTHONPATH + an explicit import.

Design
------
GC MoE is a single operator that owns all three MoE phases -- dispatch, expert
GEMM, combine. vLLM's modular MoE interface splits those across
prepare/experts/finalize, so:

  * prepare   -- quantize to FP8 (K128 blocks) and stage the tokens plus their
                 routing into GC MoE's symmetric buffer; no communication.
  * experts   -- one ``sm90_fp8_fp4_gc_moe`` call does dispatch *and* combine
                 internally; the returned BF16 output is already top-k reduced.
  * finalize  -- identity, because GC MoE already applied the routing weights
                 and summed the top-k contributions.

That makes GC MoE's measured cost directly comparable to a DeepEP LL
dispatch + expert GEMM + combine pipeline.
"""
from . import deepep_compare  # noqa: F401  (registers the backend on import)

__all__ = ['deepep_compare']
