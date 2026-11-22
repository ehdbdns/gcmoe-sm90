"""Bridge one GPU globaltimer reading to a named Kineto kernel interval."""
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer


@triton.jit
def sm100_phase_clock_anchor(ptr):
    tl.store(ptr, globaltimer())


anchor = sm100_phase_clock_anchor
