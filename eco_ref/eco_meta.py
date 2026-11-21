"""Host-side EcoMeta builder used by the CPU reference tests.

The implementation lives in the kernel source tree
(`source/gc_moe/eco/meta_sm90.py`) so the code that fills the buffer during a
run is exactly the code these tests exercise.
"""
from __future__ import annotations

import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'source'))

from gc_moe.eco.meta_sm90 import (  # noqa: E402
    COPIES_WORDS_PER_EXPERT, K_EMPTY, K_PASSIVE, build_eco_meta, read_copies,
)
from gc_moe.eco.meta_sm100 import build_eco_meta_sm100  # noqa: E402


def install_into_buffer(buf, phy, num_ranks, num_logical):
    from gc_moe.eco.meta_sm90 import install_into_buffer as _install
    return _install(buf, phy, num_ranks, num_logical)
