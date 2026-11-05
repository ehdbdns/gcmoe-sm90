"""Point GC MoE's PTX assembler at the CUDA 13.4 ptxas shipped inside the env.

gc_moe/runtime/jit.py hardcodes /usr/local/cuda/bin/ptxas (13.2, PTX <= 9.2), but
CuTe DSL 4.8.1 built against cu13 libs emits PTX 9.4. The env's nvidia/cu13 tree
carries ptxas 13.4.92, which assembles it. Import before the first gc_moe launch.
"""
import os
import sys
from pathlib import Path

import gc_moe.runtime.jit as _jit

_candidates = [
    Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}'
    / 'site-packages' / 'nvidia' / 'cu13' / 'bin' / 'ptxas',
    Path('/usr/local/cuda/bin/ptxas'),
]
for _c in _candidates:
    if _c.is_file() and os.access(_c, os.X_OK):
        _jit.SYSTEM_PTXAS = str(_c)
        break
