"""# [ECO-SM90] ECO (EPLB-placed capped online split) helpers.

`meta_sm90` builds the host-side EcoMeta the SM90 plan role reads; the plan
itself lives in `impls/sm90_fp8_fp4_gc_moe.py` (role 5).
"""
from .meta_sm90 import K_EMPTY, K_PASSIVE, build_eco_meta, read_copies

__all__ = ['K_EMPTY', 'K_PASSIVE', 'build_eco_meta', 'read_copies']
