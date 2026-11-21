"""Static host tables for the Blackwell online unit planner."""
import numpy as np

from .meta_sm90 import build_eco_meta, reorder_single_copy_first
from .unit_ref import unit_section


def build_eco_meta_sm100(phy, num_ranks, num_logical):
    core = build_eco_meta(phy, num_ranks, num_logical)
    ordered, _ = reorder_single_copy_first(phy, num_ranks)
    if not np.array_equal(ordered, phy):
        raise ValueError("Blackwell ECO requires the active single-copy prefix on each rank")
    # The CUDA plan role uses int4 loads relative to meta + header[15].
    start = (len(core) + 3) // 4 * 4
    core[15] = start
    return np.concatenate((core, np.zeros(start - len(core), np.int32),
                           unit_section(phy, num_ranks, num_logical)))
