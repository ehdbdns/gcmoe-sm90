"""Check per-budget placement, RR ordinal counting, and token IR aggregation."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/sm90"))
from sweep_eplb_extra import exact_layer, layouts, rebalance_experts_global


def test_token_ir():
    torch.set_num_threads(1)
    one = torch.randint(0, 150, (5, 1, 384), generator=torch.Generator().manual_seed(40))
    h = one.expand(-1, 40, -1).contiguous()
    ids = np.random.default_rng(31).integers(0, 384, (9, 24, 6))
    logical = np.stack([8 * np.bincount(i.ravel(), minlength=384) for i in ids])
    inverses, expected = [], []
    for k, (plan, inverse) in enumerate(layouts(h, 8, "cpu")):
        direct, direct_inverse, _ = rebalance_experts_global(h, 384+8*k, 48+k)
        np.testing.assert_array_equal(plan, direct.numpy())
        np.testing.assert_array_equal(inverse, direct_inverse.numpy()[:, :, :inverse.shape[2]])
        inv = inverse[7]
        inverses.append(inv)
        copies = (inv >= 0).sum(1)
        by_slot = np.zeros((9, 384+8*k), dtype=int)
        keys = (23333*np.arange(24))[:, None]
        for step, experts in enumerate(ids):
            for source in range(8):
                dst = inv[experts, (source+keys) % copies[experts]]
                by_slot[step] += np.bincount(dst.ravel(), minlength=384+8*k)
        rank_tokens = by_slot.reshape(9, 8, 48+k).sum(2)
        expected.append(np.mean(rank_tokens.max(1) / rank_tokens.mean(1)))
    np.testing.assert_allclose(exact_layer(ids, logical, inverses, by_slot, True), expected)


if __name__ == "__main__":
    test_token_ir()
    print("H20 EPLB token IR passes all nine budgets")
