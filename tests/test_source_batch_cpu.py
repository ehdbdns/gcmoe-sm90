"""Validate source row ownership, reconstruction and M on the six H20 workloads."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from source_batch import SIZES, routed_m, run_local_moe


def test_source_reconstruction():
    for batch, height in zip(SIZES, (8, 8, 16, 16, 32, 64)):
        assert routed_m(batch) == height
        ids = torch.arange(batch * 8).view(-1, 1)
        hidden, weights = 3 * ids + 1, ids + 13
        expected = hidden * weights + ids
        gathered = []
        for rank in range(8):
            def run(output, x, w, i):
                assert torch.equal(i, ids[rank * batch:(rank + 1) * batch])
                output.copy_(x * w + i)

            def gather(result, dim):
                assert dim == 0
                gathered.append(result)
                return expected

            group = SimpleNamespace(world_size=8, rank_in_group=rank, all_gather=gather)
            output = torch.empty_like(expected)
            run_local_moe(group, rank, batch, output, hidden, weights, ids, run)
            assert torch.equal(output, expected)
        assert torch.equal(torch.cat(gathered), expected)


if __name__ == "__main__":
    test_source_reconstruction()
    print("source rows and reconstruction passed at six batch sizes")
