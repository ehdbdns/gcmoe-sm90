"""Check replica selection and vLLM load recording before loading the model."""
import argparse
import torch
from v41_rr import map_and_record


def check(device):
    ids = torch.tensor([[0, 1], [0, 1], [1, -1], [0, 2]], device=device)
    inverse = torch.tensor([[3, 0, -1], [4, 1, 2]], device=device)
    counts = torch.tensor([2, 3], device=device)
    for rank in range(8):
        expected = [[inverse[e, (rank + t * 23333) % int(counts[e])].item()
                     if 0 <= e < 2 else -1 for e in row] for t, row in enumerate(ids.tolist())]
        for enabled, limit in [(1, 3), (0, None), (1, None)]:
            loads = torch.zeros(5, dtype=torch.int32, device=device)
            out = map_and_record(rank, ids, loads, inverse, counts,
                                 torch.tensor(enabled, device=device),
                                 torch.tensor(limit, device=device) if limit is not None else None)
            assert out.tolist() == expected
            want = [0] * 5
            for row in expected[:limit]:
                for slot in row:
                    if enabled and slot >= 0:
                        want[slot] += 1
            assert loads.tolist() == want
    print("DeepInfer RR and physical load checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    check(parser.parse_args().device)
