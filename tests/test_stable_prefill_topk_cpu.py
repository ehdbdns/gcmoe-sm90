"""Compare packed/strided inputs and cutoff ties against a per-row CPU oracle."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from stable_prefill_topk import select, select_decode


def test_rows():
    torch.manual_seed(27)
    scores = torch.randint(0, 4, (12, 38)).float()[:, ::2]
    scores[:, 3:9] = -torch.inf
    starts = torch.arange(12, dtype=torch.int32)
    ends = torch.tensor([19, 8, 9, 3, 8, 10, 13, 17, 13, 17, 18, 13])
    for k in (1, 5, 19, 23):
        out = torch.empty((12, k), dtype=torch.int32)
        select(scores, starts, ends, out, len(out), *scores.stride(), k)
        decoded = torch.empty_like(out)
        select_decode(scores, ends, decoded, k)
        for r, length in enumerate(ends.tolist()):
            row = scores[r, :length].tolist()
            ids = list(range(length))
            if length > k:
                ids = sorted(ids, key=lambda i: (-row[i], i))[:k]
            assert decoded[r].tolist() == ids + [-1] * (k - len(ids))
        for r in range(len(out)):
            row = scores[r, starts[r]:ends[r]].tolist()
            ids = list(range(len(row)))
            if len(ids) > k:
                ids = sorted(ids, key=lambda i: (-row[i], i))[:k]
            assert out[r].tolist() == ids + [-1] * (k - len(ids))


if __name__ == "__main__":
    test_rows()
    print("stable prefill top-k CPU PASS")
