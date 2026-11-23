"""Compare logical MoE boundary snapshots from the first native decode step."""
import json
import sys
from pathlib import Path

import numpy as np


def compare(root):
    arms = [root / arm for arm in ("mega", "gc")]
    prefill = [json.loads((arm / "prefill.json").read_text()) for arm in arms]
    report = {"prefill_tokens_equal": prefill[0] == prefill[1], "differences": []}
    for layer in range(4):
        for rank in range(8):
            path = f"layer{layer}-rank{rank}.npz"
            with np.load(arms[0] / path) as left, np.load(arms[1] / path) as right:
                for key in ("x", "sf", "ids", "weights", "y"):
                    a, b = left[key], right[key]
                    assert a.shape == b.shape and a.dtype == b.dtype and a.shape[0] == 64
                    if key in ("ids", "weights"):
                        for arm, values in zip(arms, (a, b)):
                            routes = np.load(arm / f"stats-{key}-rank{rank}.npy", mmap_mode="r")
                            assert np.array_equal(values, routes[layer]), (arm, layer, rank, key)
                    mask = a != b
                    if not mask.any():
                        continue
                    pos = tuple(map(int, np.unravel_index(mask.argmax(), mask.shape)))
                    if key == "y" and a.dtype == np.int16:
                        a, b = [(x.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
                                for x in (a, b)]
                    report["differences"].append(dict(layer=layer, rank=rank, tensor=key,
                        different=int(mask.sum()), elements=a.size, first=pos,
                        mega=a[pos].item(), gc=b[pos].item()))
    (root / "first_diff.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    compare(Path(sys.argv[1]))
