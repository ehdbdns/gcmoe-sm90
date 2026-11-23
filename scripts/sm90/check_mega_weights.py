"""CPU-only qualification of the three routed projections of one real expert."""
import hashlib
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[2]
sys.path.extend([str(root / "integration"), str(root / "scripts/sm100")])
from real_routed_weights import ExpertCheckpoint
from sm90_mega_weights import convert_projection


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    ckpt = ExpertCheckpoint(sys.argv[1])
    for projection, n, k in (("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)):
        name = f"layers.0.ffn.experts.0.{projection}"
        w = ckpt.tensor(name+".weight", "I8", (n,k//2))
        sf = ckpt.tensor(name+".scale", "F8_E8M0", (n,k//32))
        fp8, scales, error = convert_projection(w, sf)
        print(json.dumps(dict(name=name, fp4_sha256=digest(w), scale_input_sha256=digest(sf),
                              fp8_sha256=digest(fp8), block_scale_sha256=digest(scales),
                              max_abs_error=error)), flush=True)
