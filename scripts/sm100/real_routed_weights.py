"""Read packed layer-0 expert weights from the target V4-Flash safetensors shards."""
import json
import math
from pathlib import Path
import struct

import torch


class ExpertCheckpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.shards = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def tensor(self, key, dtype, shape):
        path = self.root / self.shards[key]
        with path.open("rb") as f:
            if path not in self.headers:
                length, = struct.unpack("<Q", f.read(8))
                if not 0 < length <= 100_000_000:
                    raise ValueError(f"Invalid header length in {path}")
                self.headers[path] = (8 + length, json.loads(f.read(length)))
            base, header = self.headers[path]
            spec = header[key]
            start, end = spec["data_offsets"]
            if spec["dtype"] != dtype or tuple(spec["shape"]) != shape or end-start != math.prod(shape):
                raise ValueError(f"Unexpected checkpoint tensor {key}: {spec}")
            f.seek(base + start)
            payload = bytearray(f.read(end-start))
            if len(payload) != end-start:
                raise ValueError(f"Short read for {key}")
        tensor = torch.frombuffer(payload, dtype=torch.uint8).reshape(shape)
        return tensor.view(torch.int8 if dtype == "I8" else torch.float8_e8m0fnu)

    def packed(self, experts, projections, rows, k):
        import deep_gemm

        n = rows * len(projections)
        w = torch.empty((len(experts), n, k // 2), device="cuda", dtype=torch.int8)
        sf = torch.empty((len(experts), n, k // 32), device="cuda", dtype=torch.float32)
        for i, expert in enumerate(experts):
            for j, projection in enumerate(projections):
                prefix = f"layers.0.ffn.experts.{expert}.{projection}"
                band = slice(j*rows, (j+1)*rows)
                w[i, band].copy_(self.tensor(prefix+".weight", "I8", (rows, k//2)))
                scale = self.tensor(prefix+".scale", "F8_E8M0", (rows, k//32))
                sf[i, band].copy_(scale.float())
        return w, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), len(experts))
