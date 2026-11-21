"""Read only representative routed expert tensor headers, without loading weights."""
import json
from pathlib import Path
import re
import struct
import sys

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
pattern = re.compile(r"(?:^|\.)experts\.0\.")
keys = [k for k in sorted(index) if pattern.search(k)]
if not keys:
    raise SystemExit("No per-expert key found; report checkpoint naming scheme")
prefix = keys[0].split("experts.0.")[0]
keys = [k for k in keys if k.startswith(prefix + "experts.0.")]
headers = {}
for key in keys:
    shard = index[key]
    if shard not in headers:
        with (root / shard).open("rb") as f:
            length, = struct.unpack("<Q", f.read(8))
            if length > 100_000_000:
                raise ValueError("invalid safetensors header length")
            headers[shard] = json.loads(f.read(length))
    spec = headers[shard][key]
    print(json.dumps(dict(key=key, dtype=spec["dtype"], shape=spec["shape"])))
