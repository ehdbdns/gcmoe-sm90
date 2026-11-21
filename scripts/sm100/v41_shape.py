"""Check against the DeepSeek-V4.1-Flash used in the SM90 main table."""
import hashlib
import json
from pathlib import Path
import sys


def model_shape(root):
    raw = (Path(root) / "config.json").read_bytes()
    cfg = json.loads(raw)
    expected = dict(hidden_size=5120, moe_intermediate_size=2304, n_routed_experts=384,
                    num_experts_per_tok=6, num_hidden_layers=40)
    actual = {k: cfg.get(k) for k in expected}
    if "DeepseekV41ForCausalLM" not in cfg.get("architectures", []) or actual != expected:
        raise ValueError(f"Incompatible with SM90 V4.1-Flash: {cfg.get('architectures')}, {actual}")
    return {**actual, "config_sha256": hashlib.sha256(raw).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(model_shape(sys.argv[1])))
