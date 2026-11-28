"""Inspect the GLM checkpoint contract and the installed loader without importing CUDA."""
import argparse
import ast
from collections import Counter, defaultdict
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import re
import struct
import sys


MODEL_DIR = Path("/mnt/hf3fs/3fs/models")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.*)")
EXPERT = re.compile(r"((?:^|\.)(?:experts|routed_experts)\.)(\d+)\.")
INDEX = "model.safetensors.index.json"


def resolve(root):
    if root is not None:
        return root.resolve(strict=True)
    paths = [MODEL_DIR / name for name in ("GLM5.3 FP4moe-FP8ue8m0",
                                           "GLM5.3/FP4moe-FP8ue8m0", "GLM5.3")]
    found = {p.resolve() for p in paths if (p / INDEX).is_file() and (p / "config.json").is_file()}
    if len(found) != 1:
        raise ValueError(f"expected a unique checkpoint, found {sorted(map(str, found))}; pass --model")
    return found.pop()


def digest(data):
    return sha256(data).hexdigest()


def loader():
    info = {"python": sys.version, "executable": sys.executable, "packages": {}, "sources": {}}
    for name in ("torch", "vllm", "transformers", "safetensors", "deep-gemm", "nvidia-cutlass-dsl"):
        try:
            info["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info["packages"][name] = None
    if info["packages"]["vllm"] is None:
        return info
    base = Path(metadata.distribution("vllm").locate_file("vllm"))
    for rel in ("model_executor/models/registry.py", "models/deepseek_v32/__init__.py",
                "models/deepseek_v32/nvidia/model.py", "model_executor/models/deepseek_v2.py",
                "model_executor/layers/quantization/fp8.py", "model_executor/layers/quantization/mxfp4.py"):
        file = base / rel
        if file.is_file():
            body = file.read_bytes()
            lines = body.decode().splitlines()
            info["sources"][rel] = dict(sha256=digest(body), matches=[
                [i, line.strip()] for i, line in enumerate(lines, 1)
                if any(s in line for s in ("GlmMoeDsa", "Glm5", "self.mlp = Deepseek", "FusedMoEFactory(",
                                           "class DeepseekV2MoE", "class Mxfp4", "class Fp8MoEMethod"))])
    # Site forks sometimes register a separate GLM or mixed quantization class.
    registry = base / "model_executor/models/registry.py"
    if registry.is_file():
        for node in ast.walk(ast.parse(registry.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            for key, val in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and "glm" in key.value.lower():
                    info.setdefault("registry", {})[key.value] = ast.unparse(val)
    return info


def inspect(root, report):
    raw = (root / "config.json").read_bytes()
    cfg = json.loads(raw)
    report.update(model=str(root), config_sha256=digest(raw), config=cfg)
    report["quantization_sidecars"] = {p.name: json.loads(p.read_bytes())
                                      for p in sorted(root.glob("*quant*.json")) if p.stat().st_size < 2_000_000}
    table = (root / INDEX).read_bytes()
    index = json.loads(table)["weight_map"]
    report["index_sha256"] = digest(table)
    report["tokenizer_files"] = {name: digest((root / name).read_bytes()) for name in
                                 ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
                                 if (root / name).is_file()}
    files = defaultdict(list)
    for name, shard in index.items():
        files[shard].append(name)
    report["shard_count"] = len(files)
    layout = defaultdict(Counter)
    ids = defaultdict(set)
    examples = {}
    header_hashes = {}
    for file, names in sorted(files.items()):
        shard = root / file
        with shard.open("rb") as f:
            n, = struct.unpack("<Q", f.read(8))
            if not 1 <= n <= 100_000_000:
                raise ValueError(f"invalid safetensors header in {file}")
            header = f.read(n)
        header_hashes[file] = digest(header)
        specs = json.loads(header)
        for name in names:
            match = LAYER.search(name)
            if not match:
                continue
            layer = int(match[1])
            suffix = match[2]
            if not any(s in suffix for s in ("mlp.", "ffn.", "expert", "gate", "router")):
                continue
            entry = specs[name]
            if "expert" in suffix or "router" in suffix or ".gate." in suffix:
                if exp := EXPERT.search(suffix):
                    ids[layer].add(int(exp[2]))
            family = EXPERT.sub(r"\g<1>{id}.", suffix)
            sig = f'{family} | {entry["dtype"]} | {entry["shape"]}'
            layout[layer][sig] += 1
            examples.setdefault(sig, dict(key=name, shard=file, dtype=entry["dtype"], shape=entry["shape"]))
    report["header_sha256"] = header_hashes
    report["tensor_examples"] = examples
    report["layers"] = {i: dict(tensors=dict(counts), expert_ids=sorted(ids[i])) for i, counts in sorted(layout.items())}
    if not layout:
        report["unrecognized_key_sample"] = sorted(index)[:30]
    if not layout:
        raise ValueError("no recognizable decoder FFN keys; inspect naming")
    actual = cfg.get("text_config") or cfg
    count = actual["num_hidden_layers"]
    dense = actual.get("first_k_dense_replace", 0)
    freq = actual.get("moe_layer_freq", 1)
    layers = [i for i in range(dense, count) if i % freq == 0]
    report["configured_routed_layers"] = layers
    report["missing_expert_layers"] = [i for i in layers if not any("expert" in k for k in layout[i])]
    report["named_expert_id_mismatches"] = [i for i in layers if ids[i] and
                                            ids[i] != set(range(actual["n_routed_experts"]))]
    if report["missing_expert_layers"] or report["named_expert_id_mismatches"]:
        raise ValueError("incomplete configured routed layers or named expert IDs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = {"trace_level_requested": 2, "loader": loader(), "status": "incomplete"}
    try:
        inspect(resolve(args.model), result)
        result["status"] = "metadata_collected"
    except (OSError, ValueError, TypeError, KeyError, struct.error) as e:
        result["error"] = f"{type(e).__name__}: {e}"
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"], result.get("error", ""), args.out)
    return int(result["status"] != "metadata_collected")


if __name__ == "__main__":
    sys.exit(main())
