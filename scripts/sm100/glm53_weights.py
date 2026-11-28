"""Load GLM-5.3's native packed payloads for the SM100 GC weight transform."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch

from real_routed_weights import ExpertCheckpoint


def pack_scales(scale, row_block):
    # Pack four existing exponent bytes along K. Source rows and GLM's
    # gate/up boundary are multiples of 128, so MN needs no extra padding.
    raw = scale.view(torch.uint8)
    words = raw.contiguous().view(torch.int32)
    if row_block != 1:
        words = words.repeat_interleave(row_block, dim=-2)
    return words.transpose(-1, -2).contiguous().transpose(-1, -2)


class GlmWeights(ExpertCheckpoint):
    def __init__(self, root):
        from safetensors import safe_open

        super().__init__(root)
        directory = self.root
        self.open_shard = lru_cache(maxsize=8)(
            lambda name: safe_open(directory / name, framework="pt", device="cpu")
        )
        config = json.loads((self.root / "config.json").read_text())
        self.h = config["hidden_size"]
        self.i = config["moe_intermediate_size"]
        self.experts = config["n_routed_experts"]
        self.routed = {i for i, kind in enumerate(config["mlp_layer_types"]) if kind == "sparse"}
        q = config["quantization_config"]
        if (q["expert_dtype"], q["fmt"], q["scale_fmt"], q["weight_block_size"],
            config["n_shared_experts"], config["hidden_act"]) != ("fp4", "e4m3", "ue8m0", [128, 128], 1, "silu"):
            raise ValueError("unexpected GLM-5.3 expert format")

    def close(self):
        self.open_shard.cache_clear()

    def tensor(self, key, dtype, shape):
        value = self.open_shard(self.shards[key]).get_tensor(key)
        types = {"I8": torch.int8, "F8_E8M0": torch.float8_e8m0fnu, "F8_E4M3": torch.float8_e4m3fn}
        if tuple(value.shape) != shape or value.dtype != types[dtype]:
            raise ValueError(f"unexpected {key} dtype or shape: {value.dtype}, {value.shape}")
        return value

    def projection(self, layer, expert, name):
        if layer not in self.routed or (expert is not None and not 0 <= expert < self.experts):
            raise ValueError("invalid decoder layer or logical expert")
        shared = expert is None
        source = "shared_experts" if shared else f"experts.{expert}"
        key = f"model.layers.{layer}.mlp.{source}.{name}_proj"
        n, k = (self.h, self.i) if name == "down" else (self.i, self.h)
        dtype = "F8_E4M3" if shared else "I8"
        weight = self.tensor(key + ".weight", dtype, (n, k if shared else k // 2))
        scale = self.tensor(key + ".scale", "F8_E8M0", (n // 128, k // 128) if shared else (n, k // 32))
        return weight, pack_scales(scale, 128 if shared else 1)

    def expert_pair(self, layer, expert):
        gate, up, down = (self.projection(layer, expert, op) for op in ("gate", "up", "down"))
        # Views as bytes preserve FP8 bit patterns even on CPU torch builds.
        gate_up = torch.cat([p[0].view(torch.uint8) for p in (gate, up)]).view(gate[0].dtype)
        sf = torch.cat([gate[1], up[1]]).T.contiguous().T
        return (gate_up, sf), down

    def _pack_routed(self, layer, ids, transform):
        for slot, logical in enumerate(ids):
            pair = self.expert_pair(layer, int(logical))
            if slot == 0:
                bank = [(torch.empty((len(ids), *w.shape), dtype=w.dtype, device="cuda"),
                         torch.empty((len(ids), sf.shape[1], sf.shape[0]), dtype=sf.dtype,
                                     device="cuda").transpose(1, 2)) for w, sf in pair]
            for (w, sf), (dst_w, dst_sf) in zip(pair, bank):
                dst_w[slot].copy_(w)
                dst_sf[slot].copy_(sf)
        return transform(*bank)

    def load_gc(self, layer, local_logical_ids):
        """Use the upstream transform per four slots, writing the final graph bank once."""
        from gc_moe import transform_weights_for_gc_moe

        ids = list(local_logical_ids)
        if not ids:
            raise ValueError("empty physical placement")
        for start in range(0, len(ids), 4):
            packed = self._pack_routed(layer, ids[start:start + 4], transform_weights_for_gc_moe)
            if start == 0:
                routed = tuple(tuple(torch.empty_strided(
                    (len(ids), *t.shape[1:]), (t[0].numel(), *t.stride()[1:]),
                    dtype=t.dtype, device=t.device) for t in pair) for pair in packed)
            for outputs, chunk in zip(routed, packed):
                for dst, src in zip(outputs, chunk):
                    dst.narrow(0, start, src.shape[0]).copy_(src)
            del packed
        shared = self.expert_pair(layer, None)
        return routed, transform_weights_for_gc_moe(*[(w.cuda(), sf.cuda()) for w, sf in shared])


def cpu_check(root):
    weights = GlmWeights(root)
    reference = ExpertCheckpoint(root)
    checks = []
    for layer in (8, 20, 32):
        for expert in (0, weights.experts - 1, None):
            l1, l2 = weights.expert_pair(layer, expert)
            source = "shared_experts" if expert is None else f"experts.{expert}"
            for op in ("gate", "up", "down"):
                key = f"model.layers.{layer}.mlp.{source}.{op}_proj"
                w, sf = l2 if op == "down" else l1
                offset = weights.i if op == "up" else 0
                rows = weights.h if op == "down" else weights.i
                w = w[offset:offset + rows].view(torch.uint8)
                restored = sf[offset:offset + rows].contiguous().view(torch.uint8)
                for suffix, candidate in ((".weight", w), (".scale", restored)):
                    name = key + suffix
                    if suffix == ".weight":
                        dtype = "F8_E4M3" if expert is None else "I8"
                        shape = tuple(w.shape)
                    else:
                        dtype = "F8_E8M0"
                        shape = (rows // 128 if expert is None else rows, restored.shape[1])
                    ref = reference.tensor(name, dtype, shape).view(torch.uint8)
                    if suffix == ".scale" and expert is None:
                        ref = ref.repeat_interleave(128, dim=0)
                    if not torch.equal(ref, candidate):
                        raise ValueError(f"byte mismatch at {name}")
                    checks.append(dict(key=name, shape=list(candidate.shape),
                                       sha256=hashlib.sha256(candidate.numpy()).hexdigest()))
    return checks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.write_text(json.dumps(cpu_check(a.model), indent=2) + "\n")
