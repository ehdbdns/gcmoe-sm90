"""Load GLM-5.3's native packed payloads for the SM100 GC weight transform."""
import argparse
import hashlib
import json
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
        super().__init__(root)
        config = json.loads((self.root / "config.json").read_text())
        self.h = config["hidden_size"]
        self.i = config["moe_intermediate_size"]
        self.experts = config["n_routed_experts"]
        self.routed = {i for i, kind in enumerate(config["mlp_layer_types"]) if kind == "sparse"}
        q = config["quantization_config"]
        if (q["expert_dtype"], q["fmt"], q["scale_fmt"], q["weight_block_size"],
            config["n_shared_experts"], config["hidden_act"]) != ("fp4", "e4m3", "ue8m0", [128, 128], 1, "silu"):
            raise ValueError("unexpected GLM-5.3 expert format")

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

    def load_gc(self, layer, local_logical_ids):
        """Physical slot order is supplied by the caller for native, RR or ECO."""
        from gc_moe import transform_weights_for_gc_moe
        ids = list(local_logical_ids)
        if not ids:
            raise ValueError("empty physical placement")
        # Only one raw CPU expert is live while building the GPU slot bank.
        for slot, logical in enumerate(ids):
            pair = self.expert_pair(layer, int(logical))
            if slot == 0:
                bank = [(torch.empty((len(ids), *w.shape), dtype=w.dtype, device="cuda"),
                         torch.empty((len(ids), sf.shape[1], sf.shape[0]), dtype=sf.dtype,
                                     device="cuda").transpose(1, 2)) for w, sf in pair]
            for (w, sf), (dst_w, dst_sf) in zip(pair, bank):
                dst_w[slot].copy_(w)
                dst_sf[slot].copy_(sf)
        routed = transform_weights_for_gc_moe(*bank)
        del bank
        shared = self.expert_pair(layer, None)
        return routed, transform_weights_for_gc_moe(*[(w.cuda(), sf.cuda()) for w, sf in shared])


def cpu_check(root):
    from safetensors import safe_open
    weights = GlmWeights(root)
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
                    with safe_open(root / weights.shards[name], framework="pt", device="cpu") as shard:
                        # Slicing bytes through the public reader handles older
                        # torch builds where arithmetic on E8M0 is unsupported.
                        ref = shard.get_tensor(name).view(torch.uint8)
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
