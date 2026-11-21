"""Eight-rank, model-shaped routed-only correctness test for DeepGEMM 5081f851."""
from pathlib import Path
import argparse
import json
import sys
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "integration"), str(ROOT / "source/gc_moe/eplb")]
from mega_eplb import mega_rr_slots
from policy import rebalance_experts_global


def packed_weights(experts, n, k, seed):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    size = len(experts)
    w = torch.empty((size, n, k // 2), dtype=torch.int8, device="cuda")
    sf = torch.empty((size, n, k // 32), dtype=torch.float32, device="cuda")
    generator = torch.Generator(device="cuda")
    for slot, expert in enumerate(experts):
        generator.manual_seed(seed + expert)
        value = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.02
        w[slot], sf[slot] = per_token_cast_to_fp4(value, use_ue8m0=True, gran_k=32)
    return w, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), size)


def copy_shared_input_scales(dest, sf, block_m):
    # Upstream shared-L1 scale layout, including padding when M is below 128.
    dest.zero_()
    idx = torch.arange(sf.shape[0], device=sf.device)
    m = idx % block_m
    address = (idx // block_m * ((block_m + 127) // 128 * 128)
               + m // 128 * 128 + m % 32 * 4 + m % 128 // 32)
    dest.index_copy_(0, address, sf)


def worker(local_rank, world, checkpoint_dir, with_shared):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist

    rank, size, group = init_dist(local_rank, world)
    assert size == 8 and torch.cuda.get_device_capability() == (10, 3)
    deep_gemm.use_deterministic_algorithms(True)
    from v41_shape import model_shape
    if checkpoint_dir:
        cfg = model_shape(checkpoint_dir)
        experts, hidden = cfg["n_routed_experts"], cfg["hidden_size"]
        inter, topk = cfg["moe_intermediate_size"], cfg["num_experts_per_tok"]
    else:
        experts, hidden, inter, topk = 256, 4096, 2048, 6
    tokens = 256
    torch.manual_seed(13000 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    x, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    scores = torch.randn(tokens, experts, device="cuda")
    scores[:, :8] += 2
    weights, logical = scores.topk(topk, dim=1)
    weights = weights.softmax(dim=1)
    history = torch.bincount(logical.flatten(), minlength=experts).to(torch.int32)
    dist.all_reduce(history, group=group)
    layout, inverse, count = rebalance_experts_global(history.reshape(1, 1, experts), experts + 8 * size,
                                                       experts // size + 8)
    slots = mega_rr_slots(logical, inverse[0], count[0], rank)
    assert torch.equal(layout[0][slots].long(), logical)
    from real_routed_weights import ExpertCheckpoint
    checkpoint = ExpertCheckpoint(checkpoint_dir) if checkpoint_dir else None
    ref = None
    identity = torch.arange(experts, device="cuda")
    cases = [("identity", identity, logical), ("eplb_rr", layout[0], slots)]
    shared_args = {}
    if with_shared:
        shared_l1, shared_l2 = deep_gemm.transform_weights_for_mega_moe(
            checkpoint.shared(("w1", "w3"), inter, hidden), checkpoint.shared(("w2",), hidden, inter))
        shared_args = dict(shared_l1_weights=shared_l1, shared_l2_weights=shared_l2)
        cases.insert(0, ("routed_control", identity, logical))
    control = None
    for name, physical, route in cases:
        shared = with_shared and name != "routed_control"
        local = physical.chunk(size)[rank].tolist()
        if checkpoint is None:
            raw = packed_weights(local, 2 * inter, hidden, 100), packed_weights(local, hidden, inter, 1000)
        else:
            raw = checkpoint.packed(local, ("w1", "w3"), inter, hidden), checkpoint.packed(local, ("w2",), hidden, inter)
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(*raw)
        del raw
        buf = deep_gemm.get_symm_buffer_for_mega_moe(group, len(physical), tokens, topk, hidden, inter,
                                                      num_shared_experts=int(shared), mma_type="fp8xfp4")
        buf.x[:tokens].copy_(x)
        buf.x_sf[:tokens].copy_(sf)
        if shared:
            block_m = deep_gemm.get_block_m_for_mega_moe(size, len(physical), buf.num_max_tokens_per_rank,
                                                          tokens, topk, "fp8xfp4")
            copy_shared_input_scales(buf.shared_l1_acts_sf, sf, block_m)
        buf.topk_idx[:tokens].copy_(route)
        buf.topk_weights[:tokens].copy_(weights)
        y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
        received = torch.zeros(len(local), dtype=torch.int32, device="cuda")
        deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buf, cumulative_local_expert_recv_stats=received,
                                   **(shared_args if shared else {}))
        expected = torch.bincount(route.flatten(), minlength=len(physical))
        dist.all_reduce(expected, group=group)
        delta = 0 if ref is None else (y != ref).sum().item()
        row = dict(arm=name, rank=rank, different_elements=delta,
                   valid_output=bool(torch.isfinite(y).all() and y.abs().max() > 0),
                   slot_count_match=torch.equal(expected.chunk(size)[rank], received.long()))
        shared_changed = True
        if shared:
            row["changed_from_routed_only"] = int((y != control).sum().item())
            shared_changed = row["changed_from_routed_only"] > 0
        print(json.dumps(row), flush=True)
        bad = torch.tensor(int(not shared_changed or delta != 0 or not row["valid_output"] or not row["slot_count_match"]), device="cuda")
        dist.all_reduce(bad, group=group)
        if name == "routed_control":
            control = y.clone()
        elif name == "identity":
            ref = y.clone()
        dist.barrier(group=group)
        buf.destroy()
        if bad.item():
            raise RuntimeError(f"{name}: {bad.item()} ranks failed the correctness check")
    dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.device_count() != 8:
        raise SystemExit("Expected the allocated group of exactly eight idle GPUs")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    if args.shared and not args.checkpoint:
        parser.error("--shared requires --checkpoint")
    torch.multiprocessing.spawn(worker, args=(8, args.checkpoint, args.shared), nprocs=8)
