"""Replay GLM's per-source RR offline, training each EPLB capacity on the original statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/gc_moe/eplb/deepinfer_periodic"))
from policy import rebalance_experts_global, _batch_greedy_indices

sys.path.insert(0, str(ROOT / "source/gc_moe/eco"))
from meta_sm90 import reorder_single_copy_first

ARCHIVE = ROOT / "docs/sm100_megamoe/glm53_formal_results/formal.QLfEZmKt"
EXPERTS, RANKS, LOCAL, TOPK, BM = 256, 8, 32, 8, 16
STEPS, LAYERS = 1000, 75
MEAN_TOKENS = LOCAL * TOPK


def layer_plans(history, budgets):
    # DeepInfer assigns the next replica before it packs ranks. Each budget
    # has the same allocation prefix but needs its own fresh greedy packing.
    history = torch.from_numpy(history[:, None, :])
    raw = rebalance_experts_global(history, EXPERTS + RANKS * max(budgets), 1)[0]
    for extra in budgets:
        width = EXPERTS // RANKS + extra
        slots = raw[:, :RANKS * width]
        copies = torch.bincount(slots[0], minlength=EXPERTS)[None, :]
        order = _batch_greedy_indices(history, slots, copies, width)
        yield reorder_single_copy_first(slots.gather(1, order)[0].numpy(), RANKS)[0]


def replay_layer(ids, logical, plans, rr8, placement8):
    steps = ids.shape[0]
    if ids.shape != (steps, RANKS, LOCAL, TOPK) or ids.min() < 0 or ids.max() >= EXPERTS:
        raise ValueError("invalid ordered source IDs")
    # The GLM adapter numbers local rows independently on each source rank.
    order = np.argsort(ids.ravel(), kind="stable")
    counts = np.bincount(ids.ravel(), minlength=EXPERTS)
    by_expert = np.split(order // TOPK, counts.cumsum()[:-1])
    cache = {}
    for expert, positions in enumerate(by_expert):
        step, global_row = np.divmod(positions, RANKS * LOCAL)
        source, local_row = np.divmod(global_row, LOCAL)
        hist = np.bincount(step, minlength=steps)
        np.testing.assert_array_equal(hist, logical[:, expert])
        cache[expert] = (step, 23333 * local_row + source, {1: hist[:, None]})

    metrics = []
    for extra, plan in plans:
        width = len(plan) // RANKS
        physical = np.empty((steps, len(plan)), dtype=np.int32)
        for expert, (step, ordinal, splits) in cache.items():
            slots = np.flatnonzero(plan == expert)
            copies = len(slots)
            if copies not in splits:
                splits[copies] = np.bincount(step * copies + ordinal % copies,
                                             minlength=steps * copies).reshape(steps, copies)
            physical[:, slots] = splits[copies]
        if extra == 8:
            np.testing.assert_array_equal(plan, placement8)
            np.testing.assert_array_equal(physical, rr8)
        if not np.all(physical.sum(1) == RANKS * MEAN_TOKENS):
            raise ValueError("incomplete or duplicated routes")
        slots = physical.reshape(steps, RANKS, width)
        m = ((slots + BM - 1) // BM).sum(2)
        peak = slots.sum(2).max(1)
        metrics.append(np.array([peak.mean() / MEAN_TOKENS, peak.mean(), m.sum(1).mean(), m.max(1).mean()]))
    return np.stack(metrics)


def verified_counts(root, inventory, arm, phase):
    width = 40 if arm == "rr" else 32
    parts = []
    for rank in range(RANKS):
        key = f"{arm}/{phase}-rank{rank}.npz"
        path = root / key
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != inventory[key]:
                raise ValueError(f"historical receipt hash mismatch: {key}")
        with np.load(path, allow_pickle=False) as z:
            part = z["counts"]
            if part.shape != (STEPS, LAYERS, width) or np.any(part < 0) or z["bm"].item() != BM:
                raise ValueError(f"incorrect counters: {key}")
            parts.append(part)
    return np.concatenate(parts, axis=-1)


def scan(root):
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    if (root / "revision").read_text().strip() != receipt["revision"]:
        raise ValueError("wrong historical archive")
    inventory = {r["path"]: r["sha256"] for r in receipt["files"]}
    train = verified_counts(root, inventory, "vanilla", "stats")
    logical = verified_counts(root, inventory, "vanilla", "measure")
    rr8 = verified_counts(root, inventory, "rr", "measure")
    for counts in (train, logical, rr8):
        if not np.all(counts.sum(-1) == RANKS * MEAN_TOKENS):
            raise ValueError("historical counters lost routes")
    placement = np.load(ARCHIVE / "placement.npy", allow_pickle=False)
    banks = []
    for rank in range(RANKS):
        ids, _weights = torch.load(root / f"measure-reference-{rank}.pt", weights_only=True,
                                   map_location="cpu", mmap=True)
        if ids.dtype != torch.int32 or tuple(ids.shape) != (STEPS, LAYERS, LOCAL, TOPK):
            raise ValueError(f"unexpected GLM route bank on rank {rank}")
        banks.append(ids.numpy())
    budgets = list(range(33))
    per_layer = []
    for layer in range(LAYERS):
        ids = np.stack([bank[:, layer] for bank in banks], axis=1)
        plans = zip(budgets, layer_plans(train[:, layer], budgets), strict=True)
        scores = replay_layer(ids, logical[:, layer], plans, rr8[:, layer], placement[layer])
        per_layer.append(scores)
        print(f"checked decoder layer {layer + 3}/{LAYERS + 2}", flush=True)
    mean = np.mean(per_layer, axis=0)
    historical = {r["arm"]: r for r in csv.DictReader((ARCHIVE / "main.csv").read_text().splitlines())}
    keys = ["token_ir", "mean_peak_rank_tokens", "total_m", "peak_m"]
    for arm, counts, idx in (("vanilla", logical, None), ("rr", rr8, 8)):
        physical = counts.reshape(STEPS, LAYERS, RANKS, -1)
        peak = physical.sum(-1).max(-1)
        m = ((physical + BM - 1) // BM).sum(-1)
        values = [peak.mean() / MEAN_TOKENS, peak.mean(), m.sum(-1).mean(), m.max(-1).mean()]
        for i in (0, 2, 3):
            np.testing.assert_allclose(values[i], float(historical[arm][keys[i]]), atol=1e-10, rtol=0)
        if idx is not None:
            np.testing.assert_allclose(mean[idx], values, atol=1e-10, rtol=0)
        else:
            native = dict(zip(keys, values))
    # Within one routed row of the best observed mean peak. A sustained
    # plateau additionally bounds the entire remaining range, not just its best.
    peaks = mean[:, 1]
    near = int(np.flatnonzero(peaks <= peaks.min() + 1)[0])
    sustained = next((k for k in budgets[:-2] if np.ptp(peaks[k:]) <= 1), None)
    return dict(cohort="formal.QLfEZmKt", revision=receipt["revision"], rank_streams=LOCAL,
                global_streams=RANKS * LOCAL, block_m=BM, stats_steps=STEPS, measure_steps=STEPS,
                layers=LAYERS, rr8_exact_slot_checks=STEPS * LAYERS * placement.shape[1],
                native=native, rr=[dict(extra_per_rank=k, **dict(zip(keys, row))) for k, row in zip(budgets, mean)],
                min_ir_extra=int(peaks.argmin()), within_one_row_extra=near, sustained_one_row_extra=sustained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = scan(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
