"""For each H20 slot budget, train a layerwise EPLB placement then score held-out RR token IR."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/trace_executor_20261119"
sys.path.insert(0, str(ROOT / "source/gc_moe/eplb"))
from policy import rebalance_experts_global, _batch_greedy_indices


def archive(bs):
    if bs < 256:
        return ARCHIVE / f"main24_cost_results/bs{bs}"
    if bs == 256:
        return ARCHIVE / "sm90_bs256_main_five_plan_comm_results/delivery/bs256"
    return ARCHIVE / f"sm90_bs512_bs1024_full_model_four_results/delivery-{bs}/bs{bs}"


def counts(path):
    chunks = []
    for rank in range(8):
        with np.load(path / f"rank{rank}.npz") as data:
            chunks.append(data["counts"])
    return np.stack(chunks, axis=1)


def layouts(history, limit, device):
    # The replica allocation order is independent of the stopping budget.
    # Pack afresh at each capacity with the unmodified DeepInfer rank packer.
    history = history.to(device)
    batch_layers = 40 if device == "cuda" else 4
    raw = torch.cat([rebalance_experts_global(history[:, i:i+batch_layers].contiguous(), 384 + 8*limit, 1)[0]
                     for i in range(0, 40, batch_layers)])
    for k in range(limit + 1):
        slots = raw[:, :384+8*k]
        copies = torch.stack([torch.bincount(row, minlength=384) for row in slots])
        order = _batch_greedy_indices(history, slots, copies, 48+k)
        plan = slots.gather(1, order).cpu().numpy()
        inverse = np.full((40, 384, int(copies.max())), -1, dtype=np.int32)
        for layer, (experts, positions) in enumerate(zip(slots.cpu().numpy(), order.argsort(-1).cpu().numpy())):
            cursor = np.zeros(384, dtype=int)
            for expert, pos in zip(experts, positions):
                inverse[layer, expert, cursor[expert]] = pos
                cursor[expert] += 1
        yield plan, inverse


def exact_layer(ids, logical, inverses, observed_rr, verify_layout):
    # Group recorded occurrences of an expert by step and ordinal residue.
    # Cache by copy count across budgets, retaining upstream copy-index order.
    steps, bs, topk = ids.shape
    flat = ids.reshape(-1)
    if flat.min() < 0 or flat.max() >= 384:
        raise ValueError("bad logical route")
    order = np.argsort(flat, kind="stable")
    limits = np.r_[0, np.bincount(flat, minlength=384).cumsum()]
    loads = [np.zeros((steps, 8), dtype=np.int64) for _ in inverses]
    for e in range(384):
        pos = order[limits[e]:limits[e+1]] // topk
        step, ordinal = np.divmod(pos, bs)
        occ = np.bincount(step, minlength=steps)
        if not np.array_equal(8 * occ, logical[:, e]):
            raise ValueError("ordered bank differs from measured logical counters")
        cache = {}
        for k, inverse in enumerate(inverses):
            slots = inverse[e][inverse[e] >= 0]
            copies = len(slots)
            if copies not in cache:
                if 8 % copies == 0:
                    cache[copies] = np.broadcast_to((occ*(8//copies))[:, None], (steps, copies))
                else:
                    key = step*copies + (23333*ordinal) % copies
                    by_phase = np.bincount(key, minlength=steps*copies).reshape(steps, copies)
                    cache[copies] = sum(np.roll(by_phase, rank, axis=1) for rank in range(8))
            split = cache[copies]
            if verify_layout and k == 8 and not np.array_equal(split, observed_rr[:, slots]):
                raise ValueError("k=8 does not reproduce archived physical counters")
            for copy, rank in enumerate(slots // (48+k)):
                loads[k][:, rank] += split[:, copy]
    peaks = []
    for load in loads:
        if not np.all(load.sum(1) == bs*topk*8):
            raise ValueError("RR lost routes")
        peaks.append(load.max(1).mean() / (bs*topk))
    return peaks


def save_plans(path, group, limit, device):
    h = np.load(group/"stats-demand.npz")["logical"]
    arrays = {"history_sha256": hashlib.sha256(memoryview(h).cast("B")).hexdigest(), "device": device}
    for k, (plan, inv) in enumerate(layouts(torch.from_numpy(h), limit, device)):
        arrays[f"p{k}"], arrays[f"i{k}"] = plan, inv
    np.savez_compressed(path, **arrays)


def scan(bs, limit, device, exact, bank_root=None, cohort=None, saved=None):
    src = cohort or archive(bs)
    history = np.load(src / "stats-demand.npz")["logical"]
    if history.shape != (1000, 40, 384) or not np.all(history.sum(-1) == bs*48):
        raise ValueError("incomplete training history")
    vanilla = np.asarray(json.loads((src/"vanilla/installed-layout.json").read_text()))
    if vanilla.shape != (40,384) or not np.all(vanilla == np.arange(384)):
        raise ValueError("publisher physical counters are not in logical order")
    held = counts(src/"vanilla").reshape(1000,40,384)
    if np.any(held % 8) or not np.all(held.sum(-1) == bs*48):
        raise ValueError("incomplete measurement history")
    rr = counts(src/"rr").reshape(1000,40,448)
    rr_layout = np.asarray(json.loads((src/"rr/installed-layout.json").read_text()))
    def fingerprints(arm):
        rows = json.loads((src/arm/"run.json").read_text())["measured"]["audit"]
        if {r["rank"] for r in rows} != set(range(8)):
            raise ValueError("incomplete ranks")
        return {r["ids"] for r in rows}
    train_hash = hashlib.sha256(memoryview(history).cast("B")).hexdigest()
    for arm in ("vanilla", "rr"):
        stats = json.loads((src/arm/"run.json").read_text())["stats"]["audit"]
        if len(stats) != 8 or {r["demand"] for r in stats} != {train_hash}:
            raise ValueError("statistics archive differs from the EPLB window")
    hashes = fingerprints("vanilla")
    if len(hashes) != 1 or hashes != fingerprints("rr"):
        raise ValueError("TP route IDs differ between ranks or arms")
    bank = None
    if exact:
        folder = bank_root or Path((src/"state-path.txt").read_text().strip())/"replay"
        tensor = torch.load(folder/"measure-rank0.pt", mmap=True, weights_only=True, map_location="cpu")["ids"]
        if tuple(tensor.shape) != (40000,bs,6) or not tensor.is_contiguous():
            raise ValueError("unexpected TP route bank")
        bank = tensor.numpy()
        if hashlib.sha256(memoryview(bank).cast("B")).hexdigest() not in hashes:
            raise ValueError("route bank digest mismatch")
        bank = bank.reshape(1000,40,bs,6)
    if saved:
        with np.load(saved) as z:
            if str(z["history_sha256"]) != hashlib.sha256(memoryview(history).cast("B")).hexdigest():
                raise ValueError("placement training history mismatch")
            device = str(z["device"])
            plans = [(z[f"p{k}"], z[f"i{k}"]) for k in range(limit+1)]
    else:
        plans = list(layouts(torch.from_numpy(history), limit, device))
    scores = np.zeros((40, len(plans)))
    native = held.reshape(40000,8,48).sum(2).max(1).mean()/(bs*6)
    actual8 = rr.reshape(40000,8,56).sum(2).max(1).mean()/(bs*6)
    for layer in range(40):
        if exact:
            same = limit >= 8 and np.array_equal(plans[8][0][layer], rr_layout[layer])
            scores[layer] = exact_layer(bank[:,layer], held[:,layer], [inv[layer] for _,inv in plans],
                                        rr[:,layer], same)
        else:
            for k,(plan,_) in enumerate(plans):
                slots = plan[layer]
                copies = np.bincount(slots, minlength=384)[slots].astype(np.float32)
                load = held[:,layer,:][:,slots].astype(np.float32)/copies
                scores[layer,k] = load.reshape(1000,8,-1).sum(2).max(1).mean(dtype="d")/(bs*6)
    for k, value in enumerate(scores.mean(0)):
        yield dict(bs=bs,k=k,slots=48+k,token_ir=value,native_ir=native,actual_k8_ir=actual8,
                   gain_vs_native_pct=100*(1-value/native),method="ordered_rr" if exact else "mean_replica_split",
                   device=device, training_steps=1000, measurement_calls=40000,
                   same_k8_layers=int((plans[8][0] == rr_layout).all(1).sum()) if limit >= 8 else "")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, nargs="+", default=[32,64,128,256,512,1024])
    p.add_argument("--max-extra", type=int, default=32)
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--exact", action="store_true")
    p.add_argument("--bank", type=Path, help="one BS: directory containing measure-rank0.pt")
    p.add_argument("--cohort", type=Path, help="one BS: full-model delivery with vanilla, rr, stats-demand")
    p.add_argument("--save-plans", action="store_true", help="only train placements; output a single BS .npz")
    p.add_argument("--plans", type=Path, help="score a saved plan file without occupying a GPU")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.max_extra < 0 or ((a.bank or a.cohort or a.plans or a.save_plans) and len(a.bs) != 1):
        p.error("invalid budget or cohort selection")
    torch.set_num_threads(3)
    if a.save_plans:
        if a.out.suffix != ".npz":
            p.error("placement output must end in .npz")
        save_plans(a.out, a.cohort or archive(a.bs[0]), a.max_extra, a.device)
        sys.exit(0)
    with a.out.open("w", newline="") as f:
        writer = None
        for bs in a.bs:
            for row in scan(bs,a.max_extra,a.device,a.exact,a.bank,a.cohort,a.plans):
                if writer is None:
                    writer = csv.DictWriter(f,fieldnames=row,lineterminator="\n")
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
