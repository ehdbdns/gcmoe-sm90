"""Audit full graph workload, auction, and slow-rank evidence before comparing arms."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from ep8_inputs import ARMS, BATCHES, PROMPT_HASH


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def peaks(clocks, price):
    price_max = price == price.max(-1, keepdims=True)
    slow = clocks == clocks.max(-1, keepdims=True)
    hit = (slow & price_max).any(-1)
    return dict(hits=int(hit.sum()), calls=len(hit), hit_pct=float(100 * hit.mean()),
                tied_price=int((price_max.sum(-1) > 1).sum()), tied_time=int((slow.sum(-1) > 1).sum()))


def summarize(counts, clocks, bm, eco):
    g, t, u, c = clocks.transpose(2, 0, 1)
    tokens = counts.sum(-1)
    blocks = ((counts + bm - 1) // bm).sum(-1)
    active = (counts > 0).sum(-1)
    routed = np.where(tokens > 0, u - t, 0)
    graph = c - g
    if not (np.all((g > 0) & (graph > 0)) and
            np.all(np.where(tokens > 0, (g <= t) & (u > t) & (u <= c), (t == -1) & (u == 0)))):
        raise ValueError("invalid device clocks")
    price = tokens + 26 * blocks + 2 * active
    stats = dict(moe_us_per_layer=graph.max(1).mean() / 1000,
                 token_ir=(tokens.max(1) / tokens.mean(1)).mean(),
                 total_m_blocks=blocks.sum(1).mean(), peak_rank_m_blocks=blocks.max(1).mean(),
                 active_physical_slots=active.sum(1).mean(), peak_rank_active_slots=active.max(1).mean(),
                 m_block_ir=(blocks.max(1) / blocks.mean(1)).mean(),
                 routed_mae_pp=(100 * np.abs(price / price.sum(1, keepdims=True)
                                             - routed / routed.sum(1, keepdims=True)).mean()) if eco else None)
    for name, span in (("routed", routed), ("graph", graph)):
        for key, val in peaks(span, price).items():
            stats[f"{name}_price_{key}"] = val
    return stats, price


def verify_slots(group, phase, arm, counts, vanilla):
    # Preserve source rank and local token ordinal in the actual DeepInfer rule.
    # Load route banks from disk only once per source for each RR check.
    import torch
    bs = int(group.name.removeprefix("bs"))
    calls = 40000
    if not np.all(counts.sum((1, 2)) == bs * 6) or np.any(counts < 0):
        raise ValueError(f"{arm} {phase}: lost/duplicated physical rows")
    if arm == "vanilla":
        return
    with np.load(group / arm / f"{phase}-layout.npz") as data:
        phy, inverse, nc = (data[k].astype(np.int64) for k in ("layout", "inverse", "counts"))
    if phase == "measure" or arm in ("random", "eco"):
        name = ("random" if arm == "random" else "eco-initial" if phase == "stats" else "plan")
        with np.load(group / f"{name}.npz") as target:
            if not np.array_equal(phy, target["layout"]) or not np.array_equal(nc, target["counts"]):
                raise ValueError(f"{arm} {phase}: installed placement differs")
            for layer in range(40):
                for e, n in enumerate(nc[layer]):
                    if not np.array_equal(inverse[layer, e, :n], target["inverse"][layer, e, :n]):
                        raise ValueError(f"{arm} {phase}: inverse replica ordering differs")
    q = counts.reshape(calls, 448)
    demand = vanilla.reshape(calls, 384)
    if arm == "eco":
        for layer in range(40):
            for e in range(384):
                if not np.array_equal(q[layer::40, phy[layer] == e].sum(1), demand[layer::40, e]):
                    raise ValueError(f"ECO {phase} L{layer} E{e}: logical conservation failed")
        return
    expect = np.zeros((calls, 448), np.int32)
    ordinals = np.repeat(np.arange(bs // 8), 6)
    for rank in range(8):
        bank, _ = torch.load(group / f"{phase}-routes-{rank}.pt", weights_only=True, map_location="cpu", mmap=True)
        for start in range(0, calls, 200):
            ids = bank[start:start+200].numpy().reshape(-1, bs // 8 * 6)
            if np.any((ids < 0) | (ids >= 384)):
                raise ValueError("invalid reference route")
            layers = np.arange(start, start + len(ids))[:, None] % 40
            copies = (rank + ordinals * 23333) % nc[layers, ids]
            selected = inverse[layers, ids, copies] + 448 * np.arange(len(ids))[:, None]
            expect[start:start+len(ids)] += np.bincount(selected.ravel(), minlength=len(ids)*448).reshape(-1, 448)
    if not np.array_equal(q, expect):
        index = np.argwhere(q != expect)[0].tolist()
        raise ValueError(f"{arm} {phase}: physical RR count differs at call,slot {index}")


def group_report(group, bs):
    results, summary, layers, auctions = {}, [], [], []
    for phase in ("stats", "measure"):
        for arm in ARMS:
            folder = group / arm
            result = results[arm] = json.loads((folder / "result.json").read_text())
            ranks = result[phase]["ranks"]
            assert result["arm"] == arm and result["bs"] == bs and result["prompt_sha256"] == PROMPT_HASH[bs]
            assert result["fused_shared"] and len(ranks) == 8
            bm, = {r["block_m"] for r in ranks}
            assert bm == {256: 16, 512: 16, 1024: 32}[bs]
            is_eco = arm == "eco"
            if any(r["native_window"] != ranks[0]["native_window"] for r in ranks):
                raise ValueError(f"{arm} {phase}: TP workers disagree on native greedy outputs")
            physical, timestamps, auction = [], [], []
            for i, r in enumerate(ranks):
                assert r["full_graph_replays"] == 1000 and r["full_graph_observed_calls"] == 40000
                ctx = r["sm_split"]
                assert ctx["plan_ctas"] == int(is_eco) and ctx["dispatch_ctas"] == ctx["dispatch"] - int(is_eco)
                roles = ctx["roles"]
                assert [role[0] for role in roles] == ([2, 4, 1, 3] if is_eco else [2, 1, 3])
                assert roles[0] == [2, 1, ctx["compute"]] and roles[-2] == [1, 0, ctx["dispatch_ctas"]]
                assert roles[-1][1] == 1 and (not is_eco or roles[1] == [4, 0, 1])
                if is_eco:
                    assert r["eco_unit_config"] == [bm, 26, 2, 1, 6, 0]
                with np.load(folder / f"{phase}-rank{i}.npz") as f:
                    assert f["counts"].shape == (1000, 40, 48 if arm == "vanilla" else 56)
                    physical.append(f["counts"].reshape(40000, -1))
                    if phase == "measure":
                        assert f["clocks"].shape == (1000, 40, 4)
                        clock = f["clocks"].reshape(-1, 4)
                        assert np.all(clock[1:, 0] >= clock[:-1, 3])
                        timestamps.append(clock)
                if phase == "measure" and is_eco:
                    with np.load(folder / f"auction-rank{i}.npz") as f:
                        auction.append(np.stack([f[k] for k in ("offered", "accepted", "score")], axis=1))
            counts = np.stack(physical, axis=1)
            if arm == "vanilla":
                reference = counts
            verify_slots(group, phase, arm, counts, reference)
            if phase != "measure":
                continue
            clocks = np.stack(timestamps, axis=1)
            metrics, prices = summarize(counts, clocks, bm, is_eco)
            accepted = None
            if is_eco:
                if not all(np.array_equal(a[:, :2], auction[0][:, :2]) for a in auction):
                    raise ValueError("the eight replicated auction plans disagree on offers or accepts")
                assert np.array_equal(np.stack([a[:, 2] for a in auction], axis=1), prices)
                offered, accepted = auction[0][:, :2].T
                assert np.all((offered >= accepted) & (accepted >= 0))
                auctions.append(dict(bs=bs, offers=int(offered.sum()), accepted=int(accepted.sum()),
                                     acceptance_pct=100*accepted.sum()/offered.sum() if offered.any() else None,
                                     zero_accept_calls=int((accepted == 0).sum()), calls=len(accepted)))
            summary.append(dict(bs=bs, arm=arm, block_m=bm, **metrics,
                                accepted_per_call=accepted.mean() if is_eco else None))
            for layer in range(40):
                stats, _ = summarize(counts[layer::40], clocks[layer::40], bm, is_eco)
                layers.append(dict(bs=bs, arm=arm, layer=layer, **stats,
                                   accepted_per_call=accepted[layer::40].mean() if is_eco else None))
    pairs = []
    for a, b in itertools.combinations(ARMS, 2):
        failures = []
        native_diff = {}
        if results[a]["feedback_sha256"] != results[b]["feedback_sha256"]:
            failures.append("feedback")
        for phase in ("stats", "measure"):
            native = [np.load(group / arm / f"{phase}-native-rank0.npy", mmap_mode="r") for arm in (a, b)]
            if any(x.shape != (bs, 1000) for x in native):
                raise ValueError(f"incomplete native top-1 window in {phase}")
            different = native[0] != native[1]
            native_diff[f"{phase}_native_differences"] = int(different.sum())
            if different.any():
                step, req = np.argwhere(different.T)[0]
                failures.append(f"{phase}/top1 step={step} req={req} {native[0][req,step]} vs {native[1][req,step]}")
            for i, (ra, rb) in enumerate(zip(results[a][phase]["ranks"], results[b][phase]["ranks"], strict=True)):
                for name, row in ((a, ra), (b, rb)):
                    if (row["first_gate_difference"] is not None or row.get("native_diff", 0)
                            or row["prefill_difference"].get("count", 0)):
                        failures.append(f"{phase}/rank{i}/{name}/native_vs_reference")
                for key in ("logical_ids", "gate_weights", "replay_ids", "replay_weights", "batch_order",
                            "native_window", "kv_boundary", "prefill_native"):
                    if ra[key] != rb[key]:
                        failures.append(f"{phase}/rank{i}/{key}")
                if (ra["sm_split"]["compute"], ra["sm_split"]["dispatch"]) != (rb["sm_split"]["compute"], rb["sm_split"]["dispatch"]):
                    failures.append(f"{phase}/rank{i}/green_contexts")
        pairs.append(dict(bs=bs, arm_a=a, arm_b=b, passed=not failures, **native_diff, failures=" ".join(failures)))
    for row in summary:
        for ref in ("vanilla", "rr"):
            arm = row["arm"]
            passed = arm == ref or next(p["passed"] for p in pairs if {p["arm_a"], p["arm_b"]} == {ref, arm})
            base = next(r for r in summary if r["arm"] == ref)
            row[f"moe_change_vs_{ref}_pct"] = 100*(row["moe_us_per_layer"]/base["moe_us_per_layer"]-1) if passed else None
    for name, rows in (("main", summary), ("layers", layers), ("auction", auctions), ("pairs", pairs)):
        write(group / f"{name}.csv", rows)
    return all(p["passed"] for p in pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", type=Path)
    g = p.parse_args().group
    bs = int(g.name.removeprefix("bs"))
    assert bs in BATCHES
    raise SystemExit(0 if group_report(g, bs) else 2)
