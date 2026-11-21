"""Summarize completed arms and audit every pair, including partial deliveries."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from five_matrix import ARMS
from gc_moe.eco.plan import eco_unit_config


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def array(root, arm, phase, name):
    parts = []
    for r in range(8):
        with np.load(root / arm / f"{phase}-rank{r}.npz") as f:
            parts.append(f[name])
    return np.stack(parts, axis=2)


def reference_counts(root, phase):
    return array(root, "mega", phase, "counts").reshape(1000, 40, 384)


def check_counts(counts, reference, layout):
    for layer, mapping in enumerate(layout):
        slots = counts[:, layer].reshape(1000, -1)
        by_expert = np.zeros((384, 1000), dtype=np.int32)
        np.add.at(by_expert, mapping, slots.T)
        if not np.array_equal(by_expert.T, reference[:, layer]):
            raise ValueError(
                f"Incorrect logical demand in physical receive counts, layer {layer}"
            )


def summarize(root, bs):
    runs = {}
    rows = []
    checks = {}
    failures = []
    for arm in ARMS:
        file = root / arm / "result.json"
        if not file.exists():
            failures.append(f"{arm}: no completed result")
            continue
        try:
            run = json.loads(file.read_text())
            assert run["bs"] == bs and run["arm"] == arm
            ref = json.loads((root / "mega/result.json").read_text())
            meas = run["measure"]
            (bm,) = set(x["block_m"] for x in meas["ranks"])
            _, wb, wa, *_ = eco_unit_config(8, bm)
            counts = array(root, arm, "measure", "counts")
            if not ((counts >= 0).all() and (counts.sum((2, 3)) == bs * 6).all()):
                raise ValueError("invalid timing or global received token count")
            layout = (
                np.load(root / "plan.npz")["layout"]
                if "_" in arm
                else np.tile(np.arange(384), (40, 1))
            )
            # Stats use the initial placement; all moved arms verify it via route tape.
            check_counts(counts, reference_counts(root, "measure"), layout)
            blocks = ((counts + bm - 1) // bm).sum(3)
            tokens = counts.sum(3)
            active = (counts > 0).sum(3)
            price = tokens + wb * blocks + wa * active
            pred = price / price.sum(2, keepdims=True)

            def err(actual):
                d = np.abs(pred - actual / actual.sum(2, keepdims=True)) * 100
                return d.mean(), d.max()

            accepted = None
            if arm == "gc_eco":
                a = array(root, arm, "measure", "accepted")
                if not ((a == a[:, :, :1]).all() and (a >= 0).all()):
                    raise ValueError("rank-local auction records disagree")
                accepted = int(a[:, :, 0].sum())
            g, t, u, c = array(root, arm, "measure", "clocks").transpose(3, 0, 1, 2)
            work = tokens > 0
            valid = (g > 0) & (c > g) & (~work | ((t >= g) & (u > t) & (u <= c)))
            if not valid.all():
                raise ValueError(
                    f"{arm}: invalid kernel boundary rows: {np.count_nonzero(~valid)}"
                )
            graph = (c - g) / 1e6
            routed = np.where(work, u - t, 0) / 1e6
            graph_mae, graph_max = err(graph)
            routed_mae, routed_max = err(routed)
            row = dict(
                bs=bs,
                arm=arm,
                ms_per_step=meas["wall_ms"],
                vs_mega_time_pct=100
                * (meas["wall_ms"] / ref["measure"]["wall_ms"] - 1),
                block_m=bm,
                token_ir=(tokens.max(2) / tokens.mean(2)).mean(),
                total_m_blocks=blocks.sum(2).mean(),
                peak_m_blocks=blocks.max(2).mean(),
                active_slots=active.sum(2).mean(),
                m_block_ir=(blocks.max(2) / blocks.mean(2)).mean(),
                moe_graph_ms=graph.max(2).sum(1).mean(),
                graph_mae_pp=graph_mae,
                graph_max_pp=graph_max,
                routed_cta_mae_pp=routed_mae,
                routed_cta_max_pp=routed_max,
                accepted=accepted,
            )
            checks[arm] = all(
                obs["replay_ids"] == base["logical_ids"]
                and obs["replay_weights"] == base["gate_weights"]
                and obs["batch_order"] == base["batch_order"]
                for phase in ("stats", "measure")
                for obs, base in zip(
                    run[phase]["ranks"], ref[phase]["ranks"], strict=True
                )
            )
            if not checks[arm]:
                raise ValueError("route or token replay mismatch")
            rows.append(row)
            runs[arm] = run
        except Exception as exc:
            failures.append(f"{arm}: {type(exc).__name__}: {exc}")
    pairs = []
    for a, b in combinations(runs, 2):
        ra, rb = runs[a], runs[b]
        equal_native = all(
            x["native"] == y["native"]
            for x, y in zip(ra["measure"]["ranks"], rb["measure"]["ranks"], strict=True)
        )
        equal_gates = all(
            x["logical_ids"] == y["logical_ids"]
            and x["gate_weights"] == y["gate_weights"]
            for phase in ("stats", "measure")
            for x, y in zip(ra[phase]["ranks"], rb[phase]["ranks"], strict=True)
        )
        same_plan = (
            not ("placement" in ra and "placement" in rb)
            or ra["placement"]["plan_sha256"] == rb["placement"]["plan_sha256"]
        )
        controlled = (
            checks[a]
            and checks[b]
            and same_plan
            and ra["prompt_sha256"] == rb["prompt_sha256"]
        )
        pairs.append(
            dict(
                arm_a=a,
                arm_b=b,
                controlled_replay=controlled,
                native_token_equal=equal_native,
                native_gate_equal=equal_gates,
                strict=controlled and equal_native and equal_gates,
            )
        )
    write(root / "metrics.csv", rows)
    write(root / "pairs.csv", pairs)
    summary = dict(
        bs=bs,
        completed=list(runs),
        strict_pairs=sum(p["strict"] for p in pairs),
        total_pairs=len(pairs),
        failures=failures,
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    return not failures and len(pairs) == 10 and all(p["strict"] for p in pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("bs", type=int)
    a = p.parse_args()
    raise SystemExit(0 if summarize(a.root, a.bs) else 1)
