"""Check actual graph coverage and aligned workload before comparing native kernels."""
import json
from pathlib import Path
import sys

import numpy as np


def graph_evidence(path, arm):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"),
                     key=lambda e: e["ts"])
    markers = [e for e in kernels if any(name in e["name"] for name in
                                         ("full_route_stage", "full_collect_layer"))]
    def correlation(e):
        return str(e.get("args", {}).get("correlation", ""))

    ids = {correlation(e) for e in markers}
    order = ["full_route_stage" in e["name"] for e in markers]
    launches = [e for e in events if e.get("name") == "cudaGraphLaunch" and
                e.get("cat") in ("cuda_runtime", "cuda_driver") and correlation(e) in ids]
    nodes = [e for e in kernels if correlation(e) in ids]
    roles = {"attention": "core_attn::fwd_kernel", "dense": "BlockScaledPersistentDenseGemm",
             "moe": "GCMoERoleKernel" if arm == "gc" else "mega_moe_impl"}
    counts = {role: sum(name.lower() in e["name"].lower() for e in nodes)
              for role, name in roles.items()}
    foreign = sum(correlation(e) not in ids for e in kernels
                  if markers and markers[0]["ts"] <= e["ts"] <= markers[-1]["ts"])
    passed = (order == [True, False] * 40 and len(ids) == len(launches) == 1
              and not ids.intersection({"", "0", "-1", "None"}) and all(counts.values()) and foreign == 0)
    return dict(passed=bool(passed), marker_count=len(markers), correlations=sorted(ids),
                matching_launches=len(launches), role_nodes=counts, foreign_kernels=foreign)


def report(root):
    arms = ("mega", "gc")
    data = {a: json.loads((root / a / "result.json").read_text()) for a in arms}
    failures = []
    def check(ok, message):
        if not ok:
            failures.append(message)

    check(data["mega"]["prompt_sha256"] == data["gc"]["prompt_sha256"], "prompt hash differs")
    graphs, timings = {}, {}
    for arm in arms:
        graphs[arm] = [graph_evidence(root / arm / f"profile-rank{r}.json", arm) for r in range(8)]
        check(all(g["passed"] for g in graphs[arm]), f"{arm} step 2 graph coverage")
    for phase in ("stats", "measure"):
        totals = np.zeros((1000, 40), dtype=np.int64)
        durations = {arm: [] for arm in arms}
        for r in range(8):
            pair = [data[arm][phase]["ranks"][r] for arm in arms]
            for key in ("replay_ids", "replay_weights", "batch_order", "native_window", "kv_boundary",
                        "logical_ids", "gate_weights", "prefill_native"):
                check(pair[0][key] == pair[1][key], f"{phase} rank {r} {key}")
            check(pair[1]["native_diff"] == 0 and pair[1]["prefill_difference"]["count"] == 0
                  and pair[1]["first_gate_difference"] is None, f"{phase} rank {r} native outputs or routes")
            counts = []
            for arm, rank in zip(arms, pair, strict=True):
                check(data[arm][phase]["full_decode"] and rank["full_graph_observed_calls"] == 40000,
                      f"{arm} {phase} rank {r} full decode calls")
                with np.load(root / arm / f"{phase}-rank{r}.npz") as f:
                    c = f["counts"]
                    check(c.shape == (1000, 40, 48) and (c >= 0).all(), f"{arm} {phase} rank {r} counts")
                    counts.append(c)
                    if phase == "measure":
                        clk = f["clocks"]
                        check(clk.shape == (1000, 40, 19) and (clk[..., 0] > 0).all()
                              and (np.diff(clk[..., :4], axis=-1) >= 0).all()
                              and (clk.reshape(-1, 19)[1:, 0] >= clk.reshape(-1, 19)[:-1, 3]).all(),
                              f"{arm} rank {r} clocks")
                        durations[arm].append(clk[..., 3] - clk[..., 0])
            check(np.array_equal(*counts), f"{phase} rank {r} slot rows differ")
            totals += counts[0].sum(-1)
        check((totals == 3072).all(), f"{phase} global token total")
        if phase == "measure":
            for arm in arms:
                ms = np.max(durations[arm], axis=0).sum(axis=1) / 1e6
                timings[arm] = float(np.delete(ms, 2).mean())
    result = dict(strict_pair_pass=not failures, failures=failures, graph=graphs,
                  clean_steps=999, sum_of_layer_max_rank_local_moe_ms=timings)
    if not failures:
        result["gc_moe_reduction_percent_vs_mega"] = 100 * (1 - timings["gc"] / timings["mega"])
    (root / "full_decode_pair.json").write_text(json.dumps(result, indent=2) + "\n")
    return not failures


if __name__ == "__main__":
    sys.exit(0 if report(Path(sys.argv[1])) else 1)
