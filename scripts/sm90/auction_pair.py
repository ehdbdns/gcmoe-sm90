"""Stage a common input bank and audit the two full-model ECO runs for one batch size."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/trace_layer"))
from main20_report import (LAYERS, RANKS, STEPS, arm_data, auction_metrics, ep8_launch_policy,
                           load, metrics, require, write_rows)
from compare_ep8_linear import workload


def stage(bank, group, bs=256):
    # Only link immutable input files. All generated KV, layouts and captures
    # belong to this new job. An absent bank is recorded by ECO in its own run.
    archives = ROOT / "docs/trace_executor_20261119"
    names = ["tokens.json", *(f"{phase}-rank{r}.pt" for phase in ("stats", "measure") for r in range(RANKS))]
    require(group.name == f"bs{bs}", "route bank requested for a different batch size")
    for archive in ("sm90_ep8_linear_four_6bs_results", "sm90_shared_first_wa1_results",
                    "sm90_bs512_bs1024_full_model_four_results"):
        history = archives / archive / f"delivery-{bs}" / f"bs{bs}"
        path_file = history / "state-path.txt"
        if not path_file.is_file():
            continue
        source = Path(path_file.read_text().strip()) / "replay"
        if not all((source / name).is_file() for name in names):
            continue
        manifest = dict(line.split()[::-1] for line in (history / "workload.sha256").read_text().splitlines())
        for name in names:
            h = hashlib.sha256()
            with (source / name).open("rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
            require(h.hexdigest() == manifest[name], f"reference file failed checksum: {source/name}")
        for name in names:
            (bank / name).symlink_to(source / name)
        (group / "reference-source.txt").write_text(f"{archive}\n{source}\n")
        (group / "reference-mode.txt").write_text("1\n")
        return
    (group / "reference-source.txt").write_text("recorded during coarse ECO\n")
    (group / "reference-mode.txt").write_text("0\n")


def logical_rows(counts, layout, layer):
    q = counts[layer::LAYERS].reshape(STEPS, -1)
    order = layout[layer].argsort()
    experts, first = np.unique(layout[layer, order], return_index=True)
    np.testing.assert_array_equal(experts, np.arange(384))
    return np.add.reduceat(q[:, order], first, axis=1), np.add.reduceat(((q + 63) // 64)[:, order], first, axis=1)


def report(delivery, bs=256):
    group = delivery / f"bs{bs}"
    sha = (delivery / "measurement.sha").read_text().strip()
    table, auctions, layers = [], [], []
    timings, baseline_counts, baseline_controls, baseline_workload = [], None, None, None
    for flag, label in enumerate(("eco", "eco_small")):
        folder = group / label
        run = load(folder / "run.json")
        control, _, native, _, counts, spans, auction, _ = arm_data(
            folder, bs, "eco", sha, with_linear=True, linear_ep8=True)
        require(ep8_launch_policy(folder, True) == "original", "unexpected shared/plan launch order")
        common = control["config"]
        require(common.pop("GC_MOE_ECO_FINE_AUCTION") == str(flag), "wrong auction switch")
        for key in ("GC_MOE_SM90_PLAN_DISPATCH", "GC_MOE_SM90_SPLIT_SHARED", "GC_MOE_STABLE_PREFILL_TOPK",
                    "GC_MOE_STABLE_DECODE_TOPK", "GC_MOE_ORDER_PREFILL", "GC_MOE_MATRIX_PREFIX_CHECK"):
            require(common[key] == "1", f"missing {key}")
        layout = np.asarray(load(folder / "installed-layout.json"))
        require(layout.shape == (40, 448), "invalid EPLB layout")
        if flag == 0:
            baseline_counts, baseline_native, baseline_layout = counts, native, layout
            baseline_controls, baseline_workload = control, workload(run)
        else:
            require(control == baseline_controls and workload(run) == baseline_workload,
                    "different controls, inputs, checkpoints, native warm outputs, hardware or runtime")
            np.testing.assert_array_equal(native, baseline_native)
            np.testing.assert_array_equal(layout, baseline_layout)

        graph, routed, price = spans.transpose(2, 0, 1)
        timing = graph.max(1).reshape(STEPS, LAYERS)
        timings.append(timing)
        require((routed.sum(1) > 0).all(), "missing routed spans")
        errors = 100 * np.abs(price / price.sum(1)[:, None] - routed / routed.sum(1)[:, None])
        small, elapsed = None, []
        for rank in range(RANKS):
            with np.load(folder / f"rank{rank}.npz") as z:
                f, ns = z["auction_small"], z["auction_ns"]
            require(f.shape == ns.shape == (STEPS * LAYERS,) and (ns > 0).all() and (f >= 0).all()
                    and (f <= auction[:, 2]).all() and not f[:LAYERS].any(), "invalid fine-bid counters")
            if small is None:
                small = f
            else:
                np.testing.assert_array_equal(f, small)
            elapsed.append(ns)
        require(flag == 1 or not small.any(), "coarse ECO committed a fine bid")
        auction_us = np.max(elapsed, axis=0) / 1000
        for layer in range(LAYERS):
            rows, blocks = logical_rows(counts, layout, layer)
            np.testing.assert_array_equal(blocks, (rows + 63) // 64)
            if flag:
                np.testing.assert_array_equal(rows, logical_rows(baseline_counts, layout, layer)[0])
            layers.append(dict(arm=label, layer=layer, moe_us=float(timing[:, layer].mean()),
                               peak_price=float(price[layer::LAYERS].max(1).mean()),
                               small_accepts=int(small[layer::LAYERS].sum())))
        summary = metrics(counts, spans)
        summary.pop("moe_ms_per_step")
        routed_peak, routed_mean = routed.max(1), routed.mean(1)
        table.append(dict(arm=label, bs=bs, moe_us=float(timing.mean()),
                          change_pct=100 * (timing.mean() / timings[0].mean() - 1),
                          **summary, peak_active_slots=float(np.count_nonzero(counts, axis=2).max(1).mean()),
                          peak_price=float(price.max(1).mean()), routed_mae_pp=float(errors.mean()),
                          routed_mean_rank_us=float(routed_mean.mean()), routed_max_rank_us=float(routed_peak.mean()),
                          routed_gap_us=float((routed_peak - routed_mean).mean()),
                          routed_peak_over_mean_pct=float(100 * (routed_peak / routed_mean - 1).mean()),
                          graph_peak_over_mean_pct=float(100 * (graph.max(1) / graph.mean(1) - 1).mean()),
                          auction_max_rank_us=float(auction_us.mean())))
        auctions.append(dict(arm=label, **auction_metrics(auction), small_accepts=int(small.sum()),
                             calls_with_small=int(np.count_nonzero(small))))
    blocks = []
    for i, (coarse, fine) in enumerate(zip(*(t.reshape(20, -1).mean(1) for t in timings))):
        blocks.append(dict(first_step=50*i, coarse_us=coarse, fine_us=fine, change_pct=100 * (fine / coarse - 1)))
    for name, data in (("main_table", table), ("auction", auctions), ("layers", layers), ("blocks", blocks)):
        write_rows(group / f"{name}.csv", data)
    print((group / "main_table.csv").read_text(), (group / "auction.csv").read_text(), sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("stage")
    prepare.add_argument("bank", type=Path)
    prepare.add_argument("group", type=Path)
    analyze = sub.add_parser("report")
    analyze.add_argument("delivery", type=Path)
    for command in (prepare, analyze):
        command.add_argument("--bs", type=int, choices=(256, 512, 1024), default=256)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.bank, args.group, args.bs)
    else:
        report(args.delivery, args.bs)
