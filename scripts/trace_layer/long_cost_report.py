"""Verify both full windows, then report paired complete MoE and routed rank shares."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# This report reads only measurements and the published scalar price function.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/gc_moe/eco"))
from h20_valid_cost import slot_cost

STEPS, LAYERS, RANKS, BATCH = 1000, 40, 8, 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def by_rank(rows):
    ordered = sorted(rows, key=lambda r: r["rank"])
    require([r["rank"] for r in ordered] == list(range(RANKS)), "expected eight distinct ranks")
    return ordered


def read_arm(root, name, revision):
    directory = root / name
    run = read(directory / "run.json")
    mode = 0 if name == "linear" else 2
    require(
        (run["cost"], run["stream"], run["steps"], run["layers"], run["batch"], run["rearranged"])
        == (mode, 1, STEPS, LAYERS, BATCH, True)
        and (directory / "execution.sha").read_text().strip() == revision,
        "wrong workload or revision",
    )
    config = run["controls"].copy()
    for key in ("GC_MOE_ECO_COSTFN", "GC_MOE_STUDY_COSTFN"):
        require(config.pop(key) == str(mode), "wrong active price")
    required = dict(
        GC_MOE_LONG_COST="1",
        GC_MOE_ORDER_EXTRA="1",
        GC_MOE_FINE_TRACE="0",
        GC_MOE_STEP_PROFILE="1",
        GC_MOE_ECO_WB="26",
        GC_MOE_ECO_WA="0",
        GC_MEASURE_FLUSH="0",
        GC_MOE_MB_CLAIM="1",
        GC_MOE_ECO_MAX_COPIES="8",
        GC_MOE_ECO_PER_LAYER="1",
        GC_MOE_BLOCK_M_FORCE="64",
        GC_MOE_ECO_NO_AUCTION="0",
        GC_MOE_ECO_NULL_PLAN="0",
        GC_MOE_ECO_NOSPLIT="0",
    )
    require(all(config.get(k) == v for k, v in required.items()), "unexpected measured configuration")
    controls = dict(
        config=config,
        prompt=run["prompt_hash"],
        feedback=run["output_hash"],
        prefill=run["prefill_steps"],
        hardware=(directory / "hardware.csv").read_text(),
        runtime=(directory / "runtime.log").read_text(),
    )
    starts = {}
    for phase in ("stats", "measured"):
        window = run[phase]
        audit = by_rank(window["audit"])
        initial = by_rank(window["initial"])
        starts[phase] = initial[0]["epoch"]
        controls[phase] = []
        for row, entry in zip(audit, initial):
            require(
                row["first_epoch"] == entry["epoch"] == starts[phase] and row["calls"] == STEPS * LAYERS,
                "incomplete window",
            )
            control = {
                k: row[k] for k in ("rank", "first_epoch", "ids", "weights", "ordinals", "compressed_layers")
            }
            if phase == "measured":
                require(2 in row["compressed_layers"], "missing canonical attention audit")
                rank = row["rank"]
                manifest = read(root / "checkpoint" / f"rank{rank}.json")
                snap = entry["checkpoint"]
                controls_hash = hashlib.sha256(
                    json.dumps(manifest["controls"], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                require(
                    snap["source_sha256"] == snap["readback_sha256"] == manifest["sha256"]
                    and snap["controls_sha256"] == controls_hash
                    and manifest["rank"] == rank
                    and snap["kv_bytes"] == sum(manifest["layout"]["sizes"])
                    and snap["layout_sha256"]
                    == hashlib.sha256(
                        json.dumps(manifest["layout"], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    and manifest["controls"]["next_epoch"] == entry["epoch"],
                    "invalid checkpoint",
                )
                control.update(checkpoint=snap, batch_order=row["batch_order"])
            controls[phase].append(control)
        totals = np.array([r["total_rows_by_layer"] for r in audit])
        require(
            totals.shape == (RANKS, LAYERS) and np.all(totals.sum(0) == STEPS * BATCH * 6 * RANKS),
            "incomplete physical rows",
        )
        blocks = np.array(window["ten_step_ms"])
        require(
            blocks.shape == (100,)
            and np.isfinite(blocks).all()
            and (blocks > 0).all()
            and np.isclose(blocks.mean(), window["ms_per_step"]),
            "invalid synced window timing",
        )
    require(starts["measured"] == starts["stats"] + STEPS * LAYERS, "unexpected step between the two windows")
    times = []
    scores = []
    physical = []
    natives = []
    for rank in range(RANKS):
        with np.load(directory / f"rank{rank}.npz", allow_pickle=False) as archive:
            clocks = archive["clocks"]
            counts = archive["counts"]
            native = archive["native"]
            epochs = archive["epochs"]
        expected = np.arange(starts["measured"], starts["measured"] + STEPS * LAYERS)
        require(
            clocks.shape == (len(expected), 6)
            and np.array_equal(clocks[:, 4], expected)
            and np.array_equal(epochs, expected)
            and (expected[0] - 1) % LAYERS == 0,
            "invalid clock epoch coverage",
        )
        require(
            counts.shape == (len(expected), 56)
            and counts.dtype.kind == "i"
            and (counts >= 0).all()
            and counts.max() <= RANKS * BATCH * 6,
            "invalid physical slot counts",
        )
        root_ns, first, last, end = clocks[:, :4].T
        rows = counts.sum(1)
        blocks = ((counts + 63) // 64).sum(1)
        require(
            np.array_equal(rows.reshape(STEPS, LAYERS).sum(axis=0), totals[rank]),
            "rank archive differs from measured audit",
        )
        active = rows > 0
        require(
            (root_ns > 0).all()
            and (end > root_ns).all()
            and np.all(
                (root_ns[active] <= first[active])
                & (first[active] < last[active])
                & (last[active] <= end[active])
            )
            and np.all((first[~active] == -1) & (last[~active] == 0)),
            "inconsistent role or tile boundaries",
        )
        linear = rows + 26 * blocks
        lookup = np.array([slot_cost(n) for n in range(int(counts.max()) + 1)])
        fine = lookup[counts].sum(1)
        require(np.array_equal(clocks[:, 5], linear if mode == 0 else fine), "device plan score mismatch")
        require(native.shape == (STEPS, BATCH) and (native >= 0).all(), "invalid native greedy samples")
        times.append(np.stack(((end - root_ns) / 1000, np.where(active, last - first, 0) / 1000), axis=-1))
        scores.append(np.stack((linear, fine), axis=-1))
        physical.append(np.stack((rows, blocks), axis=-1))
        natives.append(native)
    require(all(np.array_equal(x, natives[0]) for x in natives), "native samples differ within TP ranks")
    shape = (STEPS, LAYERS, RANKS, 2)
    times = np.stack(times, axis=1).reshape(shape)
    scores = np.stack(scores, axis=1).reshape(shape)
    physical = np.stack(physical, axis=1).reshape(shape)
    require(
        np.all(physical[:, :, :, 0].sum(axis=2) == BATCH * 6 * RANKS), "expert routes lost in a measured call"
    )
    return controls, natives[0], np.array(run["measured"]["ten_step_ms"]), times, scores, physical


def compare(root):
    revision = (root / "measurement.sha").read_text().strip()
    metrics = {}
    trajectories = []
    layer_means = []
    reference = None
    different_controls = []
    with gzip.open(root / "ranks.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arm",
                "step",
                "layer",
                "rank",
                "tokens",
                "routed_rows",
                "m_blocks",
                "moe_graph_us",
                "routed_tile_us",
                "linear_price",
                "fine_price",
                "linear_share",
                "fine_share",
                "actual_moe_share",
                "actual_routed_share",
                "active_moe_error_pp",
                "active_routed_error_pp",
            ]
        )
        for name, mode in [("linear", 0), ("fine", 1)]:
            controls, native, host, times, scores, physical = read_arm(root, name, revision)
            if reference is None:
                reference = controls
                native_ref = native
            different_controls = [key for key in reference if reference[key] != controls[key]]
            inputs_match = not different_controls
            different = (native != native_ref).sum(axis=1)
            sums = times.sum(axis=2, keepdims=True)
            require((sums > 0).all(), "empty layer time")
            actual = times / sums
            pred = scores / scores.sum(axis=2, keepdims=True)
            error = 100 * (pred[:, :, :, :, None] - actual[:, :, :, None, :])
            row = {
                "e2e_ms_per_step": float(host.mean()),
                "moe_ms_per_step": float(times[:, :, :, 0].max(axis=2).sum(axis=1).mean() / 1000),
                "mean_layer_max_moe_us": float(times[:, :, :, 0].max(axis=2).mean()),
                "cross_scores": {},
                "native_different": int(different.sum()),
                "inputs_match": inputs_match,
            }
            for i, model in enumerate(("linear", "fine")):
                row["cross_scores"][model] = {}
                for j, target in enumerate(("moe", "routed")):
                    abs_err = np.abs(error[:, :, :, i, j])
                    row["cross_scores"][model][target] = {
                        "mae_pp": float(abs_err.mean()),
                        "max_abs_pp": float(abs_err.max()),
                    }
            metrics[name] = row
            layer_means.append(times[:, :, :, 0].max(axis=2).mean(axis=0))
            moe = times[:, :, :, 0].max(axis=2).sum(axis=1) / 1000
            routed_err = np.abs(error[:, :, :, mode, 1]).mean(axis=(1, 2))
            trajectories.append((host, moe, routed_err))
            for step, layer, rank in np.ndindex(STEPS, LAYERS, RANKS):
                ix = (step, layer, rank)
                writer.writerow(
                    [
                        name,
                        step,
                        layer,
                        rank,
                        physical[ix][0] / RANKS,
                        *physical[ix],
                        *times[ix],
                        *scores[ix],
                        *pred[ix],
                        *actual[ix],
                        *error[ix][mode],
                    ]
                )
    with (root / "step_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["step", "linear_moe_ms", "fine_moe_ms", "delta_ms", "linear_routed_mae_pp", "fine_routed_mae_pp"]
        )
        (_, b, be), (_, c, ce) = trajectories
        for i in range(STEPS):
            w.writerow((i, b[i], c[i], c[i] - b[i], be[i], ce[i]))
    with (root / "blocks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "first_step",
                "last_step",
                "linear_e2e_ms",
                "fine_e2e_ms",
                "linear_moe_ms",
                "fine_moe_ms",
                "linear_routed_mae_pp",
                "fine_routed_mae_pp",
            ]
        )
        for i in range(20):
            sl = slice(50 * i, 50 * i + 50)
            hs = slice(5 * i, 5 * i + 5)
            w.writerow(
                [
                    sl.start,
                    sl.stop - 1,
                    *(t[0][hs].mean() for t in trajectories),
                    *(t[1][sl].mean() for t in trajectories),
                    *(t[2][sl].mean() for t in trajectories),
                ]
            )
    with (root / "layer_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "linear_mean_max_rank_us", "fine_mean_max_rank_us", "delta_us", "delta_pct"])
        for layer, (old, new) in enumerate(zip(*layer_means)):
            w.writerow([layer, old, new, new - old, 100 * (new - old) / old])
    delta = c - b
    comparable = metrics["fine"]["inputs_match"] and metrics["fine"]["native_different"] == 0
    summary = dict(
        revision=revision,
        strict_pair_comparable=comparable,
        arms=metrics,
        differing_control_groups=different_controls,
        measured_steps_per_arm=STEPS,
        layers=LAYERS,
        rank_samples_per_arm=STEPS * LAYERS * RANKS,
        paired_e2e_delta_pct=float(100 * (trajectories[1][0].mean() / trajectories[0][0].mean() - 1)),
        paired_active_routed_mae_delta_pp=float((ce - be).mean()),
        moe_faster_layers=int((layer_means[1] < layer_means[0]).sum()),
        paired_moe_delta_ms=float(delta.mean()),
        paired_moe_delta_pct=float(100 * delta.mean() / b.mean()),
        moe_faster_steps=int((delta < 0).sum()),
        moe_faster_50_step_blocks=int((delta.reshape(20, 50).mean(1) < 0).sum()),
        measured_moe_faster=bool(delta.mean() < 0) if comparable else None,
    )
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    compare(parser.parse_args().root)
