"""Compare the two SM90 linear-cost arms on the same BS256 reference trajectory."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from main20_report import LAYERS, RANKS, STEPS, arm_data, auction_metrics, load, metrics, require, write_rows


def summarize(root):
    bs = 256
    group = root / "bs256"
    sha = (root / "measurement.sha").read_text().strip()
    with np.load(group / "stats-demand.npz") as data:
        history = data["logical"]
        require(history.shape == (STEPS, LAYERS, 384) and (history.sum(2) == bs * 6 * RANKS).all(),
                "incomplete logical demand history")
        demand = hashlib.sha256(np.ascontiguousarray(history)).hexdigest()
    upstream = np.asarray(load(group / "placement.json")).reshape(LAYERS, RANKS, 56)
    upstream = np.sort(upstream, axis=-1)
    reference, _, _, _, native_counts, _, _, _ = arm_data(group / "vanilla", bs, "vanilla", sha, True)
    archive = (Path(__file__).resolve().parents[2] / "docs/trace_executor_20261119" /
               "sm90_bs256_main_five_plan_comm_results/delivery/bs256/vanilla/run.json")
    require(reference["prompt"] == load(archive)["prompt_hash"], "BS256 prompt cohort differs from main table")
    native_layout = np.asarray(load(group / "vanilla/installed-layout.json"))
    require(np.array_equal(native_layout, np.tile(np.arange(384), (LAYERS, 1))), "invalid native layout")
    active_floor = (np.count_nonzero(native_counts, axis=(1, 2)) + RANKS - 1) // RANKS
    del native_counts
    rows, layers, auctions, blocks = [], [], [], []
    for wa, arm in enumerate(("eco_linear", "eco_linear_wa1")):
        directory = group / arm
        control, _, native, warm, counts, times, auction, _ = arm_data(directory, bs, arm, sha, True)
        require(all(row["demand"] == demand for row in control["stats"]), "stats demand hash")
        config = control["config"]
        require(config.pop("GC_MOE_ECO_WA") == str(wa), "active-slot weight")
        for key in ("GC_MOE_SM90_PLAN_DISPATCH", "GC_MOE_NO_PREFETCH", "GC_MOE_MATRIX_PREFIX_CHECK",
                    "GC_MOE_ORDER_PREFILL", "GC_MOE_STABLE_PREFILL_TOPK", "GC_MOE_STABLE_DECODE_TOPK"):
            require(config.get(key) == "1", f"{arm}: {key}")
        require(config["CUDAGRAPH_MODE"] == "FULL_DECODE_ONLY" and config["GC_MOE_KV_BYTES"] == "21474836480",
                "graph mode or KV budget")
        marker = "SM90 ECO plan context=communication communication_sms=6 compute_sms=72"
        require(marker in (directory / "runner.log").read_text(), f"{arm}: plan partition")
        for key in ("stats", "measured", "hardware", "runtime", "prompt", "prefill", "output"):
            require(control[key] == reference[key], f"{arm}: publisher mismatch in {key}")
        installed = np.asarray(load(directory / "installed-layout.json")).reshape(LAYERS, RANKS, 56)
        require(np.array_equal(np.sort(installed, axis=-1), upstream), "EPLB rank membership")

        moe = times[:, :, 0].max(1).reshape(STEPS, LAYERS)
        routed = times[:, :, 1]
        price = times[:, :, 3]  # arm_data validated this against the live GPU score
        abs_error = 100 * np.abs(price / price.sum(1, keepdims=True) - routed / routed.sum(1, keepdims=True))
        active = np.count_nonzero(counts, axis=2).max(1)
        require((active >= active_floor).all(), "active-slot peak below logical lower bound")
        mb = ((counts + 63) // 64).sum(2).max(1)
        if wa == 0:
            base_control, base_layout, base_native, base_warm = control, installed, native, warm
            base_moe, base_active = moe, active
        else:
            for key in base_control:
                require(control[key] == base_control[key], f"arms differ in {key}")
            require(np.array_equal(installed, base_layout), "ECO slot order differs")
        stats = auction_metrics(auction)
        auctions.append(dict(bs=bs, wa=wa, **stats))
        rows.append(dict(bs=bs, wa=wa, wb=26, slack=90 + wa, moe_us_per_layer=moe.mean(),
                         moe_change_pct=100 * (moe.mean() / base_moe.mean() - 1),
                         **metrics(counts, times), peak_rank_active_slots=active.mean(),
                         peak_active_lower_bound=active_floor.mean(), routed_mae_pp=abs_error.mean(),
                         accepted=stats["accepted"], offers=stats["offers"], acceptance_pct=stats["acceptance_pct"]))
        for layer in range(LAYERS):
            layers.append(dict(bs=bs, wa=wa, layer=layer, moe_us=moe[:, layer].mean(),
                               peak_rank_active_slots=active[layer::LAYERS].mean(),
                               peak_rank_m_blocks=mb[layer::LAYERS].mean(), routed_mae_pp=abs_error[layer::LAYERS].mean()))
        for start in range(0, STEPS, 50):
            blocks.append(dict(wa=wa, first_step=start, steps=50, moe_us_per_layer=moe[start:start+50].mean()))
    diff = int(np.count_nonzero(native != base_native))
    fair = warm == base_warm and diff == 0
    pair = dict(bs=bs, native_diff=diff, measured_tokens=native.size, warm_native_equal=warm == base_warm,
                strict_pair_comparable=fair, layer_calls=moe.size, moe_delta_us=(moe-base_moe).mean(),
                moe_change_pct=rows[-1]["moe_change_pct"], peak_active_delta=(active-base_active).mean(),
                calls_peak_active_down=int((active < base_active).sum()),
                calls_peak_active_equal=int((active == base_active).sum()),
                calls_peak_active_up=int((active > base_active).sum()))
    write_rows(group / "pair.csv", [pair])
    require(fair, "native outputs differ, see pair.csv")
    for filename, values in (("main_table", rows), ("layers", layers), ("auction", auctions), ("blocks", blocks)):
        write_rows(group / f"{filename}.csv", values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("delivery", type=Path)
    summarize(parser.parse_args().delivery)
