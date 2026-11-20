"""Verify and aggregate the README matrix from rank-local clocks, slots and device auction counters."""

import argparse
import csv
import gzip
import hashlib
import json
from itertools import combinations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/gc_moe/eco"))
from h20_valid_cost import slot_cost
from long_cost_report import by_rank, require

BATCHES = (32, 64, 128, 256)
ARMS = ("vanilla", "random", "rr", "eco", "eco_smr")
ARMS24 = ("vanilla", "random", "rr", "eco_linear", "eco", "eco_smr")
STEPS, LAYERS, RANKS = 1000, 40, 8
AUCTION = (
    "rounds",
    "offers",
    "accepted",
    "eligible_donors",
    "feasible_donors",
    "no_offer_rounds",
    "all_rejected_rounds",
)


def load(path):
    return json.loads(path.read_text())


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def metrics(counts, spans):
    rows = counts.sum(-1)
    blocks = ((counts + 63) // 64).sum(-1)
    peak = spans[:, :, 0].max(1)
    return dict(
        moe_ms_per_step=float(peak.reshape(STEPS, LAYERS).sum(1).mean() / 1000),
        token_ir=float((rows.max(1) / rows.mean(1)).mean()),
        total_m_blocks=float(blocks.sum(1).mean()),
        peak_rank_m_blocks=float(blocks.max(1).mean()),
        active_physical_slots=float(np.count_nonzero(counts, axis=(1, 2)).mean()),
        m_block_ir=float((blocks.max(1) / blocks.mean(1)).mean()),
    )


def arm_data(directory, bs, arm, sha, with_linear=False):
    run = load(directory / "run.json")
    eco = arm.startswith("eco")
    mode = 0 if arm == "eco_linear" else 2
    expected_price = dict(mode=mode, slack=90 if mode == 0 else 11469) if eco else None
    require(
        run["batch"] == bs
        and run["steps"] == STEPS
        and run["layers"] == LAYERS
        and run["stream"] == 1
        and run["cost"] == mode
        and run["rearranged"] == (arm == "rr" or eco),
        "wrong arm/window",
    )
    require((directory / "execution.sha").read_text().strip() == sha, "revision mismatch")
    config = run["controls"].copy()
    treatment = dict(
        GC_MOE_MATRIX_ARM=arm,
        GC_MOE_ECO_COSTFN=str(mode),
        GC_MOE_EPLB=str(int(arm != "vanilla")),
        GC_MOE_ECO=str(int(eco)),
        GC_MOE_SM90_SM_REUSE=str(int(arm == "eco_smr")),
        GC_MOE_SLOT_PLACEMENT="random" if arm == "random" else "identity",
        GC_MOE_EPLB_NO_REARRANGE=str(int(arm in ("vanilla", "random"))),
    )
    for key, value in treatment.items():
        require(config.pop(key) == value, f"{arm}: wrong {key}")
    fixed = dict(
        GC_MOE_MATRIX_BS=str(bs),
        GC_MOE_ECO_NO_AUCTION="0",
        GC_MOE_ECO_NULL_PLAN="0",
        GC_MOE_ECO_NOSPLIT="0",
        GC_MOE_MB_CLAIM="1",
        GC_MOE_BLOCK_M_FORCE="64",
        GC_MOE_ECO_PER_LAYER="1",
        GC_MOE_ECOPLB_PER_LAYER="1",
        GC_MOE_ORDER_EXTRA="1",
        GC_MOE_ECO_MAX_COPIES="8",
        GC_MOE_STEP_PROFILE="1",
        GC_MOE_EPLB_WINDOW="1000",
        GC_MEASURE_FLUSH="0",
        GC_MOE_SHARED_IN_GC="1",
    )
    if with_linear:
        fixed.update(
            GC_MOE_MATRIX_SIZE="24",
            GC_MOE_ECO_WB="26",
            GC_MOE_ECO_WA="0",
            GC_MOE_ECO_BLOCKS_ONLY="0",
            GC_MOE_ECO_TOKENS_ONLY="0",
            GC_MOE_ECO_ROUNDS="1",
            GC_MOE_ECO_COLD_ROUNDS="6",
            GC_MOE_SM90_DISPATCH_SMS="6",
        )
    require(all(config[k] == v for k, v in fixed.items()), "invalid common configuration")
    control = dict(
        config=config,
        prompt=run["prompt_hash"],
        output=run["output_hash"],
        prefill=run["prefill_steps"],
        hardware=(directory / "hardware.csv").read_text(),
        runtime=(directory / "runtime.log").read_text(),
    )
    installed_hash = hashlib.sha256(
        json.dumps(load(directory / "installed-layout.json"), separators=(",", ":")).encode()
    ).hexdigest()
    calls = STEPS * LAYERS
    for phase in ("stats", "measured"):
        window = run[phase]
        audit, initial = by_rank(window["audit"]), by_rank(window["initial"])
        first = initial[0]["epoch"]
        control[phase] = []
        physical_totals = np.array([r["total_rows_by_layer"] for r in audit])
        require(np.all(physical_totals.sum(0) == STEPS * bs * 6 * RANKS), "incomplete physical rows")
        for row, start in zip(audit, initial):
            require(row["calls"] == calls and row["first_epoch"] == start["epoch"] == first, "missing epochs")
            if with_linear and arm == "rr":
                require(row["rr_checked_calls"] == calls, "incomplete DeepInfer RR slot verification")
            control[phase].append(
                {k: row[k] for k in ("rank", "first_epoch", "ids", "weights", "demand", "compressed_layers")}
            )
            if phase == "measured":
                require(2 in row["compressed_layers"], "incomplete attention audit")
                boundary = start["checkpoint"]
                require(
                    boundary["installed_layout"] == installed_hash,
                    "installed layout does not match live boundary",
                )
                if with_linear:
                    require(
                        boundary["plan_price"] == expected_price, f"{arm}: wrong prepared role-5 mode/slack"
                    )
                snap = boundary["snapshot"]
                manifest = load(directory.parent / "checkpoint" / f"rank{row['rank']}.json")
                require(
                    snap["readback_sha256"] == snap["source_sha256"] == manifest["sha256"]
                    and boundary["demand"] == manifest["controls"]["demand"]
                    and boundary["logical_weights"] == manifest["controls"]["logical_weights"]
                    and manifest["controls"]["next_epoch"] == first,
                    "invalid boundary",
                )
                control[phase][-1].update(snapshot=snap, order=row["batch_order"])
        if phase == "stats":
            stats_first = first
        else:
            require(first == stats_first + calls, "unexpected steps at rearrangement")
        host = np.array(window["ten_step_ms"])
        require(
            host.shape == (100,)
            and np.isfinite(host).all()
            and (host > 0).all()
            and np.isclose(host.mean(), window["ms_per_step"]),
            "invalid E2E timing",
        )
    placement = {r["checkpoint"]["placement"] for r in initial}
    require(len(placement) == 1 and (first - 1) % LAYERS == 0, "placement or layer alignment")
    epochs = np.arange(first, first + calls)
    observers = []
    occupancies = []
    native = None
    auction = None
    for rank in range(RANKS):
        with np.load(directory / f"rank{rank}.npz", allow_pickle=False) as data:
            q = data["counts"]
            clocks = data["clocks"]
            greedy = data["native"]
            require(
                q.shape == (calls, 48 if arm == "vanilla" else 56) and q.dtype.kind == "i" and (q >= 0).all(),
                "invalid slot counters",
            )
            require(
                np.array_equal(q.reshape(STEPS, LAYERS, -1).sum((0, 2)), physical_totals[rank]), "rank totals"
            )
            require(
                clocks.shape == (calls, 6)
                and np.array_equal(clocks[:, 4], epochs)
                and np.array_equal(data["epochs"], epochs),
                "stale device clock row",
            )
            g, t, u, c = clocks[:, :4].T
            active = q.sum(1) > 0
            require(
                np.all((g > 0) & (c > g))
                and np.all((g[active] <= t[active]) & (t[active] < u[active]) & (u[active] <= c[active]))
                and np.all((t[~active] == -1) & (u[~active] == 0)),
                "bad timing boundary",
            )
            lookup = np.array([slot_cost(i) for i in range(int(q.max()) + 1)])
            fine_price = lookup[q].sum(1)
            linear_price = q.sum(1) + 26 * ((q + 63) // 64).sum(1)
            price = linear_price if mode == 0 else fine_price
            require(np.all(clocks[:, 5] == (price if eco else 0)), "incorrect role-5 score")
            observers.append(
                np.stack(
                    [(c - g) / 1000, np.where(active, u - t, 0) / 1000, fine_price, linear_price], axis=-1
                )
            )
            occupancies.append(q)
            require(greedy.shape == (STEPS, bs) and (greedy >= 0).all(), "incomplete native top-1")
            if native is None:
                native = greedy
            require(np.array_equal(greedy, native), "TP ranks disagree on native top-1")
            if eco:
                a = data["auction"]
                require(a.shape == (calls, len(AUCTION)) and (a >= 0).all(), "bad auction archive")
                rounds, offers, accepted, eligible, feasible, empty, rejected = a.T
                require(
                    np.all((eligible >= feasible) & (feasible >= offers) & (offers >= accepted))
                    and np.all((rounds >= 1) & (rounds <= 6) & (empty + rejected <= 1))
                    and np.all(rounds[LAYERS:] == 1)
                    and np.all((accepted == 0) == (rounds == empty + rejected)),
                    "invalid per-round auction counters",
                )
                if auction is None:
                    auction = a
                require(np.array_equal(auction, a), "replicated planners disagree")
            else:
                require("auction" not in data, "auction counters without ECO")
    counts = np.stack(occupancies, axis=1)
    values = np.stack(observers, axis=1)
    require(np.all(counts.sum(axis=(1, 2)) == bs * 6 * RANKS), "routes lost")
    return (
        control,
        placement.pop(),
        native,
        [r["warm_native"] for r in audit],
        counts,
        values,
        auction,
        float(host.mean()),
    )


def auction_metrics(auction):
    totals = dict(zip(AUCTION, map(int, auction.sum(0))))
    accepted, offers = totals["accepted"], totals["offers"]
    return dict(
        **totals,
        calls=len(auction),
        blocked_before_offer=totals["eligible_donors"] - totals["feasible_donors"],
        unaccepted_offers=offers - accepted,
        accepted_per_layer_call=accepted / len(auction),
        calls_zero_accepts=int(np.count_nonzero(auction[:, 2] == 0)),
        acceptance_pct=100 * accepted / offers if offers else None,
    )


def analyze_group(root, bs, sha, with_linear=False):
    arms = ARMS24 if with_linear else ARMS
    rows = []
    layers = []
    auctions = []
    reference = None
    adaptive = None
    outcomes = {}
    eco_layout = None
    eco_allocation = None
    eco_auction = None
    group = root / f"bs{bs}"
    with np.load(group / "stats-demand.npz") as data:
        history = data["logical"]
    demand_hash = hashlib.sha256(np.ascontiguousarray(history)).hexdigest()
    require(
        history.shape == (STEPS, LAYERS, 384) and np.all(history.sum(-1) == bs * 6 * RANKS),
        "incomplete EPLB history",
    )
    for arm in arms:
        control, placement, native, warm, counts, v, auction, host = arm_data(
            group / arm, bs, arm, sha, with_linear
        )
        require(all(a["demand"] == demand_hash for a in control["stats"]), "EPLB demand hash mismatch")
        if reference is None:
            reference = control
            reference_native = native
            reference_warm = warm
            reference_host = host
        require(reference == control, f"{bs}/{arm}: common workload or configuration differs")
        installed = np.asarray(load(group / arm / "installed-layout.json"))
        physical = 384 if arm == "vanilla" else 448
        require(
            installed.shape == (LAYERS, physical) and np.all((installed >= 0) & (installed < 384)),
            "invalid installed map",
        )
        require(all(np.all(np.bincount(x, minlength=384) > 0) for x in installed), "missing logical expert")
        if arm == "vanilla":
            require(np.all(installed == np.arange(384)), "native placement is not identity")
        if arm == "rr":
            adaptive = placement
            upstream = np.asarray(load(group / "placement.json"))
            require(np.array_equal(installed, upstream), "RR differs from the policy map")
        if arm.startswith("eco"):
            width = physical // RANKS
            require(
                np.array_equal(
                    np.sort(installed.reshape(LAYERS, RANKS, width), axis=-1),
                    np.sort(upstream.reshape(LAYERS, RANKS, width), axis=-1),
                ),
                "installed ECO rank membership differs from upstream",
            )
            if eco_layout is None:
                eco_layout = installed
            require(np.array_equal(installed, eco_layout), "installed ECO slot order differs")
            if arm == "eco":
                eco_allocation = hashlib.sha256(np.ascontiguousarray(counts)).digest()
                eco_auction = auction
            elif arm == "eco_smr":
                require(
                    eco_allocation == hashlib.sha256(np.ascontiguousarray(counts)).digest()
                    and np.array_equal(auction, eco_auction),
                    "SM-reuse changed ECO placement, quotas or auction",
                )
        if arm.startswith("eco"):
            require(placement == adaptive, "EPLB rank membership differs")
        spans = v[:, :, :2]
        require(np.all(spans.sum(1) > 0), "empty layer spans")
        actual = spans / spans.sum(1, keepdims=True)
        prices = v[:, :, 2:4]  # fine and linear prices on this same allocation
        shares = prices / prices.sum(1, keepdims=True)
        score_index = int(arm == "eco_linear")
        predicted = shares[:, :, score_index]
        err = np.abs(100 * (predicted[:, :, None] - actual))
        diff = int(np.count_nonzero(native != reference_native))
        rows.append(
            dict(
                bs=bs,
                arm=arm,
                e2e_ms_per_step=host,
                vs_vanilla_pct=100 * (host / reference_host - 1),
                **metrics(counts, spans),
                routed_mae_pp=float(err[:, :, 1].mean()),
                routed_max_pp=float(err[:, :, 1].max()),
                graph_mae_pp=float(err[:, :, 0].mean()),
                graph_max_pp=float(err[:, :, 0].max()),
                score_source="online ECO" if auction is not None else "offline fixed allocation",
                native_diff=diff,
                warm_native_equal=warm == reference_warm,
                measured_pair_comparable=diff == 0,
                strict_pair_comparable=(diff == 0 and warm == reference_warm),
            )
        )
        if with_linear:
            rows[-1].update(
                primary_price=("linear" if score_index else "fine"),
                auction_slack=(90 if score_index else 11469) if auction is not None else None,
            )
            for i, price_name in enumerate(("fine", "linear")):
                error = np.abs(100 * (shares[:, :, i, None] - actual))
                for target_index, target in enumerate(("graph", "routed")):
                    deviations = error[:, :, target_index]
                    rows[-1][f"{price_name}_{target}_mae_pp"] = float(deviations.mean())
                    rows[-1][f"{price_name}_{target}_max_pp"] = float(deviations.max())
        rank_errors = []
        for rank in range(RANKS):
            rank_errors.append(
                dict(
                    bs=bs,
                    arm=arm,
                    rank=rank,
                    predicted_pct=float(100 * predicted[:, rank].mean()),
                    routed_pct=float(100 * actual[:, rank, 1].mean()),
                    routed_mae_pp=float(err[:, rank, 1].mean()),
                    graph_pct=float(100 * actual[:, rank, 0].mean()),
                    graph_mae_pp=float(err[:, rank, 0].mean()),
                )
            )
        write_rows(group / arm / "rank_shares.csv", rank_errors)
        outcomes[arm] = (native, warm, rows[-1])
        peaks = spans[:, :, 0].max(1).reshape(STEPS, LAYERS)
        for layer in range(LAYERS):
            layers.append(
                dict(
                    bs=bs,
                    arm=arm,
                    layer=layer,
                    mean_slowest_rank_us=float(peaks[:, layer].mean()),
                    routed_mae_pp=float(err[layer::LAYERS, :, 1].mean()),
                    graph_mae_pp=float(err[layer::LAYERS, :, 0].mean()),
                )
            )
        if auction is not None:
            auctions.append(dict(bs=bs, arm=arm, **auction_metrics(auction)))
            write_rows(
                group / arm / "auction_phases.csv",
                [
                    dict(bs=bs, arm=arm, phase=phase, **auction_metrics(part))
                    for phase, part in (
                        ("cold_step_0", auction[:LAYERS]),
                        ("warm_steps_1_onwards", auction[LAYERS:]),
                    )
                ],
            )
            with gzip.open(group / arm / "auction_calls.csv.gz", "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "layer", *AUCTION])
                for i, values in enumerate(auction):
                    w.writerow([*divmod(i, LAYERS), *values])
    pairs = []
    for a, b in combinations(arms, 2):
        n0, w0, r0 = outcomes[a]
        n1, w1, r1 = outcomes[b]
        diff = int(np.count_nonzero(n0 != n1))
        pairs.append(
            dict(
                bs=bs,
                arm_a=a,
                arm_b=b,
                native_diff=diff,
                measured_samples=bs * STEPS,
                warm_native_equal=w0 == w1,
                strict_pair_comparable=(w0 == w1 and diff == 0),
                moe_change_pct=100 * (r1["moe_ms_per_step"] / r0["moe_ms_per_step"] - 1),
                e2e_change_pct=100 * (r1["e2e_ms_per_step"] / r0["e2e_ms_per_step"] - 1),
            )
        )
        if with_linear:
            pairs[-1].update(
                moe_delta_ms=r1["moe_ms_per_step"] - r0["moe_ms_per_step"],
                e2e_delta_ms=r1["e2e_ms_per_step"] - r0["e2e_ms_per_step"],
                routed_mae_delta_pp=r1["routed_mae_pp"] - r0["routed_mae_pp"],
            )
    return rows, layers, auctions, pairs


def report(root, batches, with_linear=None):
    if with_linear is None:
        source = load(root / f"bs{batches[0]}" / "vanilla" / "run.json")
        with_linear = source["controls"].get("GC_MOE_MATRIX_SIZE", "20") == "24"
    sha = (root / "measurement.sha").read_text().strip()
    rows = []
    layers = []
    auctions = []
    pairs = []
    for bs in batches:
        a, b, c, d = analyze_group(root, bs, sha, with_linear)
        rows.extend(a)
        layers.extend(b)
        auctions.extend(c)
        pairs.extend(d)
    out = root if len(batches) > 1 else root / f"bs{batches[0]}"
    write_rows(out / "main_table.csv", rows)
    write_rows(out / "layers.csv", layers)
    write_rows(out / "auction.csv", auctions)
    write_rows(out / "pairs.csv", pairs)
    if with_linear:
        write_rows(
            out / "cost_pairs.csv", [p for p in pairs if (p["arm_a"], p["arm_b"]) == ("eco_linear", "eco")]
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--bs", type=int, choices=BATCHES)
    p.add_argument(
        "--with-linear",
        action="store_true",
        default=None,
        help="require all 24 arms and audit the active prices",
    )
    args = p.parse_args()
    report(args.root, [args.bs] if args.bs else BATCHES, args.with_linear)
