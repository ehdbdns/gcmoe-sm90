"""Fit public GLM samples and score frozen prices on paired full-window rank clocks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("vanilla", "rr", "eco")
LAYERS = (8, 20, 32)
OLD = (249, 140)
NEW = (66, 74)
PUBLIC = Path(__file__).resolve().parents[2] / "docs/sm100_megamoe/glm53_formal_results/formal.QLfEZmKt"


def share_errors(x, duration, weights):
    score = x @ np.array((1, *weights), dtype=float)
    return 100 * (score / score.sum(-1, keepdims=True) - duration / duration.sum(-1, keepdims=True))


def mae(x, duration, weights):
    return float(np.abs(share_errors(x, duration, weights)).mean())


def fit(x, duration, fixed_active=None):
    from scipy.optimize import least_squares

    observed = duration / duration.sum(-1, keepdims=True)
    def residual(w):
        score = x @ np.array([1., w[0], w[1] if fixed_active is None else fixed_active])
        return (100 * (score / score.sum(-1, keepdims=True) - observed)).ravel()

    start = OLD if fixed_active is None else OLD[:1]
    result = least_squares(residual, start, bounds=(0, np.inf), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success:
        raise ValueError(result.message)
    return result.x if fixed_active is None else np.array([result.x[0], fixed_active])


def samples(root):
    features, target = [], []
    for arm in ARMS:
        group = []
        times = []
        for layer in LAYERS:
            rows = json.loads((root / arm / "perfetto" / f"step0002-layer{layer:02d}.ranks.json").read_text())
            if [r["rank"] for r in rows] != list(range(8)):
                raise ValueError("incomplete rank set")
            q = np.asarray([r["slot_counts"] for r in rows])
            x = np.stack([q.sum(1), ((q + 15) // 16).sum(1), (q > 0).sum(1)], axis=-1)
            if any(r["block_m"] != 16 or r["step"] != 2 or r["layer"] != layer for r in rows):
                raise ValueError("wrong sample geometry")
            np.testing.assert_array_equal(x, [[r[k] for k in ("tokens", "m_blocks", "active")] for r in rows])
            if arm == "eco":
                np.testing.assert_array_equal(x @ [1, *OLD], [r["price"] for r in rows])
            group.append(x)
            times.append([r["routed_ns"] for r in rows])
        features.append(group)
        target.append(times)
    x, y = np.array(features), np.array(target)
    if np.any(x[..., 0].sum(-1) != 2048) or np.any(y <= 0):
        raise ValueError("invalid sample counts or time")

    weights = fit(x, y)
    folds = []
    for j, layer in enumerate(LAYERS):
        keep = np.arange(len(LAYERS)) != j
        w = fit(x[:, keep], y[:, keep])
        folds.append(dict(held_layer=layer, trained_weights=w.tolist(), held_mae_pp=mae(x[:, j], y[:, j], w),
                          old_mae_pp=mae(x[:, j], y[:, j], OLD)))
    return dict(weights=weights.tolist(), integer_weights=list(NEW), old_weights=list(OLD),
                fitted_mae_pp=mae(x, y, NEW), old_mae_pp=mae(x, y, OLD), layer_folds=folds,
                by_arm={arm: dict(fitted_mae_pp=mae(x[i], y[i], NEW), old_mae_pp=mae(x[i], y[i], OLD))
                        for i, arm in enumerate(ARMS)})


def full(root, output, historical):
    from glm53_launch_compare import phase_summary

    inventory = {}
    if historical:
        receipt = json.loads((PUBLIC / "receipt.json").read_text())
        if (root / "revision").read_text().strip() != receipt["revision"]:
            raise ValueError("wrong historical cohort")
        inventory = {f["path"]: f["sha256"] for f in receipt["files"]}
    output.mkdir(parents=True, exist_ok=False)
    scores = {}
    past = []
    from glm53_matrix import SLOT_BUDGETS
    slot_budgets = not historical and (root / "scheduled-arms.json").is_file()
    for arm in (SLOT_BUDGETS if slot_budgets else ARMS):
        result = json.loads((root / arm / "result.json").read_text())
        expected_slots = 32 if arm == "vanilla" else (40 if historical else 36)
        if slot_budgets:
            expected_slots = 32 + SLOT_BUDGETS[arm]
        setup = sorted(result["setup"], key=lambda r: r["rank"])
        if len(setup) != 8 or any(s["slots"] != expected_slots or s["block_m"] != 16 for s in setup):
            raise ValueError(f"invalid {arm} geometry")
        if any(s["full_replays"] != 1000 for s in result["measure"]):
            raise ValueError("incomplete measured window")
        weights = OLD if historical else NEW
        if arm.startswith("eco") and any(s["eco_unit"] != [16, *weights, 1, 6, 0] for s in setup):
            raise ValueError("installed historical/experimental ECO price differs")
        ranks, clock_rows = [], []
        for rank in range(8):
            path = root / arm / f"measure-rank{rank}.npz"
            if historical:
                with path.open("rb") as f:
                    if hashlib.file_digest(f, "sha256").hexdigest() != inventory[str(path.relative_to(root))]:
                        raise ValueError(f"archive checksum differs for {path}")
            with np.load(path, allow_pickle=False) as data:
                q, c = data["counts"].astype(np.int32), data["clocks"]
                if q.shape != (1000, 75, expected_slots) or c.shape != (1000, 75, 21) or (q < 0).any():
                    raise ValueError("incomplete rank measurements")
                ranks.append(np.stack((q.sum(-1), ((q + 15) // 16).sum(-1), (q > 0).sum(-1)), axis=-1))
                clock_rows.append(c)
        x = np.stack(ranks, axis=2)
        c = np.stack(clock_rows, axis=2)
        duration = np.where(x[..., 0] > 0, c[..., 7] - c[..., 4], 0)
        if np.any(x[..., 0].sum(-1) != 2048) or np.any(duration[x[..., 0] > 0] <= 0):
            raise ValueError("invalid rank work or routed span")
        if arm.startswith("eco"):
            np.testing.assert_array_equal(c[..., 19], x @ [1, *weights])
        # Publish compact sufficient inputs for independent fits, without model or request payloads.
        np.savez_compressed(output / f"{arm}.npz", tokens=x[..., 0].astype(np.uint16),
                            m_blocks=x[..., 1].astype(np.uint16), active=x[..., 2].astype(np.uint16),
                            routed_ns=duration)
        scores[arm] = dict(old_mae_pp=mae(x, duration, OLD), candidate_mae_pp=mae(x, duration, NEW),
                           calls=75000, **phase_summary(c, arm.startswith("eco")))
        if historical:
            past.append((x, duration))
    if historical:
        # Fit on the old window only. The four-slot price in the GPU binary is frozen.
        x = np.stack([a for a, _ in past])
        y = np.stack([b for _, b in past])
        held_weights = fit(x[:, :800], y[:, :800])
        full_weights = fit(x, y)
        scores["historical_refit"] = dict(full_weights=full_weights.tolist(), fit_calls=225000,
            train_steps_1_800_weights=held_weights.tolist(), holdout_steps_801_1000_mae_pp=dict(
                historical_refit=mae(x[:, 800:], y[:, 800:], held_weights),
                frozen_candidate=mae(x[:, 800:], y[:, 800:], NEW),
                original=mae(x[:, 800:], y[:, 800:], OLD)))
        tests = {}
        for active in (0, 1):
            w = fit(x[:, :800], y[:, :800], fixed_active=active)
            tests[str(active)] = dict(trained_wb=float(w[0]), held_mae_pp=mae(x[:, 800:], y[:, 800:], w))
        scores["historical_refit"]["fixed_active_on_same_time_holdout"] = tests
    (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sample", "historical", "new"))
    parser.add_argument("root", type=Path, nargs="?", default=PUBLIC)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action != "sample" and args.output is None:
        parser.error("full windows require --output")
    report = samples(args.root) if args.action == "sample" else full(args.root, args.output, args.action == "historical")
    print(json.dumps(report, indent=2))
