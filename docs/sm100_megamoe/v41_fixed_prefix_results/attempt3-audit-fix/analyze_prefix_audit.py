import argparse
import json
from collections import Counter
from pathlib import Path


BACKENDS = ("auto", "deep_gemm_mega_moe")


def load_entries(path):
    entries = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("{") or '"backend"' not in line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries.append(entry)
    return entries


def load_backend(path):
    entries = load_entries(path)
    workers = next((e["workers"] for e in entries if "workers" in e), None)
    predictions = [e for e in entries if "prefix" in e and "top" in e]
    if workers is None:
        raise RuntimeError(f"missing workers record in {path}")
    if len(predictions) != 4:
        raise RuntimeError(f"expected 4 predictions in {path}, got {len(predictions)}")
    return workers, predictions


def top_rows(prediction, count=2):
    rows = []
    for token_id, logprob in prediction["top"][:count]:
        rows.append({"id": int(token_id), "logprob": float(logprob)})
    return rows


def rank_of(token_id, top):
    for index, (candidate, _) in enumerate(top, start=1):
        if int(candidate) == int(token_id):
            return index
    return None


def fusion_summary(workers):
    ranks = {}
    for rank, layers in enumerate(workers):
        fused = sum(layer.get("fused_shared") is True for layer in layers)
        seq_parallel = sum(layer.get("sequence_parallel") is True for layer in layers)
        class_counts = Counter(layer.get("expert_class") for layer in layers)
        ranks[f"rank_{rank}"] = {
            "layers": len(layers),
            "fused_shared_layers": fused,
            "sequence_parallel_layers": seq_parallel,
            "expert_classes": dict(class_counts),
        }
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", required=True)
    parser.add_argument("--mega", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    auto_workers, auto_predictions = load_backend(args.auto)
    mega_workers, mega_predictions = load_backend(args.mega)
    if len(auto_workers) != 8 or len(mega_workers) != 8:
        raise RuntimeError("both backends must report exactly 8 workers")
    if any(len(layers) != 40 for layers in auto_workers + mega_workers):
        raise RuntimeError("every worker must report exactly 40 MoE layers")

    prefixes = []
    for auto_prediction, mega_prediction in zip(auto_predictions, mega_predictions):
        if auto_prediction["prefix"] != mega_prediction["prefix"]:
            raise RuntimeError("prefix mismatch between backends")
        auto_top = auto_prediction["top"]
        mega_top = mega_prediction["top"]
        auto_first, auto_second = top_rows(auto_prediction)
        mega_first, mega_second = top_rows(mega_prediction)
        mega_logprobs = {int(token_id): float(logprob) for token_id, logprob in mega_top}
        auto_logprobs = {int(token_id): float(logprob) for token_id, logprob in auto_top}
        auto_rank_in_mega = rank_of(auto_first["id"], mega_top)
        mega_rank_in_auto = rank_of(mega_first["id"], auto_top)
        prefixes.append({
            "prefix": auto_prediction["prefix"],
            "auto": {
                "generated": int(auto_prediction["generated"]),
                "top1": auto_first,
                "top2": auto_second,
                "top1_minus_top2": auto_first["logprob"] - auto_second["logprob"],
            },
            "deep_gemm_mega_moe": {
                "generated": int(mega_prediction["generated"]),
                "top1": mega_first,
                "top2": mega_second,
                "top1_minus_top2": mega_first["logprob"] - mega_second["logprob"],
            },
            "cross_backend_logprob_delta_mega_minus_auto": {
                str(auto_first["id"]): mega_logprobs.get(auto_first["id"], None) - auto_first["logprob"]
                if auto_first["id"] in mega_logprobs else None,
                str(mega_first["id"]): mega_first["logprob"] - auto_logprobs[mega_first["id"]]
                if mega_first["id"] in auto_logprobs else None,
            },
            "auto_top1_id": auto_first["id"],
            "mega_top1_id": mega_first["id"],
            "auto_top1_rank_in_mega_top10": auto_rank_in_mega,
            "mega_top1_rank_in_auto_top10": mega_rank_in_auto,
        })

    result = {
        "backends": {
            "auto": {"workers": auto_workers, "fusion_summary": fusion_summary(auto_workers)},
            "deep_gemm_mega_moe": {
                "workers": mega_workers,
                "fusion_summary": fusion_summary(mega_workers),
            },
        },
        "prefixes": prefixes,
    }
    Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "prefixes": prefixes,
        "fusion_summary": {
            backend: result["backends"][backend]["fusion_summary"]
            for backend in BACKENDS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
