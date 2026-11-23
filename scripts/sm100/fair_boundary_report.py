"""CPU-only evaluation of the shared-prefill and common-expert-input controls."""
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load(path):
    return json.loads(path.read_text())


def report(root):
    arms = [root / name for name in ("mega", "gc")]
    results = [load(a / "result.json") for a in arms]
    checks = {}
    checks["prefill"] = load(arms[0]/"prefill.json") == load(arms[1]/"prefill.json")
    tokens = [np.load(a/"stats-tokens.npy") for a in arms]
    checks["tokens"] = tokens[0].shape == (512, 3) and np.array_equal(*tokens)
    checks["prompt"] = results[0]["prompt_sha256"] == results[1]["prompt_sha256"]
    checks["full_graph_mode"] = all(res["stats"]["full_decode"] for res in results)
    totals = np.zeros((3, 40), dtype=np.int64)
    native, replay, kv = [], [], []
    anchor = {}
    for r in range(8):
        ranks = [res["stats"]["ranks"][r] for res in results]
        a, b = [row["kv"] for row in ranks]
        checks[f"rank{r}_kv"] = (a["source_sha256"] == b["readback_sha256"] == a["readback_sha256"]
                                  and a["controls_sha256"] == b["controls_sha256"])
        kv.append(dict(rank=r, raw_before_equal=a["before_restore_sha256"] == b["before_restore_sha256"]))
        checks[f"rank{r}_calls"] = all(v["full_graph_observed_calls"] == 120 and v["block_m"] == 16 for v in ranks)
        checks[f"rank{r}_sm"] = ranks[1]["sm_split"] == {"dispatch": 20, "compute": 128}
        with np.load(arms[0]/f"stats-rank{r}.npz") as left, np.load(arms[1]/f"stats-rank{r}.npz") as right:
            checks[f"rank{r}_counts"] = left["counts"].shape == (3,40,48) and np.array_equal(left["counts"], right["counts"])
            totals += left["counts"].sum(axis=-1)
        controls = [load(p/f"replay-{r}.json") for p in arms]
        assert all(len(x) == 120 for x in controls)
        for step in range(3):
            rows = [torch.load(p/f"native-{step}-{r}.pt", map_location="cpu", weights_only=True) for p in arms]
            assert all(len(x) == 40 for x in rows)
            for layer in range(40):
                i,j = [v[layer] for v in rows]
                differences = [k for k in ("attn_in","attn_out") if i["attention"][k] != j["attention"][k]]
                details = {}
                if (step, layer, r) == (0, 3, 7):
                    for name, data in (("mega", i), ("gc", j)):
                        anchor[name] = data["tensors"]["topk_weights"][33].tolist()
                for k in ("hidden_states","topk_ids","topk_weights","y"):
                    left,right = i["tensors"][k], j["tensors"][k]
                    if left.dtype != right.dtype or left.shape != right.shape:
                        raise ValueError(f"native tensor schema mismatch at {(step,layer,r,k)}")
                    bits = left.contiguous().view(torch.uint8) != right.contiguous().view(torch.uint8)
                    if bits.any():
                        differences.append(k)
                        pos = tuple(bits.reshape(*left.shape, left.element_size()).any(-1).nonzero()[0].tolist())
                        details[k] = dict(first=pos, mega=left[pos].item(), gc=right[pos].item())
                if differences:
                    native.append(dict(step=step,layer=layer,rank=r,fields=differences,details=details))
                x,y = [v[step*40+layer] for v in controls]
                assert all((v["step"],v["layer"]) == (step,layer) for v in (x,y))
                changed = [k for k in x["inputs"] if x["inputs"][k] != y["inputs"][k]]
                slots = [k for k,(a,b) in enumerate(zip(x["partial_hashes"],y["partial_hashes"],strict=True)) if a!=b]
                verified = x["partials_verified"] and y["partials_verified"]
                if changed or slots or not verified or any(v["different"] or not v["finite"] for v in (x,y)):
                    replay.append(dict(step=step,layer=layer,rank=r,input_differences=changed,
                                verified_partials=verified, differing_slots=slots if verified else None,
                                mega=x,gc=y))
    native.sort(key=lambda v:(v["step"],v["layer"],v["rank"]))
    checks["global_load"] = bool((totals == 3072).all())
    outcome = dict(anchor=anchor, short_strict_pass=all(checks.values()) and not native and not replay,
                   checks=checks, native_differences=native, common_input_replay_differences=replay,
                   pre_restore_allocation_hashes=kv)
    (root/"fair_boundary.json").write_text(json.dumps(outcome,indent=2)+"\n")
    print("short_strict_pass", outcome["short_strict_pass"])
    return outcome["short_strict_pass"]


if __name__ == "__main__":
    sys.exit(0 if report(Path(sys.argv[1])) else 3)
