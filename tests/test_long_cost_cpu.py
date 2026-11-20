"""Regressions for independent route banks, capture replay and paired MoE aggregation."""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "integration"), str(ROOT / "scripts/trace_layer")]
import long_cost_report as report
from fixed_tokens import ReferenceTokens
from long_cost import LongRouteTape


def routes(directory, device):
    epochs = [255, 257]  # both layer-aligned, crossing the original 256-entry ring
    audits = {}
    for cost in ("0", "2"):
        with patch.dict(
            os.environ, GC_MOE_COST_STUDY="1", GC_MOE_ECO_COSTFN=cost, GC_MOE_FIXED_DIR=str(directory)
        ):
            tape = LongRouteTape(device, 2, 2, steps=1, layers=2, batch=2)
            buf = SimpleNamespace(
                eco_tl_step=torch.zeros(1, dtype=torch.int32, device=device),
                used_route_ordinal=torch.zeros((2, 2), dtype=torch.int32, device=device),
                profile_slot_counts=torch.ones((256, 2), dtype=torch.int32, device=device),
            )
            live = torch.tensor([[0, 1], [1, 2]], device=device)
            weights = torch.arange(4, dtype=torch.float32, device=device).reshape(2, 2)

            def call(layer):
                tape.select(live, weights, layer, buf.used_route_ordinal)
                tape.finish_call(buf)

            graphs = []
            if device == "cuda":
                for layer in range(2):
                    call(layer)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        call(layer)
                    graphs.append(graph)
            for stage, first in zip(("stats", "measure"), epochs):
                tape.start_phase(stage, 0)
                if stage == "stats":
                    tape.replaying.fill_(False)  # attention coverage is independent of route selection
                for layer in range(2):
                    buf.eco_tl_step.fill_(first + layer)
                    offset = 100 if cost == "2" else 0
                    live.fill_(first + layer + offset)
                    weights.fill_(first + layer + offset)
                    if graphs:
                        graphs[layer].replay()
                    else:
                        call(layer)
                    assert tape.selected_ids.eq(first + layer).all()
                    assert tape.selected_weights.eq(first + layer).all()
                    assert torch.equal(buf.used_route_ordinal.cpu(), torch.tensor([[0, 1], [2, 3]]))
                result = tape.finish_phase(first)
                if cost == "0":
                    audits[stage] = result
                else:
                    assert audits[stage] == result
    assert audits["stats"]["ids"] != audits["measure"]["ids"]
    tape.used_ordinals.fill_(-1)
    try:
        tape.finish_phase(epochs[-1])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid source ordinals accepted")


def native_feedback():
    for primary in (True, False):
        table = {"req0": [] if primary else [4, 8], "req1": [] if primary else [5, 9]}
        ref = ReferenceTokens(table, "cpu", capacity=4, follow_native=primary)
        ref.mark_replay_start()
        for step, values in enumerate(([4, 5], [8, 9])):
            raw = torch.tensor(values) + (0 if primary else 1)
            output = SimpleNamespace(sampled_token_ids=raw.clone()[:, None])
            ref.feed(list(table), output, torch.ones(2, dtype=torch.long))
            assert output.sampled_token_ids[:, 0].tolist() == values
        if primary:
            ref.lengths.copy_(ref.positions)
        assert ref.native_history(2)["replay_top1"][0] == ([4, 5] if primary else [5, 6])


def aggregation(root):
    # Use the production window length and eight ranks. Two layers deliberately
    # have their slowest ranks on different devices.
    (root / "measurement.sha").write_text("synthetic-test")
    shape = (1000, 2, 8, 2)
    times = np.full(shape, 100.0, dtype=float)
    times[:, 0, 1, 0] = 600
    times[:, 1, 7, 0] = 400

    def fixture(_, arm, revision):
        assert revision == "synthetic-test"
        scale = 1 if arm == "linear" else 0.9
        return (
            {},
            np.ones((1000, 2)),
            np.full(100, 90.0 * scale),
            times * scale,
            np.ones(shape),
            np.ones(shape),
        )

    with (
        patch.object(report, "read_arm", fixture),
        patch.object(report, "LAYERS", 2),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        report.compare(root)
    result = json.loads((root / "comparison.json").read_text())
    assert result["strict_pair_comparable"] and result["measured_moe_faster"]
    assert np.isclose(result["arms"]["linear"]["moe_ms_per_step"], 1.0)
    assert np.isclose(result["paired_moe_delta_pct"], -10)
    assert result["moe_faster_steps"] == 1000 and result["moe_faster_50_step_blocks"] == 20
    assert len((root / "step_pairs.csv").read_text().splitlines()) == 1001
    assert len((root / "layer_pairs.csv").read_text().splitlines()) == 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        routes(Path(tmp), "cuda" if args.cuda else "cpu")
        native_feedback()
        aggregation(Path(tmp))
    print("LONG_COST PASS")
