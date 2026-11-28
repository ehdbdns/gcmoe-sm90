"""DeepInfer history/replica-order regressions and the 20- and 24-arm archive verifier."""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import csv
import json
import shutil
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / s) for s in ("source", "integration", "scripts/trace_layer")]
from gc_moe.eplb.policy import rebalance_experts_global
from gc_moe.sm90_gc import install_eplb_plan
from eplb_logic import placement_window
import main20_report as report


def test_policy():
    # Expected maps from DeepInfer a679bb86, including upstream inverse copy order.
    x = torch.tensor([[[10, 0, 6, 4]], [[0, 10, 4, 6]]])
    phy, inverse, count = rebalance_experts_global(x, 6, 3)
    assert phy.tolist() == [[2, 1, 0, 3, 0, 1]] and count.tolist() == [[2, 2, 1, 1]]
    assert inverse[0, 0].tolist() == [4, 2, -1]
    assert not torch.equal(phy, rebalance_experts_global(x.sum(0, keepdim=True), 6, 3)[0])
    hot = torch.zeros(2, 1, 8, dtype=torch.int32)
    hot[:, :, 0] = 1024
    phy, inv, c = rebalance_experts_global(hot, 24, 3)
    buf = SimpleNamespace(
        num_experts=8,
        physical=24,
        slot_logical=torch.empty_like(phy),
        expert_slots=torch.empty_like(inv),
        buffer=torch.zeros(1),
    )
    _, rr = install_eplb_plan(buf, phy.tolist(), inv)
    assert c.max() > 8 and torch.equal(rr, inv) and torch.equal(buf.expert_slots, inv)
    assert torch.equal(phy[0, rr[0, 0, : c[0, 0]].long()], torch.zeros(c[0, 0]))
    ring = torch.tensor([[5, 6, 4], [7, 8, 5], [9, 10, 6], [3, 4, 3]], dtype=torch.int32)
    buf = SimpleNamespace(
        placement_history=ring, header=torch.tensor([7]), group=SimpleNamespace(size=lambda: 2)
    )
    origin = torch.tensor([[0, 1, 0, 2], [0, 2, 1, 1]])
    h = placement_window(buf, origin, 1, 3, first_epoch=3)
    assert h.tolist() == [[[3, 0, 4], [0, 11, 0]], [[7, 0, 8], [0, 19, 0]]]
    ring[1, -1] = 1
    try:
        placement_window(buf, origin, 1, 3, first_epoch=3)
    except ValueError:
        pass
    else:
        raise AssertionError("stale ring accepted")


def test_matrix_replay_checks():
    from fixed_tokens import ReferenceTokens
    from long_cost import LongRouteTape

    ref = ReferenceTokens({"req0": [4, 5, 6], "req1": [8, 9]}, "cpu", capacity=5)
    ref.positions.copy_(torch.tensor([3, 2]))
    ref.native[:, :3] = torch.tensor([[4, 5, 6], [8, 9, 77]])
    start = torch.tensor([1, 1])
    ref.verify_native_prefix("stats", start)
    ref.native[0, 0] = 0  # Before the window.
    ref.native[1, 1] = 3
    try:
        ref.verify_native_prefix("stats", start)
    except ValueError as exc:
        assert "req1, position 1, native 3, reference 9" in str(exc)
    else:
        raise AssertionError("accepted divergent native token")

    with patch.dict("os.environ", GC_MOE_MATRIX_ARM="rr"):
        tape = LongRouteTape("cpu", 2, 3, steps=2, layers=1, batch=2)
        out = torch.empty((2, 2), dtype=torch.int32)
        tape.record_ordinals(torch.tensor([[[1, 2], [1, 2]]]), out)
        buf = SimpleNamespace(used_route_ordinal=out.clone(), eco_tl_step=torch.tensor([1], dtype=torch.int32),
                              profile_slot_counts=torch.zeros((256, 3), dtype=torch.int32))
        tape.finish_call(buf)
        tape.index.fill_(1)
        buf.used_route_ordinal[1, 0] = 0
        tape.finish_call(buf)
        assert tape.ordinal_match.tolist() == [True, False]


def test_worker_export(root):
    from long_cost import register_long_rpc

    calls = 6
    epochs = torch.arange(1, calls + 1, dtype=torch.int32)
    tape = SimpleNamespace(
        steps=2,
        layers=3,
        batch=1,
        counts=torch.ones(calls, 1, dtype=torch.int32),
        ids=torch.zeros(calls, 1, 1, dtype=torch.int64),
        epochs=epochs,
        finish_phase=lambda first: {},
    )
    auction = torch.arange(calls * 7, dtype=torch.int32).reshape(calls, 7)
    buf = SimpleNamespace(
        group=SimpleNamespace(size=lambda: 1),
        rank_idx=0,
        num_topk=1,
        num_experts=1,
        fixed_routes=tape,
        eco=True,
        layout=SimpleNamespace(matrix=True, auction_granularity_audit=True),
        auction_small=torch.arange(calls),
        auction_ns=torch.arange(calls) + 10,
        replay_stream_mode=torch.tensor(1),
        header=torch.tensor([calls + 1]),
        auction_counts=auction,
        long_clocks=torch.ones(calls, 6, dtype=torch.int64),
    )

    class Worker:
        pass

    ref = SimpleNamespace(
        replay_start=torch.zeros(1),
        verify_native_prefix=lambda phase, start: None,
        native_history=lambda steps: dict(replay_top1=[[7], [8]], warm_top1_sha256="warm"),
        batch_hash=hashlib.sha256(),
    )
    worker = Worker()
    worker._gc_reference_tokens = ref
    worker._long_first = 1
    with (
        patch.dict("os.environ", GC_MOE_MATRIX_ARM="eco"),
        patch("torch.distributed.all_reduce"),
        patch("torch.cuda.synchronize"),
        patch("attention_order.order_coverage", return_value={2: [1, 1]}),
    ):
        register_long_rpc(Worker, lambda: buf, [])
        result = worker._gc_moe_long_finish("measure", str(root))
    assert result["warm_native"] == "warm"
    with np.load(root / "rank0.npz") as raw:
        np.testing.assert_array_equal(raw["auction"], auction[epochs.long() % calls])
        for key in ("auction_small", "auction_ns"):
            np.testing.assert_array_equal(raw[key], getattr(buf, key)[epochs.long() % calls])
        assert raw["native"].tolist() == [[7], [8]]


def test_prepared_prices():
    from dataclasses import replace
    from gc_moe.apis.sm90_gc_moe import ValidatedCall
    from gc_moe.layout.sm90_gc_moe import SM90GCBufferLayout
    from gc_moe.jit_kernels.impls import sm90_fp8_fp4_gc_moe as jit

    t = torch.zeros(1)
    config = ValidatedCall(
        ranks=8,
        experts=384,
        capacity=2048,
        topk=6,
        hidden=5120,
        intermediate=2304,
        tokens=32,
        bound=2048,
        clamp=10.0,
        fast_math=True,
        physical=448,
        eco_logical_experts=384,
        eco_wb=26,
        eco_wa=0,
    )
    ctx = SimpleNamespace(dispatch_sms=6, compute_sms=72)
    with (
        patch.dict("os.environ", GC_MOE_MATRIX_ARM="eco", GC_MOE_BLOCK_M_FORCE="64", GC_MOE_MB_CLAIM="1"),
        patch.object(jit, "_CACHE", {}),
        patch.object(jit, "_get_or_compile_binaries", return_value=()),
        patch.object(jit, "_compiler_identity", return_value=None),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        layout = SM90GCBufferLayout(8, 384, 2048, 6, 5120, 2304, 0, 448, True, True)
        buf = SimpleNamespace(layout=layout, buffer=t, peer_table=t, rank_idx=0)
        for batch, mode, wa, slack in [(32, 0, 0, 90), (256, 0, 1, 91), (512, 2, 0, 11469)]:
            jit.prepare_role_binaries(
                replace(config, tokens=batch, eco_cost_mode=mode, eco_wa=wa),
                t,
                (t, t, t),
                (t, t, t),
                buf,
                stats=t,
                context=ctx,
            )
            assert jit.prepared_matrix_price(layout, batch) == dict(mode=mode, slack=slack)
        assert jit.prepared_matrix_price(layout, 32) == dict(mode=0, slack=90)


def test_rr_counts():
    from matrix_workload import verify_rr_counts

    # Different upstream inverse order for the two layers. Odd length makes the
    # physical slot counts distinguish an order swap even with the same logical sum.
    tape = SimpleNamespace(
        batch=3,
        ids=torch.zeros((2, 3, 1), dtype=torch.long),
        layer_ids=torch.tensor([0, 1]),
        counts=torch.tensor([[1, 2], [2, 1]], dtype=torch.int32),
    )
    buf = SimpleNamespace(
        fixed_routes=tape,
        rank_idx=0,
        group=SimpleNamespace(size=lambda: 1),
        expert_slots=torch.tensor([[[1, 0]], [[0, 1]]], dtype=torch.int32),
    )
    with patch("torch.distributed.all_reduce"):
        assert verify_rr_counts(buf) == 2
        tape.counts[0] = torch.tensor([2, 1])
        try:
            verify_rr_counts(buf)
        except ValueError as e:
            assert "physical slot counters" in str(e)
        else:
            raise AssertionError("RR accepted a different copy order")


def test_archive(root):
    # Three steps and three layers exercise step/layer indexing and cold/warm boundaries.
    report.STEPS = report.LAYERS = 3
    bs = 32
    group = root / "bs32"
    manifest = group / "checkpoint"
    manifest.mkdir(parents=True)
    history = np.zeros((3, 3, 384), dtype=np.int32)
    history[:, :, 0] = bs * 6 * 8
    np.savez(group / "stats-demand.npz", logical=history)
    demand = hashlib.sha256(history).hexdigest()
    (root / "measurement.sha").write_text("revision")
    fixed = dict(
        GC_MOE_MATRIX_SIZE="24",
        GC_MOE_ECO_WB="26",
        GC_MOE_ECO_WA="0",
        GC_MOE_ECO_BLOCKS_ONLY="0",
        GC_MOE_ECO_TOKENS_ONLY="0",
        GC_MOE_ECO_ROUNDS="1",
        GC_MOE_ECO_COLD_ROUNDS="6",
        GC_MOE_SM90_DISPATCH_SMS="6",
        GC_MOE_MATRIX_BS="32",
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
    for arm in report.ARMS24:
        dst = group / arm
        dst.mkdir()
        layout = [list(range(384)) + ([] if arm == "vanilla" else list(range(64))) for _ in range(3)]
        (dst / "installed-layout.json").write_text(json.dumps(layout))
        if arm == "rr":
            (group / "placement.json").write_text(json.dumps(layout))
        eco = arm.startswith("eco")
        linear = arm == "eco_linear"
        cfg = dict(
            fixed,
            GC_MOE_MATRIX_ARM=arm,
            GC_MOE_ECO_COSTFN="0" if linear else "2",
            GC_MOE_EPLB=str(int(arm != "vanilla")),
            GC_MOE_ECO=str(int(eco)),
            GC_MOE_SM90_SM_REUSE=str(int(arm == "eco_smr")),
            GC_MOE_SLOT_PLACEMENT="random" if arm == "random" else "identity",
            GC_MOE_EPLB_NO_REARRANGE=str(int(arm in ("vanilla", "random"))),
        )
        run = dict(
            controls=cfg,
            batch=bs,
            steps=3,
            layers=3,
            stream=1,
            cost=0 if linear else 2,
            rearranged=(arm == "rr" or eco),
            prompt_hash="p",
            output_hash="x",
            prefill_steps=9,
        )
        for name in ("hardware.csv", "runtime.log", "execution.sha"):
            (dst / name).write_text("revision")
        for phase, start in [("stats", 1), ("measured", 10)]:
            win = dict(audit=[], initial=[], ten_step_ms=[2] * 100, ms_per_step=2)
            for rank in range(8):
                audit = dict(
                    rank=rank,
                    calls=9,
                    first_epoch=start,
                    ids="i",
                    weights="w",
                    demand=demand,
                    compressed_layers=[2],
                    total_rows_by_layer=[576] * 3,
                    batch_order="b",
                    warm_native="w",
                )
                if arm == "rr":
                    audit["rr_checked_calls"] = 9
                record = dict(rank=rank, epoch=start)
                if phase == "measured":
                    record["checkpoint"] = dict(
                        plan_price=(dict(mode=0, slack=90) if linear else dict(mode=2, slack=11469))
                        if eco
                        else None,
                        snapshot=dict(source_sha256="s", readback_sha256="s"),
                        demand=demand,
                        logical_weights="w",
                        placement="layout",
                        installed_layout=hashlib.sha256(
                            json.dumps(layout, separators=(",", ":")).encode()
                        ).hexdigest(),
                    )
                    (manifest / f"rank{rank}.json").write_text(
                        json.dumps(
                            dict(sha256="s", controls=dict(next_epoch=10, demand=demand, logical_weights="w"))
                        )
                    )
                win["audit"].append(audit)
                win["initial"].append(record)
            run[phase] = win
        (dst / "run.json").write_text(json.dumps(run))
        q = np.zeros((9, 48 if arm == "vanilla" else 56), np.int32)
        q[:, 0] = 192
        epochs = np.arange(10, 19)
        clocks = np.tile([1000, 2000, 8000, 11000, 0, 0], (9, 1))
        clocks[:, 4] = epochs
        native = np.zeros((3, bs), np.int32)
        if arm == "random":
            native[1, 3] = 1
        extra = {}
        if eco:
            clocks[:, 5] = report.slot_cost(192)
            extra["auction"] = np.tile([1, 1, 1, 3, 2, 0, 0], (9, 1))
        for rank in range(8):
            local_q, local_clocks = q.copy(), clocks.copy()
            if linear:
                if rank == 0:
                    local_q[:, :2] = [129, 63]
                local_clocks[:, 5] = local_q.sum(1) + 26 * ((local_q + 63) // 64).sum(1)
                extra["auction"] = np.tile([1, 1, 0, 3, 2, 0, 1], (9, 1))
            np.savez(
                dst / f"rank{rank}.npz",
                counts=local_q,
                clocks=local_clocks,
                native=native,
                epochs=epochs,
                **extra,
            )
    rows, layers, auctions, pairs = report.analyze_group(root, bs, "revision")
    assert len(rows) == 5 and len(layers) == 15 and len(pairs) == 10 and len(auctions) == 2
    assert rows[0]["moe_ms_per_step"] == 0.03 and rows[0]["routed_mae_pp"] == 0
    assert (
        rows[0]["total_m_blocks"] == 24
        and rows[0]["peak_rank_m_blocks"] == 3
        and rows[0]["active_physical_slots"] == 8
    )
    assert not rows[1]["strict_pair_comparable"] and rows[1]["native_diff"] == 1
    assert auctions[0]["accepted"] == 9 and auctions[0]["blocked_before_offer"] == 9
    rows, layers, auctions, pairs = report.analyze_group(root, bs, "revision", with_linear=True)
    assert len(rows) == 6 and len(layers) == 18 and len(auctions) == 3 and len(pairs) == 15
    old, new = rows[3:5]
    assert old["auction_slack"] == 90 and new["auction_slack"] == 11469
    assert (
        old["routed_mae_pp"] == old["linear_routed_mae_pp"]
        and old["fine_routed_mae_pp"] != old["routed_mae_pp"]
    )
    assert (
        auctions[0]["arm"] == "eco_linear" and auctions[0]["accepted"] == 0 and auctions[1]["accepted"] == 9
    )
    report.report(root, [bs])
    with (group / "cost_pairs.csv").open() as f:
        (cost_pair,) = csv.DictReader(f)
    assert (cost_pair["arm_a"], cost_pair["arm_b"], cost_pair["strict_pair_comparable"]) == (
        "eco_linear",
        "eco",
        "True",
    )
    linear_dir = group / "eco_linear"
    # Candidate ranks must include one charge per nonempty slot and a slack of 91.
    candidate = group / "eco_linear_wa1"
    shutil.copytree(linear_dir, candidate)
    run = json.loads((candidate / "run.json").read_text())
    run["controls"].update(GC_MOE_MATRIX_ARM=candidate.name, GC_MOE_ECO_WA="1")
    for rank in run["measured"]["initial"]:
        rank["checkpoint"]["plan_price"]["slack"] = 91
    (candidate / "run.json").write_text(json.dumps(run))
    try:
        report.arm_data(candidate, bs, candidate.name, "revision", with_linear=True)
    except ValueError as e:
        assert "role-5 score" in str(e)
    else:
        raise AssertionError("accepted wa=0 scores for the wa=1 arm")
    for path in candidate.glob("rank*.npz"):
        with np.load(path) as data:
            corrected = dict(data)
        corrected["clocks"][:, 5] += np.count_nonzero(corrected["counts"], axis=1)
        np.savez(path, **corrected)
    report.arm_data(candidate, bs, candidate.name, "revision", with_linear=True)
    run_file = linear_dir / "run.json"
    archived = run_file.read_text()
    run = json.loads(archived)
    run["measured"]["initial"][7]["checkpoint"]["plan_price"]["slack"] = 11469
    run_file.write_text(json.dumps(run))
    try:
        report.arm_data(linear_dir, bs, "eco_linear", "revision", with_linear=True)
    except ValueError as e:
        assert "mode/slack" in str(e)
    else:
        raise AssertionError("accepted a linear arm with fine slack")
    run_file.write_text(archived)
    path = linear_dir / "rank0.npz"
    with np.load(path) as data:
        wrong = dict(data)
    wrong["clocks"][:, 5] = report.slot_cost(129) + report.slot_cost(63)
    np.savez(path, **wrong)
    try:
        report.arm_data(linear_dir, bs, "eco_linear", "revision", with_linear=True)
    except ValueError as e:
        assert "role-5 score" in str(e)
    else:
        raise AssertionError("accepted a linear arm with fine scores")
    path = group / "eco" / "rank7.npz"
    with np.load(path) as f:
        data = dict(f)
    data["auction"][2] = [1, 1, 0, 3, 2, 0, 1]
    np.savez(path, **data)
    try:
        report.analyze_group(root, bs, "revision")
    except ValueError as e:
        assert "planners disagree" in str(e)
    else:
        raise AssertionError("divergent planner accepted")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_matrix_replay_checks()
    test_policy()
    test_prepared_prices()
    test_rr_counts()
    with TemporaryDirectory() as tmp:
        test_worker_export(Path(tmp))
        test_archive(Path(tmp))
    print("MAIN20_CPU PASS")
