"""Check SM90 metadata, replanning and scheduling without a device."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root/'source'), str(root/'integration')]
from eplb_logic import layer_demand_from_stats
from gc_moe.eco.meta_sm90 import install_into_buffer, prepare_meta
from gc_moe.layout.sm90_gc_moe import SM90GCBufferLayout
from gc_moe.jit_kernels.heuristics.sm90_gc_moe import get_sm90_role_config
from gc_moe.apis.sm90_gc_moe import ValidatedCall


def test_repeated_rearrangement(buf, first, second):
    # Load the normal worker entry point without importing vLLM's backend
    # registration (the module registers the GPU platform at import time).
    path = root/'integration/deepep_compare.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('rearrange_now', '_eco_enabled', '_slot_placement')]
    scope = dict(torch=torch, os=os, _ECO_SLOT_ORDER={})
    exec(compile(tree, path, 'exec'), scope)
    import gc_moe.sm90_gc as sm90
    from gc_moe.eco.meta_sm90 import reorder_single_copy_first

    buf.buffer = torch.zeros(1)
    buf.rank_idx, buf.num_experts, buf.group = 0, 4, SimpleNamespace(size=lambda: 2)
    buf.slot_logical = torch.empty((2, 6), dtype=torch.int32)
    buf.expert_slots = torch.empty((2, 4, sm90.MAX_REPLICAS), dtype=torch.int32)
    plan, slots = sm90.install_eplb_plan(buf, [first, first])
    stats = torch.zeros((2, 3), dtype=torch.int32)
    state = dict(plan=plan, slots=slots, slot_table=slots.flatten().clone(),
                 count=(slots >= 0).sum(-1), stats=stats)
    shuffler = SimpleNamespace(components=[None, None], move=Mock())
    mod = SimpleNamespace(_eplb=state, _get_buffer=lambda: buf,
                          _eplb_shuffler=shuffler, _gc=SimpleNamespace(sm90_gc=sm90))
    targets = [[first, second], [second, first]]
    scope['_EXPERT_REGISTRY'] = [mod]
    with patch.dict(os.environ, GC_MOE_ECO='1', GC_MOE_SLOT_PLACEMENT='identity'), \
            patch('torch.distributed.all_reduce'), patch('torch.distributed.broadcast'), \
            patch('gc_moe.eplb.deepinfer_periodic.policy.rebalance_experts_global') as rebalance:
        origins = plan.tolist()
        for target in targets:
            stats.copy_(torch.tensor([[7, 3, 1], [4, 8, 6]]))
            demand = layer_demand_from_stats(stats, origins, 0, 3, 4)
            expected = [list(reorder_single_copy_first(row, 2)[0]) for row in target]
            rebalance.return_value = (torch.tensor(target, dtype=torch.int32), None, None)
            assert scope['rearrange_now']()
            assert torch.equal(rebalance.call_args.args[0][0], demand)
            shuffler.move.assert_called_with(origins, expected)
            assert state['plan'].tolist() == expected and not stats.any()
            origins = expected


def main():
    args = (2, 4, 128, 2, 512, 256, 0, 6, True)
    with patch.dict(os.environ, dict(GC_MOE_LAYERS='2', GC_MOE_ECO_PER_LAYER='1',
                                     GC_MOE_PULL_TRACE='1', GC_MOE_IR_STEP='1',
                                     GC_IRSTEP_STEPS='3'), clear=True):
        layout = SM90GCBufferLayout(*args)
        assert layout.ir_steps == 3 and layout.views['ir_step_epoch'].shape == (6,)
        wide = SM90GCBufferLayout(16, 32, 128, 2, 512, 256, eco=True)
        assert wide.views['pull_tl'].shape == (256 * 40,)
        with patch.dict(os.environ, GC_MOE_STEP_PROFILE='1'):
            assert layout != SM90GCBufferLayout(*args)
        with patch.dict(os.environ, GC_MOE_FIXED_DIR='/tmp/replay'):
            variants = []
            for mode in ('0', '1'):
                with patch.dict(os.environ, GC_MOE_SM90_STREAM_M_BLOCKS=mode):
                    variants.append(SM90GCBufferLayout(*args))
            assert variants[0] == variants[1] and variants[0].stream_blocks
        a = [0, 1, 0, 2, 3, 0]
        b = [2, 3, 0, 0, 1, 2]
        buf = SimpleNamespace(layout=layout, physical=6,
                              eco_state=torch.ones(2 * 2 * 6, dtype=torch.int32),
                              eco_meta=torch.empty(layout.views['eco_meta'].shape, dtype=torch.int32))
        assert (prepare_meta(buf, a, 2, 4) == prepare_meta(buf, [a], 2, 4)).all()
        install_into_buffer(buf, [a, b], 2, 4)
        assert not buf.eco_state.any()
        assert not torch.equal(*buf.eco_meta.reshape(2, -1))
        stats = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert layer_demand_from_stats(stats, [a, b], 1, 3, 4).tolist() == [[3, 0, 1, 2], [4, 5, 6, 0]]
        with patch.dict(os.environ, GC_MOE_ECO_PER_LAYER='0'):
            buf.layout = SM90GCBufferLayout(*args)
            try:
                prepare_meta(buf, [a, b], 2, 4)
            except ValueError as exc:
                assert 'distinct' in str(exc)
            else:
                raise AssertionError('distinct layer rows in shared metadata')
        buf.layout = layout
        test_repeated_rearrangement(buf, a, b)
        context = SimpleNamespace(dispatch_sms=6, compute_sms=72)
        for bound in (1, 2):
            for eco in (0, 4):
                config = ValidatedCall(ranks=2, experts=4, capacity=128, topk=2,
                                       hidden=512, intermediate=256, tokens=bound, bound=bound,
                                       clamp=10., fast_math=True, physical=6, eco_logical_experts=eco)
                schedule = get_sm90_role_config(config, context)
                assert schedule.direct_route_scan == (eco == 0)
                assert schedule.grids[0] == (24 if eco else 3)
    print('ECO_PORT_CPU PASS')


if __name__ == '__main__':
    main()
