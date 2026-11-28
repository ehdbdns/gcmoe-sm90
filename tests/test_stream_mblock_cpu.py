"""Check that both SM90 scheduling variants prepare the expected workspace and roles."""
from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))

from gc_moe.apis.sm90_gc_moe import ValidatedCall  # noqa: E402
from gc_moe.layout.sm90_gc_moe import SM90GCBufferLayout  # noqa: E402
from gc_moe.jit_kernels.heuristics.sm90_gc_moe import get_sm90_role_config  # noqa: E402
from gc_moe.jit_kernels.impls.sm90_fp8_fp4_gc_moe import _role_specs  # noqa: E402


def main():
    conf = ValidatedCall(ranks=8, experts=384, capacity=2048, topk=6, hidden=5120,
                         intermediate=2304, tokens=256, bound=2048, clamp=10.,
                         fast_math=True, shared_experts=1, physical=448,
                         eco_logical_experts=384, split_shared_m=32, split_shared=True)
    context = SimpleNamespace(dispatch_sms=6, compute_sms=72)
    layout_args = (8, 384, 2048, 6, 5120, 2304, 1, 448, True, True)
    forced = dict(GC_MOE_MB_CLAIM='1', GC_MOE_BLOCK_M_FORCE='64')
    with patch.dict(os.environ, dict(forced, GC_MOE_SM90_STREAM_M_BLOCKS='0')):
        base = SM90GCBufferLayout(*layout_args)
        assert 'stream_l2_ready' not in base.views
    with patch.dict(os.environ, dict(forced, GC_MOE_SM90_STREAM_M_BLOCKS='1')):
        stream = SM90GCBufferLayout(*layout_args)
        assert base != stream
        capacity = stream.views['active_ready'].shape[0]
        assert capacity >= conf.ranks * conf.capacity * conf.topk // 16 + 56
        fields = ['stream_l1_next', 'stream_l2_next', 'stream_l1_done', 'stream_l2_ready']
        assert all(stream.views[name].shape == (capacity,) for name in fields)
        assert len({stream.views[name].offset for name in fields}) == len(fields)
        assert all(name not in base.views for name in fields)
        schedule = get_sm90_role_config(conf, context)
        for hidden in range(512, 8193, 512):
            geom = get_sm90_role_config(replace(conf, hidden=hidden), context)
            warp_bytes = hidden + hidden // 128 * 4
            first_row = geom.smem[0] - 384 - (geom.threads[0] // 32) * warp_bytes
            assert first_row >= (2 * conf.ranks + 2) * 4, (hidden, first_row)
            assert geom.grids[0] * (geom.smem[0] + 1024) <= context.dispatch_sms * 228 * 1024
        assert (schedule.grids[0], schedule.smem[0]) == (20 if conf.plan_dispatch else 24, 43008)
        specs = _role_specs(conf, stream, schedule, True)
        assert [spec[0] for _, spec in specs] == ([5, 7, 1, 4, 8, 2, 3] if conf.plan_dispatch else [1, 4, 8, 2, 5, 3])
        by_role = {spec[0]: (kernel, spec) for kernel, spec in specs}
        l1, l2 = by_role[4][0], by_role[8][0]
        assert (l1.max_m, l1.shared_phase_end, l2.max_m, l2.shared_phase_start) == (32, 1, 32, 1)
        assert by_role[8][1][3:] == (18, (), ((4, 2, 0, 1),))
        assert by_role[2][1][4] == (4,)
        assert by_role[4][1][4:] == ((),)
        if conf.plan_dispatch:
            assert by_role[5][1][4:] == by_role[7][1][4:] == ((),)
            assert by_role[1][1][4:] == ((7,),)
        else:
            assert by_role[5][1][4:] == by_role[1][1][4:] == ((),)
        assert by_role[4][1][3] == 144
        for eco in (0, conf.eco_logical_experts):
            original = replace(conf, eco_logical_experts=eco)
            candidate = replace(original, split_shared_m=64)
            old = {s[0]: s for _, s in _role_specs(
                original, stream, get_sm90_role_config(original, context), True)}
            new = _role_specs(candidate, stream, get_sm90_role_config(candidate, context), True)
            assert list(old) == [s[0] for _, s in new]
            for kernel, launch in new:
                role = launch[0]
                if role in (4, 8):
                    assert kernel.max_m == 64 and kernel.smem == old[role][1]
                if role == 4:
                    assert launch == (*old[role][:3], 72, *old[role][4:])
                elif role == 8:
                    assert launch == (*old[role][:3], 9, *old[role][4:])
                else:
                    assert launch == old[role]
            for tokens in (0, 32, 128, 255, 257):
                assert get_sm90_role_config(replace(candidate, tokens=tokens), context).shared_tile_m == 32
            assert not get_sm90_role_config(replace(candidate, shared_experts=0), context).split_shared
            unsplit = replace(original, split_shared=False)
            unsplit_specs = _role_specs(unsplit, stream, get_sm90_role_config(unsplit, context), True)
            assert [s[0] for _, s in unsplit_specs] == [role for role in old if role != 8]
            for kernel, launch in unsplit_specs:
                if launch[0] == 4:
                    assert (kernel.max_m, kernel.shared_phase_start, kernel.shared_phase_end) == (64, 0, 2)
                    assert launch == (4, 199680, 256, 72, ())
                else:
                    assert launch == old[launch[0]]
        with patch.dict(os.environ, GC_MOE_SM90_SPLIT_SHARED='0'):
            assert not ValidatedCall(8, 384, 512, 6, 5120, 2304, 256, 512, 10., True).split_shared
        with patch.dict(os.environ, GC_MOE_SM90_SPLIT_SHARED_M='64'):
            assert ValidatedCall(8, 384, 512, 6, 5120, 2304, 256, 512, 10., True).split_shared_m == 64
        with patch.dict(os.environ, {'GC_MOE_BLOCK_M_FORCE': '0'}):
            for bound, block_m in ((96, 16), (128, 32)):
                other = replace(conf, bound=bound, eco_cost_mode=0, plan_dispatch=True)
                geometry = get_sm90_role_config(other, context)
                assert geometry.tile_m == block_m
                roles = {spec[0]: (k, spec) for k, spec in _role_specs(other, stream, geometry, True)}
                if block_m == 32:
                    assert roles[4][0].shared_phase_end == 1
                    assert roles[8][1][3:] == (geometry.shared_grid, (4,))
                else:
                    assert 8 not in roles and roles[4][0].shared_phase_end == 2
        routed = next(kernel for kernel, spec in specs if spec[0] == 2)
        assert (routed.l1_tiles, routed.l2_tiles) == (18, 20)
        assert routed.stream_blocks and not routed.block_ready
        helper_conf = replace(conf, sm_reuse=1, dispatch_partition_sms=6)
        helper_specs = dict((s[0], (k,s)) for k,s in _role_specs(helper_conf, stream, schedule, True))
        helper, launch = helper_specs[6]
        assert (launch[2],launch[3]) == (384,6) and helper.stream_blocks
        assert (helper.l1_tiles, helper.l2_tiles) == (routed.l1_tiles, routed.l2_tiles)
        for cost_mode, slack in ((0,90), (2,11469)):
            priced = replace(conf, eco_cost_mode=cost_mode, eco_wb=26, eco_wa=0)
            plan = next(k for k,s in _role_specs(priced,stream,schedule,True) if s[0]==5)
            assert plan.eco_slack == slack
        for invalid in (replace(conf, bound=2), replace(conf, sm_reuse=2)):
            try:
                _role_specs(invalid, stream, get_sm90_role_config(invalid, context), True)
            except ValueError as exc:
                assert 'M-block streaming requires' in str(exc), exc
            else:
                raise AssertionError(f'invalid streaming options accepted: {invalid}')
        with patch.dict(os.environ, {'GC_MOE_MB_CLAIM': '0'}):
            try:
                _role_specs(conf, stream, schedule, True)
            except ValueError as exc:
                assert 'M-block streaming requires MB-CLAIM' in str(exc), exc
            else:
                raise AssertionError('streaming without MB-CLAIM accepted')
    print('STREAM_MBLOCK_CPU PASS')


if __name__ == '__main__':
    main()
