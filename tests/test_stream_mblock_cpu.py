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
                         eco_logical_experts=384)
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
        assert stream.views['stream_l2_ready'].shape == (capacity,)
        schedule = get_sm90_role_config(conf, context)
        specs = _role_specs(conf, stream, schedule, True)
        assert [spec[0] for _, spec in specs] == [1, 4, 2, 5, 3]
        routed = next(kernel for kernel, spec in specs if spec[0] == 2)
        assert (routed.l1_tiles, routed.l2_tiles) == (18, 20)
        assert routed.stream_blocks and not routed.block_ready
        for invalid in (replace(conf, bound=2), replace(conf, sm_reuse=1)):
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
