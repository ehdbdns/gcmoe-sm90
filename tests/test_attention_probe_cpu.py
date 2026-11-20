"""Exercise the sparse decode wrapper and its CUDA Graph gates before model loading."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

gpu = sys.argv[1:] == ['--cuda']
if not gpu:
    os.environ['TRITON_INTERPRET'] = '1'
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from attention_probe import read_flash_probe, reset_flash_probe, wrap_decode


def exercise(canonical):
    device = 'cuda' if gpu else 'cpu'
    n, layers = 3, 4
    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)
    ids = ints([[2, 1, 1, -1], [1, -1, -1, -1], [-1]*4])
    lens = ints([3, 1, 0])
    tape = SimpleNamespace(batch=n, layers=layers, steps=3, replaying=torch.tensor(False, device=device),
                           position=ints([2]).long())
    buf = SimpleNamespace(fixed_routes=tape, replay_stream_mode=ints([0]))
    q = torch.arange(n * 4 * 8, device=device).view(n, 4, 8).to(torch.bfloat16)
    out = torch.empty_like(q)
    seen = torch.empty_like(ids)
    def flash(**kw):
        extra = kw['extra_indices_in_kvcache']
        if extra is not None:
            seen.copy_(extra.view_as(ids))
        kw['out'].copy_(kw['q'])
        return kw['out'], None
    if gpu:
        from vllm.models.deepseek_v41.nvidia import flashmla as module
        native_flash = module.flash_mla_with_kvcache
        module.flash_mla_with_kvcache = flash
    else:
        module = SimpleNamespace(flash_mla_with_kvcache=flash)
        class Backend:
            def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
                return module.flash_mla_with_kvcache(q=q.unsqueeze(1), indices=swa_metadata.decode_swa_indices,
                        topk_length=swa_metadata.decode_swa_lens, extra_indices_in_kvcache=None if swa_only else self.topk_indices_buffer[:, None],
                        extra_topk_length=lens, out=output.unsqueeze(1))
        module.DeepseekV4FlashMLAAttention = Backend
    original_decode = module.DeepseekV4FlashMLAAttention._forward_decode
    wrap_decode(module, lambda: buf, canonical)
    cache = ints([1, 2, 3]).view(3, 1, 1)
    obj = SimpleNamespace(layer_id=2, n_local_heads=2, compress_ratio=2, topk_indices_buffer=ids,
                          swa_cache_layer=SimpleNamespace(kv_cache=cache), scale=1., attn_sink=None)
    swa = SimpleNamespace(num_decodes=n, num_decode_tokens=n, decode_swa_indices=ids.clone()[:, None],
                          decode_swa_lens=lens, token_to_req_indices=ints([0, 1, 2]),
                          is_valid_token=ints([1]*n).bool(), tile_sched_c2a=object(),
                          tile_sched_c1a=None, tile_sched_swaonly=None)
    meta = SimpleNamespace(block_size=2, block_table=ints([list(range(3))]*n))
    def run():
        module.DeepseekV4FlashMLAAttention._forward_decode(
            obj, q=q, kv_cache=cache, swa_metadata=swa, attn_metadata=meta, swa_only=False, output=out)
    run()
    reset_flash_probe(tape)
    if canonical:
        from attention_order import order_coverage, read_order, reset_order
        modules = [SimpleNamespace(layer_id=i, compress_ratio=2 if i == 2 else 0) for i in range(layers)]
        assert reset_order(tape, modules, SimpleNamespace) == [2]
    run()
    assert {tuple(mark.tolist()) for _, mark in tape.attention_fields.values()} == {(-1, -1)}
    if gpu:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            run()
        replay = graph.replay
    else:
        replay = run
    tape.replaying.fill_(True)
    replay()
    first = read_flash_probe(tape, 0)
    if canonical:
        first_order = read_order(tape, 0, first)['2']
        assert first_order['permuted_rows'] == 1
    ids[0, :3] = ints([1, 2, 1])
    # A valid but different physical id outside the SWA prefix is irrelevant.
    # The local topk buffer below stays padded because the upstream map counts valid entries.
    swa.decode_swa_indices[0, 0] = ints([1, 2, 1, 123])
    tape.position.fill_(layers + 2)
    buf.replay_stream_mode.fill_(1)
    replay()
    second = read_flash_probe(tape, 1)
    for side in ('swa', 'extra'):
        same_order = first['hashes'][side+'_order'][0] == second['hashes'][side+'_order'][0]
        assert same_order == (canonical and side == 'extra')
        assert first['hashes'][side+'_multiset'] == second['hashes'][side+'_multiset']
        assert first['lengths'][side] == second['lengths'][side] == [3, 1, 0]
    assert first['hashes']['query'] == second['hashes']['mla_output']
    assert first['mode'] == 0 and second['mode'] == 1
    if canonical:
        second_order = read_order(tape, 1, second)['2']
        assert first_order['native'] != second_order['native'] and first_order['consumed'] == second_order['consumed']
    tape.position.fill_(2*layers+2)
    q.zero_()
    replay()
    assert read_flash_probe(tape, 1) == second
    assert torch.equal(out, q)
    if canonical:
        assert order_coverage(tape) == {'2': [0, 1, 1]}
    tape.replaying.fill_(False)
    obj.layer_id = 3
    ids[0] = ints([2, 1, 1, -1])
    run()
    assert seen[0].tolist() == ([1, 1, 2, -1] if canonical else [2, 1, 1, -1])
    if canonical:
        assert set(tape.order_banks) == {2, 3}
    obj.layer_id, obj.compress_ratio = 0, 0
    swa.tile_sched_swaonly = object()
    module.DeepseekV4FlashMLAAttention._forward_decode(
        obj, q=q, kv_cache=None, swa_metadata=swa, attn_metadata=None, swa_only=True, output=out)
    if canonical:
        assert set(tape.order_banks) == {2, 3}
    module.DeepseekV4FlashMLAAttention._forward_decode = original_decode
    if gpu:
        module.flash_mla_with_kvcache = native_flash


if __name__ == '__main__':
    exercise(False)
    exercise(True)
    print('ATTENTION_PROBE_CUDA PASS' if gpu else 'ATTENTION_PROBE_CPU PASS')
