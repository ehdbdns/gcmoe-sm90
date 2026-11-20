"""Exercise post-EPLB KV restoration, cache aliases, and live request mapping."""
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
import stream_checkpoint as sc


def rejected(fn, message):
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), exc
    else:
        raise AssertionError(f'expected rejection for {message}')


def test_checkpoint(root, device):
    backing = torch.arange(20, dtype=torch.float32, device=device)
    other = torch.arange(6, dtype=torch.int32, device=device)
    views = {'k': backing[::2], 'v': backing[1::2].reshape(2, 5).T, 'compressor': other}
    controls = {'requests': 'reference'}
    with patch.object(sc, 'CHUNK', 7):
        saved = sc.checkpoint(root, 0, True, views, controls)
        source = backing.clone()
        backing.fill_(777)
        other.zero_()
        pointers = [t.data_ptr() for t in views.values()]
        loaded = sc.checkpoint(root, 0, False, dict(reversed(list(views.items()))), controls)
        assert loaded == saved and torch.equal(source, backing) and torch.equal(other, torch.arange(6, device=device))
        assert [t.data_ptr() for t in views.values()] == pointers and loaded['kv_bytes'] == 104
        other.zero_()
        before = sc.checkpoint(root, 0, False, views, controls, audit_prior=True)
        assert before['before_restore_sha256'] != saved['source_sha256'] == before['readback_sha256']
        before = sc.checkpoint(root, 0, False, views, controls, audit_prior=True)
        assert before['before_restore_sha256'] == saved['source_sha256']
        assert sc.tensor_hash(views['v']) == hashlib.sha256(views['v'].cpu().numpy().tobytes()).hexdigest()
        assert sc.tensor_hash(torch.tensor(0.5, dtype=torch.bfloat16)) == hashlib.sha256(
            torch.tensor([0.5], dtype=torch.bfloat16).view(torch.uint8).numpy()).hexdigest()
        layout = json.loads((root/'rank0.json').read_text())['layout']
        assert layout['views']['k'][0] == layout['views']['v'][0]
        rejected(lambda: sc.checkpoint(root, 0, False, views, {'requests': 'different'}), 'controls')
        rejected(lambda: sc.checkpoint(root, 0, False, dict(views, v=views['v'].clone()), controls), 'layout')
        payload = root/'rank0.bin'
        data = bytearray(payload.read_bytes())
        data[-1] ^= 1
        payload.write_bytes(data)
        rejected(lambda: sc.checkpoint(root, 0, False, views, controls), 'readback')
        payload.write_bytes(data[:-1])
        rejected(lambda: sc.checkpoint(root, 0, False, views, controls), 'byte count')


def test_named_caches():
    layer = torch.ones(2)
    eng = SimpleNamespace(use_slot_cache=True, _kv_cache_ref=None, _cache=torch.zeros(3))
    model_runner = SimpleNamespace(init_kv_cache=lambda: {'layer': layer, 'alias': layer})
    package = SimpleNamespace(model_runner=model_runner)
    runner = SimpleNamespace(kv_caches=[layer, layer], device=layer.device,
                             model=SimpleNamespace(named_modules=lambda: [('history', eng)]))
    with patch.dict(sys.modules, {'vllm.v1.worker.gpu': package}):
        sc.install_cache_names()
        sc.install_cache_names()
        model_runner.init_kv_cache()
        assert sc.named_caches(runner) == {'layer': layer, 'alias': layer, 'engram:history': eng._cache}
        runner.kv_caches.append(torch.ones(4))
        rejected(lambda: sc.named_caches(runner), 'every live KV')


def test_requests():
    def staged(values):
        t = torch.tensor(values)
        return SimpleNamespace(gpu=t, np=t.numpy())
    req = SimpleNamespace(req_id_to_index={'req0': 1, 'req1': 0}, num_speculative_steps=0,
                          free_indices=[], all_token_ids=staged([[31, 32, 0], [41, 42, 0]]),
                          last_sampled_tokens=torch.tensor([[32], [42]]))
    for field in ('prompt_len', 'prefill_len', 'num_computed_tokens', 'total_len'):
        setattr(req, field, staged([2, 2]))
    for field in ('max_seq_len', 'num_computed_tokens_np', 'num_computed_prefill_tokens'):
        setattr(req, field, np.array([2, 2]))
    blocks = staged([[7, 5, 0], [9, 8, 0]])
    tables = SimpleNamespace(block_tables=[blocks], num_blocks=staged([[2, 2]]),
                             block_sizes=[64], kernel_block_sizes=[32])
    runner = SimpleNamespace(req_states=req, block_tables=tables,
                             kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['A'])]))
    ref = SimpleNamespace(keys=req.req_id_to_index.copy(), positions=torch.arange(2), batch_hash=hashlib.sha256())
    initial = sc.runner_boundary(runner, ref)
    blocks.gpu[:, 2] = 555
    req.all_token_ids.gpu[:, 2] = 888
    assert initial == sc.runner_boundary(runner, ref)
    blocks.gpu[0, 0] += 1
    assert sc.runner_boundary(runner, ref)['physical_blocks'] != initial['physical_blocks']
    req.req_id_to_index = {'req0': 0, 'req1': 1}
    assert sc.runner_boundary(runner, ref)['requests'] != initial['requests']


if __name__ == '__main__':
    device = 'cpu'
    if sys.argv[1:] == ['--gpu-preflight']:
        import vllm
        from vllm.v1.worker.gpu import model_runner
        sc.install_cache_names()
        assert hasattr(model_runner.init_kv_cache, 'gc_cache_names')
        assert torch.cuda.device_count() == 8
        assert all(torch.cuda.get_device_capability(i)[0] == 9 for i in range(8))
        print('vLLM', vllm.__version__, 'torch', torch.__version__, 'CUDA', torch.version.cuda)
        device = 'cuda'
    elif sys.argv[1:]:
        raise SystemExit('use --gpu-preflight or run without arguments for CPU')
    with TemporaryDirectory() as tmp:
        test_checkpoint(Path(tmp), device)
    test_requests()
    test_named_caches()
    print(f'STREAM_CHECKPOINT_{device.upper()} PASS')
