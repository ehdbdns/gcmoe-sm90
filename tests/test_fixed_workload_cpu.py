"""Check the same route indices, quota rows and feedback consumed by the two workers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from fixed_tokens import install
from fixed_workload import RouteTape, recording_reference


def test_fixed_driver_startup(folder):
    """Exercise main() up to its first request without loading model weights."""
    prompts = [[i] for i in range(256)]
    prompt_hash = hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest()
    Path(folder, 'tokens.json').write_text(json.dumps({
        'prompt_hash': prompt_hash, 'tokens': {f'req{i}': [i] for i in range(256)}}))
    required = ('GC_MOE_EPLB', 'GC_MOE_ECO', 'GC_MOE_ECOPLB_PER_LAYER',
                'GC_MOE_PULL_TRACE', 'GC_MOE_ECO_TRACE', 'GC_MOE_SHARED_IN_GC')
    env = dict.fromkeys(required, '1')
    env.update(GC_MOE_FIXED_DIR=folder, GC_MEASURE_FLUSH='2', GC_MEASURE_LOAD='0',
               MODEL='cpu-fixture', VLLM_ENABLE_V1_MULTIPROCESSING='0')
    path = Path(__file__).resolve().parents[1] / 'scripts/real_decode_100_100.py'
    args = [str(path), '--output', folder + '/result.json', '--label', 'startup',
            '--warm-steps', '200', '--measure-steps', '1', '--timing-tail-steps', '100',
            '--profile-aligned-step', '--fine-layers', '0', '--profile-out', folder]

    class FirstRequest(Exception):
        pass

    engine = Mock(engine_core=SimpleNamespace(get_output=Mock()))
    engine.collective_rpc.return_value = [True] * 8
    engine.add_request.side_effect = FirstRequest
    llm = Mock(return_value=SimpleNamespace(llm_engine=engine))
    modules = {
        'vllm_warmup_compat': Mock(), 'deepep_compare': Mock(),
        'vllm': SimpleNamespace(LLM=llm, SamplingParams=Mock()),
        'vllm.sampling_params': SimpleNamespace(RequestOutputKind=SimpleNamespace(DELTA=1)),
        'vllm.v1.engine.core_client': SimpleNamespace(InprocClient=SimpleNamespace),
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True), \
            patch.object(sys, 'argv', args), patch.object(sys, 'path', sys.path.copy()):
        spec = importlib.util.spec_from_file_location('fixed_driver', path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        with patch.object(driver, 'load_prompts', return_value=prompts), \
                patch.object(driver, '_compilation_config', return_value={}):
            try:
                driver.main()
            except FirstRequest:
                pass
        llm.assert_called_once()
        assert not llm.call_args.kwargs.get('logits_processors'), 'token replay belongs in worker.sample'
        engine.collective_rpc.assert_called_once_with('_gc_moe_fixed_tokens')
        engine.add_request.assert_called_once()


def test_routes(folder):
    audits = []
    # Seed the shared tape, then run a baseline and a stream using that tape.
    for arm, (flag, reuse) in enumerate([('0', '0'), ('0', '1'), ('1', '1')]):
        tape = RouteTape('cpu', topk=2, slots=2, steps=2, layers=2, batch=3)
        buf = SimpleNamespace(eco_tl_step=torch.zeros(1, dtype=torch.int32),
                              profile_slot_counts=torch.zeros(256, 2, dtype=torch.int32))

        def call(n, live):
            ids = torch.full((3, 2), live, dtype=torch.int32)
            weights = torch.full((3, 2), live / 10, dtype=torch.float32)
            ids, weights = tape.select(ids, weights, n % 2)
            # Simulate the GPU's quota observation (including wrap at epoch 256).
            buf.eco_tl_step.fill_(250+n)
            buf.profile_slot_counts[(250+n) % 256] = ids[0].int()
            tape.finish_call(buf)
            return ids, weights

        for n in range(10):
            call(n, 20 * arm + n)
        with patch.dict(os.environ, GC_MOE_SM90_STREAM_M_BLOCKS=flag, GC_MOE_FIXED_REUSE=reuse):
            tape.freeze(folder, 0, reference=recording_reference())
        for n in range(4):
            ids, weights = call(10 + n, 90+arm)
            assert (ids == 6+n).all()
            assert torch.allclose(weights, torch.full((3, 2), (6+n)/10))
        audits.append(tape.audit(0))
    assert audits[0] == audits[1] == audits[2]


def test_tokens(folder):
    class CPURunner:
        device = 'cpu'

        def sample(self, native_ids, input_batch, grammar_output):
            output = SimpleNamespace(sampled_token_ids=native_ids[:, None])
            return output, input_batch.counts, torch.zeros_like(input_batch.counts)

    Path(folder, 'tokens.json').write_text(json.dumps({'tokens': {'req0':[5, 2], 'req1':[6, 3, 8]}}))
    runner = CPURunner()
    with patch.dict(sys.modules, {'vllm.v1.worker.gpu.model_runner': SimpleNamespace(GPUModelRunner=CPURunner)}):
        reference = install(runner, folder)

    def feed(keys, native, counts):
        batch = SimpleNamespace(req_ids=keys, num_reqs=len(keys), counts=torch.tensor(counts),
                                query_start_loc_np=torch.arange(len(keys)+1).numpy())
        output, accepted, rejected = runner.sample(torch.tensor(native, dtype=torch.int32), batch, None)
        assert accepted.tolist() == counts and not rejected.any()
        return output.sampled_token_ids[:, 0].tolist()
    assert feed(['req1','req0'], [11,12], [1,0]) == [6,12]
    assert feed(['req0','req1'], [13,14], [1,1]) == [5,3]
    assert feed(['req1','req0'], [15,16], [1,1]) == [8,2]
    assert reference.native_sample(remaining=2) == [13,14]


if __name__ == '__main__':
    with TemporaryDirectory() as folder:
        test_fixed_driver_startup(folder)
        test_routes(folder)
        test_tokens(folder)
    print('FIXED_WORKLOAD_CPU PASS')
