"""Exercise the same first-step taps and FP8 staging before loading an EP8 model."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Obj

gpu = sys.argv[1:] == ['--cuda']
if gpu:
    os.environ.pop('TRITON_INTERPRET', None)
else:
    os.environ['TRITON_INTERPRET'] = '1'

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from fp8_quant_ref import per_token_cast_to_fp8
from replay_probe import ReplayProbe, sample_first_step
from shared_input import stage_shared_input


def run(device, directory, publish):
    n, h, k, layers = 3, 160, 2, 2
    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)
    b = Obj(num_topk=k, x=empty((n,h), torch.float8_e4m3fn), x_sf=empty((n,2),torch.float32),
            shared_l1_acts=empty((n,h),torch.float8_e4m3fn), shared_l1_acts_sf=empty((n,5),torch.float32),
            shared_y=empty((n,h),torch.bfloat16), combine=empty((n,k,h),torch.bfloat16),
            used_route_ordinal=torch.arange(n*k, dtype=torch.int32,device=device).view(n,k),
            eco_tl_step=empty((1,),torch.int32))
    t = Obj(batch=n, layers=layers, layer_ids=empty((6,),torch.int32), ids=range(6),
            used_ordinals=empty((6,n,k),torch.int32), epochs=empty((6,),torch.int32),
            position=empty((1,),torch.int64), replaying=torch.tensor(False,device=device))
    b.fixed_routes = t
    probe = t.first_step_probe = ReplayProbe(b, t)
    probe.shared_pairs = {i: [(b.x_sf, b.x_sf), (b.x_sf, b.x_sf)] for i in range(layers)}
    out = empty((n,h),torch.bfloat16)
    probe.data.fill_(77)
    probe.raw_input.fill_(77)
    for step in range(2):
        t.replaying.fill_(bool(step))
        for layer in range(layers):
            # Exact E4M3 codes (including subnormals, sign and signed zero) in the interpreter.
            # On CUDA also check RN-even, exponent carry and random BF16 against the reference.
            vals = torch.tensor([0.,1.,-0.,-.125,2.,3.,.5,1./65536],device=device)
            x = vals.repeat(n*h//8).reshape(h,n).T.to(torch.bfloat16)
            if gpu:
                g = torch.Generator(device=device).manual_seed(34+layer)
                x = torch.randn((h,n), generator=g,device=device,dtype=torch.bfloat16).T
                x[0,:4] = torch.tensor([.78125,-.97265625,-.65625,1.0625],device=device)
            pos = layer
            t.position.fill_(pos)
            b.eco_tl_step.fill_(23+pos)
            t.epochs[pos] = 23+pos
            t.layer_ids[pos] = layer
            stage_shared_input(x,b)
            q,sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
            assert torch.equal(q.view(torch.uint8),b.shared_l1_acts.view(torch.uint8))
            assert torch.equal(sf,b.shared_l1_acts_sf)
            out.fill_(layer+1)
            b.combine.fill_(2)
            if step and layer == 1 and not publish:
                b.combine[1,1,4] = -2
            b.x.view(torch.uint8).fill_(5)
            b.x_sf.fill_(6)
            b.shared_y.fill_(7)
            probe.record(b,out)
            assert torch.equal(t.used_ordinals[pos],b.used_route_ordinal)
            if step:
                assert torch.equal(probe.raw_input[pos],x)
        if not step:
            assert (probe.data == 77).all() and (probe.raw_input == 77).all() and (probe.marks == -1).all()
    t.position.fill_(layers)
    saved = probe.data.clone()
    raw = probe.raw_input.clone()
    stage_shared_input(x,b)
    probe.record(b,out.zero_())
    assert torch.equal(saved,probe.data) and torch.equal(raw,probe.raw_input)
    batch = Obj(req_ids=['req1','req2','req0'], logits_indices=torch.tensor([2,0,1],device=device),
                num_reqs=3, idx_mapping_np=[1,2,0])
    class Sampler:
        def __call__(self, logits, local):
            return logits.argmax(1)
    runner = Obj(sampler=Sampler(),req_states=Obj(index_to_req_id={i:f'req{i}' for i in range(3)}))
    original_sampler = runner.sampler
    def original(hidden, local, _):
        return runner.sampler(hidden[local.logits_indices], local)
    _, sampler = sample_first_step(runner,original,x,batch,None)
    assert sampler['row_requests'] == ['req2','req0','req1'] and runner.sampler is original_sampler
    assert set(sampler['logits']) == set(batch.req_ids)
    result = probe.read(t,directory,0,publish,sampler)['layers']
    if not publish:
        diff = result[1]['different_rows']
        assert diff['routed_slots'] == dict(count=1,examples=[dict(row=1,request='req0',topk_slot=1)])
        assert sum(x['count'] for x in diff.values()) == 1
    if gpu:
        # Both modes run in the same captured graph on the executor. The
        # device counter, including the transition out of step zero, stays live.
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            stage_shared_input(x,b)
            probe.record(b,out)
        t.position.zero_()
        b.eco_tl_step.fill_(111)
        graph.replay()
        torch.cuda.synchronize()
        assert probe.marks.tolist() == [111,24]
        assert torch.equal(probe.raw_input[0], x)
        t.position.fill_(layers)
        b.eco_tl_step.fill_(222)
        graph.replay()
        torch.cuda.synchronize()
        assert probe.marks.tolist() == [111,24]

    # R7 reuses the slab for step one, then holds it across the entire timing tail.
    probe.steps = 2
    treatment = Path(directory)/'treatment'
    treatment.mkdir(exist_ok=True)
    for layer in range(layers):
        pos = layers + layer
        t.position.fill_(pos)
        t.layer_ids[pos] = layer
        t.epochs[pos] = 100 + layer
        b.eco_tl_step.fill_(100 + layer)
        stage_shared_input(x, b)
        out.fill_(11 + layer)
        probe.record(b, out)
    saved = probe.data.clone()
    t.position.fill_(2 * layers)
    probe.record(b, out.zero_())
    stage_shared_input(torch.zeros_like(x), b)
    assert torch.equal(saved, probe.data) and torch.equal(probe.raw_input[-1], x)
    t.position.fill_(len(t.ids))
    rows = probe.read(t, treatment, 0, publish, sampler, step=1)['layers']
    assert [r['epoch'] for r in rows] == [100, 101]


if __name__ == '__main__':
    with TemporaryDirectory() as d:
        for publish in (True,False):
            run('cuda' if gpu else 'cpu',d,publish)
    print('FIRST_STEP_CUDA PASS' if gpu else 'FIRST_STEP_CPU PASS')
