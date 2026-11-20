"""Check the same route indices, quota rows and feedback consumed by the two workers."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from fixed_tokens import ReferenceTokens
from fixed_workload import RouteTape


def test_routes(folder):
    audits = []
    for arm in range(2):
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
        tape.freeze(folder, 0, reference=arm == 0)
        for n in range(4):
            ids, weights = call(10 + n, 90+arm)
            assert (ids == 6+n).all()
            assert torch.allclose(weights, torch.full((3, 2), (6+n)/10))
        audits.append(tape.audit(0))
    assert audits[0] == audits[1]


def test_tokens():
    reference = ReferenceTokens({'req0':[5, 2], 'req1':[6, 3, 8]}, 'cpu')
    def feed(keys, native, counts):
        output = SimpleNamespace(sampled_token_ids=torch.tensor(native, dtype=torch.int32)[:, None])
        reference.feed(keys, output, torch.tensor(counts, dtype=torch.int32))
        return output.sampled_token_ids[:, 0].tolist()
    assert feed(['req1','req0'], [11,12], [1,0]) == [6,12]
    assert feed(['req0','req1'], [13,14], [1,1]) == [5,3]
    assert feed(['req1','req0'], [15,16], [1,1]) == [8,2]
    assert reference.native_sample(remaining=2) == [13,14]


if __name__ == '__main__':
    with TemporaryDirectory() as folder:
        test_routes(folder)
    test_tokens()
    print('FIXED_WORKLOAD_CPU PASS')
