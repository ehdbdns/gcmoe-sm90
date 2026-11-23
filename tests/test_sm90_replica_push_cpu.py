"""Check physical quota ownership against ragged, out-of-order source reservations."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/trace_layer'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from single_push_payload import verify_rows


def main():
    # Rank zero holds two physical slots of expert zero. The split falls
    # inside source one's contribution. Source one reserves before source zero.
    sources = [(torch.tensor([[1], [2]], dtype=torch.uint8) + i, torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1)) for i in range(2)]
    meta = torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 1, 0]], [[1, 1, 0], [0, 0, 0], [0, 0, 0]]])
    buf = SimpleNamespace(expert_count=torch.tensor([3, 1]), eco_quota_start=torch.tensor([0, 3]),
                          eco_quota=torch.tensor([3, 1]), route_meta=meta,
                          l1_acts=torch.tensor([[[2], [1], [2]], [[3], [0], [0]]], dtype=torch.uint8),
                          l1_acts_sf=torch.full((2, 1, 3), 64.), route_weight=torch.ones(2, 3))
    with patch('torch.distributed.all_reduce'):
        verify_rows(buf, (0, 0), sources, 0)
        meta[0, 0, 1], meta[1, 0, 1] = 1, 0
        try:
            verify_rows(buf, (0, 0), sources, 0)
        except AssertionError as exc:
            assert 'quota ownership' in str(exc)
        else:
            raise AssertionError('wrong physical destination accepted')
    print('SM90 replica quota payload oracle PASS')


if __name__ == '__main__':
    main()
