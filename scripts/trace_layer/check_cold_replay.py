"""Check the frozen R2 inputs and predict its first-step missing rows on CPU."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from fixed_workload import digest
from gc_moe.eco.meta_sm90 import reorder_single_copy_first


def old_missing_rows(rows, copies, block=64):
    """R2 split by the active copy count but emitted at most four rounded quotas."""
    if copies <= 4:
        return 0
    each, extra = divmod(rows, copies)
    full = (each + extra) // block + 3 * (each // block)
    units = (rows + block - 1) // block
    return max(0, units - full - 4) * block


def check(directory, runner):
    with runner.open() as stream:
        result = next(json.loads(line[7:]) for line in stream if line.startswith('RESULT '))
    workload = result['fixed_workload']
    recorded = sorted(workload['ranks'], key=lambda r: r['rank'])
    if [r['rank'] for r in recorded] != list(range(8)):
        raise ValueError('R2 receipt must cover eight ranks')
    placement = json.loads((directory / 'placement.json').read_text())
    rows = [reorder_single_copy_first(row, 8)[0] for row in placement]
    effective = torch.from_numpy(np.asarray(rows, dtype=np.int32))
    if effective.shape != (40, 448) or {r['placement'] for r in workload['initial']} != {digest(effective)}:
        raise ValueError('reordered placement differs from the R2 installed layout')
    demand = torch.zeros((40, 384), dtype=torch.int64)
    for rank in recorded:
        path = directory / f"routes-rank{rank['rank']}.pt"
        bank = torch.load(path, weights_only=True, map_location='cpu')
        for key, audit in [('ids', 'route_ids'), ('weights', 'route_weights'), ('layers', 'layer_order')]:
            if digest(bank[key]) != rank[audit]:
                raise ValueError(f'{path.name} {key} differs from R2 receipt')
        for position, ids in enumerate(bank['ids'][:40]):
            demand[position] += torch.bincount(ids.flatten(), minlength=384)
    starts = {r['first_epoch'] for r in recorded}
    if len(starts) != 1:
        raise ValueError('rank replay epochs disagree')
    start = starts.pop()
    predicted, observed = [], []
    for layer, counts in enumerate(demand.tolist()):
        # owned_totals is in call order; the GPU selects its placement by epoch.
        row = (start - 1 + layer) % 40
        copies = Counter(expert for slots in effective[row].reshape(8, 56).tolist()
                         for expert in set(slots))
        predicted.append(sum(old_missing_rows(n, copies[e]) for e, n in enumerate(counts)))
        if sum(counts) != 12288:
            raise ValueError(f'incomplete reference demand at layer {layer}')
        observed.append(sum(counts) - sum(r['owned_totals'][0][layer] for r in recorded))
    print('layer,old_predicted_missing_rows,r2_observed_missing_rows')
    for layer, (pred, obs) in enumerate(zip(predicted, observed)):
        print(f'{layer},{pred},{obs}')
    print(f'R2 step 0: matching layers={sum(p == o for p, o in zip(predicted, observed))}/40 '
          f'predicted_missing={sum(predicted)} observed_missing={sum(observed)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('runner', type=Path)
    args = parser.parse_args()
    check(args.directory, args.runner)
