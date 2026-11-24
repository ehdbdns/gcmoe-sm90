"""Match each source-step replay to the archived rank-local physical allocation."""
import argparse
import json
from pathlib import Path

import numpy as np

from se_replay_report import ARMS, LAYERS, write_rows


def audit(group, out):
    rows = []
    for arm in ARMS:
        eco = arm.startswith('eco')
        for rank in range(8):
            with np.load(group / arm / f'rank{rank}.npz') as archive:
                counts = archive['counts']
                auctions = archive['auction'] if eco else None
                epochs = archive['epochs']
                clocks = archive['clocks']
                for layer in LAYERS:
                    result = json.loads((out / f'layer{layer}' / arm / f'rank{rank}-device.json').read_text())
                    if not result['historical_routes'] or result['source_tokens'] != result['bs'] or len(result['calls']) != 3:
                        raise ValueError('expected three full historical calls')
                    for step, row in enumerate(result['calls']):
                        index = step * 40 + layer
                        ref = counts[index]
                        actual = np.asarray(row['slot_counts'])
                        if actual.shape != ref.shape:
                            raise ValueError('wrong number of physical slots')
                        different = np.flatnonzero(actual != ref)
                        a = auctions[index].tolist() if eco else None
                        g, _, _, end = map(int, clocks[index, :4])
                        rows.append(dict(bs=result['bs'], arm=arm, layer=layer, rank=rank,
                                         step_zero_based=step, history_epoch=int(epochs[index]), replay_epoch=row['epoch'],
                                         history_tokens=int(ref.sum()), replay_tokens=row['rank_tokens'],
                                         history_m_blocks=int(((ref+63)//64).sum()), replay_m_blocks=row['m_blocks'],
                                         history_graph_us=(end-g)/1000,
                                         replay_graph_us=(row['clocks'][3]-row['clocks'][0])/1000,
                                         different_slots=len(different), first_different_slot=next(iter(different), ''),
                                         history_auction=a, replay_auction=row['auction'], auction_equal=a==row['auction']))
    write_rows(out / 'history_comparison.csv', rows)
    mismatches = sum(bool(r['different_slots'] or not r['auction_equal']) for r in rows)
    print(f'Historical allocation/auction audit: {len(rows)-mismatches}/{len(rows)} matched')
    return mismatches


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('group', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    # Leave diagnostic graphs available even when the historical allocation differs.
    audit(args.group, args.out)
