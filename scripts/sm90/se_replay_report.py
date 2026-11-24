"""Aggregate the second profiled graph replay of each sampled layer."""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

from se_perfetto import check

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'source/gc_moe/eco'))
from h20_valid_cost import slot_cost

ARMS = ('vanilla', 'random', 'rr', 'eco', 'eco_smr')
LAYERS = (8, 20, 32)


def write_rows(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def report(root, historical=False, fresh=False, arms=ARMS):
    ranks, layers, summary = [], [], []
    reference, eco_alloc = {}, {}
    for arm in arms:
        for layer in LAYERS:
            group = []
            auctions = []
            for rank in range(8):
                d = json.loads((root / f'layer{layer}' / arm / f'rank{rank}-device.json').read_text())
                check(d.get('historical_routes', False) == historical, 'wrong replay source')
                row = d['calls'][1]
                counts = row['slot_counts']
                key = layer, rank
                check(len(counts) == (48 if arm == 'vanilla' else 56) and min(counts) >= 0, 'bad slots')
                check(sum(counts) == row['rank_tokens'] and d['layer'] == layer, 'wrong sample')
                bs = d['bs']
                check(bs in (256, 512, 1024) and (not ranks or ranks[0]['bs'] == bs), 'mixed batch sizes')
                signature = d['input_sha256'], d['output_sha256']
                expected = reference.setdefault(key, signature)
                check(expected[0] == signature[0], 'replay input mismatch')
                graph, first, last, end, epoch, online_price = row['clocks']
                blocks = sum((q+63)//64 for q in counts)
                price = sum(slot_cost(q) for q in counts)
                check(blocks == row['m_blocks'] and row['epoch'] == epoch, 'bad epoch or M blocks')
                eco = arm.startswith('eco')
                check(online_price == (price if eco else 0), 'device/offline fine price mismatch')
                if eco:
                    a = row['auction']
                    check(len(a) == 7 and min(a) >= 0 and a[1] >= a[2], 'bad auction record')
                    allocation = [(call['slot_counts'], call['auction']) for call in d['calls']]
                    check(eco_alloc.setdefault(key, allocation) == allocation, 'SM reuse changed ECO allocation')
                    auctions.append(a)
                group.append(dict(bs=bs, arm=arm, layer=layer, sample=1, epoch=epoch, rank=rank,
                                  tokens=sum(counts), m_blocks=blocks, active_slots=sum(q > 0 for q in counts),
                                  graph_us=(end-graph)/1000, routed_us=(last-first)/1000, price=price,
                                  output_equal=expected[1] == signature[1]))
            check(len({r['epoch'] for r in group}) == 1, 'ranks sampled different replays')
            check(sum(r['tokens'] for r in group) == bs*6*(8 if historical else 1), 'incomplete logical routes')
            check(all(a == auctions[0] for a in auctions), 'replicated planners disagree')
            def values(field):
                return [r[field] for r in group]
            for r in group:
                r['predicted_pct'] = 100*r['price']/sum(values('price'))
                for phase in ('routed', 'graph'):
                    r[phase+'_pct'] = 100*r[phase+'_us']/sum(values(phase+'_us'))
            ranks.extend(group)
            layers.append(dict(bs=bs, arm=arm, layer=layer, moe_us=max(values('graph_us')),
                               token_ir=max(values('tokens'))/mean(values('tokens')),
                               total_m_blocks=sum(values('m_blocks')), peak_m_blocks=max(values('m_blocks')),
                               active_slots=sum(values('active_slots')), m_block_ir=max(values('m_blocks'))/mean(values('m_blocks')),
                               routed_mae_pp=mean(abs(r['predicted_pct']-r['routed_pct']) for r in group),
                               graph_mae_pp=mean(abs(r['predicted_pct']-r['graph_pct']) for r in group),
                               auction_accepted=auctions[0][2] if auctions else None,
                               output_equal=all(values('output_equal'))))
        sample = layers[-len(LAYERS):]
        averaged = {k: mean(r[k] for r in sample) for k in sample[0]
                    if k not in ('bs', 'arm', 'layer', 'auction_accepted', 'output_equal')}
        baseline = summary[0]['moe_us'] if summary else averaged['moe_us']
        accepted = [r['auction_accepted'] for r in sample if r['auction_accepted'] is not None]
        summary.append(dict(bs=bs, arm=arm, **averaged, moe_change_pct=100*(averaged['moe_us']/baseline-1),
                            auction_accepted_mean=mean(accepted) if accepted else None,
                            auction_accepted_total=sum(accepted) if accepted else None,
                            output_equal=all(r['output_equal'] for r in sample)))
    if historical:
        with (root / ('capture_comparison.csv' if fresh else 'history_comparison.csv')).open() as f:
            audit = list(csv.DictReader(f))
        for row in summary:
            if fresh and row['arm'] != 'vanilla':
                row['capture_allocation_equal'] = None
                continue
            checks = [r for r in audit if r['arm'] == row['arm']]
            keys = {(int(r['layer']), int(r['rank']), int(r['step_zero_based'])) for r in checks}
            expected_keys = {(layer, rank, step) for layer in LAYERS for rank in range(8) for step in range(3)}
            check(keys == expected_keys and len(checks) == len(keys), 'incomplete history comparison')
            row['capture_allocation_equal' if fresh else 'historical_allocation_equal'] = all(
                int(r['different_slots']) == 0 and r['auction_equal'] == 'True' for r in checks)
    write_rows(root/'ranks.csv', ranks)
    write_rows(root/'layers.csv', layers)
    write_rows(root/'main_table.csv', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument("--four-arms", action="store_true")
    args = parser.parse_args()
    rows = report(args.root, args.historical or args.fresh, args.fresh, ARMS[:4] if args.four_arms else ARMS)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([r['arm'] for r in rows], [r['moe_us'] for r in rows], color='#276787')
    ax.bar_label(bars, labels=[f"{r['moe_us']:.1f}" for r in rows], padding=3)
    ax.set(ylabel='MoE time (µs/layer)', title=f"SM90 EP8 · BS{rows[0]['bs']} · replay 2 · mean of L8, L20, L32")
    ax.margins(y=0.14)
    fig.tight_layout()
    fig.savefig(args.root/'moe.svg')
    if args.fresh:
        if not rows[0]['capture_allocation_equal'] or not all(r['output_equal'] for r in rows):
            raise SystemExit('New collection comparison failed; inspect capture_comparison.csv and ranks.csv')
    elif args.historical and not all(r['historical_allocation_equal'] and r['output_equal'] for r in rows):
        raise SystemExit('Historical comparison failed; inspect history_comparison.csv and ranks.csv')
