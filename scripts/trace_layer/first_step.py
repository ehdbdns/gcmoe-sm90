"""Find the first observed difference in an eight-rank post-EPLB forward."""
import csv

from aligned_step import require


STAGES = (('input', ('hidden_input',)),
          ('quantization', ('quantized', 'scales', 'shared_input', 'shared_scales')),
          ('experts', ('routed_slots', 'shared_y')),
          ('combine', ('output',)))
FIELDS = tuple(field for _, fields in STAGES for field in fields)


def first_step_controls(root, audits, step=0):
    call = 'treatment_call' if step else 'first_call'
    probes = [[r[call]['first_step'] for r in audit['initial']] for audit in audits]
    reqs = {f'req{i}' for i in range(256)}
    known_logits = [set(), set()]
    shared_live = True
    weight_differences, nonfinite = [], []
    errors, layers = [], []
    for rank in range(8):
        b, s = (a[rank] for a in probes)
        rb, rs = b['sampler'], s['sampler']
        ids = rb['row_requests']
        require(b['rank'] == s['rank'] == rank and ids == rs['row_requests']
                and len(ids) == 256 and set(ids) == reqs, f'rank {rank} first-step row order')
        for arm, probe in enumerate((b, s)):
            rows = probe['layers']
            start = audits[arm]['ranks'][rank]['first_epoch'] + step * 40
            require([row['layer'] for row in rows] == list(range(40))
                    and [row['epoch'] for row in rows] == list(range(start, start+40)),
                    f'rank {rank} incomplete first-step layers')
            for row in rows:
                require(set(row['hashes']) == set(FIELDS) and all(
                    len(h) == 64 and len(bytes.fromhex(h)) == 32 for h in row['hashes'].values()),
                    'invalid layer fingerprint')
                require(set(row['nonfinite']) == set(FIELDS), 'missing first-step finite-value audit')
                for field, count in row['nonfinite'].items():
                    if count:
                        nonfinite.append(dict(arm=('baseline','stream')[arm], rank=rank, layer=row['layer'],
                                              field=field, count=count))
                shared_live &= all(row['shared_nonzero'][key] for key in ('shared_input', 'shared_y'))
            legacy = audits[arm]['initial'][rank][call]
            require(all(legacy[k] == rows[0]['hashes'][k] for k in ('quantized','scales','output')),
                    'first-call evidence differs from the layer chain')
            sampler = probe['sampler']
            require(set(sampler['hidden']) == reqs and set(sampler['logits']) <= reqs,
                    f'rank {rank} incomplete sampler state')
            require(all(len(h)==64 and len(bytes.fromhex(h))==32 for h in sampler['hidden'].values())
                    and all(len(row['sha256'])==64 and len(bytes.fromhex(row['sha256']))==32
                            and isinstance(row['argmax'], int) and row['argmax'] >= 0
                            for row in sampler['logits'].values()), 'invalid sampler fingerprint')
            known_logits[arm].update(sampler['logits'])
        for layer, (left, right) in enumerate(zip(b['layers'], s['layers'], strict=True)):
            for row in (left, right):
                require(len(row['shared_weights']) == 4 and all(
                    len(h) == 64 and len(bytes.fromhex(h)) == 32 for h in row['shared_weights']),
                    'incomplete shared expert weight pair')
            if left['shared_weights'] != right['shared_weights']:
                weight_differences.append(dict(layer=layer, rank=rank))
            for phase, (stage, fields) in enumerate(STAGES):
                for key in fields:
                    equal = left['hashes'][key] == right['hashes'][key]
                    info = right['different_rows'][key]
                    limit = 6 if key == 'routed_slots' else 1
                    require((info['count'] == 0) == equal and 0 <= info['count'] <= len(ids)*limit
                            and len(info['examples']) == min(4, info['count']), 'row audit disagrees with layer hash')
                    for sample in info['examples']:
                        require(0 <= sample['row'] < len(ids) and ids[sample['row']] == sample['request']
                                and (limit == 1 or 0 <= sample['topk_slot'] < limit), 'bad differing row')
                    layers.append(dict(layer=layer, rank=rank, stage=stage, field=key, matching=equal,
                                       different_rows=info['count']))
                    if not equal:
                        errors.append((layer, phase, dict(layer=layer, stage=stage, field=key, rank=rank,
                                                          **info)))
        require(rb['logits'].keys() == rs['logits'].keys(), 'sampler shard ownership changed')
        for index, key in enumerate(('hidden','logits')):
            requests = sorted((k for k in rb[key] if rb[key][k] != rs[key][k]), key=lambda k:int(k[3:]))
            if requests:
                errors.append((40, index, dict(stage='sampler_'+key, rank=rank, count=len(requests),
                                               examples=requests[:4])))
    require(all(seen == reqs for seen in known_logits), 'first-step logits omit requests')
    samples = [audit['ranks'][0]['native_history']['replay_top1'][step] for audit in audits]
    native = [dict(request=f'req{i}', baseline=x, stream=y) for i,(x,y) in enumerate(zip(*samples, strict=True)) if x!=y]
    if native:
        errors.append((40, 2, dict(stage='native_top1', count=len(native), examples=native[:4])))
    first = min(((layer, phase) for layer, phase, _ in errors), default=None)
    filename = 'treatment_step_layers.csv' if step else 'first_step_layers.csv'
    with (root/filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(layers[0]))
        writer.writeheader()
        writer.writerows(sorted(layers, key=lambda row:(row['layer'],row['rank'])))
    return dict(first_step_chain_match=not errors, shared_branch_active=shared_live,
                first_step_finite=not nonfinite, nonfinite_examples=nonfinite[:8],
                shared_weights_match=not weight_differences, shared_weight_differences=weight_differences,
                first_step_native_differences=native,
                first_observed_differences=[detail for layer, phase, detail in errors if (layer, phase)==first])


def attention_controls(root, audits):
    fields = ('query', 'swa_lengths', 'swa_multiset', 'swa_order',
              'extra_lengths', 'extra_multiset', 'extra_order', 'mla_output')
    table, result = [], {}
    for step, call, stage in ((0, 'first_call', 'control'), (1, 'treatment_call', 'treatment')):
        diffs = []
        for rank in range(8):
            probes = [a['initial'][rank][call]['first_step'] for a in audits]
            ids = probes[0]['sampler']['row_requests']
            require(len(ids) == 256 and ids == probes[1]['sampler']['row_requests'], 'invalid attention rows')
            values = []
            for arm, probe in enumerate(probes):
                attn = probe['attention']
                mode = 0 if step == 0 else arm
                require(attn['mode'] == probe['stream_mode'] == mode and attn['position'] == 40*step+2,
                        'FlashMLA or MoE probe observed wrong treatment or step')
                rows = dict(attn['hashes'], **{k+'_lengths': v for k, v in attn['lengths'].items()})
                require(set(rows) == set(fields) and all(len(v) == 256 for v in rows.values()),
                        'incomplete FlashMLA rows')
                for name, data in rows.items():
                    if name.endswith('_lengths'):
                        require(all(type(n) is int and n >= 0 for n in data), 'invalid active prefix length')
                    else:
                        require(all(len(h) == 64 and len(bytes.fromhex(h)) == 32 for h in data), 'bad FlashMLA hash')
                values.append(rows)
            for key in fields:
                affected = [req for req, a, b in zip(ids, values[0][key], values[1][key], strict=True) if a != b]
                row = dict(step=step, layer=2, rank=rank, field=key, different_rows=len(affected),
                           examples=','.join(affected[:4]))
                table.append(row)
                if affected:
                    diffs.append(row)
        result[stage+'_match'] = not diffs
        result[stage+'_differences'] = diffs
    with (root/'flashmla_rows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return result
