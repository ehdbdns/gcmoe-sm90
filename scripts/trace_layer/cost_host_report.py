"""Verify BS1024 fixed-workload repeats and compare exclusive host boundaries."""
from pathlib import Path
import sys

import numpy as np

from cost_boundary_report import summarize
from main20_report import arm_data, load, require, write_rows


def exclusive(directory):
    host = load(directory/'run.json')['measured']['host_boundary_ns']
    steps = np.array(host['steps'])
    values = {}
    hits = {}
    for i, name, wall, cpu in host['exclusive_host']:
        require(0 <= i < 1000 and min(wall, cpu) >= 0, 'invalid exclusive timing')
        if name not in values:
            values[name] = np.zeros((1000, 2))
            hits[name] = np.zeros(1000, dtype=int)
        values[name][i] += (wall, cpu)
        hits[name][i] += 1
    for key in ('step','schedule','process_outputs','execute_model','update_from_output'):
        require(np.all(hits[key] == 1), f'missing or repeated {key}')
    wall = sum(v[:,0] for v in values.values())
    residual = steps[:,1]-steps[:,0]-wall
    require(np.all(residual >= 0), 'exclusive host timing exceeds engine.step')
    return values, residual


def report(group, prior):
    revision = (group.parent/'measurement.sha').read_text().strip()
    reference = None
    pair = []
    for arm in ('vanilla','eco_linear','eco'):
        directory = group/arm
        control, _, native, warm, _, _, _, _ = arm_data(directory,1024,arm,revision,True)
        if reference is None:
            reference = control, native, warm
        common, tokens, prefix = reference
        require(control == common and np.array_equal(tokens,native) and warm == prefix,
                f'{arm}: strict current-run comparison failed')
        with np.load(prior/arm/'rank0.npz') as f:
            require(np.array_equal(f['native'],native), f'{arm}: previous native trajectory differs')
        require(load(prior/arm/'installed-layout.json') == load(directory/'installed-layout.json'),
                'different placement from original run')
        if arm != 'vanilla':
            pair.append(exclusive(directory))
    require(load(group/'eco/installed-layout.json') == load(group/'eco_linear/installed-layout.json'),
            'cost arms use different physical slots')
    write_rows(group/'pairs.csv',[dict(arm_a='eco_linear',arm_b='eco',strict_pair_comparable=True,
                                       measured_samples=1024000,native_diff=0,warm_native_equal=True)])
    (linear,lr),(fine,fr) = pair
    require(linear.keys()==fine.keys(),'unequal host boundary coverage')
    rows=[]
    for name in sorted(linear):
        a,b=linear[name]/1e6,fine[name]/1e6
        d=b-a
        rows.append(dict(interval=name,linear_wall_ms=a[:,0].mean(),fine_wall_ms=b[:,0].mean(),
                         delta_wall_ms=d[:,0].mean(),delta_cpu_ms=d[:,1].mean(),
                         delta_wall_less_cpu_ms=(d[:,0]-d[:,1]).mean(),
                         median_delta_ms=np.median(d[:,0]),fine_slower_steps=(d[:,0]>0).sum()))
    write_rows(group/'exclusive_host.csv',rows)
    balance=[]
    for arm, values, residual in (('eco_linear',linear,lr),('eco',fine,fr)):
        measured=load(group/arm/'run.json')['measured']
        steps=np.asarray(measured['host_boundary_ns']['steps'])
        exclusive_ms=sum(v[:,0].mean() for v in values.values())/1e6
        wrapper_ms=residual.mean()/1e6
        collect_ms=(steps[:,2]-steps[:,1]).mean()/1e6
        end_to_end=measured['ms_per_step']
        balance.append(dict(arm=arm,e2e_ms=end_to_end,exclusive_engine_ms=exclusive_ms,
                            outer_wrapper_ms=wrapper_ms,collect_ms=collect_ms,
                            interstep_and_final_sync_ms=end_to_end-exclusive_ms-wrapper_ms-collect_ms))
    write_rows(group/'host_balance.csv',balance)
    rows=[]
    for root,order in ((prior,'fine_first'),(group,'linear_first')):
        a,b=(summarize(root/arm) for arm in ('eco_linear','eco'))
        for key in a:
            d=b[key]-a[key]
            rows.append(dict(order=order,interval=key,samples=len(d),linear_ms=a[key].mean(),
                             fine_ms=b[key].mean(),delta_ms=d.mean(),paired_median_ms=np.median(d)))
    write_rows(group/'orders.csv',rows)


if __name__ == '__main__':
    report(*(Path(p) for p in sys.argv[1:]))
