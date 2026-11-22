"""Account for host time between MoE graph submissions on the submitting thread."""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def partition(events, start, end):
    edges = defaultdict(list)
    for i, e in enumerate(events):
        a, b = max(start, e['ts']), min(end, e['ts'] + e['dur'])
        if b > a:
            edges[a].append((True, i))
            edges[b].append((False, i))
    edges[start]
    edges[end]
    live = set()
    prev = start
    for t in sorted(edges):
        if t > prev:
            # Innermost recorded span. Nested runtime calls usually sit inside ATen.
            leaf = min(live, key=lambda i: (events[i]['dur'], -events[i]['ts'], i), default=None)
            parent = min((i for i in live if i != leaf and events[i]['cat'] == 'cpu_op'),
                         key=lambda i: events[i]['dur'], default=None)
            yield prev, t-prev, events[leaf] if leaf is not None else {}, events[parent]['name'] if parent is not None else ''
        for entering, i in edges[t]:
            if entering:
                live.add(i)
            else:
                live.remove(i)
        prev = t


def extract(path, writer, arm, rank, reference):
    trace = json.loads(path.read_text())
    spans = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    anchors = sorted(e['ts']+e['dur']/2 for e in spans if e.get('cat') == 'kernel' and 'sm100_phase_clock_anchor' in e['name'])
    if len(anchors) != 2:
        raise ValueError(f'{path}: expected two anchors')
    api = sorted((e for e in spans if e.get('cat') in ('cuda_runtime', 'cuda_driver') and
                  e['name'].startswith('cudaGraphLaunch') and anchors[0] < e['ts'] < anchors[1]), key=lambda e:e['ts'])
    if len(api) != 40 or len({(e['pid'],e['tid']) for e in api}) != 1:
        raise ValueError(f'{path}: ambiguous graph sequence or submitter')
    pid, tid = api[0]['pid'], api[0]['tid']
    host = [e for e in spans if e.get('pid') == pid and e.get('tid') == tid]
    kernels = defaultdict(set)
    for e in spans:
        corr = e.get('args', {}).get('correlation')
        if corr is not None and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset'):
            kernels[corr].add(e['name'])
    for layer, launch in enumerate(api):
        end = launch['ts']
        end_ns = int(trace['baseTimeNanoseconds']) + round(end*1000)
        if layer:
            row = reference[arm, rank, layer]
            expected = int(row['entry_host_ns']) - round(float(row['launch_to_entry_us'])*1000)
            if abs(expected-end_ns) > 5:
                raise ValueError(f'{path}: graph attribution disagrees at layer {layer}')
        start = api[layer-1]['ts'] if layer else anchors[0]
        for ts, dur, e, parent in partition(host,start,end):
            corr = e.get('args', {}).get('correlation')
            writer.writerow((arm,rank,layer,pid,tid,end_ns,round(ts-start,3),round(dur,6),
                             e.get('cat','unobserved'),e.get('name','unobserved'),parent,corr,
                             '|'.join(sorted(kernels[corr]))))


if __name__ == '__main__':
    root, prev, dest = map(Path,sys.argv[1:])
    with prev.open() as f:
        ref = {(r['arm'],int(r['rank']),int(r['layer'])):r for r in csv.DictReader(f)}
    with gzip.open(dest,'wt') as f:
        w=csv.writer(f)
        w.writerow('arm rank layer pid tid launch_host_ns offset_us exclusive_us category name enclosing_cpu_op correlation gpu_activity'.split())
        for arm in ('gc','mega'):
            for rank in range(8):
                extract(root/arm/f'profile-rank{rank}.json',w,arm,rank,ref)
