"""Capture a selected GC replay and export rank-aligned EP8 Perfetto and table rows."""
import argparse
import json
from pathlib import Path

import numpy as np

PHASE = {1: "routed L1", 2: "routed L2 incl. push", 3: "shared expert L1", 4: "shared expert L2"}


class ProfileWindow:
    """The same worker opens and closes the window around one selected replay."""
    def __init__(self):
        import torch
        from native_phase_clock import anchor
        self.ticks = torch.empty(2, dtype=torch.int64, device="cuda")
        self.anchor = anchor
        anchor[(1,)](self.ticks)
        torch.cuda.synchronize()
        self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA])
        self.prof.start()
        anchor[(1,)](self.ticks[:1])

    def finish(self, path):
        import torch
        self.anchor[(1,)](self.ticks[1:])
        torch.cuda.synchronize()
        self.prof.stop()
        self.prof.export_chrome_trace(str(path))
        return self.ticks.cpu().numpy()


def save_sample(values, ticks, folder, *, rank, step, layer, block_m):
    stem = f"step{step:04d}-layer{layer:02d}-rank{rank}"
    readings = {key: value.detach().cpu().numpy() for key, value in values.items()}
    if "events" in readings:
        used = int(readings["clocks"][20])
        if not 0 <= used <= len(readings["events"]):
            raise ValueError(f"absolute trace overflow ({used} records), increase GC_MOE_SM100_TRACE_EVENTS")
        readings["events"] = readings["events"][:used]
    np.savez_compressed(folder / f"{stem}.npz", **readings, ticks=ticks,
                        block_m=block_m, step=step, layer=layer, rank=rank)


def calibration(path, ticks, epoch):
    raw = json.loads(path.read_text())
    marks = sorted((e for e in raw["traceEvents"] if e.get("cat") == "kernel"
                    and e.get("ph") == "X" and "sm100_phase_clock_anchor" in e["name"]),
                   key=lambda e: e["ts"])
    if len(marks) != 2 or min(e["dur"] for e in marks) <= 0:
        raise ValueError(f"missing or invalid Kineto anchors in {path}")
    device = [int(x) for x in ticks]
    host = [(int(raw["baseTimeNanoseconds"]) - epoch) / 1000 + e["ts"] + e["dur"] / 2 for e in marks]
    elapsed = (device[1] - device[0]) / 1000
    bound = sum(e["dur"] for e in marks) / 2
    if elapsed <= 0 or host[1] <= host[0] or abs(host[1] - host[0] - elapsed) > bound + 5:
        raise ValueError(f"unstable Kineto clock alignment in {path}")

    def timestamp(t):
        if not device[0] <= int(t) <= device[1]:
            raise ValueError(f"device trace is outside the selected replay in {path}")
        return host[0] + (int(t) - device[0]) / (device[1] - device[0]) * (host[1] - host[0])
    return timestamp, max(e["dur"] for e in marks) / 2


def absolute(rows, rank, stamp, common):
    starts, finishes, pushes, events = {}, {}, {}, []
    for row in rows.view(np.uint32):
        lo, hi, kind, cta, warp, seq, slot, m, n, valid, sm, _ = map(int, row)
        now = (hi << 32) | lo
        if kind == 12:
            events.append(dict(ph="i", s="t", pid=rank, tid="ready", name="M block ready", ts=stamp(now),
                               args={**common, "slot": slot, "m_block": m, "pool_block": n, "valid_m": valid,
                                     "cta": cta, "sm": sm}))
            continue
        if kind == 13:
            pushes[cta, seq] = now
            continue
        if kind in (14, 15):
            key = ("wait", cta, warp, seq)
            name, lane = "wait for M block", f"activation loader CTA {cta}"
            args = dict(slot=slot, m_block=m, n_cluster=n, valid_m=valid)
            first = kind == 14
        elif kind in (10, 11):
            key = ("pull", cta, warp, seq)
            name = "pull token"
            lane = f"dispatch CTA {cta} warp {warp}"
            args = dict(slot=slot, m_block=m, source_rank=n, row_in_block=valid)
            first = kind == 10
        elif 1 <= kind <= 8:
            first = kind <= 4
            phase = kind if first else kind - 4
            # A leader issues one MMA for the cluster. Both member CTAs push
            # their half of the tile. Keep the two distinct completion times.
            key = ("compute", cta // 2, seq, phase)
            name = PHASE[phase]
            lane = f"compute CTA {cta}"
            args = dict(slot=slot, m_block=m, n_cluster=n, valid_m=valid, cta=cta)
        else:
            raise ValueError(f"unexpected absolute trace kind {kind}")
        args["sm"] = sm
        if first:
            if key in starts:
                raise ValueError(f"duplicate start {key}")
            starts[key] = now
        else:
            finishes.setdefault(key, []).append((now, lane, name, args))
    if starts.keys() != finishes.keys():
        raise ValueError("unpaired absolute trace events")
    for key, begin in starts.items():
        ends = finishes[key]
        if len(ends) != (2 if key[0] == "compute" else 1) or len({e[1] for e in ends}) != len(ends):
            raise ValueError(f"incomplete or duplicate cluster/warp completion {key}")
        for end, lane, name, args in ends:
            if name == PHASE[2]:
                push = pushes.pop((args["cta"], key[2]))
                if not begin <= push <= end:
                    raise ValueError("invalid routed push interval")
                events.append(dict(ph="X", pid=rank, tid=lane+" push", name="routed L2 epilogue / push",
                                   ts=stamp(push), dur=stamp(end)-stamp(push), args={**common, **args}))
            if end < begin:
                raise ValueError("negative absolute duration")
            events.append(dict(ph="X", pid=rank, tid=lane, name=name, ts=stamp(begin),
                               dur=stamp(end)-stamp(begin), args={**common, **args}))
    if pushes:
        raise ValueError("unpaired routed push events")
    return events


def export(folder, prefix):
    paths = [folder / f"{prefix.split('-layer')[0]}-rank{r}.kineto.json" for r in range(8)]
    epoch = min(int(json.loads(p.read_text())["baseTimeNanoseconds"]) for p in paths)
    overview, detail, ranks = [], [], []
    for rank, path in enumerate(paths):
        with np.load(folder / f"{prefix}-rank{rank}.npz") as sample:
            clocks = sample["clocks"].tolist()
            counts = sample["counts"].astype(np.int64)
            bm, step, layer = (int(sample[k]) for k in ("block_m", "step", "layer"))
            if rank != int(sample["rank"]) or bm <= 0 or np.any(counts < 0):
                raise ValueError("invalid rank, BM, or receive counts")
            eco = "quota" in sample
            if eco and not np.array_equal(counts, sample["quota"]):
                raise ValueError(f"rank {rank} actual pulls differ from plan quota")
            stamp, error = calibration(path, sample["ticks"], epoch)
            blocks = (counts + bm - 1) // bm
            common = dict(rank=rank, step=step, layer=layer, tokens=int(counts.sum()), m_blocks=int(blocks.sum()))
            graph = min(clocks[i] for i in ((0, 1, 15) if eco else (0, 1)))
            finish = clocks[13]
            stamp(graph)
            stamp(finish)
            if not graph <= clocks[1] <= finish:
                raise ValueError(f"rank {rank} compute entry outside the MoE call")
            overview.append(dict(ph="i", s="t", pid=rank, tid="compute", name="compute kernel entry",
                                 ts=stamp(clocks[1]), args=common))

            def span(track, name, start, end, optional=False):
                if optional and start == -1 and end == 0:
                    return
                if not graph <= start <= end <= finish:
                    raise ValueError(f"rank {rank} invalid {name} span [{start},{end}]")
                overview.append(dict(ph="X", pid=rank, tid=track, name=name, ts=stamp(start),
                                     dur=stamp(end)-stamp(start), args=common))

            if common["tokens"]:
                span("dispatch", "count exchange", clocks[0], clocks[2])
                span("dispatch", "pull token", clocks[2], clocks[3])
                span("compute", "wait token", clocks[14], clocks[4])
                span("compute", "routed GEMM", clocks[4], clocks[7])
            span("shared L1", "shared expert L1", clocks[8], clocks[9], True)
            span("shared L2", "shared expert L2", clocks[10], clocks[11], True)
            span("combine", "combine", clocks[12], finish)
            if eco:
                span("plan", "ECO plan", clocks[15], clocks[16])
            if "events" in sample:
                if len(sample["events"]) != clocks[20]:
                    raise ValueError("absolute trace record count mismatch")
                detail.extend(absolute(sample["events"], rank, stamp, common))
            ranks.append(dict(**common, slot_counts=counts.tolist(), block_m=bm, active=int((counts > 0).sum()),
                              moe_ns=finish-graph, routed_ns=clocks[7]-clocks[4] if common["tokens"] else 0,
                              price=clocks[19] if eco else None, offered=clocks[18] if eco else None,
                              accepted=clocks[17] if eco else None, anchor_half_width_us=error))
    if len({(v["step"], v["layer"], v["block_m"], v["price"] is None) for v in ranks}) != 1:
        raise ValueError("rank samples refer to different calls or configurations")
    for name, events in (("perfetto", overview), ("absolute.perfetto", detail)):
        if events:
            tracks, metadata = {}, []
            for event in events:
                rank, label = event["pid"], event["tid"]
                key = rank, label
                if key not in tracks:
                    tracks[key] = len(tracks) + 1
                    metadata.append(dict(ph="M", name="thread_name", pid=rank, tid=tracks[key],
                                         args=dict(name=label)))
                event["tid"] = tracks[key]
            metadata.extend(dict(ph="M", name="process_name", pid=r, tid=0,
                                 args=dict(name=f"rank {r}")) for r in range(8))
            (folder / f"{prefix}.{name}.json").write_text(
                json.dumps(dict(traceEvents=metadata+events, displayTimeUnit="ns")))
    (folder / f"{prefix}.ranks.json").write_text(json.dumps(ranks, indent=2))
    return ranks


def metrics(ranks):
    def column(key):
        return np.array([r[key] for r in ranks])
    t, m = column("tokens"), column("m_blocks")
    out = dict(moe_us=column("moe_ns").max()/1000, token_ir=t.max()/t.mean(), total_m=m.sum(),
               peak_rank_m=m.max(), active_slots=column("active").sum(), m_ir=m.max()/m.mean())
    if ranks[0]["price"] is not None:
        score, routed = column("price"), column("routed_ns")
        if len(set(column("offered"))) != 1 or len(set(column("accepted"))) != 1:
            raise ValueError("replicated auction counters disagree")
        out.update(routed_mae_pp=100*np.abs(score/score.sum()-routed/routed.sum()).mean(),
                   accepted=ranks[0]["accepted"], offered=ranks[0]["offered"])
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("folder", type=Path)
    cli.add_argument("prefix", help="e.g. step0002-layer20")
    args = cli.parse_args()
    print(json.dumps({k: float(v) for k, v in metrics(export(args.folder, args.prefix)).items()}, indent=2))
