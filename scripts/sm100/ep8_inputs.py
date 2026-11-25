"""Prepare the exact prompt cohorts and initial replica placements used by the EP8 matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BATCHES = (256, 512, 1024)
ARMS = ("vanilla", "random", "rr", "eco")
PROMPT_HASH = {
    256: "254af9d8a2752eab68f6419f085c1973e22e582a3f50a3c99a9853efeffdaec8",
    512: "e7c4aee9db91a34c018bd54464e67355681b74112d515401844f473c053cfcef",
    1024: "d7eeeb50bb0428af010e6cad5ed56598f8c559a06525070a4d09ebd023d7e74c",
}
FILES = ("3664daca1619617b76026e6a75e4b3738316320aceb2852c76f2897873e99073",
         "957d06fed999f9fe517dab4581a2fe79f162debc1c21468502d3e988c83511f0")


def compact(value):
    return json.dumps(value, separators=(",", ":")).encode()


def stride_sample(pool, count):
    if count >= len(pool):
        return list(pool)
    step = len(pool) / count
    return [pool[min(len(pool) - 1, int(i * step))] for i in range(count)]


def select_prompts(pools, count):
    take = max(1, count // len(pools))
    queues = [stride_sample(pool, min(take, len(pool))) for pool in pools]
    got = sum(map(len, queues))
    if got < count:
        for index, pool in enumerate(pools):
            extra = min(count - got, len(pool))
            if extra > 0:
                queues[index].extend(stride_sample(pool, extra))
                got += extra
            if got >= count:
                break
    picked = []
    cursor = [0] * len(queues)
    while len(picked) < count:
        before = len(picked)
        for index, queue in enumerate(queues):
            if len(picked) >= count:
                break
            if cursor[index] >= len(queue):
                continue
            picked.append(queue[cursor[index]])
            cursor[index] += 1
        if len(picked) == before:
            break
    if len(picked) != count:
        raise ValueError(f"only {len(picked)} prompts selected, need {count}")
    return picked


def make_prompts(source, root):
    pools = []
    for index, expected in enumerate(FILES, 1):
        path = source / f"dsv4-flash_online_input_ids_26-10-08_{index}.jsonl"
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"source digest mismatch: {path}")
        with path.open() as f:
            rows = [(json.loads(line).get("input_ids") or []) for line in f if line.strip()]
        pool = sorted((x for x in rows if 256 <= len(x) <= 2048), key=len)
        if not pool:
            raise ValueError(f"no eligible prompts: {path}")
        pools.append(pool)
    # The SM90 six-arm selector evenly strides each file, then fills a short
    # pool by striding it again rather than rejecting a valid 256-2048 cohort.
    interleaved = select_prompts(pools, 512)
    for bs, prompts in zip(BATCHES, (select_prompts(pools[:1], 256), interleaved, interleaved * 2), strict=True):
        payload = compact(prompts)
        if len(prompts) != bs or hashlib.sha256(payload).hexdigest() != PROMPT_HASH[bs]:
            raise ValueError(f"BS{bs} prompt selection differs from SM90")
        dest = root / f"bs{bs}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "prompts.json").write_bytes(payload)


def initial_plans(root):
    import torch
    from gc_moe.eco.meta_sm90 import reorder_single_copy_first

    generator = torch.Generator().manual_seed(0)
    random = torch.cat((torch.randperm(384, generator=generator),
                        torch.randint(384, (64,), generator=generator))).numpy()
    identity = np.arange(448) % 384
    for name, row in (("random", random), ("eco-initial", reorder_single_copy_first(identity, 8)[0])):
        slots = [np.flatnonzero(row == e) for e in range(384)]
        counts = np.asarray([len(s) for s in slots], dtype=np.int64)
        inverse = np.full((384, max(counts)), -1, dtype=np.int64)
        for e, copies in enumerate(slots):
            inverse[e, :len(copies)] = copies
        np.savez_compressed(root / f"{name}.npz", **{
            k: np.repeat(v.astype(np.int32)[None], 40, axis=0)
            for k, v in dict(layout=row, inverse=inverse, counts=counts).items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("--prepared", action="store_true", help="source contains bs*/prompts.json")
    args = p.parse_args()
    if args.prepared:
        for bs in BATCHES:
            data = (args.source / f"bs{bs}" / "prompts.json").read_bytes()
            if hashlib.sha256(data).hexdigest() != PROMPT_HASH[bs]:
                raise ValueError(f"prepared BS{bs} cohort differs from SM90")
            dest = args.root / f"bs{bs}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "prompts.json").write_bytes(data)
    else:
        make_prompts(args.source, args.root)
