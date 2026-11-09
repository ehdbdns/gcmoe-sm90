#!/bin/bash
# Preload the checkpoint into the OS page cache before a run.
#
# WHY: vLLM's own `_prefetch_all_checkpoints` is a BACKGROUND thread that
# starts at model init, so it races the actual weight load.  When the page
# cache already holds the checkpoint (as it did during the 16-arm sweep, after
# earlier runs) loading takes ~3 s at 15 shards/s; when the kernel has evicted
# it every worker reads ~37 GB from the NAS and the same load takes 10+ min
# (measured: 631 s, 13 s/shard, load average ~50).
#
# Doing the read here, single-threaded and BEFORE the engine starts, makes the
# load deterministic.  48 shards x ~0.9 GB at ~4.5 GB/s = ~10 s.
set -u
MODEL_DIR=${1:-${MODEL:-/mnt/nas2/models/DeepSeek-V4.1-Flash}}
# `posix_fadvise(WILLNEED)` asks the kernel to read the file into the page
# cache WITHOUT copying it into user space.  `cat f > /dev/null` does the same
# job but pays a full copy: measured 96 s (sys 94 s) versus 6 s with fadvise.
WARMDIR="$MODEL_DIR" python3 - <<'PYEOF'
import os, glob, sys
d = os.environ['WARMDIR']
n = 0
for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
    fd = os.open(f, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)
    n += 1
print(f'[warm] {n} shards queued for page cache')
PYEOF
n=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "[warm] $n shards read into page cache"
