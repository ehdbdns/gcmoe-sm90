# SM100 V4.1 EPLB decode results

Two independent cold starts at source commit
`040eba834d6e51c338e895440ef6c45646199613`:

- `deep_gemm_mega_moe/`: `--backend deep_gemm_mega_moe --eplb-decode`, exit 0.
- `gc_moe/`: `--backend gc_moe --eplb-decode`, exit 0.

Both used the same checkpoint, venv, TP8/EP8, prompt
`[100,200,300,400]`, greedy 4-token decode, and EPLB initial placement with
384 logical experts, 448 physical slots, 64 redundant experts, and 56 local
experts per rank.

Results:

- Both generated IDs are `[75, 200, 18, 200]`; first ID difference: none.
- All 40 per-layer placement hashes are identical across arms; first hash
  difference: none. The shared hash is
  `4e74b9e8c49846a9afbc6399a6d169d363412bc138d727348c2c0b55cc1afdfa`.
- `replica-comparison.json` contains the full 40-layer machine-readable
  comparison; `replica-comparison.txt` contains the table.
- Each arm loaded 48/48 checkpoint shards and returned 8 worker PIDs.
- The GC arm logged 8/8 distinct adapter installation PIDs.
- Both arms exited 0 with no `ERROR` or `Traceback` lines and no cgroup OOM.
- GPUs were released after each arm; final check is in
  `gpu-release-recheck.txt`.

`SHA256SUMS` covers every other file in this directory tree.
