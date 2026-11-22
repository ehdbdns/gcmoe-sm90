# BS1024 graph-replay run: interrupted status

- Stop requested at `2026-11-22T15:15:20+08:00`; `run_five.sh` trap recorded release at `2026-11-22T15:15:24+08:00`.
- Completed arms: `mega`, `gc` (both exit 0, 40/40 validated layer graphs on every rank).
- Interrupted arm: `mega_rr` (started, terminated before result.json).
- Not started: `gc_rr`, `gc_eco`.
- Eight GPUs released: `nvidia-smi --query-compute-apps` and `fuser /dev/nvidia0..7` are empty.
- Workdir retained for resume: `/data0/home/admin/workspace/megamoe/v41-five-graph-bs1024-01`.
- `overlay/`, `overlay.ready`, `dg-cache/`, `eco-cache/` retained because the run may be resumed; do not treat this as final cleanup.
