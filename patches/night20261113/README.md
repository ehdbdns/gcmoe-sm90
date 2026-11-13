# Night 2026-11-13 — verified code snapshot

Both patches are **diffs against `gcmoe_ir20_trace_20261111`**, the last
known-good baseline (69.39 ms / BS256, no trace overhead).

They contain only these changes:

## `impls-sm90_fp8_fp4_gc_moe.patch`
1. `[PULL-BIG-FIX2]` per-CTA pull-order permutation in its OWN shared memory
   (the old version wrote a GLOBAL array from block 0 and every CTA read it
   after a CTA-local barrier -> half-written, slots duplicated/dropped).
2. `[PULL-BIG-FIX3]` the permutation replaces the LOOP VARIABLE, not just
   `global_slot`. `expert_count` / `stats` / `route_meta` / `l1_acts` /
   `route_weight` are all indexed by that variable, so permuting only
   `global_slot` made a CTA pull slot X and write it into row Y.
3. Scratch placed at the BOTTOM of the shared window (offsets 128/384);
   the old `smem-512` overlapped the TMA staging tail and the mbarriers.
4. `[T13-FINEFIX]` the fine-trace `fine_floor` is read once per call into a
   CTA-local cache (the per-event global read cost +35 ms at BS256) and
   type-5 records are emitted once per CTA.

Default behaviour is unchanged: the permutation is behind
`GC_MOE_SM90_PULL_BIG=1` (default 0).

## `integration-deepep_compare.patch`
1. `[T13-LAYERIDX]` the per-layer counter row index is STAMPED on the packed
   tensor instead of being looked up through `id()` on every forward.
   `id()` is only unique while the object lives and the map is built on a
   different code path, so a recycled id collapsed all 40 layers onto ONE row.
   Signature of that failure:
   `per_layer_median_tokens=[0,...,159.7]`, `m_blocks=5`, `max_copies=53`,
   `ir_step_alllayers=7.88` (null = 1.05).
2. `[T13-LAYERIDX4]` the ECO arm's per-slot weight view
   (`gather_slot_weights`) is a BRAND NEW tensor set, so it is stamped too --
   without this the ECO arm collapsed while the RR arm stayed healthy.
3. `[T13-MULTIDS]` `GC_PROMPT_SLICE` + `GC_PROMPT_SLICE_BATCHES` interleave a
   length-sorted pool so every batch spans the whole length range.
4. `[T13-EPLBDBG]` / `[T13-LAYERDBG]` optional diagnostics
   (`GC_MOE_EPLB_DBG=1`, `GC_MOE_LAYER_DBG=1`).

## CPU gates that must pass
```
python tests/test_eplb_logic_cpu.py      # CPU_TEST_FINAL PASS
```
