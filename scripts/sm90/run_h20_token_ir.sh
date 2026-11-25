#!/usr/bin/env bash
# Recover full-model ordered banks first. H20 only computes the upstream placements.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: run_h20_token_ir.sh JOB" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
job=$(realpath -m "$1")
mkdir -p "$job"
exec 8>"$job/scan.lock"
flock -n 8 || { echo "scan already running" >&2; exit 1; }
sha=$(git rev-parse HEAD)
if [[ -f "$job/scan.sha" ]]; then test "$(cat "$job/scan.sha")" = "$sha"; else echo "$sha" > "$job/scan.sha"; fi
source scripts/env.sh
py="$GC_ENV/bin/python"
archive="$PWD/docs/trace_executor_20261119"
exec 9>/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123/h20-eight.lock
for bs in 256 32 64 128 512 1024; do
  out="$job/bs$bs"
  mkdir -p "$out"
  if [[ -e $out/SHA256SUMS ]]; then (cd "$out" && sha256sum --quiet -c SHA256SUMS); continue; fi
  case $bs in
    32|64|128) previous="$archive/main24_cost_results/bs$bs" ;;
    256) previous="$archive/sm90_bs256_main_five_plan_comm_results/delivery/bs256" ;;
    *) previous="$archive/sm90_bs512_bs1024_full_model_four_results/delivery-$bs/bs$bs" ;;
  esac
  group=""
  shopt -s nullglob
  for g in "${SM90_SIX_BS_JOB:-/nonexistent}/delivery-$bs/bs$bs" "$previous" /mnt/fuse/sm90-ep8-linear-four-*/delivery-"$bs"/bs"$bs" "$job/ref-$bs/delivery-$bs/bs$bs"; do
    if [[ -s "$g/rr/run.json" && -s "$g/state-path.txt" && -s "$(cat "$g/state-path.txt")/replay/measure-rank0.pt" ]]; then group=$g; break; fi
  done
  if [[ -z $group ]]; then
    bash scripts/sm90/run_full_model_four.sh --ir-reference "$bs" "$job/ref-$bs" 8>&- 9>&- > "$out/reference.log" 2>&1
    group="$job/ref-$bs/delivery-$bs/bs$bs"
  fi
  echo "$group" > "$out/cohort.txt"
  while true; do
    flock 9
    nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$out/gpus.csv"
    awk -F, '$2 !~ /NVIDIA H20/ { bad=1 } END {exit (bad || NR!=8)}' "$out/gpus.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > "$out/idle.csv"
    [[ -s "$out/idle.csv" ]] || break
    flock -u 9; sleep 60
  done
  "$py" scripts/sm90/sweep_eplb_extra.py --save-plans --device cuda --bs "$bs" --cohort "$group" --out "$out/plans.npz"
  flock -u 9
  CUDA_VISIBLE_DEVICES='' "$py" scripts/sm90/sweep_eplb_extra.py --exact --plans "$out/plans.npz" --bs "$bs" --cohort "$group" --out "$out/token-ir.csv"
  (cd "$out" && sha256sum cohort.txt plans.npz token-ir.csv gpus.csv idle.csv > SHA256SUMS)
done
