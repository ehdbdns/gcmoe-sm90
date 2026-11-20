#!/usr/bin/env bash
# CPU and synthetic device preflight. No language-model launch.
set -eo pipefail
[[ $# == 1 ]] || exit 2
out=$(realpath -m "$1")
cd "$(git rev-parse --show-toplevel)"
mkdir "$out"
source scripts/env.sh
py="$GC_ENV/bin/python"
df -h "$out" "${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$out/preexisting-processes.csv"
test ! -s "$out/preexisting-processes.csv"
if [[ ${GC_MOE_MATRIX_SIZE:-20} == 24 ]]; then
  # Byte-identical to DeepInfer a679bb86 (deepinfer/model_executor/eplb/policy.py).
  echo '52241603827a1a1d19a395d48f5d618a7336fd1ceee631127e6782c644f736dc  source/gc_moe/eplb/policy.py' | sha256sum -c -
fi
for test_file in tests/test_*_cpu.py; do "$py" "$test_file"; done
"$py" tests/test_stream_checkpoint_cpu.py --gpu-preflight
"$py" tests/test_attention_order_cpu.py --cuda
"$py" tests/test_long_cost_cpu.py --cuda
export GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 GC_MOE_ECO_COSTFN=2 GC_MOE_SM90_STREAM_M_BLOCKS=1
export GC_MOE_ECO_MAX_COPIES=8 GC_MOE_ECO_PER_LAYER=1 GC_MOE_LAYERS=40 GC_MOE_FINE_TRACE=0
export GC_MOE_SM90_PULL_TMA=1 GC_MOE_IR_STEP=0 GC_MOE_SM90_DISPATCH_SMS=6 SMOKE_ECO_WB=26
arms=(vanilla random rr)
case ${GC_MOE_MATRIX_SIZE:-20} in
  20) ;;
  24) arms+=(eco_linear) ;;
  *) exit 2 ;;
esac
arms+=(eco eco_smr)
for arm in "${arms[@]}"; do
  eco=0; reuse=0; mode=2
  case "$arm" in
    eco_linear) eco=1; mode=0 ;;
    eco) eco=1 ;;
    eco_smr) eco=1; reuse=1 ;;
  esac
  GC_MOE_ECO_COSTFN="$mode" GC_MOE_LONG_COST=1 GC_MOE_FIXED_DIR="$out" GC_MOE_MATRIX_ARM="$arm" GC_MOE_EPLB_WINDOW=1000 \
  GC_MOE_STEP_PROFILE=1 GC_MOE_ECO_TRACE=1 GC_MOE_PULL_TRACE=1 SMOKE_ECO="$eco" GC_MOE_SM90_SM_REUSE="$reuse" \
    "$py" scripts/smoke_compile_eco.py 2>&1 | tee "$out/compile-$arm.log"
  roles='[1, 4, 2, 3]'
  if [[ $eco == 1 ]]; then roles='[1, 4, 2, 5, 3]'; fi
  if [[ $arm == eco_smr ]]; then roles='[1, 4, 2, 5, 6, 3]'; fi
  grep -Fq "compiled roles $roles" "$out/compile-$arm.log"
done
for reuse in 0 1; do
  GC_MOE_STEP_PROFILE=0 GC_MOE_ECO_TRACE=0 GC_MOE_PULL_TRACE=0 GC_MOE_SM90_SM_REUSE="$reuse" \
  TL_TOKENS=17 TL_CAP=128 TL_HOT_BLOCKS=1 timeout 300 "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 \
    scripts/verify_eco_capture.py 2>&1 | tee "$out/reuse-$reuse.log"
  grep -oE 'STREAM_CHECK rank=[0-7] sha256=[0-9a-f]{64}' "$out/reuse-$reuse.log" | sort > "$out/reuse-$reuse.sha"
  test "$(wc -l < "$out/reuse-$reuse.sha")" = 8
  for rank in {0..7}; do grep -q "^STREAM_CHECK rank=$rank " "$out/reuse-$reuse.sha"; done
done
diff -u "$out/reuse-0.sha" "$out/reuse-1.sha"
