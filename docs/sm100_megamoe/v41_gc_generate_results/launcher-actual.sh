#!/bin/bash
set -eo pipefail

repo=/tmp/sm100-native-retry-20261121-1945
work=$(cat /tmp/sm100-v41-gc-generate-workdir.txt)
cd "$repo"
echo "$$" > "$work/launcher-pid.txt"
git rev-parse HEAD > "$work/source.sha"

source /home/admin/workspace/megamoe/env.sh
unset PYTHONPATH
source /data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv/bin/activate
export SM100_V41_MODEL=/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model
test -r "$SM100_V41_MODEL/model.safetensors.index.json"
export PYTHONPATH="$repo/scripts/sm100:$repo/source"

{
  echo 'executor=SM100'
  echo 'launcher=persistent-pty-session'
  echo 'external_timeout=none'
  echo "repo=$repo"
  echo "work=$work"
  echo "start=$(date -Is)"
  echo "source_commit=$(cat "$work/source.sha")"
  echo "model=$SM100_V41_MODEL"
  echo 'backend=gc_moe'
  echo 'install=not_required_existing_verified_vllm_venv'
  echo "python=$(python -V 2>&1)"
  echo "vllm=$(python -c 'import vllm; print(vllm.__version__)')"
  echo "torch=$(python -c 'import torch; print(torch.__version__)')"
  echo "torch_cuda=$(python -c 'import torch; print(torch.version.cuda)')"
  echo "flashinfer=$(python -c 'import flashinfer; print(flashinfer.__version__)')"
  echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"
  echo "triton=$(python -c 'import triton; print(triton.__version__)')"
  echo "gc_moe=$(python -c 'import gc_moe; print(gc_moe.__version__, gc_moe.__file__)')"
} > "$work/versions.txt"

python - <<'PY' > "$work/import.log" 2>&1
import sys, torch, vllm, gc_moe
from gc_moe import fp8_fp4_gc_moe, get_block_m_for_gc_moe, get_symm_buffer_for_gc_moe, transform_weights_for_gc_moe
print('python', sys.version.split()[0])
print('vllm', vllm.__version__, vllm.__file__)
print('torch', torch.__version__, torch.version.cuda)
print('gc_moe', gc_moe.__version__, gc_moe.__file__)
print('api', fp8_fp4_gc_moe.__module__, get_block_m_for_gc_moe.__module__, get_symm_buffer_for_gc_moe.__module__, transform_weights_for_gc_moe.__module__)
print('devices', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_capability(i), torch.cuda.get_device_name(i))
PY

python - <<'PY' > "$work/gpu-capability.txt"
import torch
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_capability(i), torch.cuda.get_device_name(i))
PY

findmnt -T "$SM100_V41_MODEL" -o TARGET,FSTYPE > "$work/load-filesystem.txt"
python scripts/sm100/v41_shape.py "$SM100_V41_MODEL" | tee "$work/config.log"
sha256sum "$SM100_V41_MODEL/model.safetensors.index.json" > "$work/index.sha256"
python -m pip freeze > "$work/pip-freeze.log" 2>&1
echo 'No dependency installation performed; using the existing verified vLLM venv.' > "$work/install.log"

journalctl -k --since "$(date -d '2 hours ago' '+%Y-%m-%d %H:%M:%S')" --no-pager > "$work/journal-probe.log" 2> "$work/journal-probe.err" || true
{
  echo "journal_probe_bytes=$(wc -c < "$work/journal-probe.log")"
  echo "journal_stderr_bytes=$(wc -c < "$work/journal-probe.err")"
  if grep -q 'not seeing messages from other users' "$work/journal-probe.err"; then
    echo 'system_journal_access=PERMISSION_LIMITED'
  elif [[ -s "$work/journal-probe.log" ]]; then
    echo 'system_journal_access=READABLE'
  else
    echo 'system_journal_access=EMPTY_OR_UNREADABLE'
  fi
  echo 'coredump_readable=no_coredumps_or_permission_limited'
} > "$work/journal-access.txt"

{
  echo "source_cgroup=$(cat /proc/self/cgroup)"
  cg=/sys/fs/cgroup$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)
  echo "resolved_cgroup=$cg"
  for f in memory.events memory.current memory.max; do
    if [[ -r "$cg/$f" ]]; then echo "--- $cg/$f"; cat "$cg/$f"; else echo "unreadable=$cg/$f"; fi
  done
} > "$work/cgroup-access.txt"

export VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1

set +e
(
  set -eo pipefail
  exec 9>>/tmp/gcmoe-sm100-gpu-group.lock
  while true; do
    flock 9
    if busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && test -z "$busy" && ! fuser /dev/nvidia{0..7} >/dev/null 2>&1; then break; fi
    flock -u 9
    sleep 20
  done
  export CUDA_VISIBLE_DEVICES=$(python -c 'import json,re; d=json.load(open("docs/sm100_megamoe/ENV.json")); print(",".join(re.findall(r"GPU-[0-9A-Fa-f-]{36}", d["commands"]["gpu_list"]["output"])))')
  test "$(tr , '\n' <<< "$CUDA_VISIBLE_DEVICES" | sort -u | wc -l)" = 8
  printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$work/cuda-visible-devices.txt"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader > "$work/gpu-before.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$work/gpu-compute-before.txt"

  cgroup_path=$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)
  cg="/sys/fs/cgroup${cgroup_path:-/}"
  if [[ ! -r "$cg/memory.events" ]]; then cg=/sys/fs/cgroup; fi
  snapshot() {
    date -Is
    printf 'cgroup=%s\n' "$cg"
    for f in memory.events memory.current memory.max; do
      if [[ -r "$cg/$f" ]]; then cat "$cg/$f"; fi
    done
    awk '/MemAvailable|SwapFree/ {print}' /proc/meminfo
  }
  snapshot > "$work/memory.before"
  export PYTHONFAULTHANDLER=1 VLLM_LOGGING_LEVEL=DEBUG
  echo "START backend=gc_moe at $(date -Is)"
  python -u scripts/sm100/v41_native_smoke.py --backend gc_moe --prefix-audit > "$work/model-full.log" 2>&1 &
  model_pid=$!
  echo "$model_pid" > "$work/model-pid.txt"
  while kill -0 "$model_pid" 2>/dev/null; do
    snapshot >> "$work/memory.log"
    sleep 20
  done
  code=0
  wait "$model_pid" || code=$?
  echo "$code" > "$work/exit-code.txt"
  snapshot > "$work/memory.after"
  tail -80 "$work/model-full.log"
  echo "END backend=gc_moe exit=$code at $(date -Is)"
  exit "$code"
) 2>&1 | tee "$work/model.log"
launcher_code=${PIPESTATUS[0]}
set -e

{
  date -Is
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$work/gpu-release.txt"
printf '%s\n' "$launcher_code" > "$work/launcher-exit-code.txt"
echo "FINISHED=$(date -Is) EXIT=$launcher_code" >> "$work/model.log"
exit "$launcher_code"
