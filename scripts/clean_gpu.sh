#!/bin/bash
# Timestamped stale-process scavenger.
# NOTE: the vLLM workers rename their OS thread to `VLLM::Worker_TP`, so
# `pkill -f real_decode` does NOT match them and they keep 55 GB per GPU. Kill
# by the PIDs the driver reports instead.
for _ in 1 2 3 4 5 6; do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  [ -z "$pids" ] && break
  for p in $pids; do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f real_decode_100_100 2>/dev/null
  pkill -9 -f run_arm_eco 2>/dev/null
  pkill -9 -f "VLLM::Worker" 2>/dev/null
  sleep 5
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk -F' ' '{print $1}' | sort -n | tail -1)
echo "[clean] max gpu mem used = ${used} MiB"
