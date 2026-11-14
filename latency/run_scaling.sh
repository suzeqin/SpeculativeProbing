#!/usr/bin/env bash
# Probe-only device-time scaling with sequence length, for every (model, task).
set -u
cd "$(dirname "$0")/.." || exit 1
GPU="${1:-6}"
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}"
mkdir -p latency/results/logs
SP_MM="mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc"
SP_MM_CORE="mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc,sp1_core,sp2_core,sp5_core"
for spec in "qwen3.5-4b|verih|$SP_MM" "qwen3.5-4b|nemotron_safety|$SP_MM" \
            "qwen3.5-9b|verih|$SP_MM" "qwen3.5-9b|nemotron_safety|$SP_MM" \
            "minicpm4.1-8b|verih|$SP_MM_CORE" "minicpm4.1-8b|nemotron_safety|$SP_MM_CORE"; do
  IFS='|' read -r model task methods <<< "$spec"
  echo "=== $(date '+%F %T') scaling $model / $task ==="
  "$PY" latency/profile_scaling.py --model "$model" --task "$task" \
    --methods "$methods" --points 6 --trials 5 \
    --out "latency/results/scaling_${model}_${task}.json" \
    > "latency/results/logs/scaling_${model}_${task}.log" 2>&1
  echo "[$(date '+%F %T')] exit=$? -> latency/results/scaling_${model}_${task}.json"
done
echo "=== scaling done $(date '+%F %T') ==="
