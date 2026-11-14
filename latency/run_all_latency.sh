#!/usr/bin/env bash
# Measure probe inference-time overhead for every (model, task) pair.
#
# Usage:
#   bash latency/run_all_latency.sh [GPU] [EXTRA_ARGS...]
#
# Results go to latency/results/<model>_<task>.json and logs to
# latency/results/logs/.  Safe to re-run: existing files are kept unless
# --force is passed.
set -u
cd "$(dirname "$0")/.." || exit 1

GPU="${1:-6}"
shift || true
FORCE=0
for a in "$@"; do [[ "$a" == "--force" ]] && FORCE=1; done
PY="${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

COMMON_ARGS="--reps 5 --reps_base 2 --cpu_examples 4 --reps_cpu 3 \
--decode_examples 5 --decode_tokens 8 --profile_reps 5"

run_one() {
  local model="$1" task="$2" methods="$3"
  local out="latency/results/${model}_${task}.json"
  local log="latency/results/logs/${model}_${task}.log"
  mkdir -p latency/results/logs
  if [[ -f "$out" && $FORCE -eq 0 ]]; then
    echo "[skip] $out exists (use --force to redo)"
    return 0
  fi
  echo "=== $(date '+%F %T')  $model / $task  (GPU $GPU) ==="
  "$PY" latency/bench_latency.py \
    --model "$model" --task "$task" --methods "$methods" \
    $COMMON_ARGS \
    --out "$out" > "$log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] $model/$task exit=$rc  -> $out"
  tail -n 4 "$log"
  return $rc
}

SP_MM="mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc"
SP_MM_CORE="mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc,sp1_core,sp2_core,sp5_core"

run_one qwen3.5-4b verih "$SP_MM"
run_one qwen3.5-4b nemotron_safety "$SP_MM"
run_one qwen3.5-9b verih "$SP_MM"
run_one qwen3.5-9b nemotron_safety "$SP_MM"
run_one minicpm4.1-8b verih "$SP_MM_CORE"
run_one minicpm4.1-8b nemotron_safety "$SP_MM_CORE"

echo "=== all runs finished at $(date '+%F %T') ==="
"$PY" latency/merge_latency.py
