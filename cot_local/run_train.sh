#!/bin/bash
# Train SP-1/2/5 + MLP + MultiMax for one model on one CoT task.
# Mirrors scripts/run_qwen.sh / scripts/run_minicpm.sh (EPOCHS=4, TE=400).
#
# Usage: bash cot_local/run_train.sh <gpu> <model> <task>
#   model in {4b, 9b, mcpm}; task in {cot_repetitive, cot_branched}
set -euo pipefail
ROOT=/data1/suzeqin/SpeculativeProbing
cd "$ROOT"
PY=${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}
GPU=$1
MODEL=$2
TASK=$3
export CUDA_VISIBLE_DEVICES=$GPU

EPOCHS=4
TE=400

case "$MODEL" in
  4b)
    CACHE=cache_${TASK}_4b.pt
    MTP=models/Qwen3.5-4B-mtp
    OUT=results/${TASK}_4b
    ;;
  9b)
    CACHE=cache_${TASK}_9b.pt
    MTP=models/Qwen3.5-9B-mtp
    OUT=results/${TASK}_9b
    ;;
  mcpm)
    E3DIR=models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16
    CACHE=cache_${TASK}_mcpm_eagle3.pt
    OUT=results/${TASK}_mcpm
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac

# Idempotent: a method that already produced final_results.json is skipped, so
# a unit can be safely relaunched after an OOM kill by the supervisor.
run_if_missing() {
  local name="$1"; shift
  if [ -f "$OUT/$name/final_results.json" ]; then
    echo "SKIP $OUT/$name (already has final_results.json)"
  else
    echo "RUN  $OUT/$name"
    "$@"
  fi
}

if [ "$MODEL" = mcpm ]; then
  for K in 1 2 5; do
    run_if_missing "sp${K}" \
      $PY fast_softprompt/train_eagle3_minicpm.py \
        --cache "$CACHE" --mmap_cache --eagle3_dir "$E3DIR" \
        --base_model models/MiniCPM4.1-8B \
        --output_dir "$OUT/sp${K}" \
        --soft_prompt_length "$K" --epochs "$EPOCHS" --batch_size 1 --dropout 0.2 --target_eval "$TE"
  done
else
  for K in 1 2 5; do
    run_if_missing "sp${K}" \
      $PY fast_softprompt/train_softprompt.py \
        --cache "$CACHE" --mmap_cache --mtp_dir "$MTP" \
        --output_dir "$OUT/sp${K}" \
        --soft_prompt_length "$K" --recursive --classifier_head linear \
        --lr 3e-4 --min_lr 1e-4 --epochs "$EPOCHS" --batch_size 1 --dropout 0.2 --target_eval "$TE"
  done
fi

run_if_missing mlp \
  $PY fast_softprompt/train_mlp.py \
    --cache "cache_${TASK}_${MODEL}.pt" --mmap_cache --output_dir "$OUT/mlp" \
    --epochs 10 --batch_size 32 --lr 5e-4 --target_eval "$TE"

run_if_missing multimax \
  $PY fast_softprompt/train_multimax.py \
    --cache "cache_${TASK}_${MODEL}_mid.pt" --mmap_cache --output_dir "$OUT/multimax" \
    --num_heads 10 --mlp_width 100 --epochs 10 --batch_size 32 --lr 5e-4 --target_eval "$TE"

echo "TRAIN_DONE $MODEL $TASK"
