#!/bin/bash
# Precompute hidden states for ONE model on ONE CoT task.
# Mirrors scripts/run_qwen.sh / scripts/run_minicpm.sh cot_* branches exactly
# (num_train 5000, num_test 400, max_length 8192, mid_layer 16), batch_size 4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# for throughput only.
# Usage: bash cot_local/run_precompute_one.sh <gpu> <model> <task>
#   model in {4b, 9b, mcpm}; task in {cot_repetitive, cot_branched}
set -euo pipefail
ROOT=/data1/suzeqin/SpeculativeProbing
cd "$ROOT"
PY=${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}
GPU=$1
MODEL=$2
TASK=$3
export CUDA_VISIBLE_DEVICES=$GPU

LABEL=$( [ "$TASK" = cot_repetitive ] && echo repetitive_thinking || echo branched_thinking )
DATA=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_${LABEL}_fixed_all.jsonl

case "$MODEL" in
  4b)
    $PY fast_softprompt/precompute_hidden_states_cot.py \
      --model models/Qwen3.5-4B --task "$TASK" --data "$DATA" \
      --output "cache_${TASK}_4b.pt" \
      --num_train 5000 --num_test 400 --max_length 8192 --batch_size 4 --mid_layer 16
    ;;
  9b)
    $PY fast_softprompt/precompute_hidden_states_cot.py \
      --model models/Qwen3.5-9B --task "$TASK" --data "$DATA" \
      --output "cache_${TASK}_9b.pt" \
      --num_train 5000 --num_test 400 --max_length 8192 --batch_size 4 --mid_layer 16
    ;;
  mcpm)
    $PY fast_softprompt/precompute_hidden_states_minicpm.py \
      --task "$TASK" --data "$DATA" \
      --base_model models/MiniCPM4.1-8B \
      --output_prefix "cache_${TASK}_mcpm" \
      --num_train 5000 --num_test 400 --max_length 8192 --batch_size 4
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac
echo "PRECOMPUTE_ONE_DONE $MODEL $TASK"
