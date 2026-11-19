#!/bin/bash
# Precompute hidden states for the WebInstruct-verified CoT tasks on 3 models.
# Mirrors scripts/run_qwen.sh / scripts/run_minicpm.sh exactly (num_train 5000,
# num_test 400, max_length 8192, mid_layer 16), with the labeled corpus produced
# by the locally deployed judge model instead of qwen3.5-plus.
#
# Usage: bash cot_local/run_precompute.sh <gpu> <which>
#   which in {4b, 9b, mcpm}
set -euo pipefail
ROOT=/data1/suzeqin/SpeculativeProbing
cd "$ROOT"
PY=${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}
GPU=$1
WHICH=$2

DATA_DIR=cot_inconsistency/data
CORPUS_GEN=Qwen3_5_122B_A10B_GPTQ_Int4   # slug of the local judge model name
REP_DATA=$DATA_DIR/generated_webinstruct_verified_train_${CORPUS_GEN}_labeled_repetitive_thinking_fixed_all.jsonl
BR_DATA=$DATA_DIR/generated_webinstruct_verified_train_${CORPUS_GEN}_labeled_branched_thinking_fixed_all.jsonl

PRECOMMON="--num_train 5000 --num_test 400 --max_length 8192 --batch_size 4"

case "$WHICH" in
  4b)
    export CUDA_VISIBLE_DEVICES=$GPU
    $PY fast_softprompt/precompute_hidden_states_cot.py \
      --model models/Qwen3.5-4B --task cot_repetitive --data "$REP_DATA" \
      --output cache_cot_repetitive_4b.pt $PRECOMMON --mid_layer 16
    $PY fast_softprompt/precompute_hidden_states_cot.py \
      --model models/Qwen3.5-4B --task cot_branched --data "$BR_DATA" \
      --output cache_cot_branched_4b.pt $PRECOMMON --mid_layer 16
    ;;
  9b)
    export CUDA_VISIBLE_DEVICES=$GPU
    $PY fast_softprompt/precompute_hidden_states_cot.py \
      --model models/Qwen3.5-9B --task cot_repetitive --data "$REP_DATA" \
      --output cache_cot_repetitive_9b.pt $PRECOMMON --mid_layer 16
    $PY fast_softprompt/precompute_hidden_states_cot.py \
      --model models/Qwen3.5-9B --task cot_branched --data "$BR_DATA" \
      --output cache_cot_branched_9b.pt $PRECOMMON --mid_layer 16
    ;;
  mcpm)
    export CUDA_VISIBLE_DEVICES=$GPU
    $PY fast_softprompt/precompute_hidden_states_minicpm.py \
      --task cot_repetitive --data "$REP_DATA" \
      --base_model models/MiniCPM4.1-8B \
      --output_prefix cache_cot_repetitive_mcpm $PRECOMMON
    $PY fast_softprompt/precompute_hidden_states_minicpm.py \
      --task cot_branched --data "$BR_DATA" \
      --base_model models/MiniCPM4.1-8B \
      --output_prefix cache_cot_branched_mcpm $PRECOMMON
    ;;
  *) echo "unknown which: $WHICH" >&2; exit 1 ;;
esac
echo "PRECOMPUTE_DONE $WHICH"
