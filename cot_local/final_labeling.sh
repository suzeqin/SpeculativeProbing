#!/bin/bash
# Final labeling pass on instance A (port 8000, large KV pool) after the
# generation client has stopped (.gen_done marker present).
# The labeling client is resume-safe; several rounds mop up transient errors.
# Usage: bash cot_local/final_labeling.sh <task> [workers]
set -u
ROOT=/data1/suzeqin/SpeculativeProbing
cd "$ROOT"
PY=/home/suzeqin/miniconda3/envs/sp/bin/python
TASK=$1
WORKERS=${2:-100}
CORPUS=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl

for i in 1 2 3; do
  echo "--- final round $i for $TASK: $(date '+%H:%M:%S')"
  $PY cot_local/label_traces.py \
    --input "$CORPUS" \
    --task "$TASK" \
    --base-url "http://127.0.0.1:8000/v1" \
    --workers "$WORKERS" 2>&1 | tail -3
done
echo "FINAL_LABEL_DONE $TASK"
