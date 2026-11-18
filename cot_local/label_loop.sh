#!/bin/bash
# Continuously label newly generated traces (resume-safe) while generation runs.
# Usage: bash cot_local/label_loop.sh <task> <port> <workers>
set -u
ROOT=/data1/suzeqin/SpeculativeProbing
cd "$ROOT"
PY=/home/suzeqin/miniconda3/envs/sp/bin/python
TASK=$1
PORT=$2
WORKERS=$3
CORPUS=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl

while true; do
  # Stop when generation is finished (marker file written by the monitor).
  [ -f cot_local/.gen_done ] && break
  $PY cot_local/label_traces.py \
    --input "$CORPUS" \
    --task "$TASK" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --workers "$WORKERS" >> "cot_local/label_${TASK}_p${PORT}.log" 2>&1
  sleep 60
done
echo "LABEL_LOOP_EXIT $TASK port=$PORT"
