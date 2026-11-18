#!/bin/bash
# Phase 3: after .labels_done - kill servers, precompute hidden states for all
# 6 (model, task) cells in parallel, then train SP-1/2/5 + MLP + MultiMax for
# each cell, then compile results.
# Sanity gates: corpus >= 12000 records, repetitive pos >= 1500, branched pos >= 1500.
set -u
cd /data1/suzeqin/SpeculativeProbing
PY=/home/suzeqin/miniconda3/envs/sp/bin/python
CORPUS=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl
REP=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_repetitive_thinking.jsonl
BR=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_branched_thinking.jsonl

while [ ! -f cot_local/.labels_done ]; do sleep 120; done
echo "phase3: labels done at $(date '+%H:%M:%S')"

NC=$(wc -l < "$CORPUS")
PR=$(grep -c '"label": 1' "$REP" || echo 0)
PB=$(grep -c '"label": 1' "$BR" || echo 0)
echo "phase3 gates: corpus=$NC rep_pos=$PR br_pos=$PB"
if [ "$NC" -lt 12000 ] || [ "$PR" -lt 1500 ] || [ "$PB" -lt 1500 ]; then
  echo "phase3: GATES FAILED - aborting, manual review required"
  exit 1
fi

echo "phase3: stopping vllm servers"
pkill -f "vllm serve models/Qwen3.5-122B" || true
sleep 15
pkill -9 -f "vllm serve models/Qwen3.5-122B" || true
sleep 10

echo "phase3: launching 6 precompute jobs"
bash cot_local/run_precompute_one.sh 2 4b  cot_repetitive > cot_local/pre_4b_rep.log 2>&1 &
bash cot_local/run_precompute_one.sh 3 4b  cot_branched   > cot_local/pre_4b_br.log 2>&1 &
bash cot_local/run_precompute_one.sh 4 9b  cot_repetitive > cot_local/pre_9b_rep.log 2>&1 &
bash cot_local/run_precompute_one.sh 5 9b  cot_branched   > cot_local/pre_9b_br.log 2>&1 &
bash cot_local/run_precompute_one.sh 6 mcpm cot_repetitive > cot_local/pre_mcpm_rep.log 2>&1 &
bash cot_local/run_precompute_one.sh 7 mcpm cot_branched   > cot_local/pre_mcpm_br.log 2>&1 &
wait
echo "phase3: all precompute done at $(date '+%H:%M:%S')"

echo "phase3: launching 6 training jobs"
bash cot_local/run_train.sh 2 4b  cot_repetitive > cot_local/train_4b_rep.log 2>&1 &
bash cot_local/run_train.sh 3 4b  cot_branched   > cot_local/train_4b_br.log 2>&1 &
bash cot_local/run_train.sh 4 9b  cot_repetitive > cot_local/train_9b_rep.log 2>&1 &
bash cot_local/run_train.sh 5 9b  cot_branched   > cot_local/train_9b_br.log 2>&1 &
bash cot_local/run_train.sh 6 mcpm cot_repetitive > cot_local/train_mcpm_rep.log 2>&1 &
bash cot_local/run_train.sh 7 mcpm cot_branched   > cot_local/train_mcpm_br.log 2>&1 &
wait
echo "phase3: all training done at $(date '+%H:%M:%S')"

$PY cot_local/compile_results.py > cot_local/RESULTS.txt 2>&1
cat cot_local/RESULTS.txt
echo "phase3: COMPLETE"
