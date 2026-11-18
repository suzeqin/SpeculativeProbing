#!/bin/bash
# Phase 2: wait for .gen_done (generation stopped), then finish labeling both
# tasks on instance A (port 8000). Writes .labels_done when both label files
# cover the whole corpus. Servers are left running for manual inspection.
set -u
cd /data1/suzeqin/SpeculativeProbing
CORPUS=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl
REP=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_repetitive_thinking.jsonl
BR=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_branched_thinking.jsonl

while [ ! -f cot_local/.gen_done ]; do sleep 120; done
echo "phase2: generation stopped at $(date '+%H:%M:%S'); corpus=$(wc -l < $CORPUS)"

bash cot_local/final_labeling.sh branched_thinking 100   >> cot_local/phase2.log 2>&1
bash cot_local/final_labeling.sh repetitive_thinking 100 >> cot_local/phase2.log 2>&1

NC=$(wc -l < "$CORPUS")
NB=$(wc -l < "$BR" 2>/dev/null || echo 0)
NR=$(wc -l < "$REP" 2>/dev/null || echo 0)
echo "phase2: corpus=$NC branched=$NB repetitive=$NR"
if [ "$NB" -ge "$NC" ] && [ "$NR" -ge "$NC" ]; then
  touch cot_local/.labels_done
  echo "phase2: .labels_done written"
else
  echo "phase2: WARNING - label coverage incomplete; second mop-up round"
  bash cot_local/final_labeling.sh branched_thinking 100   >> cot_local/phase2.log 2>&1
  bash cot_local/final_labeling.sh repetitive_thinking 100 >> cot_local/phase2.log 2>&1
  NB=$(wc -l < "$BR" 2>/dev/null || echo 0)
  NR=$(wc -l < "$REP" 2>/dev/null || echo 0)
  echo "phase2: after mop-up: corpus=$NC branched=$NB repetitive=$NR"
  if [ "$NB" -ge "$NC" ] && [ "$NR" -ge "$NC" ]; then
    touch cot_local/.labels_done
    echo "phase2: .labels_done written (after mop-up)"
  fi
fi
