#!/bin/bash
cd /data1/suzeqin/SpeculativeProbing
while true; do
  C=$(wc -l < cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl 2>/dev/null || echo 0)
  L=$(cat cot_inconsistency/data/*labeled_repetitive*.jsonl 2>/dev/null | wc -l)
  B=$(cat cot_inconsistency/data/*labeled_branched*.jsonl 2>/dev/null | wc -l)
  echo "$(date '+%H:%M') corpus=$C rep=$L br=$B" >> cot_local/monitor.log
  sleep 600
done
