#!/bin/bash
# One-shot status snapshot for the clean-label WebInstruct CoT pipeline.
cd /data1/suzeqin/SpeculativeProbing
DATA=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4
date '+%F %T'
echo "corpus:        $(wc -l < ${DATA}.jsonl 2>/dev/null || echo 0)"
for t in repetitive_thinking branched_thinking; do
  p="${DATA}_labeled_${t}_fixed_all.jsonl"
  if [ -f "$p" ]; then
    total=$(wc -l < "$p")
    pos=$(grep -c '"label": 1' "$p" || true)
    echo "clean_${t}: $total (pos=$pos)"
  else
    echo "clean_${t}: MISSING"
  fi
done
echo "result cells:  $(find results -maxdepth 3 -path '*cot_*' -name final_results.json 2>/dev/null | wc -l)/30"
echo "--- cot_local/RESULTS.txt ---"
cat cot_local/RESULTS.txt 2>/dev/null || echo "(not generated yet)"
