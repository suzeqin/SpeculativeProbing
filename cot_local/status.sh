#!/bin/bash
# One-shot pipeline status snapshot.
cd /data1/suzeqin/SpeculativeProbing
CORPUS=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl
REP=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_repetitive_thinking.jsonl
BR=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4_labeled_branched_thinking.jsonl
date '+%H:%M:%S'
echo "corpus:        $(wc -l < $CORPUS 2>/dev/null || echo 0)"
for f in "$REP:repetitive" "$BR:branched"; do
  p=${f%%:*}; n=${f##*:}
  if [ -f "$p" ]; then
    total=$(wc -l < "$p")
    pos=$(grep -c '"label": 1' "$p" || true)
    echo "labeled_$n: $total (pos=$pos)"
  else
    echo "labeled_$n: 0"
  fi
done
echo "gen client: $(grep -E 'Generated' cot_local/gen_traces.log | tail -1)"
echo "A(generation): $(grep -oE 'Avg generation throughput: [0-9.]+' cot_local/vllm_122b.log | tail -1)"
echo "B(labeling):   $(grep -oE 'Avg generation throughput: [0-9.]+' cot_local/vllm_122b_b.log | tail -1)"
