#!/bin/bash
# Stop the generation client once the corpus reaches TARGET records.
# The corpus JSONL is resume-safe; the vLLM server (instance A) stays up for
# the final labeling pass.
set -u
cd /data1/suzeqin/SpeculativeProbing
CORPUS=cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl
TARGET=${1:-13000}

GENPID=$(pgrep -f "gen_traces.py --base-url" | head -1)
echo "watchdog: gen client pid=$GENPID target=$TARGET"
while true; do
  N=$(wc -l < "$CORPUS" 2>/dev/null || echo 0)
  if [ "$N" -ge "$TARGET" ]; then
    break
  fi
  sleep 300
done
if [ -n "$GENPID" ]; then
  kill "$GENPID" && echo "watchdog: killed gen client $GENPID at corpus=$N"
fi
touch cot_local/.gen_done
echo "watchdog: .gen_done written"
