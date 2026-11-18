#!/bin/bash
# Full clean-label rerun: regenerate caches from *_fixed_all.jsonl with the
# robust balanced sampler, then train all 6 (model, task) cells, then compile.
#
# Prerequisites (already applied):
#   * make_fixed_labels.py produced _labeled_<task>_fixed_all.jsonl
#     (keeps only exact single-word judge verdicts; drops truncated essays)
#   * precompute_hidden_states_cot.load_cot_data balances robustly
#   * eval_split.py keeps val/test stratified and balanced
#   * cache_io._load_bundle uses MAP_SHARED (large mmap caches, no ENOMEM)
#   * run_train.sh passes --mmap_cache to MultiMax and skips completed methods
#
# GPUs 0-1 belong to another user; 2-7 are ours.
set -u
cd /data1/suzeqin/SpeculativeProbing
PY=/home/suzeqin/miniconda3/envs/sp/bin/python

echo "rerun_all: START $(date '+%F %T')"

# Preserve the original (label-bug) numbers for the record.
if [ -f cot_local/RESULTS.txt ]; then
  cp -f cot_local/RESULTS.txt cot_local/RESULTS_original_buggy.txt
fi

# Discard results produced under the corrupted labels so the idempotent
# supervisor actually retrains every cell.
find results -maxdepth 3 -path "*cot_*" -name final_results.json -delete
echo "rerun_all: cleared old cot_* results ($(find results -maxdepth 3 -path '*cot_*' -name final_results.json | wc -l) left)"

# Old caches were built from the corrupted labels and are invalid. Removing
# them also avoids temporary .hidden.raw + stale .hidden doubling disk usage
# (mcpm eagle3 alone is ~600 GB).
before=$(df -BG --output=avail /data1 | tail -1 | tr -dc '0-9')
rm -f cache_cot_*.pt cache_cot_*.pt.hidden cache_cot_*.hidden.raw
after=$(df -BG --output=avail /data1 | tail -1 | tr -dc '0-9')
echo "rerun_all: cleared old cot caches (free ${before}GB -> ${after}GB)"

echo "=== PRECOMPUTE wave $(date '+%F %T') ==="
pids=()
specs=(
  "2 4b cot_repetitive" "3 4b cot_branched"
  "4 9b cot_repetitive" "5 9b cot_branched"
  "6 mcpm cot_repetitive" "7 mcpm cot_branched"
)
for u in "${specs[@]}"; do
  set -- $u
  bash cot_local/run_precompute_one.sh "$1" "$2" "$3" > "cot_local/repre_${2}_${3}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do
  wait "$p" && rc=0 || rc=$?
  [ "$rc" -eq 0 ] || echo "  precompute pid $p FAILED rc=$rc"
done
echo "=== PRECOMPUTE done $(date '+%F %T') ==="
ok=0
for f in cot_local/repre_*.log; do
  echo "  $f: $(tr '\r' '\n' < "$f" | grep -aE 'PRECOMPUTE_ONE_DONE|Traceback|Error' | tail -2 | tr '\n' ' ')"
  grep -aq "PRECOMPUTE_ONE_DONE" "$f" && ok=$((ok+1))
done
if [ "$ok" -ne 6 ]; then
  echo "rerun_all: only $ok/6 precompute jobs succeeded - ABORT before training"
  exit 1
fi

echo "=== TRAIN wave $(date '+%F %T') ==="
bash cot_local/rerun_training.sh > cot_local/retrain_main.log 2>&1
tail -40 cot_local/retrain_main.log

echo "=== FINAL RESULTS ==="
$PY cot_local/compile_results.py > cot_local/RESULTS.txt 2>&1
cat cot_local/RESULTS.txt
echo "rerun_all: COMPLETE $(date '+%F %T')"
