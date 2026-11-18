#!/bin/bash
# Repair/resume phase3 training after the 2026-11-18 OOM kills.
#
# Fixes already applied:
#   * cache_io._load_bundle forces MAP_SHARED for mmap loads (MAP_PRIVATE was
#     charged against the ~282 GiB commit limit -> 608 GiB cache got ENOMEM).
#   * run_train.sh passes --mmap_cache to train_multimax.py (was loading the
#     whole 100-200 GiB mid cache into RAM) and skips methods that already have
#     a final_results.json (idempotent resume).
#
# Each of the 6 (model, task) units runs on its own GPU (2-7); GPUs 0-1 belong
# to another user. Units are relaunched in rounds so an OOM-killed unit can
# resume without redoing methods that already finished.
set -u
cd /data1/suzeqin/SpeculativeProbing
PY=/home/suzeqin/miniconda3/envs/sp/bin/python

UNITS=(
  "2 4b cot_repetitive"
  "3 4b cot_branched"
  "4 9b cot_repetitive"
  "5 9b cot_branched"
  "6 mcpm cot_repetitive"
  "7 mcpm cot_branched"
)

count_results() {
  find results -maxdepth 3 -path "*cot_*" -name final_results.json | wc -l
}

all_done() {
  for t in cot_repetitive cot_branched; do
    for m in 4b 9b mcpm; do
      for meth in sp1 sp2 sp5 mlp multimax; do
        [ -f "results/${t}_${m}/${meth}/final_results.json" ] || return 1
      done
    done
  done
  return 0
}

echo "rerun: start $(date '+%F %T'), existing results: $(count_results)/30"

for round in 1 2 3; do
  if all_done; then break; fi
  echo "=== round $round START $(date '+%F %T') ==="
  pids=()
  for u in "${UNITS[@]}"; do
    set -- $u
    gpu=$1; model=$2; task=$3
    log="cot_local/rerun_${model}_${task}.log"
    bash cot_local/run_train.sh "$gpu" "$model" "$task" >> "$log" 2>&1 &
    pids+=("$!")
    echo "  launched gpu=$gpu model=$model task=$task pid=${pids[-1]} log=$log"
  done
  for p in "${pids[@]}"; do
    wait "$p" && rc=0 || rc=$?
    [ "$rc" -eq 0 ] || echo "  pid $p exited with rc=$rc"
  done
  echo "=== round $round END $(date '+%F %T'), results: $(count_results)/30 ==="
done

if all_done; then
  $PY cot_local/compile_results.py > cot_local/RESULTS.txt 2>&1
  cat cot_local/RESULTS.txt
  echo "rerun: COMPLETE $(date '+%F %T')"
else
  echo "rerun: INCOMPLETE - missing cells remain, manual review required"
  for t in cot_repetitive cot_branched; do
    for m in 4b 9b mcpm; do
      for meth in sp1 sp2 sp5 mlp multimax; do
        [ -f "results/${t}_${m}/${meth}/final_results.json" ] || \
          echo "  MISSING results/${t}_${m}/${meth}"
      done
    done
  done
fi
